"""Flask web app: authentication, chat with conversation memory, history.

    python app.py            # http://127.0.0.1:5000

Environment (.env): GROQ_API_KEY, FLASK_SECRET_KEY, FLASK_DEBUG=0/1, HOST, PORT.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import threading
import uuid
from functools import wraps
from pathlib import Path

from flask import Flask, g, jsonify, redirect, render_template, request, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash

from recipe_rag.config import PROJECT_ROOT

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("app")

DATABASE = Path(os.getenv("DATABASE_PATH", PROJECT_ROOT / "recipe_bot.db"))
HISTORY_TURNS = 3  # previous (question, answer) pairs given to the assistant

app = Flask(__name__)
secret = os.getenv("FLASK_SECRET_KEY")
if not secret:
    log.warning("FLASK_SECRET_KEY not set: using a random key (sessions reset on restart)")
app.secret_key = secret or secrets.token_hex(32)
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")


# ----------------------------------------------------------------- database
def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db() -> None:
    with sqlite3.connect(DATABASE) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users (username)
            );
        """)
        # Lightweight migration of the v1 schema.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(messages)")}
        if "conversation_id" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN conversation_id TEXT")
        if "sources" not in cols:
            conn.execute("ALTER TABLE messages ADD COLUMN sources TEXT")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id)")


# ------------------------------------------------------------------ auth
def hash_password(password: str) -> str:
    return generate_password_hash(password)  # salted scrypt (werkzeug default)


def verify_password(stored: str, password: str) -> tuple[bool, bool]:
    """Returns (valid, needs_rehash). v1 stored unsalted SHA-256 hex digests;
    those still validate once and are then upgraded transparently."""
    if len(stored) == 64 and all(c in "0123456789abcdef" for c in stored):
        legacy = hashlib.sha256(password.encode()).hexdigest()
        return hmac.compare_digest(stored, legacy), True
    return check_password_hash(stored, password), False


def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "username" not in session:
            return jsonify({"error": "Not authenticated"}), 401
        return view(*args, **kwargs)
    return wrapped


# ------------------------------------------------------------- assistant
_assistant = None
_assistant_lock = threading.Lock()


def get_assistant():
    global _assistant
    if _assistant is None:
        with _assistant_lock:
            if _assistant is None:
                from recipe_rag.assistant import RecipeAssistant
                _assistant = RecipeAssistant.load()
    return _assistant


# ----------------------------------------------------------------- routes
@app.route("/")
def index():
    if "username" not in session:
        return render_template("login.html")
    return render_template("chat.html", username=session["username"])


@app.route("/register", methods=["POST"])
def register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""
    if not username or not email or not password:
        return jsonify({"error": "All fields are required"}), 400
    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400
    db = get_db()
    if db.execute("SELECT 1 FROM users WHERE username = ? OR email = ?", (username, email)).fetchone():
        return jsonify({"error": "Username or email already exists"}), 400
    db.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
               (username, email, hash_password(password)))
    db.commit()
    return jsonify({"success": True, "message": "Registration successful"}), 201


@app.route("/login", methods=["POST"])
def login():
    data = request.get_json(silent=True) or {}
    username, password = data.get("username"), data.get("password")
    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    db = get_db()
    row = db.execute("SELECT password_hash FROM users WHERE username = ?", (username,)).fetchone()
    valid, rehash = verify_password(row["password_hash"], password) if row else (False, False)
    if not valid:
        return jsonify({"error": "Invalid credentials"}), 401
    if rehash:
        db.execute("UPDATE users SET password_hash = ? WHERE username = ?",
                   (hash_password(password), username))
        db.commit()
    session.clear()
    session["username"] = username
    return jsonify({"success": True, "message": "Login successful"})


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("index"))


@app.route("/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json(silent=True) or {}
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400
    if len(prompt) > 2000:
        return jsonify({"error": "Message too long"}), 400
    conversation_id = data.get("conversation_id") or uuid.uuid4().hex
    username = session["username"]

    db = get_db()
    rows = db.execute(
        "SELECT prompt, response FROM messages WHERE username = ? AND conversation_id = ? "
        "ORDER BY id DESC LIMIT ?", (username, conversation_id, HISTORY_TURNS)).fetchall()
    history = []
    for r in reversed(rows):
        history += [{"role": "user", "content": r["prompt"]},
                    {"role": "assistant", "content": r["response"]}]

    try:
        result = get_assistant().answer(prompt, history)
    except Exception:
        log.exception("Assistant failed")
        return jsonify({"error": "The assistant failed to answer. Please try again."}), 500

    db.execute("INSERT INTO messages (username, prompt, response, conversation_id, sources) "
               "VALUES (?, ?, ?, ?, ?)",
               (username, prompt, result.answer, conversation_id, json.dumps(result.sources)))
    db.commit()
    return jsonify({"response": result.answer, "conversation_id": conversation_id,
                    **{k: v for k, v in result.to_dict().items() if k != "answer"}})


@app.route("/recipe/<recipe_id>")
@login_required
def recipe(recipe_id: str):
    recipes = get_assistant().engine.recipes
    match = recipes[recipes["recipe_id"] == recipe_id]
    if match.empty:
        return jsonify({"error": "Recipe not found"}), 404
    r = match.iloc[0]
    return jsonify({
        "recipe_id": r["recipe_id"], "name": r["name"], "description": r["description"],
        "source": r["source"], "ingredients": list(r["ingredients"]), "steps": list(r["steps"]),
        "minutes": None if r["minutes"] != r["minutes"] else int(r["minutes"]),
        "calories": None if r["calories"] != r["calories"] else int(r["calories"]),
    })


@app.route("/history")
@login_required
def get_history():
    rows = get_db().execute(
        "SELECT prompt, response, timestamp, conversation_id FROM messages WHERE username = ? "
        "ORDER BY id DESC LIMIT 50", (session["username"],)).fetchall()
    return jsonify({"messages": [dict(r) for r in rows]})


@app.route("/profile")
@login_required
def get_profile():
    row = get_db().execute("SELECT username, email, created_at FROM users WHERE username = ?",
                           (session["username"],)).fetchone()
    if not row:
        return jsonify({"error": "User not found"}), 404
    return jsonify({"username": row["username"], "email": row["email"],
                    "member_since": row["created_at"]})


@app.route("/health")
def health():
    a = _assistant
    return jsonify({"status": "ok", "assistant_loaded": a is not None,
                    "llm_available": bool(a and a.llm.available),
                    "reranker": bool(a and a.engine.reranker is not None)})


if __name__ == "__main__":
    init_db()
    get_assistant()  # load indexes/models before accepting requests
    app.run(host=os.getenv("HOST", "127.0.0.1"), port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG", "0") == "1", use_reloader=False)
