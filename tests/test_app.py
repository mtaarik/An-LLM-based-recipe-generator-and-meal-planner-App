import hashlib
import sqlite3

import pytest

import app as webapp
from recipe_rag.assistant import AssistantResponse


class FakeAssistant:
    def __init__(self):
        self.calls = []

    def answer(self, message, history=None):
        self.calls.append((message, history))
        return AssistantResponse(answer=f"echo: {message}", sources=[{"n": 1, "name": "Harira"}])


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(webapp, "DATABASE", tmp_path / "test.db")
    fake = FakeAssistant()
    monkeypatch.setattr(webapp, "_assistant", fake)
    webapp.init_db()
    webapp.app.config["TESTING"] = True
    with webapp.app.test_client() as c:
        c.fake = fake
        yield c


def register_and_login(c, user="amine", pw="s3cret-pass"):
    c.post("/register", json={"username": user, "email": f"{user}@x.ma", "password": pw})
    return c.post("/login", json={"username": user, "password": pw})


def test_chat_requires_login(client):
    assert client.post("/chat", json={"prompt": "harira"}).status_code == 401


def test_register_login_chat_and_history(client):
    assert register_and_login(client).status_code == 200
    r = client.post("/chat", json={"prompt": "recette de harira"}).get_json()
    assert r["response"] == "echo: recette de harira" and r["sources"][0]["name"] == "Harira"
    # follow-up in the same conversation receives the history
    client.post("/chat", json={"prompt": "et sans lentilles ?", "conversation_id": r["conversation_id"]})
    _, history = client.fake.calls[-1]
    assert history[0] == {"role": "user", "content": "recette de harira"}
    assert len(client.get("/history").get_json()["messages"]) == 2


def test_passwords_are_salted(client):
    register_and_login(client)
    with sqlite3.connect(webapp.DATABASE) as conn:
        stored = conn.execute("SELECT password_hash FROM users").fetchone()[0]
    assert stored.startswith("scrypt:") or stored.startswith("pbkdf2:")


def test_legacy_sha256_password_is_upgraded(client):
    legacy = hashlib.sha256(b"old-password").hexdigest()
    with sqlite3.connect(webapp.DATABASE) as conn:
        conn.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
                     ("v1user", "v1@x.ma", legacy))
    assert client.post("/login", json={"username": "v1user", "password": "wrong"}).status_code == 401
    assert client.post("/login", json={"username": "v1user", "password": "old-password"}).status_code == 200
    with sqlite3.connect(webapp.DATABASE) as conn:
        stored = conn.execute("SELECT password_hash FROM users WHERE username='v1user'").fetchone()[0]
    assert stored != legacy and not (len(stored) == 64)


def test_short_password_rejected(client):
    r = client.post("/register", json={"username": "a", "email": "a@x.ma", "password": "123"})
    assert r.status_code == 400
