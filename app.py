import sys
import os
from dotenv import load_dotenv
# Add project root to Python path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from flask import Flask, request, jsonify, session, render_template, redirect, url_for
from flask_cors import CORS
import sqlite3
import hashlib
import secrets
from datetime import datetime
from src.rag_interface import generate_rag_answer

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", os.urandom(24)) # Generate a secure secret key
CORS(app, supports_credentials=True)

DATABASE = 'recipe_bot.db'

def init_db():
    """Initialize the database with required tables"""
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    
    # Users table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Messages table
    cur.execute('''
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    ''')
    
    conn.commit()
    conn.close()

def hash_password(password):
    """Hash password with salt"""
    return hashlib.sha256(password.encode()).hexdigest()

# Routes
@app.route('/')
def index():
    if 'username' not in session:
        return render_template('login.html')
    return render_template('chat.html', username=session['username'])

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    
    data = request.json
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    
    if not username or not email or not password:
        return jsonify({'error': 'All fields are required'}), 400
    
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    
    # Check if user already exists
    cur.execute("SELECT username FROM users WHERE username = ? OR email = ?", (username, email))
    if cur.fetchone():
        conn.close()
        return jsonify({'error': 'Username or email already exists'}), 400
    
    # Create new user
    password_hash = hash_password(password)
    try:
        cur.execute("INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)", 
                   (username, email, password_hash))
        conn.commit()
        conn.close()
        return jsonify({'success': True, 'message': 'Registration successful'}), 201
    except Exception as e:
        conn.close()
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')
    
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    
    password_hash = hash_password(password)
    cur.execute("SELECT username FROM users WHERE username = ? AND password_hash = ?", 
               (username, password_hash))
    user = cur.fetchone()
    conn.close()
    
    if user:
        session['username'] = username
        return jsonify({'success': True, 'message': 'Login successful'})
    else:
        return jsonify({'error': 'Invalid credentials'}), 401

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    data = request.json
    prompt = data.get("prompt")
    username = session.get("username")

    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        answer, source = generate_rag_answer(prompt, top_k=2)
        
        # Save the prompt/response to DB
        conn = sqlite3.connect(DATABASE)
        cur = conn.cursor()
        cur.execute("INSERT INTO messages (username, prompt, response) VALUES (?, ?, ?)", 
                   (username, prompt, answer))
        conn.commit()
        conn.close()

        return jsonify({"response": answer})
    except Exception as e:
        print("Error while calling RAG:", e)  # Show error in terminal
        return jsonify({"error": str(e)}), 500

@app.route('/history')
def get_history():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session.get("username")
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    
    cur.execute("""
        SELECT prompt, response, timestamp 
        FROM messages 
        WHERE username = ? 
        ORDER BY timestamp DESC 
        LIMIT 50
    """, (username,))
    
    messages = []
    for row in cur.fetchall():
        messages.append({
            'prompt': row[0],
            'response': row[1],
            'timestamp': row[2]
        })
    
    conn.close()
    return jsonify({'messages': messages})

@app.route('/profile')
def get_profile():
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    username = session.get("username")
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    
    cur.execute("SELECT username, email, created_at FROM users WHERE username = ?", (username,))
    user_data = cur.fetchone()
    
    if user_data:
        profile = {
            'username': user_data[0],
            'email': user_data[1],
            'member_since': user_data[2]
        }
        conn.close()
        return jsonify(profile)
    
    conn.close()
    return jsonify({'error': 'User not found'}), 404

if __name__ == '__main__':
    init_db()
    app.run(debug=True, host='0.0.0.0', port=5000)