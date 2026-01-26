# 🍳 CulinaryRAG: Intelligent Recipe Generator & Meal Planner

[cite_start]An innovative web application that leverages a **Retrieval-Augmented Generation (RAG)** system paired with the **LLAMA 4** large language model[cite: 17, 470]. [cite_start]This project was developed as a 1st-year engineering project at **ENSIAS** to address the limitations of general AI models in handling the rich, specific cultural nuances of **Moroccan cuisine**[cite: 8, 17, 196].

---

## 🌟 Key Features

* [cite_start]**Ingredient-Based Generation**: Input available ingredients to receive customized recipe suggestions[cite: 44, 69].
* [cite_start]**Cultural Authenticity**: Specialized processing for Moroccan culinary terms (Darija) to provide reliable recipes for traditional dishes like Tajine, Couscous, and Harira[cite: 18, 301, 441].
* [cite_start]**Nutritional Accuracy**: Integrates exact nutritional information sourced from a verified dataset of over 230,000 recipes[cite: 52, 234].
* [cite_start]**Smart Meal Planning**: Generates personalized weekly meal plans, including breakfasts, dinners, and snacks[cite: 46, 77].
* [cite_start]**Step-by-Step Guidance**: Provides clear, concise cooking instructions and techniques for every generated meal[cite: 63, 79].
* [cite_start]**User Management**: Secure authentication system with persistent conversation history and user profiles[cite: 18, 389, 476].

---

## 🏗️ Technical Architecture

[cite_start]The system follows a modern RAG pipeline to minimize hallucinations and ensure cultural relevance[cite: 17, 203, 310].



* [cite_start]**Model**: **LLAMA 4** (via Groq Cloud API) for high-performance natural language generation[cite: 17, 307].
* [cite_start]**Embeddings**: `all-MiniLM-L6-v2` for lightweight, efficient semantic vectorization[cite: 207, 299].
* [cite_start]**Vector Store**: **FAISS** (Facebook AI Similarity Search) for high-speed local indexing and retrieval[cite: 153, 236].
* [cite_start]**Backend**: **Flask** (Python) with a **SQLite** database for user data and chat persistence[cite: 18, 318].
* [cite_start]**Query Expansion**: Custom linguistic filtering that expands Moroccan terms (e.g., "Tajine" ➔ "Tagine", "Moroccan", "Stew") to improve retrieval accuracy[cite: 301, 475].

---

## 📂 Project Structure

```text
📁 projet_recettes/
  📄 app.py                 # Flask Main Application
  📁 data/                  # Local Data (Ignored by Git)
    📁 embeddings/          # FAISS index and metadata
    📁 processed/           # Cleaned datasets
    📁 raw/                 # Original Kaggle & Moroccan datasets
  📁 notebooks/             # Data exploration & AI pipeline development
  📁 src/                   # Modular Python logic (RAG, Preprocessing, Search)
  📁 templates/             # Responsive HTML interfaces
  📄 recipe_bot.db          # SQLite Database for users/history
  📄 requirements.txt       # Python dependencies
  📄 .env                   # Private API Keys (Excluded from Git)
  📄 .gitignore             # Git exclusion rules

  🚀 Setup and Installation
1. Clone the Repository
Bash

git clone "link"
cd CulinaryRAG
2. Configure Environment Variables
Create a .env file in the root directory to store your credentials securely. Do not share this file.

Plaintext

GROQ_API_KEY=your_groq_api_key_here
FLASK_SECRET_KEY=your_random_secret_string
3. Install Dependencies
It is recommended to use a virtual environment:

Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

## 📊 Data Setup

Because of file size limits and security, the recipe datasets and FAISS indices are not included in this repository. Follow these steps to set up the data:

1. **Download Raw Data**: 
   - Download the [Food.com Recipes Dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions) and place `RAW_recipes.csv` into `data/raw/`.
   - Download your Moroccan Recipes and place them in the same folder. (https://www.kaggle.com/datasets/idrisskh/moroccan-recipes-dataset)
2. **Run the Pipeline**:
   - Run `notebooks/02_recipe_preprocessing.ipynb` to clean the data.
   - Run `notebooks/03_embedding_generation.ipynb` to create the local FAISS vector index.

   🎮 How to Run
Start the Flask server:

Bash

python app.py
Open your browser and navigate to http://127.0.0.1:5000 to interact with the Culinary AI.


👥 Authors

Mohamed Amine Tarik
Mohamed Oussama Rafiq 


Supervised by: Prof. H. Benbrahim 


Institution: ENSIAS - Université Mohammed V de Rabat