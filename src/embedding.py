# src/embedding.py

import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os

# Get absolute path to the root of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def generate_embeddings_and_save_faiss(df, model_name='all-MiniLM-L6-v2', text_column='document', output_dir='data/embeddings'):
    """
    Génère les embeddings à partir d'une colonne texte d'un DataFrame et les sauvegarde dans un index FAISS + .npy.
    Les métadonnées (le DataFrame lui-même) sont sauvegardées dans un fichier CSV ou JSON séparé
    pour être facilement récupérées lors de la recherche.
    """
    # Convert relative path to absolute path
    output_dir = os.path.join(PROJECT_ROOT, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Charger le modèle
    model = SentenceTransformer(model_name)
    print(f"Modèle {model_name} chargé.")

    # Nettoyage des textes
    texts = df[text_column].apply(lambda x: str(x) if pd.notna(x) else "").tolist()

    # Génération des embeddings
    print(f"Génération des embeddings pour {len(texts)} documents...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    print(f"Embeddings générés. Forme : {embeddings.shape}")

    # Création de l'index FAISS
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Chemins complets
    faiss_index_path = os.path.join(output_dir, 'recipe_index.faiss')
    metadata_path = os.path.join(output_dir, 'recipe_metadata.csv')

    # Sauvegarde
    faiss.write_index(index, faiss_index_path)
    df.to_csv(metadata_path, index=False)

    print(f"Index FAISS sauvegardé dans : {faiss_index_path}")
    print(f"Métadonnées (DataFrame) sauvegardées dans : {metadata_path}")
