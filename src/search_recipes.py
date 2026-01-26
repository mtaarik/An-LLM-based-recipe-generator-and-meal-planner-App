import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import sys

from src.rag_system import preprocess_query2

# Get absolute path to the root of the project
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def search_recipes(query, top_k=5):
    """
    Recherche les recettes les plus similaires à une requête donnée.

    Args:
        query (str): La requête textuelle de l'utilisateur.
        top_k (int): Le nombre de recettes les plus similaires à retourner.

    Returns:
        pd.DataFrame: Un DataFrame contenant les recettes les plus similaires.
    """
    # Path-safe loading of metadata and FAISS index
    metadata_path = os.path.join(PROJECT_ROOT, "data", "embeddings", "recipe_metadata.csv")
    faiss_index_path = os.path.join(PROJECT_ROOT, "data", "embeddings", "recipe_index.faiss")

    df_metadata = pd.read_csv(metadata_path)
    index = faiss.read_index(faiss_index_path)

    embedding_model_name = 'all-MiniLM-L6-v2'
    query_model = SentenceTransformer(embedding_model_name)

    processed_query = preprocess_query2(query)
    print(f"Requête originale: '{query}'")
    print(f"Requête prétraitée: '{processed_query}'")

    # 1. Encode query
    query_embedding = query_model.encode([processed_query], convert_to_numpy=True, normalize_embeddings=True)

    # 2. Search using FAISS
    distances, indices = index.search(query_embedding, top_k)

    # 3. Retrieve matching recipes
    retrieved_recipes = df_metadata.iloc[indices[0]].copy()
    retrieved_recipes['faiss_distance'] = distances[0]
    retrieved_recipes = retrieved_recipes.sort_values(by='faiss_distance').reset_index(drop=True)

    return retrieved_recipes
