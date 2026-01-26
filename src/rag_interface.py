# %%
import os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
from src.search_recipes import search_recipes
import pandas as pd
from groq import Groq
import groq
from dotenv import load_dotenv
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
dotenv_path = os.path.join(PROJECT_ROOT, '.env')
load_dotenv(dotenv_path)
api_key = os.getenv("GROQ_API_KEY")
client = groq.Client(api_key=api_key)

def generate_rag_answer(query, top_k=5):

    llm_model_name = 'meta-llama/llama-4-scout-17b-16e-instruct'
    

    """
    Exécute un pipeline RAG :
    1. Récupère les recettes pertinentes à l'aide de FAISS.
    2. Construit un prompt pour le LLM en utilisant la requête et les recettes récupérées.
    3. Appelle le LLM pour générer une réponse.

    Args:
        query (str): La question de l'utilisateur.
        top_k (int): Le nombre de documents à récupérer.

    Returns:
        tuple: (Réponse du LLM, pd.DataFrame des documents sources)
    """
    print(f"\n--- Processus RAG pour la question : '{query}' ---")
    print("1. Récupération des recettes pertinentes...")
    # Récupération des documents (utilise la fonction search_recipes définie plus haut)
    retrieved_docs_df = search_recipes(query, top_k=top_k)

    if retrieved_docs_df.empty:
        print("Aucune recette pertinente trouvée.")
        return "Je n'ai pas d'information suffisante dans les recettes disponibles pour répondre à cette question.", pd.DataFrame()

    # 2. Construction du Contexte pour le LLM
    context_parts = []
    for idx, row in retrieved_docs_df.iterrows():
        # Utilise la colonne 'document' qui contient le texte prétraité de la recette
        context_parts.append(f"Recette {idx+1} (Nom: {row['name']}):\n{row['document']}")

    # Concaténer tous les documents pertinents en un seul contexte
    context = "\n\n".join(context_parts)

    print(f"2. Contexte construit à partir de {len(retrieved_docs_df)} recettes.")

    # 3. Construction du Prompt pour le LLM
    # Un prompt structuré pour guider le LLM.
    # Il est crucial d'instruire le LLM à n'utiliser que le contexte fourni.
    prompt_template = f"""
You are a helpful culinary assistant who answers cooking-related questions using recipe information as your primary reference.

Answer the question (the Question below that a user has writen to you . it's the only output tha he has given you this whole prompt and context it's just me )naturally and completely, without referencing that you're using a context or dataset. Combine and synthesize useful details from the recipes provided to give a clear, informative, and engaging response.

If the recipes do not fully address the question, you may use your general culinary knowledge to provide the best possible answer(don't mention that you have used your general knowledge) — but always prioritize and integrate any useful information found in the recipes.

Do not mention recipe numbers, IDs, or refer to the recipes explicitly. Just answer the question as if you're drawing from your own expertise.

Recipe Context:
{context}

Question: {query}

Answer:
"""





    # NOUVEAU : Afficher le prompt complet avant l'appel au LLM
    print("\n" + "="*20 + " PROMPT COMPLET ENVOYÉ AU LLM " + "="*20)
    print(prompt_template)
    print("="*20 + " FIN DU PROMPT " + "="*20 + "\n")

    # 4. Appel au LLM de GroqCloud
    print("3. Appel au modèle de langage (LLM) de GroqCloud pour générer la réponse...")
    try:
        response = client.chat.completions.create(
            model=llm_model_name,
            messages=[
                {"role": "system", "content": "Tu es un assistant culinaire utile qui répond aux questions sur les recettes marocaines, en utilisant uniquement le contexte fourni."},
                {"role": "user", "content": prompt_template}
            ],
            temperature=0.9, # Température basse pour des réponses plus factuelles
            max_tokens=500 # Limiter la longueur de la réponse
        )
        llm_answer = response.choices[0].message.content
        print("4. Réponse du LLM reçue.")
        # NOUVEAU : Afficher la réponse complète du LLM
        print("\n" + "="*20 + " RÉPONSE COMPLÈTE DU LLM " + "="*20)
        print(llm_answer)
        print("="*20 + " FIN DE LA RÉPONSE " + "="*20 + "\n")

        return llm_answer, retrieved_docs_df
    except Exception as e:
        print(f"Erreur lors de l'appel au LLM de GroqCloud : {e}")
        return "Désolé, une erreur est survenue lors de la génération de la réponse. Veuillez réessayer.", pd.DataFrame()
