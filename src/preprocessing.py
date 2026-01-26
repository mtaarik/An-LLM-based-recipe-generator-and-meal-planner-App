import pandas as pd
import ast
import re
import os

# Define project root to resolve relative paths safely
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def load_and_clean_raw_recipes(file_path):
    """
    Charge le dataset RAW_recipes.csv, effectue le nettoyage initial,
    et prépare les colonnes pour la construction du document.
    """
    file_path = os.path.join(PROJECT_ROOT, file_path)  # Convert to absolute path
    df = pd.read_csv(file_path)

    df.dropna(subset=["ingredients", "steps", "name"], inplace=True)
    df["ingredients"] = df["ingredients"].apply(ast.literal_eval)
    df["steps"] = df["steps"].apply(ast.literal_eval)
    df["tags"] = df["tags"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else [])

    columns_to_drop = ["contributor_id", "submitted"]
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    df["ingredients_text"] = df["ingredients"].apply(lambda x: "\n".join(x))
    df["steps_text"] = df["steps"].apply(lambda x: "\n".join([f"{i+1}. {step}" for i, step in enumerate(x)]))

    return df

def build_recipe_document(row):
    return f"""
    Title: {row['name']}
    Tags: {', '.join(row['tags'])}
    Ingredients:
    {row['ingredients_text']}

    Instructions:
    {row['steps_text']}
    """

def process_raw_recipes_pipeline(input_file_path):
    df_processed = load_and_clean_raw_recipes(input_file_path)
    df_processed["document"] = df_processed.apply(build_recipe_document, axis=1)
    return df_processed

def parse_moroccan_recipe_text(recipe_text_content):
    name = "Nom de recette inconnu"
    ingredients = []
    steps = []
    tags = ["marocaine", "north african"]

    if not isinstance(recipe_text_content, str):
        return {'name': name, 'ingredients': [], 'steps': [], 'tags': tags}

    name_match = re.search(r"Recipe Name\s*:\s*(.+?)(?:\n|$)", recipe_text_content)
    if name_match:
        name = name_match.group(1).strip()

    ingredients_match = re.search(r"Ingredients\s*:\s*(\[.*?\])", recipe_text_content, re.DOTALL)
    if ingredients_match:
        try:
            ing_list_str = ingredients_match.group(1).replace('\xa0', ' ')
            ingredients = ast.literal_eval(ing_list_str)
            ingredients = [item.strip() for item in ingredients if item.strip()]
        except (ValueError, SyntaxError):
            ingredients = []

    steps_match = re.search(r"Steps for preparation\s*:\s*(\[.*?\]|nan)", recipe_text_content, re.DOTALL)
    if steps_match:
        steps_str = steps_match.group(1).strip()
        if steps_str.lower() != 'nan':
            try:
                steps = ast.literal_eval(steps_str.replace('\xa0', ' '))
                steps = [re.sub(r'^\d+\.\s*', '', step).strip() for step in steps if re.sub(r'^\d+\.\s*', '', step).strip()]
            except (ValueError, SyntaxError):
                steps = []

    return {
        'name': name,
        'ingredients': ingredients,
        'steps': steps,
        'tags': list(set(tags))
    }

def load_and_process_moroccan_recipes(file_path, recipe_text_column='prompt'):
    file_path = os.path.join(PROJECT_ROOT, file_path)  # Convert to absolute path
    df_moroccan_raw = pd.read_csv(file_path)

    parsed_data = df_moroccan_raw[recipe_text_column].apply(parse_moroccan_recipe_text).tolist()
    df_moroccan_processed = pd.DataFrame(parsed_data)

    for col in ['ingredients', 'steps', 'tags']:
        df_moroccan_processed[col] = df_moroccan_processed[col].apply(lambda x: x if isinstance(x, list) else [])

    df_moroccan_processed["ingredients_text"] = df_moroccan_processed["ingredients"].apply(lambda x: "\n".join(x))
    df_moroccan_processed["steps_text"] = df_moroccan_processed["steps"].apply(lambda x: "\n".join([f"{i+1}. {step}" for i, step in enumerate(x)]))

    return df_moroccan_processed
