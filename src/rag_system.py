def preprocess_query2( query):
    query = query.lower().strip()
    
    expansions_to_add = set() # Utilise un set pour garantir l'unicité des termes

    expansions = {
        'tajine': ['tajine', 'tagine', 'moroccan', 'stew'],
        'tagine': ['tajine', 'tagine', 'moroccan', 'stew'], # Pourrait être redondant si 'tajine' et 'tagine' ont les mêmes expansions
        'couscous': ['couscous', 'moroccan', 'semolina', 'grain'],
        'marocain': ['moroccan', 'marocain', 'maghreb', 'north african'],
        'marocaine': ['moroccan', 'marocaine', 'maghreb', 'north african'],
        'kefta': ['kofta', 'spiced', 'meatball'],
        'pastilla': ['pastilla', 'pigeon', 'pie', 'filo', 'pastry'],
        'harira': ['harira', 'soup', 'lentil', 'chickpea'],
        'tanjia': ['tanjia', 'marrakchi', 'stew', 'lamb']
    }

    # Commence avec les mots de la requête originale
    current_terms = set(query.split())

    for term, expansion_list in expansions.items():
        if term in current_terms: # Vérifie si le terme original est dans la requête
            for exp_term in expansion_list:
                expansions_to_add.add(exp_term) # Ajoute chaque terme d'expansion au set

    # Combine la requête originale avec les termes d'expansion uniques
    # Filtre les termes qui sont déjà dans la requête originale pour éviter les doubles
    final_terms = list(current_terms) + [t for t in expansions_to_add if t not in current_terms]
    
    return " ".join(final_terms)