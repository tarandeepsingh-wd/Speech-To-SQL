from module1_asr.extractor import extract_ingredients

FLAVOR_MOLECULES = {"limonene", "capsaicin", "vanillin", "menthol", "cinnamaldehyde"}
DIETARY_MARKERS  = {"vegan", "vegetarian", "gluten-free", "keto", "diabetic", "low-sodium"}

def tag_entities(text: str) -> dict:
    tokens = text.lower().split()

    ingredient_matches = extract_ingredients(text)

    flavor  = []
    dietary = []
    for token in tokens:
        clean = token.strip(".,?!")
        if clean in FLAVOR_MOLECULES:
            flavor.append(clean)
        if clean in DIETARY_MARKERS:
            dietary.append(clean)

    return {
        "ingredients":      [m["canonical"] for m in ingredient_matches],
        "flavor_molecules": flavor,
        "dietary_markers":  dietary,
        "raw_spans":        ingredient_matches
    }