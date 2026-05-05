# Mock entity lists (will be replaced with real DB values when access arrives)
INGREDIENTS = {"tomato", "onion", "garlic", "chicken", "rice", "cumin", "salt", "pepper", "turmeric", "ginger"}
FLAVOR_MOLECULES = {"limonene", "capsaicin", "vanillin", "menthol", "cinnamaldehyde"}
DIETARY_MARKERS = {"vegan", "vegetarian", "gluten-free", "keto", "diabetic", "low-sodium"}

def tag_entities(text: str) -> dict:
    """Extract food domain entities from transcribed text using BIO tagging."""
    tokens = text.lower().split()
    found = {
        "ingredients": [],
        "flavor_molecules": [],
        "dietary_markers": []
    }

    for token in tokens:
        clean = token.strip(".,?!")
        if clean in INGREDIENTS:
            found["ingredients"].append(clean)
        elif clean in FLAVOR_MOLECULES:
            found["flavor_molecules"].append(clean)
        elif clean in DIETARY_MARKERS:
            found["dietary_markers"].append(clean)

    return found