from __future__ import annotations

import sqlite3
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
SCHEMA_PATH = PROJECT_ROOT / "config" / "schema.sql"
DB_PATH = THIS_DIR / "recipe_test.db"
TOTAL_RECIPE_ROWS = 100
SQLITE_DUPLICATE_COLUMNS: set[tuple[str, str]] = set()
TABLE_COLUMN_CACHE: dict[str, list[tuple[str, str]]] = {}

BASE_RECIPE_INGREDIENTS = {
    1001: ["chicken", "spinach", "garlic", "olive oil"],
    1002: ["spinach", "lemon juice", "olive oil", "garlic"],
    1003: ["salmon", "garlic", "lemon juice"],
    1004: ["tomato", "garlic", "pasta", "basil"],
    1005: ["lentil", "carrot", "onion", "turmeric"],
    1006: ["sugar", "water", "lemon juice"],
}

INGREDIENT_META = {
    "chicken": ("ing001", 5001, "Animal Product", "poultry"),
    "spinach": ("ing002", 5002, "Vegetable", "leafy green"),
    "garlic": ("ing003", 5003, "Vegetable", "allium"),
    "olive oil": ("ing004", 5004, "Oil", "oil"),
    "lemon juice": ("ing005", 5005, "Fruit", "citrus"),
    "salmon": ("ing006", 5006, "Seafood", "fish"),
    "tomato": ("ing007", 5007, "Vegetable", "nightshade"),
    "pasta": ("ing008", 5008, "Cereal", "grain"),
    "basil": ("ing009", 5009, "Herb", "herb"),
    "lentil": ("ing010", 5010, "Legume", "pulse"),
    "carrot": ("ing011", 5011, "Vegetable", "root vegetable"),
    "onion": ("ing012", 5012, "Vegetable", "allium"),
    "turmeric": ("ing013", 5013, "Spice", "spice"),
    "sugar": ("ing014", 5014, "Sweetener", "sweetener"),
    "water": ("ing015", 5015, "Beverage", "liquid"),
}

GENERATED_RECIPE_BLUEPRINTS = [
    {
        "title": "Garlic Chicken Spinach Bowl",
        "ingredients": ["chicken", "spinach", "garlic", "olive oil"],
        "region": "Indian Subcontinent",
        "sub_region": "Indian",
        "continent": "Asian",
        "processes": "chop saute simmer",
        "utensils": "skillet,bowl,knife",
        "vegan": "false",
        "pescetarian": "false",
        "ovo_vegetarian": "false",
        "lacto_vegetarian": "false",
        "ovo_lacto_vegetarian": "false",
        "calories": 410,
        "protein": 40,
        "carbs": 30,
        "fat": 14,
        "fiber": 7,
        "sodium": 510,
        "sugar": 5,
        "cholesterol": 82,
        "magnesium": 88,
        "folate": 125,
        "thiamin": 0.42,
        "vitamin_c": 24,
        "vitamin_k": 88,
        "water": 245,
    },
    {
        "title": "Vegan Spinach Lemon Salad",
        "ingredients": ["spinach", "lemon juice", "olive oil", "garlic"],
        "region": "Mediterranean",
        "sub_region": "Mediterranean",
        "continent": "European",
        "processes": "chop mix toss",
        "utensils": "bowl,knife",
        "vegan": "true",
        "pescetarian": "false",
        "ovo_vegetarian": "false",
        "lacto_vegetarian": "false",
        "ovo_lacto_vegetarian": "true",
        "calories": 255,
        "protein": 8,
        "carbs": 20,
        "fat": 16,
        "fiber": 9,
        "sodium": 170,
        "sugar": 4,
        "cholesterol": 0,
        "magnesium": 145,
        "folate": 220,
        "thiamin": 0.22,
        "vitamin_c": 36,
        "vitamin_k": 122,
        "water": 305,
    },
    {
        "title": "Garlic Lemon Salmon",
        "ingredients": ["salmon", "garlic", "lemon juice", "olive oil"],
        "region": "American",
        "sub_region": "American",
        "continent": "North American",
        "processes": "bake roast",
        "utensils": "oven,pan",
        "vegan": "false",
        "pescetarian": "true",
        "ovo_vegetarian": "false",
        "lacto_vegetarian": "false",
        "ovo_lacto_vegetarian": "false",
        "calories": 380,
        "protein": 35,
        "carbs": 6,
        "fat": 24,
        "fiber": 1,
        "sodium": 430,
        "sugar": 2,
        "cholesterol": 68,
        "magnesium": 62,
        "folate": 28,
        "thiamin": 0.16,
        "vitamin_c": 14,
        "vitamin_k": 5,
        "water": 195,
    },
    {
        "title": "Tomato Basil Pasta",
        "ingredients": ["tomato", "garlic", "pasta", "basil"],
        "region": "Italian",
        "sub_region": "Italian",
        "continent": "European",
        "processes": "boil simmer saute",
        "utensils": "pot,skillet",
        "vegan": "false",
        "pescetarian": "false",
        "ovo_vegetarian": "false",
        "lacto_vegetarian": "true",
        "ovo_lacto_vegetarian": "true",
        "calories": 535,
        "protein": 17,
        "carbs": 70,
        "fat": 17,
        "fiber": 6,
        "sodium": 600,
        "sugar": 9,
        "cholesterol": 8,
        "magnesium": 58,
        "folate": 85,
        "thiamin": 0.34,
        "vitamin_c": 19,
        "vitamin_k": 21,
        "water": 255,
    },
    {
        "title": "Thiamine Lentil Turmeric Stew",
        "ingredients": ["lentil", "carrot", "onion", "turmeric"],
        "region": "Indian Subcontinent",
        "sub_region": "Indian",
        "continent": "Asian",
        "processes": "boil simmer",
        "utensils": "pot,bowl",
        "vegan": "true",
        "pescetarian": "false",
        "ovo_vegetarian": "false",
        "lacto_vegetarian": "false",
        "ovo_lacto_vegetarian": "true",
        "calories": 305,
        "protein": 18,
        "carbs": 43,
        "fat": 6,
        "fiber": 14,
        "sodium": 380,
        "sugar": 6,
        "cholesterol": 0,
        "magnesium": 118,
        "folate": 245,
        "thiamin": 0.76,
        "vitamin_c": 17,
        "vitamin_k": 18,
        "water": 345,
    },
    {
        "title": "Sugar Water Lemonade",
        "ingredients": ["sugar", "water", "lemon juice"],
        "region": "American",
        "sub_region": "American",
        "continent": "North American",
        "processes": "mix stir",
        "utensils": "pitcher,spoon",
        "vegan": "true",
        "pescetarian": "false",
        "ovo_vegetarian": "false",
        "lacto_vegetarian": "false",
        "ovo_lacto_vegetarian": "true",
        "calories": 155,
        "protein": 0,
        "carbs": 41,
        "fat": 0,
        "fiber": 0,
        "sodium": 5,
        "sugar": 37,
        "cholesterol": 0,
        "magnesium": 4,
        "folate": 7,
        "thiamin": 0,
        "vitamin_c": 28,
        "vitamin_k": 0,
        "water": 420,
    },
    {
        "title": "Carrot Lentil Soup",
        "ingredients": ["lentil", "carrot", "onion", "garlic"],
        "region": "Middle Eastern",
        "sub_region": "Lebanese",
        "continent": "Asian",
        "processes": "boil simmer blend",
        "utensils": "pot,bowl,blender",
        "vegan": "true",
        "pescetarian": "false",
        "ovo_vegetarian": "false",
        "lacto_vegetarian": "false",
        "ovo_lacto_vegetarian": "true",
        "calories": 285,
        "protein": 16,
        "carbs": 40,
        "fat": 5,
        "fiber": 12,
        "sodium": 360,
        "sugar": 7,
        "cholesterol": 0,
        "magnesium": 105,
        "folate": 215,
        "thiamin": 0.62,
        "vitamin_c": 11,
        "vitamin_k": 15,
        "water": 360,
    },
    {
        "title": "Chicken Tomato Skillet",
        "ingredients": ["chicken", "tomato", "garlic", "basil"],
        "region": "American",
        "sub_region": "American",
        "continent": "North American",
        "processes": "chop saute simmer",
        "utensils": "skillet,knife",
        "vegan": "false",
        "pescetarian": "false",
        "ovo_vegetarian": "false",
        "lacto_vegetarian": "false",
        "ovo_lacto_vegetarian": "false",
        "calories": 365,
        "protein": 38,
        "carbs": 18,
        "fat": 13,
        "fiber": 4,
        "sodium": 470,
        "sugar": 8,
        "cholesterol": 78,
        "magnesium": 66,
        "folate": 70,
        "thiamin": 0.3,
        "vitamin_c": 25,
        "vitamin_k": 30,
        "water": 285,
    },
    {
        "title": "Spinach Garlic Pasta Bowl",
        "ingredients": ["pasta", "spinach", "garlic", "basil"],
        "region": "Italian",
        "sub_region": "Italian",
        "continent": "European",
        "processes": "boil saute toss",
        "utensils": "pot,skillet,bowl",
        "vegan": "false",
        "pescetarian": "false",
        "ovo_vegetarian": "false",
        "lacto_vegetarian": "true",
        "ovo_lacto_vegetarian": "true",
        "calories": 480,
        "protein": 20,
        "carbs": 64,
        "fat": 15,
        "fiber": 8,
        "sodium": 520,
        "sugar": 6,
        "cholesterol": 5,
        "magnesium": 95,
        "folate": 190,
        "thiamin": 0.45,
        "vitamin_c": 24,
        "vitamin_k": 95,
        "water": 270,
    },
    {
        "title": "Pescetarian Tomato Salmon",
        "ingredients": ["salmon", "tomato", "garlic", "basil"],
        "region": "Mediterranean",
        "sub_region": "Mediterranean",
        "continent": "European",
        "processes": "bake simmer",
        "utensils": "oven,pan",
        "vegan": "false",
        "pescetarian": "true",
        "ovo_vegetarian": "false",
        "lacto_vegetarian": "false",
        "ovo_lacto_vegetarian": "false",
        "calories": 405,
        "protein": 34,
        "carbs": 14,
        "fat": 25,
        "fiber": 3,
        "sodium": 440,
        "sugar": 7,
        "cholesterol": 72,
        "magnesium": 70,
        "folate": 58,
        "thiamin": 0.2,
        "vitamin_c": 22,
        "vitamin_k": 24,
        "water": 220,
    },
]


def quote_identifier(name: str) -> str:
    return "`" + name.replace("`", "``") + "`"


def table_columns(conn: sqlite3.Connection, table: str) -> list[tuple[str, str]]:
    if table not in TABLE_COLUMN_CACHE:
        TABLE_COLUMN_CACHE[table] = [
            (str(row[1]), str(row[2]).upper())
            for row in conn.execute(f"PRAGMA table_info({quote_identifier(table)})")
        ]
    return TABLE_COLUMN_CACHE[table]


def default_value(column_type: str) -> object:
    if any(token in column_type for token in ("INT", "DOUBLE", "REAL", "NUMERIC", "FLOAT")):
        return 0
    return ""


def insert_row(conn: sqlite3.Connection, table: str, row: dict[str, object]) -> None:
    table_meta = table_columns(conn, table)
    columns = [column for column, _column_type in table_meta] or list(row)
    lowered_row = {key.lower(): value for key, value in row.items()}
    complete_row = {
        column: row[column]
        if column in row
        else lowered_row.get(column.lower(), default_value(column_type))
        for column, column_type in table_meta
    } if table_meta else row
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(quote_identifier(column) for column in columns)
    sql = f"INSERT INTO {quote_identifier(table)} ({column_sql}) VALUES ({placeholders})"
    conn.execute(sql, [complete_row[column] for column in columns])


def sqlite_schema_text() -> str:
    lines: list[str] = []
    current_table: str | None = None

    for line in SCHEMA_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("CREATE TABLE `"):
            current_table = stripped.split("`", 2)[1]
        elif current_table and stripped == ");":
            current_table = None

        if current_table and stripped.startswith("`"):
            column = stripped.split("`", 2)[1]
            if (current_table, column) in SQLITE_DUPLICATE_COLUMNS:
                continue
        lines.append(line)

    return "\n".join(lines)


def seed_recipes(conn: sqlite3.Connection) -> None:
    recipes = [
        {
            "_id": "r1001",
            "recipe_id": 1001,
            "recipe_index": 1,
            "recipe_title": "Vitamin K Chicken Spinach Bowl",
            "calories": 420,
            "cook_time": 20,
            "prep_time": 15,
            "total_time": 35,
            "servings": 2,
            "url": "https://example.test/recipes/vitamin-k-chicken",
            "region": "Indian Subcontinent",
            "sub_region": "Indian",
            "continent": "Asian",
            "source": "temp_fixture",
            "img_url": "https://example.test/images/chicken.jpg",
            "processes": "chop saute simmer",
            "utensils": "skillet,bowl,knife",
            "vegan": "false",
            "pescetarian": "false",
            "ovo_vegetarian": "false",
            "lacto_vegetarian": "false",
            "ovo_lacto_vegetarian": "false",
            "protein (g)": 42,
            "carbohydrate, by difference (g)": 35,
            "total lipid (fat) (g)": 14,
            "fiber, total dietary (g)": 7,
            "sodium, na (mg)": 520,
            "sugars, total (g)": 5,
            "cholesterol (mg)": 85,
            "energy (kcal)": 420,
            "magnesium, mg (mg)": 90,
            "folate, total (g)": 120,
            "thiamin (mg)": 0.4,
            "vitamin c, total ascorbic acid (mg)": 22,
            "vitamin k (phylloquinone) (g)": 85,
            "water (g)": 240,
        },
        {
            "_id": "r1002",
            "recipe_id": 1002,
            "recipe_index": 2,
            "recipe_title": "Vegan Magnesium Spinach Salad",
            "calories": 260,
            "cook_time": 0,
            "prep_time": 12,
            "total_time": 12,
            "servings": 2,
            "url": "https://example.test/recipes/spinach-salad",
            "region": "Mediterranean",
            "sub_region": "Mediterranean",
            "continent": "European",
            "source": "temp_fixture",
            "img_url": "https://example.test/images/salad.jpg",
            "processes": "chop mix toss",
            "utensils": "bowl,knife",
            "vegan": "true",
            "pescetarian": "false",
            "ovo_vegetarian": "false",
            "lacto_vegetarian": "false",
            "ovo_lacto_vegetarian": "true",
            "protein (g)": 8,
            "carbohydrate, by difference (g)": 22,
            "total lipid (fat) (g)": 16,
            "fiber, total dietary (g)": 9,
            "sodium, na (mg)": 180,
            "sugars, total (g)": 4,
            "cholesterol (mg)": 0,
            "energy (kcal)": 260,
            "magnesium, mg (mg)": 140,
            "folate, total (g)": 210,
            "thiamin (mg)": 0.2,
            "vitamin c, total ascorbic acid (mg)": 35,
            "vitamin k (phylloquinone) (g)": 120,
            "water (g)": 300,
        },
        {
            "_id": "r1003",
            "recipe_id": 1003,
            "recipe_index": 3,
            "recipe_title": "Low Carb Garlic Salmon",
            "calories": 390,
            "cook_time": 18,
            "prep_time": 8,
            "total_time": 26,
            "servings": 2,
            "url": "https://example.test/recipes/garlic-salmon",
            "region": "American",
            "sub_region": "American",
            "continent": "North American",
            "source": "temp_fixture",
            "img_url": "https://example.test/images/salmon.jpg",
            "processes": "bake roast",
            "utensils": "oven,pan",
            "vegan": "false",
            "pescetarian": "true",
            "ovo_vegetarian": "false",
            "lacto_vegetarian": "false",
            "ovo_lacto_vegetarian": "false",
            "protein (g)": 36,
            "carbohydrate, by difference (g)": 5,
            "total lipid (fat) (g)": 24,
            "fiber, total dietary (g)": 1,
            "sodium, na (mg)": 450,
            "sugars, total (g)": 1,
            "cholesterol (mg)": 70,
            "energy (kcal)": 390,
            "magnesium, mg (mg)": 65,
            "folate, total (g)": 25,
            "thiamin (mg)": 0.15,
            "vitamin c, total ascorbic acid (mg)": 12,
            "vitamin k (phylloquinone) (g)": 4,
            "water (g)": 190,
        },
        {
            "_id": "r1004",
            "recipe_id": 1004,
            "recipe_index": 4,
            "recipe_title": "Tomato Garlic Pasta",
            "calories": 550,
            "cook_time": 25,
            "prep_time": 10,
            "total_time": 35,
            "servings": 3,
            "url": "https://example.test/recipes/tomato-pasta",
            "region": "Italian",
            "sub_region": "Italian",
            "continent": "European",
            "source": "temp_fixture",
            "img_url": "https://example.test/images/pasta.jpg",
            "processes": "boil simmer saute",
            "utensils": "pot,skillet",
            "vegan": "false",
            "pescetarian": "false",
            "ovo_vegetarian": "false",
            "lacto_vegetarian": "true",
            "ovo_lacto_vegetarian": "true",
            "protein (g)": 18,
            "carbohydrate, by difference (g)": 72,
            "total lipid (fat) (g)": 18,
            "fiber, total dietary (g)": 6,
            "sodium, na (mg)": 610,
            "sugars, total (g)": 10,
            "cholesterol (mg)": 10,
            "energy (kcal)": 550,
            "magnesium, mg (mg)": 55,
            "folate, total (g)": 80,
            "thiamin (mg)": 0.35,
            "vitamin c, total ascorbic acid (mg)": 18,
            "vitamin k (phylloquinone) (g)": 20,
            "water (g)": 260,
        },
        {
            "_id": "r1005",
            "recipe_id": 1005,
            "recipe_index": 5,
            "recipe_title": "Thiamine Lentil Stew",
            "calories": 310,
            "cook_time": 35,
            "prep_time": 12,
            "total_time": 47,
            "servings": 4,
            "url": "https://example.test/recipes/lentil-stew",
            "region": "Indian Subcontinent",
            "sub_region": "Indian",
            "continent": "Asian",
            "source": "temp_fixture",
            "img_url": "https://example.test/images/lentil.jpg",
            "processes": "boil simmer",
            "utensils": "pot,bowl",
            "vegan": "true",
            "pescetarian": "false",
            "ovo_vegetarian": "false",
            "lacto_vegetarian": "false",
            "ovo_lacto_vegetarian": "true",
            "protein (g)": 18,
            "carbohydrate, by difference (g)": 44,
            "total lipid (fat) (g)": 6,
            "fiber, total dietary (g)": 14,
            "sodium, na (mg)": 390,
            "sugars, total (g)": 6,
            "cholesterol (mg)": 0,
            "energy (kcal)": 310,
            "magnesium, mg (mg)": 115,
            "folate, total (g)": 240,
            "thiamin (mg)": 0.75,
            "vitamin c, total ascorbic acid (mg)": 16,
            "vitamin k (phylloquinone) (g)": 18,
            "water (g)": 340,
        },
        {
            "_id": "r1006",
            "recipe_id": 1006,
            "recipe_index": 6,
            "recipe_title": "Sugar Water Lemonade",
            "calories": 160,
            "cook_time": 0,
            "prep_time": 6,
            "total_time": 6,
            "servings": 2,
            "url": "https://example.test/recipes/lemonade",
            "region": "American",
            "sub_region": "American",
            "continent": "North American",
            "source": "temp_fixture",
            "img_url": "https://example.test/images/lemonade.jpg",
            "processes": "mix stir",
            "utensils": "pitcher,spoon",
            "vegan": "true",
            "pescetarian": "false",
            "ovo_vegetarian": "false",
            "lacto_vegetarian": "false",
            "ovo_lacto_vegetarian": "true",
            "protein (g)": 0,
            "carbohydrate, by difference (g)": 42,
            "total lipid (fat) (g)": 0,
            "fiber, total dietary (g)": 0,
            "sodium, na (mg)": 5,
            "sugars, total (g)": 38,
            "cholesterol (mg)": 0,
            "energy (kcal)": 160,
            "magnesium, mg (mg)": 4,
            "folate, total (g)": 8,
            "thiamin (mg)": 0,
            "vitamin c, total ascorbic acid (mg)": 28,
            "vitamin k (phylloquinone) (g)": 0,
            "water (g)": 420,
        },
    ]
    recipes.extend(generate_recipe_rows(len(recipes), TOTAL_RECIPE_ROWS))

    for recipe in recipes:
        insert_row(conn, "micronutrients", recipe)
        insert_row(conn, "recipes", recipe_as_recipes_row(recipe))


def recipe_as_recipes_row(recipe: dict[str, object]) -> dict[str, object]:
    return {
        "_id": recipe["_id"],
        "recipe_id": str(recipe["recipe_id"]),
        "calories": str(recipe["calories"]),
        "cook_time": str(recipe["cook_time"]),
        "prep_time": str(recipe["prep_time"]),
        "servings": str(recipe["servings"]),
        "recipe_title": recipe["recipe_title"],
        "total_time": str(recipe["total_time"]),
        "url": recipe["url"],
        "region": recipe["region"],
        "sub_region": recipe["sub_region"],
        "continent": recipe["continent"],
        "source": recipe["source"],
        "img_url": recipe["img_url"],
        "carbohydrate, by difference (g)": str(recipe["carbohydrate, by difference (g)"]),
        "energy (kcal)": str(recipe["energy (kcal)"]),
        "protein (g)": str(recipe["protein (g)"]),
        "total lipid (fat) (g)": str(recipe["total lipid (fat) (g)"]),
        "processes": recipe["processes"],
        "vegan": recipe["vegan"],
        "pescetarian": recipe["pescetarian"],
        "ovo_vegetarian": recipe["ovo_vegetarian"],
        "lacto_vegetarian": recipe["lacto_vegetarian"],
        "ovo_lacto_vegetarian": recipe["ovo_lacto_vegetarian"],
        "utensils": recipe["utensils"],
    }


def generate_recipe_rows(existing_count: int, total_rows: int) -> list[dict[str, object]]:
    generated: list[dict[str, object]] = []
    for recipe_index in range(existing_count + 1, total_rows + 1):
        recipe_id = 1000 + recipe_index
        blueprint = GENERATED_RECIPE_BLUEPRINTS[
            (recipe_index - existing_count - 1) % len(GENERATED_RECIPE_BLUEPRINTS)
        ]
        variant = ((recipe_index - existing_count - 1) // len(GENERATED_RECIPE_BLUEPRINTS)) + 1
        title = f"{blueprint['title']} {variant:02d}"
        nutrition_shift = (variant % 5) * 7 + (recipe_index % 4) * 3
        time_shift = (recipe_index % 6) * 2
        generated.append({
            "_id": f"r{recipe_id}",
            "recipe_id": recipe_id,
            "recipe_index": recipe_index,
            "recipe_title": title,
            "calories": float(blueprint["calories"]) + nutrition_shift,
            "cook_time": 10 + time_shift if "mix" not in str(blueprint["processes"]) else 0,
            "prep_time": 6 + (recipe_index % 5) * 2,
            "total_time": 16 + time_shift + (recipe_index % 5) * 2,
            "servings": 2 + (recipe_index % 4),
            "url": f"https://example.test/recipes/{recipe_id}",
            "region": blueprint["region"],
            "sub_region": blueprint["sub_region"],
            "continent": blueprint["continent"],
            "source": "temp_fixture_generated",
            "img_url": f"https://example.test/images/{recipe_id}.jpg",
            "processes": blueprint["processes"],
            "utensils": blueprint["utensils"],
            "vegan": blueprint["vegan"],
            "pescetarian": blueprint["pescetarian"],
            "ovo_vegetarian": blueprint["ovo_vegetarian"],
            "lacto_vegetarian": blueprint["lacto_vegetarian"],
            "ovo_lacto_vegetarian": blueprint["ovo_lacto_vegetarian"],
            "protein (g)": float(blueprint["protein"]) + (recipe_index % 3),
            "carbohydrate, by difference (g)": float(blueprint["carbs"]) + (recipe_index % 8),
            "total lipid (fat) (g)": float(blueprint["fat"]) + (variant % 3),
            "fiber, total dietary (g)": float(blueprint["fiber"]) + (recipe_index % 2),
            "sodium, na (mg)": float(blueprint["sodium"]) + (recipe_index % 7) * 10,
            "sugars, total (g)": float(blueprint["sugar"]) + (recipe_index % 4),
            "cholesterol (mg)": float(blueprint["cholesterol"]),
            "energy (kcal)": float(blueprint["calories"]) + nutrition_shift,
            "magnesium, mg (mg)": float(blueprint["magnesium"]) + (recipe_index % 5) * 2,
            "folate, total (g)": float(blueprint["folate"]) + (recipe_index % 6) * 3,
            "thiamin (mg)": float(blueprint["thiamin"]) + (recipe_index % 4) * 0.02,
            "vitamin c, total ascorbic acid (mg)": float(blueprint["vitamin_c"]) + (recipe_index % 5),
            "vitamin k (phylloquinone) (g)": float(blueprint["vitamin_k"]) + (recipe_index % 4),
            "water (g)": float(blueprint["water"]) + (recipe_index % 10) * 5,
        })
    return generated


def all_recipe_ingredients() -> dict[int, list[str]]:
    recipe_ingredients = dict(BASE_RECIPE_INGREDIENTS)
    for recipe_index in range(len(BASE_RECIPE_INGREDIENTS) + 1, TOTAL_RECIPE_ROWS + 1):
        recipe_id = 1000 + recipe_index
        blueprint = GENERATED_RECIPE_BLUEPRINTS[
            (recipe_index - len(BASE_RECIPE_INGREDIENTS) - 1) % len(GENERATED_RECIPE_BLUEPRINTS)
        ]
        recipe_ingredients[recipe_id] = list(blueprint["ingredients"])
    return recipe_ingredients


def seed_ingredients(conn: sqlite3.Connection) -> None:
    recipe_ingredients = all_recipe_ingredients()
    ingredient_meta = INGREDIENT_META

    for ingredient, (ing_id, _ndb_id, category, generic_name) in ingredient_meta.items():
        insert_row(conn, "ingredient_flavor", {
            "_id": f"flavor_{ing_id}",
            "ingid": ing_id,
            "ingredient": ingredient,
            "generic_name": generic_name,
            "dietrx_category": category,
            "flavordb_category": generic_name,
        })

    row_number = 1
    for recipe_id, ingredients in recipe_ingredients.items():
        for ingredient in ingredients:
            ing_id, ndb_id, _category, _generic_name = ingredient_meta[ingredient]
            insert_row(conn, "ingredient_phrase", {
                "_id": f"ip{row_number:03d}",
                "recipe_no": recipe_id,
                "ingredient_phrase": f"1 serving {ingredient}",
                "ingredient": ingredient,
                "quantity": "1",
                "unit": "serving",
                "ing_id": ing_id,
                "ndb_id": ndb_id,
                "m_or_a": "manual",
                "state": "",
                "size": "",
                "temp": "",
                "df": "",
            })
            insert_row(conn, "ingredient_nutrition", ingredient_nutrition_row(
                row_number,
                recipe_id,
                ingredient,
                ndb_id,
            ))
            row_number += 1


def ingredient_nutrition_row(
    row_number: int,
    recipe_id: int,
    ingredient: str,
    ndb_id: int,
) -> dict[str, object]:
    protein_defaults = {
        "chicken": 31,
        "salmon": 25,
        "lentil": 9,
        "pasta": 6,
        "spinach": 3,
    }
    carb_defaults = {
        "sugar": 100,
        "pasta": 25,
        "lentil": 20,
        "lemon juice": 7,
        "tomato": 4,
        "carrot": 10,
        "onion": 9,
    }
    return {
        "_id": f"inut{row_number:03d}",
        "recipe_id": recipe_id,
        "ndb_id": ndb_id,
        "protein (g)": protein_defaults.get(ingredient, 1),
        "carbohydrate, by difference (g)": carb_defaults.get(ingredient, 2),
        "total lipid (fat) (g)": 12 if ingredient in {"olive oil", "salmon"} else 1,
        "fiber, total dietary (g)": 4 if ingredient in {"lentil", "spinach"} else 0,
        "sodium, na (mg)": 30,
        "cholesterol (mg)": 70 if ingredient in {"chicken", "salmon"} else 0,
        "magnesium, mg (mg)": 80 if ingredient in {"spinach", "lentil"} else 10,
        "folate, total (g)": 180 if ingredient in {"spinach", "lentil"} else 5,
        "thiamin (mg)": 0.35 if ingredient in {"lentil", "pasta"} else 0.05,
        "vitamin c, total ascorbic acid (mg)": 20 if ingredient in {"lemon juice", "spinach"} else 1,
        "vitamin k (phylloquinone) (g)": 90 if ingredient == "spinach" else 0,
        "water (g)": 85 if ingredient in {"water", "lemon juice"} else 40,
    }


def seed_instructions(conn: sqlite3.Connection) -> None:
    steps = {
        1001: "Chop spinach and garlic. Saute chicken in a skillet. Simmer until cooked.",
        1002: "Chop spinach. Mix lemon juice and olive oil. Toss in a bowl.",
        1003: "Bake salmon with garlic and lemon juice in an oven.",
        1004: "Boil pasta. Saute garlic and tomato. Simmer sauce.",
        1005: "Boil lentils with turmeric, carrot, and onion. Simmer until tender.",
        1006: "Mix sugar, water, and lemon juice. Stir until dissolved.",
    }
    steps.update(generated_instruction_steps())
    for recipe_id, text in steps.items():
        insert_row(conn, "instructions", {
            "_id": f"ins{recipe_id}",
            "recipe_id": recipe_id,
            "steps": text,
        })


def generated_instruction_steps() -> dict[int, str]:
    steps: dict[int, str] = {}
    for recipe_index in range(len(BASE_RECIPE_INGREDIENTS) + 1, TOTAL_RECIPE_ROWS + 1):
        recipe_id = 1000 + recipe_index
        blueprint = GENERATED_RECIPE_BLUEPRINTS[
            (recipe_index - len(BASE_RECIPE_INGREDIENTS) - 1) % len(GENERATED_RECIPE_BLUEPRINTS)
        ]
        ingredients = ", ".join(str(item) for item in blueprint["ingredients"])
        processes = str(blueprint["processes"]).replace(",", " ")
        utensils = str(blueprint["utensils"]).replace(",", ", ")
        steps[recipe_id] = (
            f"Prepare {ingredients}. Use {utensils}. "
            f"Follow these process cues: {processes}."
        )
    return steps


def build_database() -> Path:
    TABLE_COLUMN_CACHE.clear()
    if DB_PATH.exists():
        DB_PATH.unlink()

    THIS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.executescript(sqlite_schema_text())
        seed_recipes(conn)
        seed_ingredients(conn)
        seed_instructions(conn)
        conn.commit()
    finally:
        conn.close()
    return DB_PATH


def main() -> None:
    db_path = build_database()
    print(f"Created {db_path}")
    with sqlite3.connect(db_path) as conn:
        for table in [
            "micronutrients",
            "ingredient_phrase",
            "ingredient_nutrition",
            "ingredient_flavor",
            "instructions",
            "recipes",
        ]:
            count = conn.execute(f"SELECT COUNT(*) FROM {quote_identifier(table)}").fetchone()[0]
            print(f"{table}: {count} rows")


if __name__ == "__main__":
    main()
