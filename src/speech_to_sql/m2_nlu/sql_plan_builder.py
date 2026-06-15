NUTRIENT_COL_MAP = {
    "calories": "calories",
    "protein": "protein_g",
    "fat": "fat_g",
    "carbs": "carbs_g",
    "fiber": "fiber_g",
    "sugar": "sugar_g",
    "sodium": "sodium_mg",
}

ALLOWED_NUTRITION_OPS = {"<", "<=", "=", ">=", ">", "~"}


def build_unified_sql_plan(
    sql_filter_plan: dict,
    nutrition_filters: list[dict],
    dietary_styles: list[str],
    recipe_id_col: str = "recipe_id",
    nutrition_table: str = "recipe_nutrition",
    dietary_table: str = "recipe_dietary",
    dietary_tag_col: str = "dietary_tag",
) -> dict:
    """
    Combine ingredient, nutrition, and dietary constraints into one SQL plan.

    The returned where_clause intentionally omits the leading WHERE keyword.
    All user-controlled values are returned separately in params.
    """
    clauses = []
    params = []

    if sql_filter_plan.get("where_clause"):
        clauses.append(f"({sql_filter_plan['where_clause']})")
        params.extend(sql_filter_plan.get("params", []))

    for nutrition_filter in nutrition_filters:
        col = NUTRIENT_COL_MAP.get(nutrition_filter.get("nutrient"))
        op = nutrition_filter.get("operator")
        if not col or op not in ALLOWED_NUTRITION_OPS:
            continue

        try:
            value = float(nutrition_filter["value"])
        except (KeyError, TypeError, ValueError):
            continue

        if op == "~":
            low = round(value * 0.9, 1)
            high = round(value * 1.1, 1)
            clauses.append(
                f"{recipe_id_col} IN ("
                f"SELECT {recipe_id_col} FROM {nutrition_table} "
                f"WHERE {col} BETWEEN ? AND ?)"
            )
            params.extend([low, high])
        else:
            clauses.append(
                f"{recipe_id_col} IN ("
                f"SELECT {recipe_id_col} FROM {nutrition_table} "
                f"WHERE {col} {op} ?)"
            )
            params.append(value)

    for style in dietary_styles:
        clauses.append(
            f"{recipe_id_col} IN ("
            f"SELECT {recipe_id_col} FROM {dietary_table} "
            f"WHERE {dietary_tag_col} = ?)"
        )
        params.append(style)

    return {
        "where_clause": " AND ".join(clauses),
        "params": params,
        "components": {
            "ingredients": sql_filter_plan,
            "nutrition": nutrition_filters,
            "dietary": dietary_styles,
        },
    }
