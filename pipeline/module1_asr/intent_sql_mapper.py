def _placeholders(count: int) -> str:
    return ", ".join(["?"] * count)


def build_sql_filter_plan(
    ingredient_logic: dict,
    recipe_id_col: str = "recipe_id",
    recipe_ingredient_table: str = "recipe_ingredients",
    ingredient_col: str = "ingredient",
) -> dict:
    """
    Build a parameterized SQL WHERE fragment from ingredient boolean intent.

    This does not execute SQL and does not assume the final schema beyond a
    recipe-ingredient bridge table. Module 3 can swap table/column names while
    keeping the AND/OR/NOT semantics intact.
    """
    include = ingredient_logic.get("include", [])
    exclude = ingredient_logic.get("exclude", [])
    include_operator = ingredient_logic.get("include_operator", "AND")
    if include_operator == "MIXED":
        include_operator_used = "AND"
        ambiguous = True
    else:
        include_operator_used = include_operator
        ambiguous = False

    clauses = []
    params = []

    if include:
        placeholders = _placeholders(len(include))
        if include_operator_used == "OR":
            clauses.append(
                f"{recipe_id_col} IN ("
                f"SELECT {recipe_id_col} FROM {recipe_ingredient_table} "
                f"WHERE {ingredient_col} IN ({placeholders})"
                f")"
            )
            params.extend(include)
        else:
            clauses.append(
                f"{recipe_id_col} IN ("
                f"SELECT {recipe_id_col} FROM {recipe_ingredient_table} "
                f"WHERE {ingredient_col} IN ({placeholders}) "
                f"GROUP BY {recipe_id_col} "
                f"HAVING COUNT(DISTINCT {ingredient_col}) = ?"
                f")"
            )
            params.extend(include)
            params.append(len(include))

    if exclude:
        placeholders = _placeholders(len(exclude))
        clauses.append(
            f"{recipe_id_col} NOT IN ("
            f"SELECT {recipe_id_col} FROM {recipe_ingredient_table} "
            f"WHERE {ingredient_col} IN ({placeholders})"
            f")"
        )
        params.extend(exclude)

    where_clause = " AND ".join(clauses)
    return {
        "where_clause": where_clause,
        "params": params,
        "include_operator": include_operator,
        "include_operator_used": include_operator_used,
        "ambiguous": ambiguous,
        "include": include,
        "exclude": exclude,
    }
