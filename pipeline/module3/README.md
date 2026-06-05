# Module 3 Recipe Search

Module 3 turns `module1_asr.entity_tagger.tag_entities()` output into RecipeDB2
API calls and a SQL-style debug query.

Important point: the live RecipeDB2 surface is HTTP, not direct SQL. The SQL
builder is therefore explainability/debug output. The executable path is:

1. `query_planner.py` chooses the most selective API endpoints.
2. `client.py` executes requests with a small TTL cache.
3. `result_merger.py` intersects/deduplicates API result sets.
4. `post_filter.py` applies constraints that are missing from a response.
5. `ranker.py` applies `ORDER BY`/`LIMIT`-style intent.

Use it without making network calls:

```powershell
cd C:\Users\wadhw\IP_Project\pipeline
python -c "from module3 import search_text; import json; print(json.dumps(search_text('vegan spinach under 300 calories', execute=False), indent=2))"
```

Use it against RecipeDB2:

```powershell
$env:RDB2_API_KEY="your Foodoscope key"
cd C:\Users\wadhw\IP_Project\pipeline
python -c "from module3 import search_text; print(search_text('vegan spinach under 300 calories')['count'])"
```

Build the virtual schema cache from Python code by calling
`module3.schema_discovery.discover_virtual_schema(...)`. The generated cache is
written to `module3/cache/virtual_schema.json`. It stores field/type evidence
only, not the API key.
