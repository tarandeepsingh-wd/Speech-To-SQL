"""
Recipe search UI.
Run: python apps/api/search_app.py
Then open: http://localhost:5050
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

app = Flask(__name__)

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RecipeDB Speech-to-SQL Search</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0f1117;
    --surface: #181c27;
    --surface2: #1e2435;
    --border: #2a3045;
    --border2: #3a4460;
    --accent: #4ade80;
    --accent-dim: #14532d30;
    --accent2: #60a5fa;
    --accent2-dim: #1e3a5f30;
    --text: #e8ecf4;
    --text2: #8b93a8;
    --text3: #5a6278;
    --red: #f87171;
    --amber: #fbbf24;
    --purple: #a78bfa;
    --mono: 'JetBrains Mono', monospace;
    --sans: 'Plus Jakarta Sans', sans-serif;
    --radius: 10px;
  }

  body {
    font-family: var(--sans);
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  /* -- Header -- */
  header {
    border-bottom: 1px solid var(--border);
    padding: 1rem 2rem;
    display: flex;
    align-items: center;
    gap: 1rem;
    background: var(--surface);
  }
  .logo {
    display: flex;
    align-items: center;
    gap: .6rem;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: -.02em;
  }
  .logo-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--accent); box-shadow: 0 0 8px var(--accent); }
  .badge {
    font-size: 11px;
    font-weight: 600;
    padding: 2px 9px;
    border-radius: 20px;
    border: 1px solid var(--border2);
    color: var(--text2);
    letter-spacing: .05em;
    text-transform: uppercase;
  }

  /* -- Main layout -- */
  main {
    flex: 1;
    display: grid;
    grid-template-columns: 420px 1fr;
    grid-template-rows: 1fr;
    min-height: 0;
  }

  /* -- Left panel -- */
  .left-panel {
    border-right: 1px solid var(--border);
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    overflow-y: auto;
    background: var(--surface);
  }

  .section-label {
    font-size: 10.5px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: .1em;
    color: var(--text3);
    margin-bottom: .5rem;
  }

  /* Search box */
  .search-wrap { display: flex; flex-direction: column; gap: .75rem; }
  textarea {
    width: 100%;
    min-height: 80px;
    background: var(--bg);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    padding: .75rem 1rem;
    resize: vertical;
    outline: none;
    line-height: 1.6;
    transition: border-color .15s;
  }
  textarea:focus { border-color: var(--accent); }
  textarea::placeholder { color: var(--text3); }

  .btn-row { display: flex; gap: .5rem; }
  button {
    font-family: var(--sans);
    cursor: pointer;
    border: none;
    border-radius: var(--radius);
    font-size: 13px;
    font-weight: 600;
    padding: .55rem 1.1rem;
    transition: opacity .15s, transform .1s;
  }
  button:active { transform: scale(.97); }
  .btn-primary {
    background: var(--accent);
    color: #000;
    flex: 1;
  }
  .btn-primary:hover { opacity: .88; }
  .btn-secondary {
    background: var(--surface2);
    color: var(--text2);
    border: 1px solid var(--border2);
  }
  .btn-secondary:hover { background: var(--border); }

  /* Execute toggle */
  .toggle-row {
    display: flex;
    align-items: center;
    gap: .6rem;
    font-size: 13px;
    color: var(--text2);
  }
  .toggle {
    position: relative;
    width: 36px;
    height: 20px;
    flex-shrink: 0;
  }
  .toggle input { opacity: 0; width: 0; height: 0; }
  .slider {
    position: absolute;
    inset: 0;
    background: var(--border2);
    border-radius: 20px;
    cursor: pointer;
    transition: .2s;
  }
  .slider::before {
    content: '';
    position: absolute;
    height: 14px; width: 14px;
    left: 3px; top: 3px;
    background: #fff;
    border-radius: 50%;
    transition: .2s;
  }
  input:checked + .slider { background: var(--accent); }
  input:checked + .slider::before { transform: translateX(16px); }

  /* Intent chips */
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip {
    font-size: 12px;
    font-weight: 500;
    padding: 3px 10px;
    border-radius: 20px;
    border: 1px solid;
  }
  .chip-green { background: #14532d22; border-color: #4ade8040; color: #4ade80; }
  .chip-blue  { background: #1e3a5f22; border-color: #60a5fa40; color: #60a5fa; }
  .chip-purple{ background: #4c1d9522; border-color: #a78bfa40; color: #a78bfa; }
  .chip-amber { background: #78350f22; border-color: #fbbf2440; color: #fbbf24; }
  .chip-red   { background: #7f1d1d22; border-color: #f8717140; color: #f87171; }

  /* SQL block */
  .sql-box {
    background: var(--bg);
    border: 1px solid var(--border2);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .sql-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: .5rem .75rem;
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
  }
  .sql-bar span { font-size: 11px; font-weight: 600; color: var(--text3); text-transform: uppercase; letter-spacing: .08em; }
  .copy-btn {
    font-size: 11px;
    padding: 2px 9px;
    background: var(--border);
    color: var(--text2);
    border: 1px solid var(--border2);
    border-radius: 5px;
  }
  .copy-btn:hover { background: var(--border2); }
  pre.sql {
    font-family: var(--mono);
    font-size: 11.5px;
    line-height: 1.9;
    padding: .9rem 1rem;
    overflow-x: auto;
    white-space: pre-wrap;
    word-break: break-all;
  }
  .kw  { color: #60a5fa; font-weight: 600; }
  .tbl { color: #fb923c; }
  .ph  { color: #4ade80; }
  .fn  { color: #a78bfa; font-weight: 600; }

  /* Params */
  .param-list { display: flex; flex-wrap: wrap; gap: 6px; }
  .param-chip {
    font-family: var(--mono);
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 5px;
    background: var(--surface2);
    border: 1px solid var(--border2);
    color: var(--text2);
  }

  /* -- Right panel -- */
  .right-panel {
    padding: 1.5rem;
    overflow-y: auto;
    background: var(--bg);
  }

  /* Empty / error states */
  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100%;
    gap: .75rem;
    color: var(--text3);
    text-align: center;
  }
  .empty-icon { font-size: 2.5rem; opacity: .3; }
  .empty-state p { font-size: 14px; line-height: 1.6; max-width: 300px; }

  /* Stats row */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
    margin-bottom: 1.25rem;
  }
  .stat-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: .75rem 1rem;
    text-align: center;
  }
  .stat-val { font-size: 22px; font-weight: 700; color: var(--accent); }
  .stat-lbl { font-size: 11px; color: var(--text3); margin-top: 2px; text-transform: uppercase; letter-spacing: .06em; }

  /* Recipe grid */
  .recipe-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 12px;
  }
  .recipe-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 1rem;
    transition: border-color .15s, transform .1s;
    cursor: default;
  }
  .recipe-card:hover { border-color: var(--border2); transform: translateY(-2px); }
  .recipe-name { font-size: 14px; font-weight: 600; margin-bottom: .5rem; line-height: 1.4; }
  .recipe-meta { font-size: 12px; color: var(--text2); display: flex; flex-wrap: wrap; gap: .4rem .7rem; }
  .recipe-meta span { display: flex; align-items: center; gap: 3px; }
  .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }
  .recipe-tags { display: flex; flex-wrap: wrap; gap: 5px; margin-top: .6rem; }

  /* Loading spinner */
  .spinner {
    width: 20px; height: 20px;
    border: 2px solid var(--border2);
    border-top-color: var(--accent);
    border-radius: 50%;
    animation: spin .6s linear infinite;
    display: none;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* Error banner */
  .error-banner {
    background: #7f1d1d22;
    border: 1px solid #f8717140;
    border-radius: var(--radius);
    padding: .75rem 1rem;
    font-size: 13px;
    color: var(--red);
    display: none;
  }

  /* Scrollbar */
  ::-webkit-scrollbar { width: 5px; height: 5px; }
  ::-webkit-scrollbar-track { background: transparent; }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 99px; }

  /* Responsive */
  @media (max-width: 900px) {
    main { grid-template-columns: 1fr; grid-template-rows: auto 1fr; }
    .left-panel { border-right: none; border-bottom: 1px solid var(--border); }
  }
</style>
</head>
<body>

<header>
  <div class="logo">
    <div class="logo-dot"></div>
    RecipeDB
  </div>
  <div class="badge">M1-M5 &nbsp;·&nbsp; SQL Pipeline</div>
</header>

<main>
  <!-- -- LEFT PANEL -- -->
  <div class="left-panel">

    <!-- Search -->
    <div>
      <div class="section-label">Search query</div>
      <div class="search-wrap">
        <textarea id="query" placeholder="e.g. top 5 vegan recipes with chicken and milk&#10;or: quick pasta under 30 minutes&#10;or: high protein breakfast without eggs"></textarea>
        <div class="toggle-row">
          <label class="toggle">
            <input type="checkbox" id="execute-toggle" checked>
            <span class="slider"></span>
          </label>
          <span>Execute API call &nbsp;<span style="color:var(--text3);font-size:12px">(off = SQL preview only)</span></span>
        </div>
        <div class="btn-row">
          <button class="btn-primary" onclick="runSearch()">
            <span id="btn-text">Search</span>
            <div class="spinner" id="spinner" style="display:inline-block;vertical-align:middle;margin-left:6px"></div>
          </button>
          <button class="btn-secondary" onclick="clearAll()">Clear</button>
        </div>
        <div class="error-banner" id="error-banner"></div>
      </div>
    </div>

    <!-- Intent -->
    <div id="intent-section" style="display:none">
      <div class="section-label">Extracted intent</div>
      <div id="intent-chips" class="chips"></div>
    </div>

    <!-- SQL -->
    <div id="sql-section" style="display:none">
      <div class="section-label">Generated SQL</div>
      <div class="sql-box">
        <div class="sql-bar">
          <span>Parameterised query</span>
          <button class="copy-btn" onclick="copySQL()">Copy</button>
        </div>
        <pre class="sql" id="sql-pre"></pre>
      </div>
    </div>

    <!-- Params -->
    <div id="params-section" style="display:none">
      <div class="section-label">Bound parameters</div>
      <div class="param-list" id="param-list"></div>
    </div>

  </div>

  <!-- -- RIGHT PANEL -- -->
  <div class="right-panel" id="right-panel">
    <div class="empty-state" id="empty-state">
      <div class="empty-icon">🍽️</div>
      <p>Type a natural language query on the left and hit Search.</p>
    </div>
    <div id="results-area" style="display:none"></div>
  </div>

</main>

<script>
const SAMPLES = [
  "top 5 vegan recipes with chicken and milk",
  "quick pasta under 30 minutes without mushrooms",
  "high protein breakfast with eggs and spinach",
  "vegetarian curry with chickpeas",
  "low calorie salad with avocado",
];

let rawSQL = "";

async function runSearch() {
  const query = document.getElementById("query").value.trim();
  if (!query) return;

  const execute = document.getElementById("execute-toggle").checked;

  // Loading state
  document.getElementById("btn-text").textContent = "Searching...";
  document.getElementById("spinner").style.display = "inline-block";
  document.getElementById("error-banner").style.display = "none";
  document.getElementById("intent-section").style.display = "none";
  document.getElementById("sql-section").style.display = "none";
  document.getElementById("params-section").style.display = "none";
  document.getElementById("empty-state").style.display = "none";
  document.getElementById("results-area").style.display = "none";

  try {
    const res = await fetch("/api/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, execute }),
    });
    const data = await res.json();

    if (!res.ok) throw new Error(data.error || "Request failed");
    renderResults(data, execute);
  } catch (err) {
    showError(err.message);
  } finally {
    document.getElementById("btn-text").textContent = "Search";
    document.getElementById("spinner").style.display = "none";
  }
}

function renderResults(data, executed) {
  // Intent chips
  const intent = data.intent || {};
  const chips = [];
  (intent.ingredients || []).forEach(i => chips.push({ label: i, cls: "chip-green" }));
  (intent.dietary_styles || []).forEach(d => chips.push({ label: d, cls: "chip-blue" }));
  (intent.categories || []).forEach(c => chips.push({ label: c, cls: "chip-purple" }));
  if (intent.include_operator) chips.push({ label: intent.include_operator, cls: "chip-amber" });
  (intent.exclude_ingredients || []).forEach(e => chips.push({ label: "❌ " + e, cls: "chip-red" }));
  if (intent.limit) chips.push({ label: "TOP " + intent.limit, cls: "chip-amber" });

  const chipsHtml = chips.map(c => `<span class="chip ${c.cls}">${c.label}</span>`).join("");
  document.getElementById("intent-chips").innerHTML = chipsHtml || "<span style='color:var(--text3);font-size:13px'>No intent extracted</span>";
  document.getElementById("intent-section").style.display = "block";

  // SQL
  const sql = data.sql || {};
  if (sql.query) {
    rawSQL = sql.query;
    document.getElementById("sql-pre").innerHTML = highlightSQL(sql.query);
    document.getElementById("sql-section").style.display = "block";

    if (sql.params && sql.params.length) {
      const paramHtml = sql.params.map(p => `<span class="param-chip">${JSON.stringify(p)}</span>`).join("");
      document.getElementById("param-list").innerHTML = paramHtml;
      document.getElementById("params-section").style.display = "block";
    }
  }

  // Right panel
  const ra = document.getElementById("results-area");
  ra.style.display = "block";

  if (!executed) {
    ra.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:.75rem;color:var(--text3);text-align:center;padding:3rem 0">
        <div style="font-size:2rem;opacity:.4">🔍</div>
        <p style="font-size:14px;max-width:300px;line-height:1.6">SQL preview mode - toggle <strong style="color:var(--text)">Execute API</strong> on to see recipes.</p>
      </div>`;
    return;
  }

  const recipes = data.recipes || [];
  if (!recipes.length) {
    ra.innerHTML = `
      <div style="display:flex;flex-direction:column;align-items:center;justify-content:center;height:100%;gap:.75rem;color:var(--text3);text-align:center;padding:3rem 0">
        <div style="font-size:2rem;opacity:.4">🤷</div>
        <p style="font-size:14px;max-width:300px;line-height:1.6">No recipes found. Try changing the query or relaxing the filters.</p>
      </div>`;
    return;
  }

  // Stats row
  const statsHtml = `
    <div class="stats-row">
      <div class="stat-card"><div class="stat-val">${recipes.length}</div><div class="stat-lbl">Results</div></div>
      <div class="stat-card"><div class="stat-val">${(data.intent?.ingredients||[]).length}</div><div class="stat-lbl">Ingredients</div></div>
      <div class="stat-card"><div class="stat-val">${(data.intent?.dietary_styles||[]).length ? "Yes" : "No"}</div><div class="stat-lbl">Diet filter</div></div>
      <div class="stat-card"><div class="stat-val" style="color:var(--accent2)">${data.executed ? "Y" : "-"}</div><div class="stat-lbl">Executed</div></div>
    </div>`;

  // Recipe cards
  const cardsHtml = recipes.map(r => {
    const name = r.recipe_name || r.Title || r.title || r.name || r.Recipe_name || "Untitled";
    const diet = [r.vegan && "Vegan", r.vegetarian && "Vegetarian", r.glutenFree && "GF"].filter(Boolean);
    const cal  = r.calories ? Math.round(r.calories) : null;
    const time = r.readyInMinutes || r.cook_time || null;
    const cuis = r.cuisine || r.Cuisine_region || null;

    const tagHtml = diet.map(d => `<span class="chip chip-green" style="font-size:11px;padding:2px 7px">${d}</span>`).join("");
    return `
      <div class="recipe-card">
        <div class="recipe-name">${escHtml(name)}</div>
        <div class="recipe-meta">
          ${cal  ? `<span><div class="dot"></div>${cal} cal</span>` : ""}
          ${time ? `<span><div class="dot"></div>${time} min</span>` : ""}
          ${cuis ? `<span><div class="dot"></div>${escHtml(String(cuis))}</span>` : ""}
        </div>
        ${tagHtml ? `<div class="recipe-tags">${tagHtml}</div>` : ""}
      </div>`;
  }).join("");

  ra.innerHTML = statsHtml + `<div class="recipe-grid">${cardsHtml}</div>`;
}

function highlightSQL(sql) {
  const kws = ["SELECT","DISTINCT","FROM","JOIN","WHERE","GROUP BY","HAVING","ORDER BY",
    "UNION ALL","UNION","LIMIT","AND","OR","ON","AS","IN","NOT","LIKE","COUNT","CASE","WHEN","THEN","ELSE","END"];
  let s = escHtml(sql);
  kws.forEach(k => {
    const re = new RegExp(`\\b(${k})\\b`, "g");
    s = s.replace(re, `<span class="kw">$1</span>`);
  });
  // table names (backtick or uppercase words after FROM/JOIN)
  s = s.replace(/`([^`]+)`/g, '<span class="tbl">$1</span>');
  s = s.replace(/(\bRecipeDB\w*)/g, '<span class="tbl">$1</span>');
  // placeholders
  s = s.replace(/\?/g, '<span class="ph">?</span>');
  // functions
  s = s.replace(/\b(COUNT|SUM|AVG|MAX|MIN)\b/g, '<span class="fn">$1</span>');
  return s;
}

function escHtml(s) {
  return String(s).replace(/&/g,"&amp;").replace(/</g,"&lt;").replace(/>/g,"&gt;");
}

function showError(msg) {
  const el = document.getElementById("error-banner");
  el.textContent = "Error: " + msg;
  el.style.display = "block";
  document.getElementById("empty-state").style.display = "flex";
}

function copySQL() {
  if (rawSQL) navigator.clipboard.writeText(rawSQL).catch(()=>{});
  const btn = document.querySelector(".copy-btn");
  btn.textContent = "Copied!";
  setTimeout(() => btn.textContent = "Copy", 1500);
}

function clearAll() {
  document.getElementById("query").value = "";
  document.getElementById("intent-section").style.display = "none";
  document.getElementById("sql-section").style.display = "none";
  document.getElementById("params-section").style.display = "none";
  document.getElementById("results-area").style.display = "none";
  document.getElementById("error-banner").style.display = "none";
  document.getElementById("empty-state").style.display = "flex";
  rawSQL = "";
}

// Enter key submits
document.addEventListener("DOMContentLoaded", () => {
  document.getElementById("query").addEventListener("keydown", e => {
    if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) runSearch();
  });
  // Random placeholder rotation
  let i = 0;
  const ta = document.getElementById("query");
  setInterval(() => {
    i = (i + 1) % SAMPLES.length;
    ta.placeholder = SAMPLES[i];
  }, 4000);
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/api/search", methods=["POST"])
def api_search():
    body = request.get_json(force=True, silent=True) or {}
    query = (body.get("query") or "").strip()
    execute = bool(body.get("execute", True))

    if not query:
        return jsonify({"error": "query is required"}), 400

    try:
        # Try full pipeline with M2 NLU tagging.
        try:
            from speech_to_sql.m2_nlu.entity_tagger import tag_entities
            tag_result = tag_entities(query)
            tag_result["transcript"] = query
        except ImportError:
            # Fallback: minimal tag_result so SQL generation can still run.
            tag_result = {
                "transcript": query,
                "ingredients": [],
                "dietary_styles": [],
                "categories": [],
                "utensils": [],
                "cooking_actions": [],
                "nutrition_filters": [],
                "ingredient_logic": {
                    "include": [],
                    "exclude": [],
                    "include_operator": "AND",
                    "exclude_operator": "AND",
                    "connectors": [],
                    "ambiguous": False,
                },
            }

        from speech_to_sql.m5_execution.search import search_recipes
        result = search_recipes(
            tag_result,
            transcript=query,
            execute=execute,
        )

        # Build intent summary for the UI
        intent = {
            "ingredients": tag_result.get("ingredients", []),
            "dietary_styles": tag_result.get("dietary_styles", []),
            "categories": tag_result.get("categories", []),
            "include_operator": (
                tag_result.get("ingredient_logic") or {}
            ).get("include_operator"),
            "exclude_ingredients": (
                tag_result.get("ingredient_logic") or {}
            ).get("exclude", []),
            "limit": _extract_limit(result),
        }

        return jsonify({
            "ok": True,
            "intent": intent,
            "sql": {
                "query": (result.get("sql") or {}).get("query"),
                "params": (result.get("sql") or {}).get("params", []),
            },
            "recipes": result.get("recipes", []),
            "count": result.get("count", 0),
            "executed": result.get("executed", False),
        })

    except Exception as exc:
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


def _extract_limit(result: dict) -> int | None:
    for plan in result.get("plans", []):
        params = plan.get("params", {})
        if "limit" in params:
            try:
                return int(params["limit"])
            except (TypeError, ValueError):
                pass
    return None


if __name__ == "__main__":
    print("\n  RecipeDB Speech-to-SQL UI")
    print("  -----------------------------")
    print("  Open:  http://localhost:5050\n")
    app.run(debug=True, port=5050)
