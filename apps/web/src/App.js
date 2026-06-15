import { useMemo, useState } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

const EXAMPLES = [
  'vegan recipes with spinach under 300 calories',
  'recipes containing onion but no garlic with milk and jackfruit',
  'top 5 high protein recipes with chicken',
  'quick Indian recipes with paneer under 30 minutes',
];

function App() {
  const [query, setQuery] = useState(EXAMPLES[0]);
  const [execute, setExecute] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);

  const intent = useMemo(() => result?.intent || result?.tag_result || {}, [result]);
  const sql = useMemo(() => result?.sql || {}, [result]);
  const apiPlans = useMemo(() => result?.api_plans || result?.plans || [], [result]);

  const summary = useMemo(() => {
    if (!result) return [];
    return [
      ['Ingredients', intent.ingredients || []],
      ['Diet', intent.dietary_styles || []],
      ['Nutrition', intent.nutrition_filters || []],
      ['SQL valid', sql.validation?.ok],
      ['Rows/Results', result.count ?? result.total_count ?? result.results?.length ?? 0],
    ];
  }, [intent, result, sql]);

  async function runQuery(event) {
    event.preventDefault();
    const text = query.trim();
    if (!text) return;

    setLoading(true);
    setError('');
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: text, execute }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || payload.error || 'Query failed');
      }
      setResult(payload);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <section className="query-panel">
        <div className="brand-row">
          <div>
            <p className="eyebrow">Speech-to-SQL</p>
            <h1>Recipe Query Console</h1>
          </div>
          <span className="status-pill">M1-M5</span>
        </div>

        <form onSubmit={runQuery} className="query-form">
          <label htmlFor="query">Natural language query</label>
          <textarea
            id="query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={5}
          />

          <div className="controls">
            <label className="toggle-row">
              <input
                type="checkbox"
                checked={execute}
                onChange={(event) => setExecute(event.target.checked)}
              />
              Execute RecipeDB API calls
            </label>
            <button type="submit" disabled={loading}>
              {loading ? 'Running...' : 'Run query'}
            </button>
          </div>
        </form>

        <div className="examples">
          {EXAMPLES.map((item) => (
            <button key={item} type="button" onClick={() => setQuery(item)}>
              {item}
            </button>
          ))}
        </div>
      </section>

      <section className="result-panel">
        {error && <div className="error-box">{error}</div>}

        {!result && !error && (
          <div className="empty-state">
            Start the Python API, run a query, and inspect intent, SQL, and
            planned execution output here.
          </div>
        )}

        {result && (
          <>
            <div className="summary-grid">
              {summary.map(([label, value]) => (
                <div className="summary-card" key={label}>
                  <span>{label}</span>
                  <strong>{formatValue(value)}</strong>
                </div>
              ))}
            </div>

            <section className="detail-block">
              <h2>Generated SQL</h2>
              <pre>{sql.query || 'No SQL returned'}</pre>
              <h3>Params</h3>
              <pre>{JSON.stringify(sql.params || [], null, 2)}</pre>
              {sql.validation?.warnings?.length > 0 && (
                <>
                  <h3>Warnings</h3>
                  <pre>{JSON.stringify(sql.validation.warnings, null, 2)}</pre>
                </>
              )}
            </section>

            <section className="detail-block">
              <h2>API Plans</h2>
              <pre>{JSON.stringify(apiPlans, null, 2)}</pre>
            </section>

            <section className="detail-block">
              <h2>Full Response</h2>
              <pre>{JSON.stringify(result, null, 2)}</pre>
            </section>
          </>
        )}
      </section>
    </main>
  );
}

function formatValue(value) {
  if (Array.isArray(value)) {
    return value.length ? value.map((item) => {
      if (typeof item === 'object') return JSON.stringify(item);
      return String(item);
    }).join(', ') : 'None';
  }
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (value === null || value === undefined || value === '') return 'None';
  return String(value);
}

export default App;
