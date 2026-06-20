import { useEffect, useMemo, useState } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_BASE || 'http://localhost:8000';

function App() {
  const [query, setQuery] = useState('');
  const [activeView, setActiveView] = useState('proof');
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [feedbackSummary, setFeedbackSummary] = useState(null);
  const [correctedQuery, setCorrectedQuery] = useState('');
  const [correctedIngredients, setCorrectedIngredients] = useState('');

  const tagResult = useMemo(() => result?.tag_result || {}, [result]);
  const sql = useMemo(() => result?.sql || {}, [result]);
  const validation = useMemo(() => sql.validation || {}, [sql]);
  const normalization = useMemo(() => result?.normalization || {}, [result]);
  const ingredients = useMemo(() => tagResult.ingredients || [], [tagResult]);
  const diet = useMemo(() => tagResult.dietary_styles || [], [tagResult]);
  const nutrition = useMemo(() => tagResult.nutrition_filters || [], [tagResult]);
  const plans = useMemo(() => result?.selected_plans || result?.plans || [], [result]);

  const metrics = useMemo(() => ([
    ['SQL', validation.ok === false ? 'Needs check' : 'Valid'],
    ['Ingredients', String(ingredients.length)],
    ['Schema links', String((result?.schema_links || []).length)],
    ['Run mode', 'SQL preview'],
  ]), [ingredients.length, result, validation.ok]);

  useEffect(() => {
    const recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    setSpeechSupported(Boolean(recognition));
    loadFeedbackSummary();
  }, []);

  useEffect(() => {
    if (!result) return;
    setCorrectedQuery(result.normalized_query || result.input_query || query);
    setCorrectedIngredients(ingredients.join(', '));
  }, [result, query, ingredients]);

  async function runQuery(event) {
    event?.preventDefault();
    const text = query.trim();
    if (!text) return;

    setLoading(true);
    setError('');
    setFeedbackStatus('');
    setResult(null);

    try {
      const response = await fetch(`${API_BASE}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: text,
          execute: false,
          max_api_calls: 2,
          fetch_limit: 50,
          final_limit: 5,
        }),
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

  async function saveFeedback(event) {
    event.preventDefault();
    if (!result) return;

    setFeedbackStatus('Saving...');
    setError('');

    const parsedIngredients = correctedIngredients
      .split(',')
      .map((item) => item.trim().toLowerCase())
      .filter(Boolean);

    try {
      const response = await fetch(`${API_BASE}/feedback/search`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          original_query: result.input_query || query,
          corrected_query: correctedQuery,
          predicted_ingredients: ingredients,
          corrected_ingredients: parsedIngredients,
          source: 'web-ui',
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail || payload.error || 'Feedback failed');
      }
      setFeedbackSummary(payload.summary);
      setFeedbackStatus(`Saved ${payload.corrections?.length || 0} transcript correction(s).`);
    } catch (err) {
      setFeedbackStatus('');
      setError(err.message);
    }
  }

  async function loadFeedbackSummary() {
    try {
      const response = await fetch(`${API_BASE}/feedback/summary`);
      if (!response.ok) return;
      setFeedbackSummary(await response.json());
    } catch {
      setFeedbackSummary(null);
    }
  }

  function startVoiceInput() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) return;
    const recognition = new Recognition();
    recognition.lang = 'en-IN';
    recognition.interimResults = false;
    recognition.continuous = false;
    recognition.onstart = () => setListening(true);
    recognition.onend = () => setListening(false);
    recognition.onerror = () => setListening(false);
    recognition.onresult = (event) => {
      const transcript = event.results?.[0]?.[0]?.transcript || '';
      if (transcript) setQuery(transcript);
    };
    recognition.start();
  }

  return (
    <main className="app-shell">
      <aside className="query-panel">
        <div className="brand-row">
          <div>
            <p className="eyebrow">RecipeDB Speech-to-SQL</p>
            <h1>Query Workbench</h1>
          </div>
          <span className="status-pill">M1-M5</span>
        </div>

        <div className="phase-tabs" role="tablist" aria-label="Demo phase">
          <button
            type="button"
            className={activeView === 'proof' ? 'active' : ''}
            onClick={() => setActiveView('proof')}
          >
            Phase 1 SQL
          </button>
          <button
            type="button"
            className={activeView === 'learn' ? 'active' : ''}
            onClick={() => setActiveView('learn')}
          >
            Phase 2 Learn
          </button>
        </div>

        <form onSubmit={runQuery} className="query-form">
          <label htmlFor="query">Transcript or typed query</label>
          <textarea
            id="query"
            placeholder="Type or speak a recipe query"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            rows={6}
          />

          <div className="controls">
            <button
              className="ghost-button"
              type="button"
              onClick={startVoiceInput}
              disabled={!speechSupported || listening}
            >
              {listening ? 'Listening' : 'Mic'}
            </button>
            <button type="submit" disabled={loading}>
              {loading ? 'Running' : 'Run'}
            </button>
          </div>
        </form>

      </aside>

      <section className="result-panel">
        {error && <div className="error-box">{error}</div>}

        {!result && !error && (
          <div className="empty-state">
            <h2>Ready for local SQL proof</h2>
            <p>Run a query to inspect extraction, schema grounding, generated SQL, and the feedback loop.</p>
          </div>
        )}

        {result && (
          <>
            <div className="summary-grid">
              {metrics.map(([label, value]) => (
                <div className="summary-card" key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>

            {activeView === 'proof' ? (
              <ProofView
                result={result}
                sql={sql}
                validation={validation}
                ingredients={ingredients}
                diet={diet}
                nutrition={nutrition}
                plans={plans}
              />
            ) : (
              <LearningView
                result={result}
                ingredients={ingredients}
                normalization={normalization}
                correctedQuery={correctedQuery}
                correctedIngredients={correctedIngredients}
                feedbackStatus={feedbackStatus}
                feedbackSummary={feedbackSummary}
                setCorrectedQuery={setCorrectedQuery}
                setCorrectedIngredients={setCorrectedIngredients}
                saveFeedback={saveFeedback}
              />
            )}
          </>
        )}
      </section>
    </main>
  );
}

function ProofView({ result, sql, validation, ingredients, diet, nutrition, plans }) {
  return (
    <>
      <section className="band">
        <div className="band-heading">
          <h2>Intent</h2>
          <span>{result.normalization?.changed ? 'Normalized transcript' : 'Direct transcript'}</span>
        </div>
        <ChipRow label="Ingredients" values={ingredients} tone="green" />
        <ChipRow label="Diet" values={diet} tone="blue" />
        <ChipRow label="Nutrition" values={nutrition.map(formatNutrition)} tone="amber" />
      </section>

      <section className="band">
        <div className="band-heading">
          <h2>Generated SQL</h2>
          <span>{validation.ok === false ? 'Validation failed' : 'Validation passed'}</span>
        </div>
        <pre className="code-block">{sql.query || 'No SQL returned'}</pre>
        <div className="param-row">
          {(sql.params || []).map((param, index) => (
            <span className="param-chip" key={`${param}-${index}`}>{JSON.stringify(param)}</span>
          ))}
        </div>
        {validation.warnings?.length > 0 && (
          <div className="notice-list">
            {validation.warnings.map((warning) => <p key={warning}>{warning}</p>)}
          </div>
        )}
      </section>

      <section className="band">
        <div className="band-heading">
          <h2>Execution Plan</h2>
          <span>{plans.length} plan(s)</span>
        </div>
        <div className="plan-list">
          {plans.map((plan, index) => (
            <div className="plan-row" key={`${plan.label}-${index}`}>
              <strong>{plan.label || 'plan'}</strong>
              <code>{plan.method || 'GET'} {plan.path}</code>
              <span>{formatParams(plan.params)}</span>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}

function LearningView({
  result,
  ingredients,
  normalization,
  correctedQuery,
  correctedIngredients,
  feedbackStatus,
  feedbackSummary,
  setCorrectedQuery,
  setCorrectedIngredients,
  saveFeedback,
}) {
  return (
    <>
      <section className="band">
        <div className="band-heading">
          <h2>Transcript</h2>
          <span>{normalization.changed ? 'Correction applied' : 'No correction applied'}</span>
        </div>
        <div className="transcript-grid">
          <div>
            <label>Original</label>
            <p className="text-surface">{result.input_query || ''}</p>
          </div>
          <div>
            <label>Pipeline text</label>
            <p className="text-surface">{result.normalized_query || result.input_query || ''}</p>
          </div>
        </div>
        <ChipRow
          label="Applied corrections"
          values={(normalization.corrections || []).map((item) => `${item.source} -> ${item.target}`)}
          tone="purple"
        />
      </section>

      <section className="band">
        <div className="band-heading">
          <h2>Save Correction</h2>
          <span>{ingredients.length} predicted ingredient(s)</span>
        </div>
        <form className="feedback-form" onSubmit={saveFeedback}>
          <label htmlFor="corrected-query">Correct transcript</label>
          <textarea
            id="corrected-query"
            rows={3}
            value={correctedQuery}
            onChange={(event) => setCorrectedQuery(event.target.value)}
          />
          <label htmlFor="corrected-ingredients">Correct ingredients</label>
          <input
            id="corrected-ingredients"
            value={correctedIngredients}
            onChange={(event) => setCorrectedIngredients(event.target.value)}
          />
          <button type="submit">Save correction</button>
        </form>
        {feedbackStatus && <p className="success-line">{feedbackStatus}</p>}
      </section>

      <section className="band">
        <div className="band-heading">
          <h2>Learned Replacements</h2>
          <span>{feedbackSummary?.feedback_count || 0} feedback row(s)</span>
        </div>
        <div className="plan-list">
          {(feedbackSummary?.learned_replacements || []).map((item) => (
            <div className="plan-row" key={`${item.source}-${item.target}`}>
              <strong>{item.source}</strong>
              <code>{item.target}</code>
              <span>{item.count}x</span>
            </div>
          ))}
          {!(feedbackSummary?.learned_replacements || []).length && (
            <p className="muted-line">No approved corrections saved yet.</p>
          )}
        </div>
      </section>
    </>
  );
}

function ChipRow({ label, values, tone }) {
  const cleanValues = (values || []).filter(Boolean);
  return (
    <div className="chip-row">
      <span>{label}</span>
      <div>
        {cleanValues.length ? cleanValues.map((value) => (
          <b className={`chip ${tone}`} key={value}>{value}</b>
        )) : <em>None</em>}
      </div>
    </div>
  );
}

function formatNutrition(item) {
  if (!item) return '';
  const name = item.nutrient || item.nutrient_group || 'nutrition';
  return `${name} ${item.operator || ''} ${item.value ?? ''}`.trim();
}

function formatParams(params = {}) {
  const entries = Object.entries(params);
  if (!entries.length) return 'No params';
  return entries.map(([key, value]) => `${key}=${value}`).join(', ');
}

export default App;
