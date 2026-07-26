import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Activity,
  AlertTriangle,
  Brain,
  CheckCircle2,
  Clipboard,
  Code2,
  Database,
  FileJson,
  Keyboard,
  Leaf,
  Mic,
  Network,
  Play,
  RefreshCcw,
  Route,
  Save,
  Search,
  Server,
  Settings2,
  Square,
  Table2,
  Timer,
  Utensils,
} from 'lucide-react';
import './App.css';

const API_BASE = (process.env.REACT_APP_API_BASE || 'http://localhost:5000/api').replace(/\/$/, '');

const EXAMPLES = [
  'Show Indian recipes that use turmeric and whose instructions mention simmer.',
  'Find vegan recipes with spinach where spinach is categorized as a leafy green.',
  'Show recipes with salmon where the ingredient-level protein is above 20g.',
  'Find recipes under 500 calories that use an ingredient from the seafood category and have ingredient-level cholesterol.',
  'Find recipes with sugar and water ingredients where the instruction steps mention stir.',
];

const MODULE_ORDER = ['m1', 'm2', 'm3', 'm4', 'm5'];

function App() {
  const [query, setQuery] = useState('');
  const [executeApi, setExecuteApi] = useState(false);
  const [fetchLimit, setFetchLimit] = useState(50);
  const [finalLimit, setFinalLimit] = useState(5);
  const [maxApiCalls, setMaxApiCalls] = useState(2);
  const [activeModule, setActiveModule] = useState('m1');
  const [inputMode, setInputMode] = useState('text');
  const [loading, setLoading] = useState(false);
  const [listening, setListening] = useState(false);
  const [speechSupported, setSpeechSupported] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const [copyState, setCopyState] = useState('');
  const [athenaLoading, setAthenaLoading] = useState(false);
  const [athenaResult, setAthenaResult] = useState(null);
  const [athenaError, setAthenaError] = useState('');
  const [athenaSqlDraft, setAthenaSqlDraft] = useState('');
  const [feedbackStatus, setFeedbackStatus] = useState('');
  const [feedbackSummary, setFeedbackSummary] = useState(null);
  const [correctedQuery, setCorrectedQuery] = useState('');
  const [correctedIngredients, setCorrectedIngredients] = useState('');
  const recognitionRef = useRef(null);
  const manualSpeechStopRef = useRef(false);
  const finalTranscriptRef = useRef('');

  const tagResult = useMemo(() => result?.tag_result || {}, [result]);
  const sql = useMemo(() => result?.sql || {}, [result]);
  const generatedAthenaSql = sql.athena?.query || '';
  const validation = useMemo(() => sql.validation || {}, [sql]);
  const queryIr = useMemo(() => sql.ir || {}, [sql]);
  const sqlPlan = useMemo(() => sql.plan || {}, [sql]);
  const plans = useMemo(() => result?.selected_plans || result?.plans || [], [result]);
  const apiResults = useMemo(() => result?.api_results || [], [result]);
  const recipes = useMemo(() => result?.recipes || [], [result]);
  const normalization = useMemo(() => result?.normalization || {}, [result]);
  const modules = useMemo(
    () => buildModules({ result, loading, inputMode, executeApi }),
    [result, loading, inputMode, executeApi],
  );
  const activeModuleConfig = modules.find((item) => item.id === activeModule) || modules[0];

  useEffect(() => {
    const recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    setSpeechSupported(Boolean(recognition));
    loadFeedbackSummary();
  }, []);

  useEffect(() => () => {
    manualSpeechStopRef.current = true;
    if (recognitionRef.current?.abort) {
      recognitionRef.current.abort();
    }
    recognitionRef.current = null;
  }, []);

  useEffect(() => {
    if (!result) return;
    setCorrectedQuery(result.normalized_query || result.input_query || query);
    setCorrectedIngredients((tagResult.ingredient_logic?.include || tagResult.ingredients || []).join(', '));
    setActiveModule('m2');
  }, [result, query, tagResult]);

  useEffect(() => {
    setAthenaSqlDraft(generatedAthenaSql);
  }, [generatedAthenaSql]);

  async function runQuery(event, overrideQuery) {
    event?.preventDefault();
    const text = (overrideQuery || query).trim();
    if (!text) return;

    setLoading(true);
    setError('');
    setFeedbackStatus('');
    setCopyState('');
    setAthenaResult(null);
    setAthenaError('');
    setAthenaSqlDraft('');
    setResult(null);
    setActiveModule('m1');

    try {
      const response = await fetch(apiPath('/query'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: text,
          execute: executeApi,
          max_api_calls: Number(maxApiCalls),
          fetch_limit: Number(fetchLimit),
          final_limit: Number(finalLimit),
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(apiErrorMessage(payload, 'Query failed'));
      }
      setQuery(text);
      setResult(payload);
      setActiveModule(payload.sql?.validation?.ok === false ? 'm4' : 'm5');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  async function saveFeedback(event) {
    event.preventDefault();
    if (!result) return;

    setFeedbackStatus('Saving');
    setError('');

    const parsedIngredients = correctedIngredients
      .split(',')
      .map((item) => item.trim().toLowerCase())
      .filter(Boolean);

    try {
      const response = await fetch(apiPath('/feedback/search'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          original_query: result.input_query || query,
          corrected_query: correctedQuery,
          predicted_ingredients: tagResult.ingredient_logic?.include || tagResult.ingredients || [],
          corrected_ingredients: parsedIngredients,
          source: 'web-ui',
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(apiErrorMessage(payload, 'Feedback failed'));
      }
      setFeedbackSummary(payload.summary);
      setFeedbackStatus(`Saved ${payload.corrections?.length || 0} correction(s)`);
    } catch (err) {
      setFeedbackStatus('');
      setError(err.message);
    }
  }

  async function loadFeedbackSummary() {
    try {
      const response = await fetch(apiPath('/feedback/summary'));
      if (!response.ok) return;
      setFeedbackSummary(await response.json());
    } catch {
      setFeedbackSummary(null);
    }
  }

  function toggleVoiceInput() {
    if (listening) {
      stopVoiceInput();
      return;
    }
    startVoiceInput();
  }

  function stopVoiceInput() {
    manualSpeechStopRef.current = true;
    if (recognitionRef.current?.stop) {
      recognitionRef.current.stop();
      return;
    }
    recognitionRef.current = null;
    setListening(false);
  }

  function startVoiceInput() {
    const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!Recognition) return;
    const recognition = new Recognition();
    recognitionRef.current = recognition;
    manualSpeechStopRef.current = false;
    finalTranscriptRef.current = query.trim();
    recognition.lang = 'en-IN';
    recognition.interimResults = true;
    recognition.continuous = true;
    recognition.onstart = () => {
      setInputMode('speech');
      setListening(true);
    };
    recognition.onend = () => {
      if (!manualSpeechStopRef.current && recognitionRef.current === recognition) {
        try {
          recognition.start();
          return;
        } catch {
          // Browser may reject immediate restarts; fall through to stopped state.
        }
      }
      if (recognitionRef.current === recognition) {
        recognitionRef.current = null;
      }
      setListening(false);
    };
    recognition.onerror = (event) => {
      manualSpeechStopRef.current = true;
      recognitionRef.current = null;
      setListening(false);
      setError(event?.error ? `Speech input failed: ${event.error}` : 'Speech input failed');
    };
    recognition.onresult = (event) => {
      let finalText = finalTranscriptRef.current;
      let interimText = '';
      for (let index = event.resultIndex || 0; index < event.results.length; index += 1) {
        const transcript = event.results[index]?.[0]?.transcript || '';
        if (!transcript) continue;
        if (event.results[index].isFinal) {
          finalText = `${finalText} ${transcript}`.trim();
        } else {
          interimText = `${interimText} ${transcript}`.trim();
        }
      }
      finalTranscriptRef.current = finalText;
      const combined = `${finalText} ${interimText}`.trim();
      if (combined) setQuery(combined);
    };
    try {
      recognition.start();
    } catch (err) {
      recognitionRef.current = null;
      setListening(false);
      setError(err.message || 'Speech input failed');
    }
  }

  function resetWorkbench() {
    setQuery('');
    setResult(null);
    setError('');
    setFeedbackStatus('');
    setCopyState('');
    setAthenaResult(null);
    setAthenaError('');
    setAthenaSqlDraft('');
    setActiveModule('m1');
    setInputMode('text');
  }

  async function copySql(kind = 'sql') {
    const text = kind === 'athena' ? athenaSqlDraft : sql.query;
    if (!text) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopyState(kind === 'athena' ? 'Athena copied' : 'SQL copied');
    } catch {
      setCopyState('Copy failed');
    }
  }

  async function runAthena() {
    const queryText = athenaSqlDraft.trim();
    if (!queryText) return;
    setAthenaLoading(true);
    setAthenaResult(null);
    setAthenaError('');
    try {
      const response = await fetch(apiPath('/athena/execute'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          query: queryText,
          max_rows: Number(finalLimit) || 20,
          timeout_seconds: 120,
        }),
      });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(apiErrorMessage(payload, 'Athena execution failed'));
      }
      setAthenaResult(payload);
    } catch (err) {
      setAthenaError(err.message);
    } finally {
      setAthenaLoading(false);
    }
  }

  return (
    <main className="app-shell">
      <aside className="left-rail">
        <header className="brand-block">
          <div className="brand-mark">
            <Leaf size={20} strokeWidth={2.4} />
          </div>
          <div>
            <p className="eyebrow">RecipeDB</p>
            <h1>Query Console</h1>
          </div>
        </header>

        <form className="query-form" onSubmit={runQuery}>
          <div className="field-row">
            <label htmlFor="query">Natural language query</label>
            <span className="mode-pill">{inputMode === 'speech' ? 'Speech' : 'Text'}</span>
          </div>
          <textarea
            id="query"
            placeholder="Show vegan recipes under 300 calories with more than 10g protein."
            value={query}
            onChange={(event) => {
              setInputMode('text');
              setQuery(event.target.value);
            }}
            rows={7}
          />
          <div className="primary-actions">
            <button
              className={`icon-button ${listening ? 'is-listening' : ''}`}
              type="button"
              aria-label={listening ? 'Stop listening' : 'Start voice input'}
              title={listening ? 'Stop listening' : 'Start voice input'}
              onClick={toggleVoiceInput}
              disabled={!speechSupported}
            >
              {listening ? <Square size={18} /> : <Mic size={18} />}
            </button>
            <button className="run-button" type="submit" disabled={loading || !query.trim()}>
              <Play size={18} />
              <span>{loading ? 'Running' : 'Run pipeline'}</span>
            </button>
          </div>
        </form>

        <section className="control-section">
          <div className="section-title">
            <Settings2 size={16} />
            <span>Run settings</span>
          </div>
          <label className="toggle-row">
            <span>Live API execution</span>
            <input
              type="checkbox"
              checked={executeApi}
              onChange={(event) => setExecuteApi(event.target.checked)}
            />
          </label>
          <div className="number-grid">
            <label>
              API calls
              <input
                type="number"
                min="1"
                max="5"
                value={maxApiCalls}
                onChange={(event) => setMaxApiCalls(event.target.value)}
              />
            </label>
            <label>
              Fetch limit
              <input
                type="number"
                min="1"
                max="200"
                value={fetchLimit}
                onChange={(event) => setFetchLimit(event.target.value)}
              />
            </label>
            <label>
              Result limit
              <input
                type="number"
                min="1"
                max="50"
                value={finalLimit}
                onChange={(event) => setFinalLimit(event.target.value)}
              />
            </label>
          </div>
        </section>

        <details className="control-section examples-section">
          <summary className="section-title">
            <Search size={16} />
            <span>Example statements</span>
          </summary>
          <div className="example-list">
            {EXAMPLES.map((item) => (
              <button
                type="button"
                className="example-button"
                key={item}
                onClick={(event) => runQuery(event, item)}
                disabled={loading}
              >
                {item}
              </button>
            ))}
          </div>
        </details>

        <button className="reset-button" type="button" onClick={resetWorkbench}>
          <RefreshCcw size={16} />
          <span>Reset</span>
        </button>
      </aside>

      <section className="workspace">
        <header className="workspace-header">
          <div>
            <p className="eyebrow">Speech-to-SQL pipeline</p>
            <h2>RecipeDB Workbench</h2>
          </div>
          <div className="health-row">
            <StatusBadge ok={!error} label={error ? 'Needs attention' : 'App ready'} />
            <span className="api-chip">
              <Server size={14} />
              {API_BASE}
            </span>
          </div>
        </header>

        {error && (
          <div className="alert-box">
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        )}

        <MetricsBar
          result={result}
          tagResult={tagResult}
          sql={sql}
          validation={validation}
          plans={plans}
          apiResults={apiResults}
          recipes={recipes}
        />

        <div className={`work-grid ${result ? '' : 'single-column'}`}>
          <section className="module-strip" aria-label="Pipeline modules">
            {modules.map((module) => (
              <button
                key={module.id}
                type="button"
                className={`module-card ${activeModule === module.id ? 'active' : ''}`}
                onClick={() => setActiveModule(module.id)}
              >
                <span className={`status-dot ${module.status}`} />
                <span className="module-icon">{module.icon}</span>
                <span>
                  <strong>{module.title}</strong>
                  <em>{module.subtitle}</em>
                </span>
              </button>
            ))}
          </section>

          <section className="detail-surface">
            <div className="detail-heading">
              <div>
                <p>{activeModuleConfig.kicker}</p>
                <h3>{activeModuleConfig.title}</h3>
              </div>
              <StatusBadge ok={activeModuleConfig.status !== 'fail'} label={activeModuleConfig.label} />
            </div>

            {activeModule === 'm1' && (
              <SpeechPanel
                query={query}
                result={result}
                normalization={normalization}
                inputMode={inputMode}
                listening={listening}
              />
            )}
            {activeModule === 'm2' && <NluPanel tagResult={tagResult} />}
            {activeModule === 'm3' && <GroundingPanel tagResult={tagResult} queryIr={queryIr} result={result} />}
            {activeModule === 'm4' && (
              <SqlPanel
                sql={sql}
                sqlPlan={sqlPlan}
                validation={validation}
                copySql={copySql}
                copyState={copyState}
                runAthena={runAthena}
                athenaLoading={athenaLoading}
                athenaResult={athenaResult}
                athenaError={athenaError}
                athenaSqlDraft={athenaSqlDraft}
                setAthenaSqlDraft={setAthenaSqlDraft}
                generatedAthenaSql={generatedAthenaSql}
              />
            )}
            {activeModule === 'm5' && (
              <ExecutionPanel
                result={result}
                plans={plans}
                apiResults={apiResults}
                recipes={recipes}
                executeApi={executeApi}
              />
            )}
          </section>

          {result && (
            <aside className="inspector">
              <LearningPanel
                result={result}
                tagResult={tagResult}
                normalization={normalization}
                correctedQuery={correctedQuery}
                correctedIngredients={correctedIngredients}
                feedbackStatus={feedbackStatus}
                feedbackSummary={feedbackSummary}
                setCorrectedQuery={setCorrectedQuery}
                setCorrectedIngredients={setCorrectedIngredients}
                saveFeedback={saveFeedback}
              />
              <RawPanel result={result} />
            </aside>
          )}
        </div>
      </section>
    </main>
  );
}

function SpeechPanel({ query, result, normalization, inputMode, listening }) {
  const transcript = result?.input_query || query || '';
  const normalized = result?.normalized_query || transcript;
  const corrections = normalization.corrections || [];

  return (
    <div className="panel-stack">
      <KeyValueGrid
        items={[
          ['Input mode', listening ? 'Listening' : inputMode === 'speech' ? 'Speech' : 'Typed text'],
          ['Transcript', transcript || 'Awaiting query'],
          ['Pipeline text', normalized || 'Awaiting query'],
          ['Corrections', String(corrections.length)],
        ]}
      />
      <TokenList
        title="Transcript corrections"
        icon={<Keyboard size={16} />}
        empty="None"
        values={corrections.map((item) => `${item.source} -> ${item.target}`)}
        tone="amber"
      />
    </div>
  );
}

function NluPanel({ tagResult }) {
  const ingredientLogic = tagResult.ingredient_logic || {};
  const nutrition = tagResult.nutrition_filters || [];
  const categories = [...(tagResult.categories || []), ...(tagResult.category_filters || [])];
  const rawSpans = tagResult.raw_spans || [];
  const sqlOperators = tagResult.sql_operators || [];

  return (
    <div className="panel-stack">
      <div className="entity-grid">
        <TokenList title="Ingredients" icon={<Leaf size={16} />} values={ingredientLogic.include || tagResult.ingredients || []} tone="green" />
        <TokenList title="Excluded" icon={<AlertTriangle size={16} />} values={ingredientLogic.exclude || []} tone="red" />
        <TokenList title="Dietary" icon={<Utensils size={16} />} values={tagResult.dietary_styles || []} tone="blue" />
        <TokenList title="Utensils" icon={<Timer size={16} />} values={tagResult.utensils || []} tone="yellow" />
        <TokenList title="Actions" icon={<Activity size={16} />} values={(tagResult.actions || []).map((item) => item.action)} tone="coral" />
        <TokenList title="Categories" icon={<Table2 size={16} />} values={categories.map(formatCategory)} tone="violet" />
      </div>
      <DataTable
        title="Nutrition filters"
        columns={['Nutrient', 'Operator', 'Value', 'Scope']}
        rows={nutrition.map((item) => [
          item.nutrient || item.nutrient_group || '',
          item.operator || '',
          item.value ?? '',
          item.table_source || item.group || 'recipe',
        ])}
      />
      <DataTable
        title="Raw spans"
        columns={['Text', 'Canonical', 'Layer', 'Confidence']}
        rows={rawSpans.map((item) => [
          item.text,
          item.canonical,
          item.layer,
          item.reranker_confidence ?? item.confidence ?? '',
        ])}
      />
      <TokenList
        title="SQL operators"
        icon={<Code2 size={16} />}
        values={sqlOperators.map((item) => `${item.canonical} (${item.layer || item.type})`)}
        tone="gray"
      />
    </div>
  );
}

function GroundingPanel({ tagResult, queryIr, result }) {
  const grounding = tagResult._grounding || [];
  const schemaLinks = result?.schema_links || tagResult._schema_links || [];
  const predicates = queryIr.predicates || [];
  const enrichments = result?.intent_enrichments || tagResult._intent_enrichments || [];

  return (
    <div className="panel-stack">
      <KeyValueGrid
        items={[
          ['Root', queryIr.root || 'recipe'],
          ['Predicates', String(predicates.length)],
          ['Schema links', String(schemaLinks.length)],
          ['Enrichments', String(enrichments.length)],
        ]}
      />
      <DataTable
        title="QueryIR predicates"
        columns={['Source', 'Field', 'Operator', 'Values']}
        rows={predicates.map((item) => [
          item.source,
          item.field,
          item.operator,
          (item.values || []).join(', '),
        ])}
      />
      <DataTable
        title="Schema grounding"
        columns={['Source', 'Table', 'Column', 'Reason']}
        rows={grounding.map((item) => [
          item.source,
          item.table,
          item.column || item.field,
          item.reason,
        ])}
      />
      <DataTable
        title="Schema links"
        columns={['Source', 'Original', 'Resolved', 'Reason']}
        rows={schemaLinks.map((item) => [
          item.source,
          item.original,
          item.resolved,
          item.reason,
        ])}
      />
    </div>
  );
}

function SqlPanel({
  sql,
  sqlPlan,
  validation,
  copySql,
  copyState,
  runAthena,
  athenaLoading,
  athenaResult,
  athenaError,
  athenaSqlDraft,
  setAthenaSqlDraft,
  generatedAthenaSql,
}) {
  const joins = sqlPlan.joins || [];
  const where = sqlPlan.where || [];
  const params = sql.params || [];
  const editableAthenaSql = athenaSqlDraft || '';
  const warnings = validation.warnings || [];
  const errors = validation.errors || [];
  const formatAthenaNumber = (value) => {
    if (value === undefined || value === null || value === '') return '-';
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return String(value);
    return numeric.toLocaleString();
  };

  return (
    <div className="panel-stack">
      <div className="sql-toolbar">
        <StatusBadge ok={validation.ok !== false} label={validation.ok === false ? 'Invalid SQL' : 'Valid SQL'} />
        <div className="sql-actions">
          <button className="compact-button" type="button" onClick={() => copySql('sql')} disabled={!sql.query}>
            <Clipboard size={15} />
            <span>{copyState === 'SQL copied' ? copyState : 'Copy SQL'}</span>
          </button>
          <button className="compact-button" type="button" onClick={() => copySql('athena')} disabled={!editableAthenaSql.trim()}>
            <Clipboard size={15} />
            <span>{copyState === 'Athena copied' ? copyState : 'Copy Athena'}</span>
          </button>
          <button
            className="compact-button"
            type="button"
            onClick={() => setAthenaSqlDraft(generatedAthenaSql)}
            disabled={!generatedAthenaSql || generatedAthenaSql === editableAthenaSql}
          >
            <RefreshCcw size={15} />
            <span>Reset Athena</span>
          </button>
          <button className="compact-button" type="button" onClick={runAthena} disabled={!editableAthenaSql.trim() || athenaLoading}>
            <Play size={15} />
            <span>{athenaLoading ? 'Running Athena' : 'Run Athena'}</span>
          </button>
        </div>
      </div>
      <section className="sql-code-section">
        <div className="sql-block-heading">
          <Code2 size={15} />
          <span>Parameterized SQL</span>
        </div>
        <pre className="code-block">{sql.query || 'No SQL generated yet'}</pre>
      </section>
      <section className="sql-code-section">
        <div className="sql-block-heading">
          <Database size={15} />
          <span>Athena SQL</span>
        </div>
        <textarea
          className="code-block athena-block athena-editor"
          value={editableAthenaSql}
          onChange={(event) => setAthenaSqlDraft(event.target.value)}
          placeholder="Paste or edit Athena SQL here, then run it."
          spellCheck="false"
          wrap="soft"
        />
      </section>
      {athenaError && (
        <div className="message-list error">
          <p>{athenaError}</p>
        </div>
      )}
      {athenaResult && (
        <section className="athena-results">
          <KeyValueGrid
            items={[
              ['Athena state', athenaResult.state || 'Unknown'],
              ['Rows shown', String(athenaResult.row_count ?? 0)],
              ['Bytes scanned', formatAthenaNumber(athenaResult.bytes_scanned)],
              ['Execution ID', athenaResult.query_execution_id || '-'],
            ]}
          />
          <AthenaRowsTable
            columns={athenaResult.columns || []}
            rows={athenaResult.rows || []}
          />
        </section>
      )}
      <TokenList title="Parameters" icon={<Database size={16} />} values={params.map((param) => JSON.stringify(param))} tone="gray" />
      <DataTable
        title="Joins"
        columns={['Table', 'Alias', 'Source', 'On']}
        rows={joins.map((item) => [item.table, item.alias, item.source, item.on])}
      />
      <DataTable
        title="Where clauses"
        columns={['Source', 'SQL', 'Params']}
        rows={where.map((item) => [item.source, item.sql, (item.params || []).join(', ')])}
      />
      {(warnings.length > 0 || errors.length > 0) && (
        <div className="message-list">
          {[...warnings, ...errors].map((item) => (
            <p key={item}>{item}</p>
          ))}
        </div>
      )}
    </div>
  );
}

function ExecutionPanel({ result, plans, apiResults, recipes, executeApi }) {
  const executed = Boolean(result?.executed);
  const count = result?.count ?? recipes.length;
  const apiFailures = (apiResults || []).filter((item) => item.ok === false);
  const verificationWarnings = result?.api_verification?.warnings || [];

  return (
    <div className="panel-stack">
      <KeyValueGrid
        items={[
          ['Mode', executed ? 'Live API' : executeApi ? 'Requested' : 'Planned only'],
          ['Selected plans', String(plans.length)],
          ['Returned recipes', String(count)],
          ['Post filter', executed ? 'Applied' : 'Preview'],
        ]}
      />
      <DataTable
        title="API plans"
        columns={['Role', 'Label', 'Path', 'Params']}
        rows={plans.map((plan) => [
          plan.role || 'plan',
          plan.label,
          `${plan.method || 'GET'} ${plan.path}`,
          formatParams(plan.params),
        ])}
      />
      {executed && (
        <DataTable
          title="Live API responses"
          columns={['Label', 'Status', 'Recipes', 'Error']}
          rows={(apiResults || []).map((item) => [
            item.label,
            item.status_code ?? (item.ok ? 'ok' : 'network'),
            item.recipe_count ?? 0,
            item.error || '',
          ])}
        />
      )}
      {(apiFailures.length > 0 || verificationWarnings.length > 0) && (
        <div className={`message-list ${apiFailures.length ? 'error' : ''}`}>
          {apiFailures.map((item) => (
            <p key={`${item.label}-${item.error}`}>{item.label}: {shortenError(item.error)}</p>
          ))}
          {verificationWarnings.map((warning) => (
            <p key={warning}>{warning}</p>
          ))}
        </div>
      )}
      <RecipeResults
        recipes={recipes}
        executed={executed}
        executeApi={executeApi}
        apiFailures={apiFailures}
        apiResults={apiResults}
      />
    </div>
  );
}

function LearningPanel({
  result,
  tagResult,
  normalization,
  correctedQuery,
  correctedIngredients,
  feedbackStatus,
  feedbackSummary,
  setCorrectedQuery,
  setCorrectedIngredients,
  saveFeedback,
}) {
  const replacements = feedbackSummary?.learned_replacements || [];
  const disabled = !result;

  return (
    <section className="side-panel">
      <div className="side-heading">
        <Brain size={17} />
        <h3>Learning loop</h3>
      </div>
      <KeyValueGrid
        compact
        items={[
          ['Normalization', normalization.changed ? 'Applied' : 'None'],
          ['Feedback rows', String(feedbackSummary?.feedback_count || 0)],
        ]}
      />
      <form className="feedback-form" onSubmit={saveFeedback}>
        <label htmlFor="corrected-query">Correct transcript</label>
        <textarea
          id="corrected-query"
          rows={3}
          value={correctedQuery}
          disabled={disabled}
          onChange={(event) => setCorrectedQuery(event.target.value)}
        />
        <label htmlFor="corrected-ingredients">Correct ingredients</label>
        <input
          id="corrected-ingredients"
          value={correctedIngredients}
          disabled={disabled}
          onChange={(event) => setCorrectedIngredients(event.target.value)}
        />
        <button className="compact-button save-button" type="submit" disabled={disabled}>
          <Save size={15} />
          <span>Save correction</span>
        </button>
      </form>
      {feedbackStatus && <p className="success-line">{feedbackStatus}</p>}
      <div className="learned-list">
        {replacements.slice(0, 4).map((item) => (
          <div className="learned-row" key={`${item.source}-${item.target}`}>
            <span>{item.source}</span>
            <strong>{item.target}</strong>
            <em>{item.count}x</em>
          </div>
        ))}
        {!replacements.length && <p className="muted-line">No saved replacements</p>}
      </div>
      <span className="hidden-data">{(tagResult.raw_spans || []).length}</span>
    </section>
  );
}

function RawPanel({ result }) {
  return (
    <section className="side-panel">
      <div className="side-heading">
        <FileJson size={17} />
        <h3>Raw payload</h3>
      </div>
      <details>
        <summary>JSON</summary>
        <pre className="raw-block">{JSON.stringify(result || {}, null, 2)}</pre>
      </details>
    </section>
  );
}

function MetricsBar({ result, tagResult, sql, validation, plans, apiResults, recipes }) {
  const ingredientCount = (tagResult.ingredient_logic?.include || tagResult.ingredients || []).length;
  const tableCount = extractTables(sql.query).length;
  const apiFailures = (apiResults || []).filter((item) => item.ok === false).length;
  const metricItems = [
    ['Transcript', result?.normalized_query ? 'Ready' : 'Waiting', <Keyboard size={17} />],
    ['Entities', String(ingredientCount + (tagResult.nutrition_filters || []).length), <Brain size={17} />],
    ['Tables', String(tableCount), <Table2 size={17} />],
    ['SQL', validation.ok === false ? 'Fail' : result ? 'Pass' : 'Waiting', <Code2 size={17} />],
    ['Plans', apiFailures ? 'API issue' : String(plans.length), <Route size={17} />],
    ['Rows', String(result?.count ?? recipes.length), <Database size={17} />],
  ];

  return (
    <section className="metrics-bar">
      {metricItems.map(([label, value, icon]) => (
        <div className="metric-tile" key={label}>
          <span>{icon}</span>
          <em>{label}</em>
          <strong>{value}</strong>
        </div>
      ))}
    </section>
  );
}

function RecipeResults({ recipes, executed, executeApi, apiFailures, apiResults }) {
  if (!recipes?.length) {
    let message = 'No live recipes loaded';
    const tokenIssue = apiFailures?.some((item) => isTokenIssue(item.error));
    if (!executed) {
      message = executeApi
        ? 'Live API execution was requested but no response has loaded yet'
        : 'Live API not executed. Enable Live API execution to fetch recipes.';
    } else if (tokenIssue) {
      message = 'RecipeDB API token quota is exhausted. Ask for API token/access refresh.';
    } else if (apiFailures?.length) {
      message = 'Live API request failed. Check network/API access.';
    } else if (apiResults?.length) {
      message = 'Live API returned 0 recipes for this query.';
    }
    return (
      <section className="results-empty">
        <Database size={18} />
        <span>{message}</span>
      </section>
    );
  }

  return (
    <section className="recipe-list">
      {recipes.slice(0, 6).map((recipe, index) => {
        const id = recipe.Recipe_id || recipe.recipe_id || recipe.id || index;
        const title = recipe.Recipe_title || recipe.recipe_title || recipe.title || `Recipe ${id}`;
        const calories = recipe.Calories || recipe.calories;
        const region = recipe.Region || recipe.region;
        return (
          <article className="recipe-row" key={id}>
            <strong>{title}</strong>
            <span>{region || 'Region unavailable'}</span>
            <em>{calories ? `${calories} calories` : 'Nutrition unavailable'}</em>
          </article>
        );
      })}
    </section>
  );
}

function isTokenIssue(error = '') {
  return String(error).toLowerCase().includes('not enough tokens');
}

function TokenList({ title, icon, values, tone = 'gray', empty = 'None' }) {
  const cleanValues = (values || []).filter((value) => value !== undefined && value !== null && String(value).trim());
  return (
    <section className="token-section">
      <div className="token-title">
        {icon}
        <span>{title}</span>
      </div>
      <div className="token-wrap">
        {cleanValues.length ? cleanValues.map((value) => (
          <span className={`token ${tone}`} key={`${title}-${value}`}>
            {String(value)}
          </span>
        )) : <em>{empty}</em>}
      </div>
    </section>
  );
}

function DataTable({ title, columns, rows }) {
  const safeRows = rows || [];
  return (
    <section className="data-section">
      <div className="data-title">
        <Table2 size={16} />
        <span>{title}</span>
      </div>
      {!safeRows.length ? (
        <p className="muted-line">No rows</p>
      ) : (
        <div className="table-scroll">
          <table>
            <thead>
              <tr>
                {columns.map((column) => <th key={column}>{column}</th>)}
              </tr>
            </thead>
            <tbody>
              {safeRows.map((row, index) => (
                <tr key={`${title}-${index}`}>
                  {row.map((cell, cellIndex) => (
                    <td key={`${title}-${index}-${cellIndex}`}>{formatCell(cell)}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function AthenaRowsTable({ columns, rows }) {
  const safeColumns = columns || [];
  const safeRows = rows || [];
  return (
    <section className="data-section athena-row-section">
      <div className="data-title">
        <Database size={16} />
        <span>Athena rows</span>
      </div>
      {!safeRows.length ? (
        <p className="muted-line">No rows</p>
      ) : (
        <div className="table-scroll athena-table-scroll">
          <table className="athena-table">
            <thead>
              <tr>
                {safeColumns.map((column) => (
                  <th key={column} title={column}>{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {safeRows.map((row, index) => (
                <tr key={`athena-${index}`}>
                  {safeColumns.map((column, cellIndex) => {
                    const value = row[cellIndex] ?? '';
                    return (
                      <td key={`athena-${index}-${column}`} title={String(value || '-')}>
                        {formatCell(value)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

function KeyValueGrid({ items, compact = false }) {
  return (
    <div className={`kv-grid ${compact ? 'compact' : ''}`}>
      {items.map(([label, value]) => (
        <div className="kv-item" key={label}>
          <span>{label}</span>
          <strong>{value}</strong>
        </div>
      ))}
    </div>
  );
}

function StatusBadge({ ok, label }) {
  return (
    <span className={`status-badge ${ok ? 'ok' : 'bad'}`}>
      {ok ? <CheckCircle2 size={14} /> : <AlertTriangle size={14} />}
      {label}
    </span>
  );
}

function buildModules({ result, loading, inputMode, executeApi }) {
  const validation = result?.sql?.validation || {};
  const hasSql = Boolean(result?.sql?.query);
  const hasIr = Boolean(result?.sql?.ir);
  const executed = Boolean(result?.executed);
  const hasApiFailure = Boolean((result?.api_results || []).some((item) => item.ok === false));
  const failed = validation.ok === false;

  const statusFor = (id) => {
    if (loading && id === 'm1') return 'run';
    if (!result) return id === 'm1' && inputMode === 'speech' ? 'skip' : 'idle';
    if (id === 'm4' && failed) return 'fail';
    if (id === 'm5' && hasApiFailure) return 'fail';
    if (id === 'm5' && !executeApi && !executed) return 'plan';
    return 'pass';
  };

  const moduleMeta = {
    m1: {
      title: 'M1 Speech',
      subtitle: inputMode === 'speech' ? 'transcript capture' : 'typed transcript',
      kicker: 'Input layer',
      icon: <Mic size={18} />,
    },
    m2: {
      title: 'M2 NLU',
      subtitle: 'entities and operators',
      kicker: 'Understanding layer',
      icon: <Brain size={18} />,
    },
    m3: {
      title: 'M3 QueryIR',
      subtitle: hasIr ? 'schema grounded' : 'waiting',
      kicker: 'Schema layer',
      icon: <Network size={18} />,
    },
    m4: {
      title: 'M4 SQL',
      subtitle: hasSql ? 'query rendered' : 'waiting',
      kicker: 'SQL layer',
      icon: <Code2 size={18} />,
    },
    m5: {
      title: 'M5 Verification',
      subtitle: executed ? 'live results' : 'API plan',
      kicker: 'Execution layer',
      icon: <Database size={18} />,
    },
  };

  return MODULE_ORDER.map((id) => {
    const status = statusFor(id);
    return {
      id,
      status,
      label: statusLabel(status),
      ...moduleMeta[id],
    };
  });
}

function statusLabel(status) {
  if (status === 'pass') return 'Pass';
  if (status === 'fail') return 'Fail';
  if (status === 'run') return 'Running';
  if (status === 'plan') return 'Planned';
  if (status === 'skip') return 'Ready';
  return 'Waiting';
}

function apiPath(path) {
  return `${API_BASE}${path}`;
}

function apiErrorMessage(payload, fallback) {
  const detail = payload?.detail || payload?.error;
  if (!detail) return fallback;
  if (typeof detail === 'string') return detail;
  try {
    return JSON.stringify(detail);
  } catch {
    return fallback;
  }
}

function formatCategory(item) {
  if (!item) return '';
  if (typeof item === 'string') return item;
  return item.value || item.term || item.category || item.field || '';
}

function formatParams(params = {}) {
  const entries = Object.entries(params || {});
  if (!entries.length) return 'No params';
  return entries.map(([key, value]) => `${key}=${value}`).join(', ');
}

function shortenError(error = '') {
  if (!error) return 'Request failed';
  const text = String(error);
  const timeout = text.match(/Connection to [^)]+ timed out[^)]*/i);
  if (timeout) return timeout[0];
  return text.length > 180 ? `${text.slice(0, 177)}...` : text;
}

function formatCell(cell) {
  if (cell === undefined || cell === null || cell === '') return '-';
  if (Array.isArray(cell)) return cell.join(', ');
  if (typeof cell === 'object') return JSON.stringify(cell);
  return String(cell);
}

function extractTables(sql = '') {
  const matches = [...String(sql).matchAll(/(?:FROM|JOIN)\s+`([^`]+)`/gi)];
  return [...new Set(matches.map((match) => match[1]))];
}

export default App;
