import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';

afterEach(() => {
  jest.restoreAllMocks();
  delete window.SpeechRecognition;
  delete window.webkitSpeechRecognition;
});

test('renders the MERN query console workbench', () => {
  global.fetch = jest.fn(() => Promise.resolve({
    ok: false,
    json: async () => ({}),
  }));

  render(<App />);

  expect(screen.getByText(/Query Console/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Run pipeline/i })).toBeInTheDocument();
  expect(screen.getAllByText(/M1 Speech/i).length).toBeGreaterThan(0);
  expect(screen.getByText(/M5 Verification/i)).toBeInTheDocument();
});

test('runs a query and renders module SQL proof output', async () => {
  global.fetch = jest.fn((url) => {
    if (String(url).endsWith('/feedback/summary')) {
      return Promise.resolve({
        ok: true,
        json: async () => ({ feedback_count: 0, learned_replacements: [] }),
      });
    }
    return Promise.resolve({
      ok: true,
      json: async () => ({
        recipes: [],
        count: 0,
        input_query: 'show recipes with chicken',
        normalized_query: 'show recipes with chicken',
        normalization: { changed: false, corrections: [] },
        executed: false,
        schema_links: [{ source: 'ingredient', original: 'chicken', resolved: 'ingredient' }],
        tag_result: {
          ingredients: ['chicken'],
          ingredient_logic: { include: ['chicken'], exclude: [], include_operator: 'AND' },
          dietary_styles: [],
          nutrition_filters: [],
          categories: [],
          category_filters: [],
          raw_spans: [{ text: 'chicken', canonical: 'chicken', layer: 'exact_aho' }],
          sql_operators: [],
          _grounding: [
            {
              source: 'ingredients',
              table: 'ingredient_phrase',
              column: 'ingredient',
              reason: 'ingredient_logic.include',
            },
          ],
        },
        sql: {
          query: 'SELECT r.* FROM `recipes` AS r JOIN `ingredient_phrase` AS ip ON ip.`recipe_no` = r.`recipe_id`',
          params: ['chicken'],
          validation: { ok: true, warnings: [], errors: [] },
          ir: {
            root: 'recipe',
            predicates: [
              { source: 'ingredients', field: 'ingredient', operator: '=', values: ['chicken'] },
            ],
          },
          plan: {
            joins: [
              {
                table: '`ingredient_phrase`',
                alias: 'ip',
                source: 'ingredients',
                on: 'ip.`recipe_no` = r.`recipe_id`',
              },
            ],
            where: [],
          },
        },
        plans: [
          {
            label: 'ingredient_category_title',
            method: 'GET',
            role: 'primary',
            path: 'recipe2-api/recipebyingredient/by-ingredients-categories-title',
            params: { includeIngredients: 'chicken' },
          },
        ],
      }),
    });
  });

  render(<App />);
  fireEvent.change(screen.getByLabelText(/Natural language query/i), {
    target: { value: 'show recipes with chicken' },
  });
  fireEvent.click(screen.getByRole('button', { name: /Run pipeline/i }));

  await waitFor(() => expect(screen.getByText(/ingredient_category_title/i)).toBeInTheDocument());

  fireEvent.click(screen.getByRole('button', { name: /M4 SQL/i }));

  await waitFor(() => expect(screen.getByText(/Valid SQL/i)).toBeInTheDocument());
  expect(screen.getAllByText(/SELECT r\.\*/i).length).toBeGreaterThan(0);
  expect(screen.getAllByText(/ingredient_phrase/i).length).toBeGreaterThan(0);
});

test('voice input button toggles recognition start and stop', async () => {
  global.fetch = jest.fn(() => Promise.resolve({
    ok: true,
    json: async () => ({ feedback_count: 0, learned_replacements: [] }),
  }));

  const instances = [];
  class MockRecognition {
    constructor() {
      instances.push(this);
      this.start = jest.fn(() => this.onstart?.());
      this.stop = jest.fn(() => this.onend?.());
      this.abort = jest.fn();
    }
  }
  window.SpeechRecognition = MockRecognition;

  render(<App />);

  const startButton = await screen.findByRole('button', { name: /Start voice input/i });
  await waitFor(() => expect(startButton).not.toBeDisabled());

  fireEvent.click(startButton);
  expect(instances).toHaveLength(1);

  const stopButton = await screen.findByRole('button', { name: /Stop listening/i });
  const finalResult = [{ transcript: 'show recipes with milk' }];
  finalResult.isFinal = true;
  await act(async () => {
    instances[0].onresult({ resultIndex: 0, results: [finalResult] });
  });

  expect(screen.getByLabelText(/Natural language query/i)).toHaveValue('show recipes with milk');

  fireEvent.click(stopButton);
  expect(instances[0].stop).toHaveBeenCalledTimes(1);
  expect(await screen.findByRole('button', { name: /Start voice input/i })).toBeInTheDocument();

  delete window.SpeechRecognition;
});
