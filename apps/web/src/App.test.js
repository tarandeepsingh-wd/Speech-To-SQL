import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import App from './App';

test('renders the query workbench', () => {
  render(<App />);
  expect(screen.getByText(/Query Workbench/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /^Run$/i })).toBeInTheDocument();
});

test('runs a query and renders SQL proof output', async () => {
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
        input_query: 'show recipes with chicken',
        normalized_query: 'show recipes with chicken',
        normalization: { changed: false, corrections: [] },
        executed: false,
        schema_links: [{ table: 'recipes' }],
        tag_result: {
          ingredients: ['chicken'],
          dietary_styles: [],
          nutrition_filters: [],
        },
        sql: {
          query: 'SELECT r.* FROM `recipes` AS r',
          params: ['chicken'],
          validation: { ok: true, warnings: [] },
        },
        plans: [
          {
            label: 'ingredient_category_title',
            method: 'GET',
            path: 'recipe2-api/recipebyingredient/by-ingredients-categories-title',
            params: { includeIngredients: 'chicken' },
          },
        ],
      }),
    });
  });

  render(<App />);
  fireEvent.change(screen.getByLabelText(/Transcript or typed query/i), {
    target: { value: 'show recipes with chicken' },
  });
  fireEvent.click(screen.getByRole('button', { name: /^Run$/i }));

  await waitFor(() => expect(screen.getByText(/Generated SQL/i)).toBeInTheDocument());
  await waitFor(() => expect(screen.getByText(/SELECT/i)).toBeInTheDocument());
  expect(screen.getByText(/ingredient_category_title/i)).toBeInTheDocument();
});
