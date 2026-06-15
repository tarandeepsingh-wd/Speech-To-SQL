import { render, screen } from '@testing-library/react';
import App from './App';

test('renders the recipe query console', () => {
  render(<App />);
  expect(screen.getByText(/Recipe Query Console/i)).toBeInTheDocument();
  expect(screen.getByRole('button', { name: /Run query/i })).toBeInTheDocument();
});
