import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';
import { StatCard } from './StatCard';

describe('StatCard', () => {
  it('renders the label as a term and the value as its definition', () => {
    render(
      <dl>
        <StatCard label="Spend" value="$12.40" detail="Last 30 days" />
      </dl>,
    );

    expect(screen.getByRole('term')).toHaveTextContent('Spend');
    const definitions = screen.getAllByRole('definition');
    expect(definitions[0]).toHaveTextContent('$12.40');
    expect(definitions[1]).toHaveTextContent('Last 30 days');
  });

  it('omits the detail line when not provided', () => {
    render(
      <dl>
        <StatCard label="Requests" value="1.2K" />
      </dl>,
    );
    expect(screen.getAllByRole('definition')).toHaveLength(1);
  });
});
