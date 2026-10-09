import { render, screen } from '@testing-library/react';

import { ConfidenceMeter, confidenceLevel } from './ConfidenceMeter';

describe('confidenceLevel', () => {
  it('classifies against the threshold and the low cut-off', () => {
    expect(confidenceLevel(0.85)).toBe('high');
    expect(confidenceLevel(0.849)).toBe('medium');
    expect(confidenceLevel(0.6)).toBe('medium');
    expect(confidenceLevel(0.59)).toBe('low');
    expect(confidenceLevel(null)).toBe('unknown');
    expect(confidenceLevel(0.9, 0.95)).toBe('medium');
  });
});

describe('ConfidenceMeter', () => {
  it('exposes the value as an accessible meter', () => {
    render(<ConfidenceMeter value={0.724} label="Total amount confidence" />);
    const meter = screen.getByRole('meter', { name: 'Total amount confidence' });

    expect(meter).toHaveAttribute('aria-valuenow', '72');
    expect(meter).toHaveAttribute('aria-valuemin', '0');
    expect(meter).toHaveAttribute('aria-valuemax', '100');
    expect(meter).toHaveAttribute('aria-valuetext', '72% (needs check)');
    expect(meter.parentElement).toHaveAttribute('data-level', 'medium');
  });

  it('clamps out-of-range values', () => {
    render(<ConfidenceMeter value={1.4} />);
    expect(screen.getByRole('meter')).toHaveAttribute('aria-valuenow', '100');
  });

  it('renders an unknown state without a numeric value', () => {
    render(<ConfidenceMeter value={null} />);
    const meter = screen.getByRole('meter');

    expect(meter).not.toHaveAttribute('aria-valuenow');
    expect(meter).toHaveAttribute('aria-valuetext', 'Not available');
    expect(screen.getByText('—')).toBeInTheDocument();
  });

  it('flags low confidence', () => {
    render(<ConfidenceMeter value={0.3} />);
    expect(screen.getByRole('meter')).toHaveAttribute('aria-valuetext', '30% (low)');
  });
});
