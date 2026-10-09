import {
  addDecimals,
  compareDecimals,
  decimalsEqual,
  isDecimalString,
  multiplyDecimals,
  parseDecimal,
  roundDecimal,
  subtractDecimals,
  toDecimalString,
} from './decimal';

describe('decimal', () => {
  it('validates decimal strings', () => {
    expect(isDecimalString('1530.40')).toBe(true);
    expect(isDecimalString('-0.5')).toBe(true);
    expect(isDecimalString('12')).toBe(true);
    expect(isDecimalString('1e3')).toBe(false);
    expect(isDecimalString('1,530.40')).toBe(false);
    expect(isDecimalString('.5')).toBe(false);
    expect(isDecimalString('')).toBe(false);
  });

  it('round-trips through parse and format, preserving scale', () => {
    for (const value of ['0', '0.00', '1530.40', '-0.05', '1000000000000000000.01']) {
      expect(toDecimalString(parseDecimal(value))).toBe(value);
    }
  });

  it('throws on malformed input', () => {
    expect(() => parseDecimal('abc')).toThrow(RangeError);
  });

  it('adds without floating point drift', () => {
    expect(addDecimals('0.1', '0.2')).toBe('0.3');
    expect(addDecimals('1530.40', '0.6', '-31')).toBe('1500.00');
    expect(addDecimals()).toBe('0');
  });

  it('subtracts across scales', () => {
    expect(subtractDecimals('1530.40', '230.4')).toBe('1300.00');
    expect(subtractDecimals('0.3', '0.1')).toBe('0.2');
    expect(subtractDecimals('1', '1.005')).toBe('-0.005');
  });

  it('multiplies exactly', () => {
    expect(multiplyDecimals('3', '19.99')).toBe('59.97');
    expect(multiplyDecimals('1.5', '-0.25')).toBe('-0.375');
    expect(multiplyDecimals('0.1', '0.1')).toBe('0.01');
  });

  it('rounds half away from zero', () => {
    expect(roundDecimal('2.345', 2)).toBe('2.35');
    expect(roundDecimal('2.344', 2)).toBe('2.34');
    expect(roundDecimal('-2.345', 2)).toBe('-2.35');
    expect(roundDecimal('-0.004', 2)).toBe('0.00');
    expect(roundDecimal('7', 2)).toBe('7.00');
  });

  it('compares across scales', () => {
    expect(compareDecimals('10', '10.00')).toBe(0);
    expect(compareDecimals('9.999', '10')).toBe(-1);
    expect(compareDecimals('-1', '-2')).toBe(1);
    expect(decimalsEqual('1530.4', '1530.40')).toBe(true);
  });
});
