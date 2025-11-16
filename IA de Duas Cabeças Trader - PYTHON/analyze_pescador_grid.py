import csv
import argparse
from typing import List, Tuple, Optional
from datetime import datetime


def read_csv_ohlc(path: str) -> Tuple[List[float], List[float], List[float], List[Optional[float]], List[Optional[float]], List[Optional[int]]]:
    close: List[float] = []
    high: List[float] = []
    low: List[float] = []
    atr: List[Optional[float]] = []
    spread: List[Optional[float]] = []
    hour: List[Optional[int]] = []
    with open(path, 'r', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        # Map columns
        idx_close = header.index('close')
        idx_high = header.index('high')
        idx_low = header.index('low')
        idx_time = header.index('time') if 'time' in header else -1
        idx_atr = header.index('atr_14') if 'atr_14' in header else -1
        idx_spread = header.index('spread') if 'spread' in header else -1
        for row in reader:
            try:
                close.append(float(row[idx_close]))
                high.append(float(row[idx_high]))
                low.append(float(row[idx_low]))
                # ATR
                if idx_atr >= 0:
                    try:
                        atr.append(float(row[idx_atr]))
                    except Exception:
                        atr.append(None)
                else:
                    atr.append(None)
                # Spread
                if idx_spread >= 0:
                    try:
                        spread.append(float(row[idx_spread]))
                    except Exception:
                        spread.append(None)
                else:
                    spread.append(None)
                # Hour from time
                if idx_time >= 0:
                    try:
                        t = row[idx_time]
                        # Try ISO or common formats
                        try:
                            dt = datetime.fromisoformat(t.replace('Z', '+00:00'))
                        except Exception:
                            # Fallback: split by space
                            # e.g., 2021-01-01 12:34:00
                            dt = datetime.strptime(t[:19], '%Y-%m-%d %H:%M:%S')
                        hour.append(dt.hour)
                    except Exception:
                        hour.append(None)
                else:
                    hour.append(None)
            except Exception:
                # Skip malformed rows
                continue
    return close, high, low, atr, spread, hour


def detect_events(close: List[float], alpha: float, ema_span: int) -> Tuple[List[int], List[int], List[float]]:
    ema: List[float] = [0.0] * len(close)
    if not close:
        return [], [], []
    ema[0] = close[0]
    for i in range(1, len(close)):
        ema[i] = alpha * close[i] + (1.0 - alpha) * ema[i - 1]
    up_idx: List[int] = []
    dn_idx: List[int] = []
    slope: List[float] = [0.0] * len(close)
    for i in range(1, len(close)):
        slope[i] = ema[i] - ema[i - 1]
        if close[i - 1] < ema[i - 1] and close[i] >= ema[i] and slope[i] > 0:
            up_idx.append(i)
        if close[i - 1] > ema[i - 1] and close[i] <= ema[i] and slope[i] < 0:
            dn_idx.append(i)
    return up_idx, dn_idx, slope


def grid_hit_rates(close: List[float], high: List[float], low: List[float], up_idx: List[int], dn_idx: List[int],
                   sl_vals: List[float], tp_vals: List[float], horizon: int,
                   slope: List[float], min_slope: float,
                   atr: List[Optional[float]], min_atr: Optional[float],
                   spread: List[Optional[float]], max_spread: Optional[float],
                   hour: List[Optional[int]], session_start: Optional[int], session_end: Optional[int]) -> Tuple[Tuple[float, float, float, int], List[str]]:
    best_ev = (-1e18, 0.0, 0.0, 0.0, 0)  # ev, sl, tp, hit, trials
    lines: List[str] = []
    n = len(close)
    for sl in sl_vals:
        for tp in tp_vals:
            total = 0
            wins = 0
            # Long events
            for i in up_idx:
                # Filters
                if abs(slope[i]) < min_slope:
                    continue
                if min_atr is not None and atr[i] is not None and atr[i] < min_atr:
                    continue
                if max_spread is not None and spread[i] is not None and spread[i] > max_spread:
                    continue
                if session_start is not None and session_end is not None and hour[i] is not None:
                    h = hour[i]
                    if not (session_start <= h < session_end):
                        continue
                entry = close[i]
                end = min(i + horizon, n - 1)
                hit_tp = False
                hit_sl = False
                j = i + 1
                while j <= end:
                    if high[j] >= entry + tp:
                        hit_tp = True
                        break
                    if low[j] <= entry - sl:
                        hit_sl = True
                        break
                    j += 1
                if hit_tp and not hit_sl:
                    wins += 1
                total += 1
            # Short events
            for i in dn_idx:
                # Filters
                if abs(slope[i]) < min_slope:
                    continue
                if min_atr is not None and atr[i] is not None and atr[i] < min_atr:
                    continue
                if max_spread is not None and spread[i] is not None and spread[i] > max_spread:
                    continue
                if session_start is not None and session_end is not None and hour[i] is not None:
                    h = hour[i]
                    if not (session_start <= h < session_end):
                        continue
                entry = close[i]
                end = min(i + horizon, n - 1)
                hit_tp = False
                hit_sl = False
                j = i + 1
                while j <= end:
                    if low[j] <= entry - tp:
                        hit_tp = True
                        break
                    if high[j] >= entry + sl:
                        hit_sl = True
                        break
                    j += 1
                if hit_tp and not hit_sl:
                    wins += 1
                total += 1
            hit = (wins / total) if total > 0 else 0.0
            ev = tp * hit - sl * (1 - hit)
            line = f"SL={sl:0.2f} TP={tp:0.2f} | hit={hit*100:5.1f}% | EV~{ev:0.3f} | trials={total}"
            lines.append(line)
            if ev > best_ev[0]:
                best_ev = (ev, sl, tp, hit, total)
    ev, sl, tp, hit, trials = best_ev
    return (sl, tp, hit, trials), lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--ema-span', type=int, default=9)
    ap.add_argument('--horizon', type=int, default=10)
    ap.add_argument('--sl', type=str, default='0.5,1.0,1.5')
    ap.add_argument('--tp', type=str, default='0.8,1.2,1.6,2.0,2.4')
    # Filters
    ap.add_argument('--min-slope', type=float, default=0.02, help='Minimum EMA slope magnitude (price units)')
    ap.add_argument('--min-atr', type=float, default=0.3, help='Minimum ATR_14 (price units). Use 0 to disable.')
    ap.add_argument('--max-spread', type=float, default=1.0, help='Maximum spread allowed. Use 0 to disable.')
    ap.add_argument('--session', type=str, default='7-18', help='Trading session hour range [start-end), 0-23; empty disables')
    args = ap.parse_args()

    # alpha from span: 2/(span+1)
    alpha = 2.0 / (args.ema_span + 1)
    close, high, low, atr, spread, hour = read_csv_ohlc(args.data)
    up_idx, dn_idx, slope = detect_events(close, alpha, args.ema_span)
    print(f"Events detected (pre-filters): up={len(up_idx)}, dn={len(dn_idx)}")
    sl_vals = [float(x) for x in args.sl.split(',') if x.strip()]
    tp_vals = [float(x) for x in args.tp.split(',') if x.strip()]
    # Filters
    min_atr = args.min_atr if args.min_atr > 0 else None
    max_spread = args.max_spread if args.max_spread > 0 else None
    session_start: Optional[int] = None
    session_end: Optional[int] = None
    if args.session and '-' in args.session:
        try:
            s, e = args.session.split('-', 1)
            session_start = int(s)
            session_end = int(e)
        except Exception:
            session_start = None
            session_end = None
    best, grid = grid_hit_rates(
        close, high, low, up_idx, dn_idx, sl_vals, tp_vals, args.horizon,
        slope, args.min_slope, atr, min_atr, spread, max_spread, hour, session_start, session_end
    )
    for line in grid:
        print(line)
    sl_b, tp_b, hit_b, trials_b = best
    # EV recompute for print
    ev_b = 0.0
    # approximate EV at hit=hit_b
    ev_b = tp_b * hit_b - (sl_b) * (1 - hit_b)
    print(f"\nBest by EV: SL={sl_b:0.2f} TP={tp_b:0.2f} | hit={hit_b*100:0.1f}% | EV~{ev_b:0.3f} | trials={trials_b}")


if __name__ == '__main__':
    main()
