import argparse
import os
import sys
import numpy as np
import pandas as pd


def load_dataset(path: str | None = None) -> pd.DataFrame:
    if path:
        ext = os.path.splitext(path)[1].lower()
        if ext in [".csv", ".txt"]:
            df = pd.read_csv(path)
        else:
            df = pd.read_pickle(path)
        # Normalize index/time
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
        elif 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df.set_index('time', inplace=True)
        return normalize_columns(df)

    # Fallback: try project loaders
    try:
        import importlib
        d4 = importlib.import_module('4dim')  # type: ignore
    except Exception:
        d4 = None

    if d4 is not None:
        try:
            df = d4.load_optimized_data()
            return normalize_columns(df)
        except Exception:
            pass

    raise RuntimeError("No dataset path provided and could not import a loader. Provide --data.")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Ensure required OHLCV columns exist with _5m suffix
    mapping = {}
    for base in ['open', 'high', 'low', 'close', 'volume']:
        with_suffix = f"{base}_5m"
        if with_suffix in df.columns:
            continue
        if base in df.columns:
            mapping[base] = with_suffix
    if mapping:
        df = df.rename(columns=mapping)
    missing = [c for c in ['open_5m', 'high_5m', 'low_5m', 'close_5m'] if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")
    return df


def detect_cross_events(df: pd.DataFrame, ema_span: int = 9) -> pd.DataFrame:
    close = df['close_5m'].astype(float)
    ema = close.ewm(span=ema_span, adjust=False).mean()
    slope = ema.diff()
    prev_close_below = close.shift(1) < ema.shift(1)
    prev_close_above = close.shift(1) > ema.shift(1)
    cross_up = (prev_close_below) & (close >= ema) & (slope > 0)
    cross_dn = (prev_close_above) & (close <= ema) & (slope < 0)
    events = pd.DataFrame(index=df.index)
    events['price'] = close
    events['ema'] = ema
    events['slope'] = slope
    events['cross_up'] = cross_up
    events['cross_dn'] = cross_dn
    return events


def compute_mfe_mae(df: pd.DataFrame, events: pd.DataFrame, horizons: list[int]) -> dict:
    high = df['high_5m'].astype(float).to_numpy()
    low = df['low_5m'].astype(float).to_numpy()
    close = df['close_5m'].astype(float).to_numpy()
    idx = np.arange(len(df))

    up_idx = np.where(events['cross_up'].to_numpy())[0]
    dn_idx = np.where(events['cross_dn'].to_numpy())[0]

    results = {h: {'long': [], 'short': []} for h in horizons}

    def per_event(i, side, h):
        entry = close[i]
        j_end = min(i + h, len(df) - 1)
        if j_end <= i:
            return None
        window_high = np.max(high[i+1:j_end+1])
        window_low = np.min(low[i+1:j_end+1])
        if side == 'long':
            mfe = max(0.0, window_high - entry)
            mae = max(0.0, entry - window_low)
        else:
            mfe = max(0.0, entry - window_low)
            mae = max(0.0, window_high - entry)
        return mfe, mae

    # Iterate events (vectorized per horizon would be complex; loop is fine)
    for h in horizons:
        for i in up_idx:
            val = per_event(i, 'long', h)
            if val:
                results[h]['long'].append(val)
        for i in dn_idx:
            val = per_event(i, 'short', h)
            if val:
                results[h]['short'].append(val)
    return results


def simulate_sl_tp_hit(df: pd.DataFrame, events: pd.DataFrame, sl_vals: list[float], tp_vals: list[float], horizon: int = 10) -> dict:
    high = df['high_5m'].astype(float).to_numpy()
    low = df['low_5m'].astype(float).to_numpy()
    close = df['close_5m'].astype(float).to_numpy()

    up_idx = np.where(events['cross_up'].to_numpy())[0]
    dn_idx = np.where(events['cross_dn'].to_numpy())[0]

    grid = {}
    for sl in sl_vals:
        for tp in tp_vals:
            key = (sl, tp)
            total = 0
            wins = 0
            for i in up_idx:
                entry = close[i]
                j_end = min(i + horizon, len(df) - 1)
                hit_tp = False
                hit_sl = False
                for j in range(i+1, j_end+1):
                    if high[j] >= entry + tp:
                        hit_tp = True
                        break
                    if low[j] <= entry - sl:
                        hit_sl = True
                        break
                wins += 1 if hit_tp and not hit_sl else 0
                total += 1
            for i in dn_idx:
                entry = close[i]
                j_end = min(i + horizon, len(df) - 1)
                hit_tp = False
                hit_sl = False
                for j in range(i+1, j_end+1):
                    if low[j] <= entry - tp:
                        hit_tp = True
                        break
                    if high[j] >= entry + sl:
                        hit_sl = True
                        break
                wins += 1 if hit_tp and not hit_sl else 0
                total += 1
            hit_rate = (wins / total) if total else 0.0
            grid[key] = {'trials': total, 'wins': wins, 'hit_rate': hit_rate}
    return grid


def summarize(results: dict) -> list[dict]:
    rows = []
    for h, d in results.items():
        for side in ['long', 'short']:
            arr = np.array(d[side]) if d[side] else np.zeros((0, 2))
            if len(arr) == 0:
                rows.append({'horizon': h, 'side': side, 'count': 0})
                continue
            mfe = arr[:, 0]
            mae = arr[:, 1]
            rows.append({
                'horizon': h,
                'side': side,
                'count': int(len(arr)),
                'mfe_mean': float(np.mean(mfe)),
                'mfe_p50': float(np.percentile(mfe, 50)),
                'mfe_p75': float(np.percentile(mfe, 75)),
                'mfe_p90': float(np.percentile(mfe, 90)),
                'mae_mean': float(np.mean(mae)),
                'mae_p50': float(np.percentile(mae, 50)),
                'mae_p75': float(np.percentile(mae, 75)),
                'mae_p90': float(np.percentile(mae, 90)),
            })
    return rows


def main():
    ap = argparse.ArgumentParser(description="Analyze EMA-cross events for Pescador strategy")
    ap.add_argument('--data', type=str, default=None, help='Path to dataset (csv/pkl). If omitted, tries project loader.')
    ap.add_argument('--ema-span', type=int, default=9, help='EMA span for fast MA')
    ap.add_argument('--horizons', type=str, default='5,10,15', help='Comma-separated forward horizons (bars)')
    ap.add_argument('--sl-range', type=str, default='2,6,0.5', help='SL min,max,step (points)')
    ap.add_argument('--tp-range', type=str, default='3,10,0.5', help='TP min,max,step (points)')
    ap.add_argument('--save', type=str, default=None, help='Optional path to save per-horizon summary CSV')
    args = ap.parse_args()

    df = load_dataset(args.data)
    events = detect_cross_events(df, ema_span=args.ema_span)

    horizons = [int(x) for x in args.horizons.split(',') if x.strip()]
    results = compute_mfe_mae(df, events, horizons)
    rows = summarize(results)
    summary_df = pd.DataFrame(rows)

    print("=== Pescador Event Summary (MFE/MAE in price points) ===")
    if len(summary_df):
        print(summary_df.to_string(index=False))
    else:
        print("No events found.")

    # SL/TP grid simulation for middle horizon (use median horizon)
    sl_min, sl_max, sl_step = [float(x) for x in args.sl_range.split(',')]
    tp_min, tp_max, tp_step = [float(x) for x in args.tp_range.split(',')]
    sl_vals = list(np.arange(sl_min, sl_max + 1e-9, sl_step))
    tp_vals = list(np.arange(tp_min, tp_max + 1e-9, tp_step))
    horizon = int(np.median(horizons))
    grid = simulate_sl_tp_hit(df, events, sl_vals, tp_vals, horizon=horizon)

    print(f"\n=== SL/TP Hit Rate (horizon={horizon}) ===")
    best = None
    for (sl, tp), stats in sorted(grid.items()):
        hit = stats['hit_rate']
        print(f"SL {sl:>4.1f} | TP {tp:>4.1f} -> hit {hit*100:5.1f}% (trials={stats['trials']})")
        # Simple expected value proxy: tp*hit - sl*(1-hit)
        ev = tp * hit - sl * (1 - hit)
        if best is None or ev > best[0]:
            best = (ev, sl, tp, hit)
    if best:
        ev, sl_b, tp_b, hit_b = best
        print(f"\nSuggested SL/TP by EV proxy: SL={sl_b:.1f}, TP={tp_b:.1f} (hit={hit_b*100:.1f}%, EV~{ev:.2f})")

    if args.save:
        summary_df.to_csv(args.save, index=False)
        print(f"\nSaved summary to: {args.save}")


if __name__ == '__main__':
    main()
