import csv
from datetime import datetime


DATA_PATH = 'data/GC=F_YAHOO_20250821_161220.csv'


def parse_time(s: str) -> datetime:
    # Handles ISO or 'YYYY-MM-DD HH:MM:SS...'
    s = s.strip()
    try:
        return datetime.fromisoformat(s.replace('Z', '+00:00'))
    except Exception:
        return datetime.strptime(s[:19], '%Y-%m-%d %H:%M:%S')


def analyze(path: str = DATA_PATH, ema_span: int = 9, block: int = 10000,
            min_slope: float = 0.02, min_atr: float = 0.3, max_spread: float = 1.0,
            sess_start: int = 7, sess_end: int = 18):
    with open(path, 'r', newline='') as f:
        r = csv.reader(f)
        header = next(r)
        idx_time = header.index('time') if 'time' in header else -1
        idx_close = header.index('close')
        idx_high = header.index('high')
        idx_low = header.index('low')
        idx_atr = header.index('atr_14') if 'atr_14' in header else -1
        idx_spread = header.index('spread') if 'spread' in header else -1

        alpha = 2.0 / (ema_span + 1)
        ema_prev = None
        ema_curr = None
        close_prev = None
        time_prev: datetime | None = None

        total_rows = 0
        cross_total = 0
        cross_up_total = 0
        cross_dn_total = 0
        filt_total = 0
        filt_up_total = 0
        filt_dn_total = 0

        # For block summaries
        blk = 0
        def print_block():
            nonlocal blk, cross_total, cross_up_total, cross_dn_total, filt_total, filt_up_total, filt_dn_total
            if blk == 0:
                return
            rate = (filt_total / cross_total * 100) if cross_total else 0.0
            print(f"Block {blk:>5d}: crosses={cross_total} (up={cross_up_total}, dn={cross_dn_total}) | filtered={filt_total} (up={filt_up_total}, dn={filt_dn_total}) | pass_rate={rate:.1f}%")
            # reset block counters
            cross_total = cross_up_total = cross_dn_total = 0
            filt_total = filt_up_total = filt_dn_total = 0

        for row in r:
            try:
                total_rows += 1
                c = float(row[idx_close])
                # ema update
                if ema_curr is None:
                    ema_curr = c
                    ema_prev = c
                else:
                    ema_prev = ema_curr
                    ema_curr = alpha * c + (1 - alpha) * ema_curr

                # slope and crosses
                slope = ema_curr - ema_prev if ema_prev is not None else 0.0
                cross_up = False
                cross_dn = False
                if close_prev is not None:
                    if close_prev < ema_prev and c >= ema_curr and slope > 0:
                        cross_up = True
                    if close_prev > ema_prev and c <= ema_curr and slope < 0:
                        cross_dn = True

                # block accounting
                if cross_up or cross_dn:
                    cross_total += 1
                    cross_up_total += int(cross_up)
                    cross_dn_total += int(cross_dn)

                    # filters
                    atr_ok = True
                    if idx_atr >= 0:
                        try:
                            atr = float(row[idx_atr])
                            atr_ok = atr >= min_atr
                        except Exception:
                            atr_ok = True
                    spread_ok = True
                    if idx_spread >= 0:
                        try:
                            spr = float(row[idx_spread])
                            spread_ok = spr <= max_spread
                        except Exception:
                            spread_ok = True
                    sess_ok = True
                    if idx_time >= 0:
                        try:
                            hour = parse_time(row[idx_time]).hour
                            sess_ok = (sess_start <= hour < sess_end)
                        except Exception:
                            sess_ok = True
                    slope_ok = abs(slope) >= min_slope

                    if slope_ok and atr_ok and spread_ok and sess_ok:
                        filt_total += 1
                        if cross_up:
                            filt_up_total += 1
                        else:
                            filt_dn_total += 1

                # rolling block
                if total_rows % block == 0:
                    blk += 1
                    print_block()

                close_prev = c
                time_prev = parse_time(row[idx_time]) if idx_time >= 0 else None
            except Exception:
                continue

        # flush last block
        blk += 1
        print_block()


if __name__ == '__main__':
    analyze()

