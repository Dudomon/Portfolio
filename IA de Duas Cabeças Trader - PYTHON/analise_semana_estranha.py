#!/usr/bin/env python3
"""
📊 ANÁLISE COMPLETA DA SEMANA ESTRANHA - SEVENTEEN vs EIGHTEEN
Compara performance de diferentes configurações lado a lado
"""

import re
import os
from collections import defaultdict
from datetime import datetime

# Logs localizados
LOGS = {
    '777712': {
        'path': r'D:/Projeto/Modelo PPO Trader/logs/trading_session_20251109_202631_15452_19c94b13.txt',
        'desc': 'Seventeen com filtro ANTIGO [8, 9, 10, 11, 17, 21]',
        'experiment': 'SEVENTEEN',
        'filter': 'ANTIGO'
    },
    '777344': {
        'path': r'D:/Projeto/Modelo PPO Trader/logs/trading_session_20251110_161139_19212_a0e93942.txt',
        'desc': 'Seventeen com filtro ATUAL [0, 1, 2, 5, 7, 8, 11, 12, 14, 16, 21, 23]',
        'experiment': 'SEVENTEEN',
        'filter': 'ATUAL'
    },
    '777528': {
        'path': r'D:/Projeto/Modelo PPO Trader/logs/trading_session_20251112_155713_33400_12456316.txt',
        'desc': 'Eighteen SEM filtro',
        'experiment': 'EIGHTEEN',
        'filter': 'SEM FILTRO'
    }
}

def parse_log(log_path):
    """Parse log file and extract trade data"""
    trades = []
    trades_by_hour = defaultdict(list)

    if not os.path.exists(log_path):
        print(f"❌ Log not found: {log_path}")
        return None, None

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    current_hour = None
    for i, line in enumerate(lines):
        # Extract hour
        time_match = re.search(r'\[(\d{2}):\d{2}:\d{2}\]', line)
        if time_match:
            current_hour = int(time_match.group(1))

        # Detect CLOSE with PnL
        if 'CLOSE |' in line and 'pnl=' in line:
            pnl_match = re.search(r'pnl=([-\d.]+)', line)
            if pnl_match:
                pnl = float(pnl_match.group(1))

                # Get hour from previous lines if needed
                hour_to_use = current_hour
                for j in range(max(0, i-3), i):
                    prev_time_match = re.search(r'\[(\d{2}):\d{2}:\d{2}\]', lines[j])
                    if prev_time_match:
                        hour_to_use = int(prev_time_match.group(1))
                        break

                if hour_to_use is not None:
                    trade = {
                        'hour': hour_to_use,
                        'pnl': pnl,
                        'is_win': pnl > 0
                    }
                    trades.append(trade)
                    trades_by_hour[hour_to_use].append(trade)

    return trades, trades_by_hour

def analyze_trades(trades, trades_by_hour):
    """Analyze trades and return stats"""
    if not trades:
        return None

    total_trades = len(trades)
    wins = [t for t in trades if t['is_win']]
    losses = [t for t in trades if not t['is_win']]

    total_wins = len(wins)
    total_losses = len(losses)

    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

    total_profit = sum(t['pnl'] for t in wins)
    total_loss = sum(abs(t['pnl']) for t in losses)
    net_pnl = sum(t['pnl'] for t in trades)

    avg_win = total_profit / total_wins if total_wins > 0 else 0
    avg_loss = total_loss / total_losses if total_losses > 0 else 0
    profit_factor = total_profit / total_loss if total_loss > 0 else 0
    pnl_per_trade = net_pnl / total_trades if total_trades > 0 else 0

    # Hour analysis
    hour_stats = {}
    for hour in range(24):
        hour_trades = trades_by_hour.get(hour, [])
        if hour_trades:
            hour_wins = len([t for t in hour_trades if t['is_win']])
            hour_wr = (hour_wins / len(hour_trades) * 100)
            hour_pnl = sum(t['pnl'] for t in hour_trades)
            hour_stats[hour] = {
                'trades': len(hour_trades),
                'wr': hour_wr,
                'pnl': hour_pnl
            }

    return {
        'total_trades': total_trades,
        'wins': total_wins,
        'losses': total_losses,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_pnl': net_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'pnl_per_trade': pnl_per_trade,
        'hour_stats': hour_stats
    }

print("=" * 120)
print("📊 ANÁLISE COMPLETA DA SEMANA ESTRANHA - SEVENTEEN vs EIGHTEEN")
print("=" * 120)
print()

# Parse all logs
results = {}

for magic, config in LOGS.items():
    print(f"📂 Processing {magic}: {config['desc']}...")
    trades, trades_by_hour = parse_log(config['path'])

    if trades:
        stats = analyze_trades(trades, trades_by_hour)
        results[magic] = {
            **config,
            'stats': stats
        }
        print(f"   ✅ {stats['total_trades']} trades found")
    else:
        print(f"   ❌ No trades found")

    print()

# Generate comparison report
print("=" * 120)
print("📊 COMPARAÇÃO LADO A LADO")
print("=" * 120)
print()

# Header
print(f"{'METRIC':<25} {'777712 (17-OLD)':<25} {'777344 (17-CURR)':<25} {'777528 (18-NONE)':<25}")
print("-" * 120)

# Compare metrics
if results:
    metrics = [
        ('Total Trades', 'total_trades', ''),
        ('Wins', 'wins', ''),
        ('Losses', 'losses', ''),
        ('Win Rate %', 'win_rate', '.1f'),
        ('Net PnL $', 'net_pnl', '.2f'),
        ('PnL/Trade $', 'pnl_per_trade', '.2f'),
        ('Profit Factor', 'profit_factor', '.2f'),
        ('Avg Win $', 'avg_win', '.2f'),
        ('Avg Loss $', 'avg_loss', '.2f'),
    ]

    for metric_name, metric_key, fmt in metrics:
        line = f"{metric_name:<25}"

        for magic in ['777712', '777344', '777528']:
            if magic in results and results[magic]['stats']:
                value = results[magic]['stats'][metric_key]
                if fmt:
                    value_str = f"{value:{fmt}}"
                else:
                    value_str = str(value)
                line += f"{value_str:<25}"
            else:
                line += f"{'N/A':<25}"

        print(line)

print()
print("=" * 120)
print("📊 ANÁLISE POR HORÁRIO - COMPARAÇÃO")
print("=" * 120)
print()

# Top 5 best hours by Net PnL for each configuration
for magic in ['777712', '777344', '777528']:
    if magic in results and results[magic]['stats']:
        config = results[magic]
        hour_stats = config['stats']['hour_stats']

        if hour_stats:
            print(f"\n🏆 {config['desc']} (Magic: {magic})")
            print(f"   {'HOUR':<6} {'TRADES':<8} {'WIN%':<8} {'NET PnL':<12}")
            print(f"   {'-'*40}")

            # Sort by Net PnL
            sorted_hours = sorted(hour_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)

            for hour, stats in sorted_hours[:10]:
                emoji = "✅" if stats['pnl'] > 0 else "🔴"
                print(f"   {emoji} {hour:02d}:00  {stats['trades']:<8} {stats['wr']:<7.1f}% ${stats['pnl']:<11.2f}")

print()
print("=" * 120)
print("💡 CONCLUSÕES")
print("=" * 120)
print()

# Find best configuration
best_magic = None
best_pnl = float('-inf')

for magic in ['777712', '777344', '777528']:
    if magic in results and results[magic]['stats']:
        net_pnl = results[magic]['stats']['net_pnl']
        if net_pnl > best_pnl:
            best_pnl = net_pnl
            best_magic = magic

if best_magic:
    best_config = results[best_magic]
    print(f"🏆 MELHOR CONFIGURAÇÃO: {best_config['desc']} (Magic: {best_magic})")
    print(f"   Net PnL: ${best_config['stats']['net_pnl']:.2f}")
    print(f"   Win Rate: {best_config['stats']['win_rate']:.1f}%")
    print(f"   Profit Factor: {best_config['stats']['profit_factor']:.2f}")

print()

# Analysis per experiment
print("📊 ANÁLISE POR EXPERIMENTO:")
print()

# Seventeen comparison
seventeen_configs = [magic for magic in ['777712', '777344'] if magic in results]
if len(seventeen_configs) == 2:
    old = results['777712']['stats']
    curr = results['777344']['stats']

    print("🔬 SEVENTEEN - Filtro ANTIGO vs ATUAL:")
    print(f"   Trades: {old['total_trades']} (OLD) vs {curr['total_trades']} (CURR)")
    print(f"   Win Rate: {old['win_rate']:.1f}% (OLD) vs {curr['win_rate']:.1f}% (CURR) → {curr['win_rate']-old['win_rate']:+.1f}%")
    print(f"   Net PnL: ${old['net_pnl']:.2f} (OLD) vs ${curr['net_pnl']:.2f} (CURR) → ${curr['net_pnl']-old['net_pnl']:+.2f}")
    print(f"   PF: {old['profit_factor']:.2f} (OLD) vs {curr['profit_factor']:.2f} (CURR) → {curr['profit_factor']-old['profit_factor']:+.2f}")
    print()

# Seventeen vs Eighteen
if '777344' in results and '777528' in results:
    sev = results['777344']['stats']
    eig = results['777528']['stats']

    print("🔬 SEVENTEEN vs EIGHTEEN (ambos com filtro ATUAL / SEM filtro):")
    print(f"   Trades: {sev['total_trades']} (17) vs {eig['total_trades']} (18)")
    print(f"   Win Rate: {sev['win_rate']:.1f}% (17) vs {eig['win_rate']:.1f}% (18) → {eig['win_rate']-sev['win_rate']:+.1f}%")
    print(f"   Net PnL: ${sev['net_pnl']:.2f} (17) vs ${eig['net_pnl']:.2f} (18) → ${eig['net_pnl']-sev['net_pnl']:+.2f}")
    print(f"   PF: {sev['profit_factor']:.2f} (17) vs {eig['profit_factor']:.2f} (18) → {eig['profit_factor']-sev['profit_factor']:+.2f}")

print()
print("=" * 120)

# Save detailed report
output_file = "D:/Projeto/analise_semana_estranha.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write("ANÁLISE COMPLETA DA SEMANA ESTRANHA\n")
    f.write("=" * 120 + "\n\n")

    for magic, config in results.items():
        f.write(f"{config['desc']} (Magic: {magic})\n")
        f.write(f"Experiment: {config['experiment']} | Filter: {config['filter']}\n")
        f.write(f"Log: {config['path']}\n\n")

        stats = config['stats']
        f.write(f"Total Trades: {stats['total_trades']}\n")
        f.write(f"Win Rate: {stats['win_rate']:.1f}%\n")
        f.write(f"Net PnL: ${stats['net_pnl']:.2f}\n")
        f.write(f"PnL/Trade: ${stats['pnl_per_trade']:.2f}\n")
        f.write(f"Profit Factor: {stats['profit_factor']:.2f}\n")
        f.write("\n" + "="*120 + "\n\n")

print(f"💾 Detailed report saved to: {output_file}")
