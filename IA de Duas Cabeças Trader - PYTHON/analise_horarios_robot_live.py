#!/usr/bin/env python3
"""
Análise de horários de wins/losses do Robot em tempo real
"""
import re
import os
from collections import defaultdict
from datetime import datetime

# Procurar logs recentes
log_dir = "D:/Projeto/Modelo PPO Trader/logs"
log_files = []

for f in os.listdir(log_dir):
    if f.endswith('.txt'):
        path = os.path.join(log_dir, f)
        log_files.append((path, os.path.getmtime(path)))

# Pegar os 5 logs mais recentes
log_files.sort(key=lambda x: x[1], reverse=True)
recent_logs = [f[0] for f in log_files[:5]]

print("📊 ANÁLISE DE HORÁRIOS - ROBOT LIVE")
print("=" * 80)

# Procurar balance 552.85
target_balance = "552.85"
found_log = None

for log_path in recent_logs:
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            if target_balance in content:
                found_log = log_path
                print(f"\n✅ LOG ENCONTRADO: {os.path.basename(log_path)}")
                print(f"📍 Balance {target_balance} USD identificado")
                break
    except:
        continue

if not found_log:
    print(f"\n⚠️  Balance {target_balance} não encontrado. Analisando log mais recente...")
    found_log = recent_logs[0]
    print(f"📁 Usando: {os.path.basename(found_log)}")

# Analisar trades
print(f"\n{'='*80}")
print("📈 ANÁLISE DE TRADES POR HORÁRIO")
print(f"{'='*80}\n")

wins_by_hour = defaultdict(list)
losses_by_hour = defaultdict(list)
total_trades = 0
total_wins = 0
total_losses = 0

with open(found_log, 'r', encoding='utf-8', errors='ignore') as f:
    for line in f:
        # Padrão: [HH:MM:SS] ... WIN/LOSS
        time_match = re.search(r'\[(\d{2}):(\d{2}):\d{2}\]', line)

        if time_match:
            hour = int(time_match.group(1))

            # Detectar WIN
            win_match = re.search(r'WIN|✅.*TP|PROFIT.*\$(\d+\.\d+)', line)
            if win_match:
                # Extrair valor do profit
                profit_match = re.search(r'\$(\d+\.\d+)', line)
                profit = float(profit_match.group(1)) if profit_match else 0.0
                wins_by_hour[hour].append(profit)
                total_wins += 1
                total_trades += 1
                continue

            # Detectar LOSS
            loss_match = re.search(r'LOSS|❌.*SL|STOP.*LOSS.*\$(\d+\.\d+)', line)
            if loss_match:
                # Extrair valor da perda
                loss_match_value = re.search(r'\$(\d+\.\d+)', line)
                loss = float(loss_match_value.group(1)) if loss_match_value else 0.0
                losses_by_hour[hour].append(loss)
                total_losses += 1
                total_trades += 1

# Calcular estatísticas por horário
hourly_stats = {}
for hour in range(24):
    wins = wins_by_hour.get(hour, [])
    losses = losses_by_hour.get(hour, [])

    total_hour_trades = len(wins) + len(losses)
    if total_hour_trades == 0:
        continue

    win_rate = (len(wins) / total_hour_trades) * 100
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    net_pnl = sum(wins) - sum(losses)

    hourly_stats[hour] = {
        'trades': total_hour_trades,
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'net_pnl': net_pnl
    }

# Ordenar por net_pnl
sorted_hours = sorted(hourly_stats.items(), key=lambda x: x[1]['net_pnl'], reverse=True)

print(f"📊 Total de trades analisados: {total_trades}")
print(f"✅ Wins: {total_wins} ({total_wins/total_trades*100:.1f}%)" if total_trades > 0 else "")
print(f"❌ Losses: {total_losses} ({total_losses/total_trades*100:.1f}%)\n" if total_trades > 0 else "")

print(f"{'HORA':<6} {'TRADES':<8} {'WINS':<6} {'LOSS':<6} {'WR%':<8} {'AVG WIN':<10} {'AVG LOSS':<10} {'NET PNL':<10}")
print("-" * 80)

for hour, stats in sorted_hours:
    wr_color = "🟢" if stats['win_rate'] >= 50 else "🔴"
    pnl_color = "💰" if stats['net_pnl'] > 0 else "💸"

    print(f"{hour:02d}h   {stats['trades']:<8} {stats['wins']:<6} {stats['losses']:<6} "
          f"{wr_color} {stats['win_rate']:>5.1f}%  "
          f"${stats['avg_win']:>7.2f}    "
          f"${stats['avg_loss']:>7.2f}    "
          f"{pnl_color} ${stats['net_pnl']:>7.2f}")

# Identificar melhores e piores horários
print(f"\n{'='*80}")
print("🏆 TOP 5 MELHORES HORÁRIOS (por Net PnL)")
print(f"{'='*80}")
for hour, stats in sorted_hours[:5]:
    print(f"  {hour:02d}h: ${stats['net_pnl']:>7.2f} | WR: {stats['win_rate']:.1f}% | Trades: {stats['trades']}")

print(f"\n{'='*80}")
print("💀 TOP 5 PIORES HORÁRIOS (por Net PnL)")
print(f"{'='*80}")
for hour, stats in sorted_hours[-5:]:
    print(f"  {hour:02d}h: ${stats['net_pnl']:>7.2f} | WR: {stats['win_rate']:.1f}% | Trades: {stats['trades']}")

print(f"\n{'='*80}")
