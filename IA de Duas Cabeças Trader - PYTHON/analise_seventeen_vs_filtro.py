#!/usr/bin/env python3
"""
Análise Seventeen 1.55M vs Filtro de Atividade
Comparar performance por horário com os BLOCKED_HOURS
"""
import re
from collections import defaultdict

# Configuração do filtro de atividade
BLOCKED_HOURS = [8, 9, 10, 11, 17, 21]

# Log a analisar (SEM FILTRO)
log_path = "D:/Projeto/Modelo PPO Trader/logs/trading_session_20251031_160208_43368_8fcc7702.txt"

print("=" * 90)
print("📊 ANÁLISE: SEVENTEEN 1.55M SEM FILTRO vs CONFIGURAÇÃO DO FILTRO DE ATIVIDADE")
print("=" * 90)
print(f"\n📁 Log: trading_session_20251031_160208_43368_8fcc7702.txt (SEM FILTRO)")
print(f"🚫 Horários bloqueados pelo filtro: {BLOCKED_HOURS}\n")

# Coletar dados por horário
wins_by_hour = defaultdict(int)
losses_by_hour = defaultdict(int)
profit_by_hour = defaultdict(float)
loss_by_hour = defaultdict(float)

# Ler arquivo e processar linha por linha
with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

current_hour = None
for i, line in enumerate(lines):
    # Extrair hora da linha atual
    time_match = re.search(r'\[(\d{2}):\d{2}:\d{2}\]', line)
    if time_match:
        current_hour = int(time_match.group(1))

    # Detectar CLOSE com PnL (timestamp está 2 linhas antes)
    if 'CLOSE |' in line and 'pnl=' in line:
        # Extrair PnL
        pnl_match = re.search(r'pnl=([-\d.]+)', line)
        if pnl_match:
            pnl = float(pnl_match.group(1))

            # Tentar pegar hora das linhas anteriores
            hour_to_use = current_hour
            for j in range(max(0, i-3), i):
                prev_time_match = re.search(r'\[(\d{2}):\d{2}:\d{2}\]', lines[j])
                if prev_time_match:
                    hour_to_use = int(prev_time_match.group(1))
                    break

            if hour_to_use is not None:
                if pnl > 0:
                    # WIN
                    wins_by_hour[hour_to_use] += 1
                    profit_by_hour[hour_to_use] += pnl
                else:
                    # LOSS
                    losses_by_hour[hour_to_use] += 1
                    loss_by_hour[hour_to_use] += abs(pnl)

# Calcular estatísticas
hourly_stats = []
for hour in range(24):
    wins = wins_by_hour.get(hour, 0)
    losses = losses_by_hour.get(hour, 0)
    total_trades = wins + losses

    if total_trades == 0:
        continue

    win_rate = (wins / total_trades) * 100
    total_profit = profit_by_hour.get(hour, 0)
    total_loss = loss_by_hour.get(hour, 0)
    net_pnl = total_profit - total_loss

    is_blocked = hour in BLOCKED_HOURS

    hourly_stats.append({
        'hour': hour,
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'is_blocked': is_blocked
    })

# Ordenar por win rate
hourly_stats.sort(key=lambda x: x['win_rate'], reverse=True)

print("=" * 90)
print("📈 PERFORMANCE POR HORÁRIO (Ordenado por Win Rate)")
print("=" * 90)
print(f"{'HORA':<6} {'TRADES':<8} {'WINS':<6} {'LOSS':<6} {'WIN RATE':<12} {'NET PNL':<12} {'FILTRO':<10}")
print("-" * 90)

for stats in hourly_stats:
    hour = stats['hour']
    wr_emoji = "🟢" if stats['win_rate'] >= 50 else "🔴"
    pnl_emoji = "💰" if stats['net_pnl'] > 0 else "💸"
    filter_status = "🚫 BLOCKED" if stats['is_blocked'] else "✅ ALLOWED"

    print(f"{hour:02d}h   {stats['trades']:<8} {stats['wins']:<6} {stats['losses']:<6} "
          f"{wr_emoji} {stats['win_rate']:>6.1f}%   "
          f"{pnl_emoji} ${stats['net_pnl']:>8.2f}  "
          f"{filter_status}")

# Análise de eficácia do filtro
print("\n" + "=" * 90)
print("🎯 VALIDAÇÃO DO FILTRO DE ATIVIDADE")
print("=" * 90)

# Separar horários bloqueados vs permitidos
blocked_stats = [s for s in hourly_stats if s['is_blocked']]
allowed_stats = [s for s in hourly_stats if not s['is_blocked']]

# Calcular médias
if blocked_stats:
    blocked_avg_wr = sum(s['win_rate'] for s in blocked_stats) / len(blocked_stats)
    blocked_total_trades = sum(s['trades'] for s in blocked_stats)
    blocked_total_pnl = sum(s['net_pnl'] for s in blocked_stats)
else:
    blocked_avg_wr = 0
    blocked_total_trades = 0
    blocked_total_pnl = 0

if allowed_stats:
    allowed_avg_wr = sum(s['win_rate'] for s in allowed_stats) / len(allowed_stats)
    allowed_total_trades = sum(s['trades'] for s in allowed_stats)
    allowed_total_pnl = sum(s['net_pnl'] for s in allowed_stats)
else:
    allowed_avg_wr = 0
    allowed_total_trades = 0
    allowed_total_pnl = 0

print(f"\n🚫 HORÁRIOS BLOQUEADOS {BLOCKED_HOURS}:")
print(f"   • Total de trades: {blocked_total_trades}")
print(f"   • Win Rate médio: {blocked_avg_wr:.1f}%")
print(f"   • Net PnL total: ${blocked_total_pnl:.2f}")

print(f"\n✅ HORÁRIOS PERMITIDOS:")
print(f"   • Total de trades: {allowed_total_trades}")
print(f"   • Win Rate médio: {allowed_avg_wr:.1f}%")
print(f"   • Net PnL total: ${allowed_total_pnl:.2f}")

# Verificar se filtro seria benéfico
wr_diff = allowed_avg_wr - blocked_avg_wr
pnl_per_trade_blocked = blocked_total_pnl / blocked_total_trades if blocked_total_trades > 0 else 0
pnl_per_trade_allowed = allowed_total_pnl / allowed_total_trades if allowed_total_trades > 0 else 0

print(f"\n📊 COMPARAÇÃO:")
print(f"   • Diferença de WR: {wr_diff:+.1f}% (Permitidos vs Bloqueados)")
print(f"   • PnL por trade bloqueado: ${pnl_per_trade_blocked:.2f}")
print(f"   • PnL por trade permitido: ${pnl_per_trade_allowed:.2f}")

# Validar se bloqueio faz sentido
print(f"\n" + "=" * 90)
print("🔍 VALIDAÇÃO INDIVIDUAL DOS HORÁRIOS BLOQUEADOS")
print("=" * 90)

for hour in BLOCKED_HOURS:
    hour_data = next((s for s in hourly_stats if s['hour'] == hour), None)
    if hour_data:
        verdict = "❌ CORRETO" if hour_data['win_rate'] < 50 else "⚠️  REVISAR"
        if hour_data['win_rate'] < 50:
            reason = f"WR {hour_data['win_rate']:.1f}% < 50% - Bloqueio justificado"
        else:
            reason = f"WR {hour_data['win_rate']:.1f}% >= 50% - Pode ser permitido"

        print(f"{hour:02d}h: {verdict} | WR: {hour_data['win_rate']:>5.1f}% | Trades: {hour_data['trades']:<4} | {reason}")
    else:
        print(f"{hour:02d}h: ⚠️  SEM DADOS no log")

print("\n" + "=" * 90)

# Recomendação final
if wr_diff > 5:
    print("✅ RECOMENDAÇÃO: Filtro de atividade é BENÉFICO")
    print(f"   Ganho de {wr_diff:.1f}% WR ao bloquear horários ruins")
elif wr_diff < -5:
    print("❌ RECOMENDAÇÃO: Filtro de atividade pode PREJUDICAR")
    print(f"   Perda de {abs(wr_diff):.1f}% WR ao bloquear esses horários")
else:
    print("⚠️  RECOMENDAÇÃO: Filtro tem impacto NEUTRO/PEQUENO")
    print(f"   Diferença de apenas {wr_diff:.1f}% WR")

print("=" * 90)
