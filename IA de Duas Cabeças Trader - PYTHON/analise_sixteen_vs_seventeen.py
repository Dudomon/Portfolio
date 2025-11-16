#!/usr/bin/env python3
"""
🎯 ANÁLISE COMPARATIVA: SIXTEEN 1.55M vs SEVENTEEN 1.55M
Objetivo: Validar se Entry Timing Rewards (Seventeen) melhorou:
- SL Hit Rate (meta: 61.5% → <48%)
- TP Hit Rate (meta: 38.5% → >52%)
- Win Rate geral
- Performance por horário
"""
import re
from collections import defaultdict

# Logs a comparar
LOG_SIXTEEN = "D:/Projeto/Modelo PPO Trader/logs/trading_session_20251029_205803_20876_ebda150b.txt"
LOG_SEVENTEEN_SEM_FILTRO = "D:/Projeto/Modelo PPO Trader/logs/trading_session_20251031_160208_43368_8fcc7702.txt"

# Horários ruins identificados na análise de 32,865 trades
BAD_HOURS = [8, 9, 10, 11, 17, 21]
EXCELLENT_HOURS = [15, 12, 19, 20, 4]
GOOD_HOURS = [13, 14, 18, 22, 23, 0, 1, 2, 3, 5, 7]

def parse_log_detailed(log_path, model_name):
    """Parse log com foco em entry timing"""

    wins_by_hour = defaultdict(int)
    losses_by_hour = defaultdict(int)
    profit_by_hour = defaultdict(float)
    loss_by_hour = defaultdict(float)

    sl_hits = 0  # Quantas vezes bateu SL
    tp_hits = 0  # Quantas vezes bateu TP
    total_wins = 0
    total_losses = 0
    total_profit = 0.0
    total_loss = 0.0

    # Estatísticas de horários
    entries_bad_hours = 0
    entries_excellent_hours = 0
    entries_good_hours = 0

    wins_bad_hours = 0
    wins_excellent_hours = 0
    wins_good_hours = 0

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    current_hour = None
    for i, line in enumerate(lines):
        # Extrair hora
        time_match = re.search(r'\[(\d{2}):\d{2}:\d{2}\]', line)
        if time_match:
            current_hour = int(time_match.group(1))

        # Detectar entradas (LONG/SHORT executado)
        if re.search(r'LONG executado|SHORT executado', line):
            if current_hour is not None:
                if current_hour in BAD_HOURS:
                    entries_bad_hours += 1
                elif current_hour in EXCELLENT_HOURS:
                    entries_excellent_hours += 1
                elif current_hour in GOOD_HOURS:
                    entries_good_hours += 1

        # Detectar fechamentos (CLOSE com PnL)
        if 'CLOSE |' in line and 'pnl=' in line:
            pnl_match = re.search(r'pnl=([-\d.]+)', line)
            if pnl_match:
                pnl = float(pnl_match.group(1))

                # Pegar hora das linhas anteriores
                hour_to_use = current_hour
                for j in range(max(0, i-3), i):
                    prev_time_match = re.search(r'\[(\d{2}):\d{2}:\d{2}\]', lines[j])
                    if prev_time_match:
                        hour_to_use = int(prev_time_match.group(1))
                        break

                if hour_to_use is not None:
                    # Determinar se foi TP ou SL (olhando linhas anteriores)
                    hit_type = "UNKNOWN"
                    for j in range(max(0, i-5), i):
                        if 'TP' in lines[j] and 'HIT' in lines[j]:
                            hit_type = "TP"
                            tp_hits += 1
                            break
                        elif 'SL' in lines[j] and 'HIT' in lines[j]:
                            hit_type = "SL"
                            sl_hits += 1
                            break

                    # Inferir se não encontrou explicitamente
                    if hit_type == "UNKNOWN":
                        if pnl > 0:
                            tp_hits += 1
                            hit_type = "TP"
                        else:
                            sl_hits += 1
                            hit_type = "SL"

                    # Contabilizar win/loss
                    if pnl > 0:
                        wins_by_hour[hour_to_use] += 1
                        profit_by_hour[hour_to_use] += pnl
                        total_wins += 1
                        total_profit += pnl

                        # Win em qual tipo de horário?
                        if hour_to_use in BAD_HOURS:
                            wins_bad_hours += 1
                        elif hour_to_use in EXCELLENT_HOURS:
                            wins_excellent_hours += 1
                        elif hour_to_use in GOOD_HOURS:
                            wins_good_hours += 1
                    else:
                        losses_by_hour[hour_to_use] += 1
                        loss_by_hour[hour_to_use] += abs(pnl)
                        total_losses += 1
                        total_loss += abs(pnl)

    total_trades = total_wins + total_losses
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    net_pnl = total_profit - total_loss

    # Calcular SL/TP rates
    total_closes = sl_hits + tp_hits
    sl_hit_rate = (sl_hits / total_closes * 100) if total_closes > 0 else 0
    tp_hit_rate = (tp_hits / total_closes * 100) if total_closes > 0 else 0

    # Entry quality score
    total_entries = entries_bad_hours + entries_excellent_hours + entries_good_hours
    if total_entries > 0:
        bad_entry_rate = (entries_bad_hours / total_entries * 100)
        excellent_entry_rate = (entries_excellent_hours / total_entries * 100)
        good_entry_rate = (entries_good_hours / total_entries * 100)
    else:
        bad_entry_rate = 0
        excellent_entry_rate = 0
        good_entry_rate = 0

    return {
        'model_name': model_name,
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': win_rate,
        'net_pnl': net_pnl,
        'sl_hits': sl_hits,
        'tp_hits': tp_hits,
        'sl_hit_rate': sl_hit_rate,
        'tp_hit_rate': tp_hit_rate,
        'entries_bad_hours': entries_bad_hours,
        'entries_excellent_hours': entries_excellent_hours,
        'entries_good_hours': entries_good_hours,
        'bad_entry_rate': bad_entry_rate,
        'excellent_entry_rate': excellent_entry_rate,
        'good_entry_rate': good_entry_rate,
        'wins_bad_hours': wins_bad_hours,
        'wins_excellent_hours': wins_excellent_hours,
        'wins_good_hours': wins_good_hours,
        'wins_by_hour': wins_by_hour,
        'losses_by_hour': losses_by_hour
    }

print("=" * 100)
print("🎯 COMPARAÇÃO: SIXTEEN 1.55M vs SEVENTEEN 1.55M (Entry Timing Rewards)")
print("=" * 100)
print("\n📊 OBJETIVO DO SEVENTEEN:")
print("   • Reduzir SL Hit Rate: 61.5% → <48%")
print("   • Aumentar TP Hit Rate: 38.5% → >52%")
print("   • Melhorar Win Rate geral")
print("   • Evitar entradas em horários ruins\n")

# Parse logs
print("⏳ Analisando logs...\n")
stats_sixteen = parse_log_detailed(LOG_SIXTEEN, "Sixteen 1.55M")
stats_seventeen = parse_log_detailed(LOG_SEVENTEEN_SEM_FILTRO, "Seventeen 1.55M")

# Tabela comparativa principal
print("=" * 100)
print("📈 MÉTRICAS PRINCIPAIS")
print("=" * 100)
print(f"{'MÉTRICA':<35} {'SIXTEEN 1.55M':<25} {'SEVENTEEN 1.55M':<25} {'DIFERENÇA':<15}")
print("-" * 100)

# Total trades
diff_trades = stats_seventeen['total_trades'] - stats_sixteen['total_trades']
print(f"{'Total de Trades':<35} {stats_sixteen['total_trades']:<25} {stats_seventeen['total_trades']:<25} {diff_trades:+<15}")

# Win Rate
diff_wr = stats_seventeen['win_rate'] - stats_sixteen['win_rate']
wr_emoji = "🟢" if diff_wr > 0 else "🔴"
print(f"{'Win Rate':<35} {stats_sixteen['win_rate']:<24.1f}% {stats_seventeen['win_rate']:<24.1f}% {wr_emoji} {diff_wr:+<13.1f}%")

# Net PnL
diff_pnl = stats_seventeen['net_pnl'] - stats_sixteen['net_pnl']
pnl_emoji = "🟢" if diff_pnl > 0 else "🔴"
print(f"{'Net PnL':<35} ${stats_sixteen['net_pnl']:<23.2f} ${stats_seventeen['net_pnl']:<23.2f} {pnl_emoji} ${diff_pnl:+<13.2f}")

print("\n" + "=" * 100)
print("🎯 MÉTRICAS DE ENTRY TIMING (OBJETIVO PRINCIPAL)")
print("=" * 100)
print(f"{'MÉTRICA':<35} {'SIXTEEN':<25} {'SEVENTEEN':<25} {'DIFERENÇA':<15}")
print("-" * 100)

# SL Hit Rate (META: reduzir de 61.5% para <48%)
diff_sl = stats_seventeen['sl_hit_rate'] - stats_sixteen['sl_hit_rate']
sl_emoji = "✅" if diff_sl < 0 else "⚠️ "
sl_target = "TARGET!" if stats_seventeen['sl_hit_rate'] < 48 else "Ainda acima"
print(f"{'🚨 SL Hit Rate':<35} {stats_sixteen['sl_hit_rate']:<24.1f}% {stats_seventeen['sl_hit_rate']:<24.1f}% {sl_emoji} {diff_sl:+<13.1f}%")
print(f"{'   Meta: <48%':<35} {'':<25} {sl_target:<25}")

# TP Hit Rate (META: aumentar de 38.5% para >52%)
diff_tp = stats_seventeen['tp_hit_rate'] - stats_sixteen['tp_hit_rate']
tp_emoji = "✅" if diff_tp > 0 else "⚠️ "
tp_target = "TARGET!" if stats_seventeen['tp_hit_rate'] > 52 else "Ainda abaixo"
print(f"{'💰 TP Hit Rate':<35} {stats_sixteen['tp_hit_rate']:<24.1f}% {stats_seventeen['tp_hit_rate']:<24.1f}% {tp_emoji} {diff_tp:+<13.1f}%")
print(f"{'   Meta: >52%':<35} {'':<25} {tp_target:<25}")

print("\n" + "=" * 100)
print("📊 QUALIDADE DAS ENTRADAS (Entry Quality Score)")
print("=" * 100)
print(f"{'MÉTRICA':<35} {'SIXTEEN':<25} {'SEVENTEEN':<25} {'DIFERENÇA':<15}")
print("-" * 100)

# Entradas em horários ruins
diff_bad = stats_seventeen['bad_entry_rate'] - stats_sixteen['bad_entry_rate']
bad_emoji = "✅" if diff_bad < 0 else "🔴"
print(f"{'❌ Entradas em horários RUINS':<35} {stats_sixteen['bad_entry_rate']:<24.1f}% {stats_seventeen['bad_entry_rate']:<24.1f}% {bad_emoji} {diff_bad:+<13.1f}%")
print(f"{'   [8,9,10,11,17,21]':<35} {stats_sixteen['entries_bad_hours']:<25} {stats_seventeen['entries_bad_hours']:<25}")

# Entradas em horários excelentes
diff_exc = stats_seventeen['excellent_entry_rate'] - stats_sixteen['excellent_entry_rate']
exc_emoji = "✅" if diff_exc > 0 else "🔴"
print(f"{'✨ Entradas em horários EXCELENTES':<35} {stats_sixteen['excellent_entry_rate']:<24.1f}% {stats_seventeen['excellent_entry_rate']:<24.1f}% {exc_emoji} {diff_exc:+<13.1f}%")
print(f"{'   [15,12,19,20,4]':<35} {stats_sixteen['entries_excellent_hours']:<25} {stats_seventeen['entries_excellent_hours']:<25}")

# Entradas em horários bons
diff_good = stats_seventeen['good_entry_rate'] - stats_sixteen['good_entry_rate']
print(f"{'🟢 Entradas em horários BONS':<35} {stats_sixteen['good_entry_rate']:<24.1f}% {stats_seventeen['good_entry_rate']:<24.1f}% {diff_good:+<14.1f}%")

print("\n" + "=" * 100)
print("🏆 VEREDICTO FINAL")
print("=" * 100)

# Calcular pontuação
score = 0
improvements = []
regressions = []

# SL Hit Rate
if diff_sl < -5:
    score += 3
    improvements.append(f"SL Hit Rate reduziu {abs(diff_sl):.1f}% 🎯")
elif diff_sl < 0:
    score += 1
    improvements.append(f"SL Hit Rate reduziu levemente {abs(diff_sl):.1f}%")
else:
    score -= 2
    regressions.append(f"SL Hit Rate AUMENTOU {diff_sl:.1f}% ❌")

# TP Hit Rate
if diff_tp > 5:
    score += 3
    improvements.append(f"TP Hit Rate aumentou {diff_tp:.1f}% 🎯")
elif diff_tp > 0:
    score += 1
    improvements.append(f"TP Hit Rate aumentou levemente {diff_tp:.1f}%")
else:
    score -= 2
    regressions.append(f"TP Hit Rate DIMINUIU {abs(diff_tp):.1f}% ❌")

# Win Rate
if diff_wr > 5:
    score += 2
    improvements.append(f"Win Rate geral +{diff_wr:.1f}% 💰")
elif diff_wr > 0:
    score += 1
    improvements.append(f"Win Rate +{diff_wr:.1f}%")
else:
    score -= 1
    regressions.append(f"Win Rate -{abs(diff_wr):.1f}%")

# Entry Quality
if diff_bad < -5:
    score += 2
    improvements.append(f"Reduziu entradas ruins em {abs(diff_bad):.1f}% ✨")
elif diff_bad < 0:
    score += 1

if diff_exc > 5:
    score += 2
    improvements.append(f"Aumentou entradas excelentes em {diff_exc:.1f}% ✨")
elif diff_exc > 0:
    score += 1

print(f"\n📊 SCORE: {score}/10\n")

if improvements:
    print("✅ MELHORIAS:")
    for imp in improvements:
        print(f"   • {imp}")

if regressions:
    print("\n⚠️  REGRESSÕES:")
    for reg in regressions:
        print(f"   • {reg}")

print("\n" + "=" * 100)

if score >= 7:
    print("🎉 CONCLUSÃO: Entry Timing Rewards FUNCIONOU! Seventeen é SUPERIOR ao Sixteen.")
    print("   Recomendação: Usar Seventeen em produção.")
elif score >= 4:
    print("🟡 CONCLUSÃO: Entry Timing Rewards teve impacto MODERADO.")
    print("   Recomendação: Continuar monitorando com mais trades.")
else:
    print("❌ CONCLUSÃO: Entry Timing Rewards NÃO melhorou significativamente.")
    print("   Recomendação: Revisar implementação ou manter Sixteen.")

print("=" * 100)
