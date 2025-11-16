#!/usr/bin/env python3
"""
🎯 ANÁLISE DE QUALIDADE DAS ENTRADAS - SEVENTEEN
Avalia se o experimento de entry timing rewards melhorou a qualidade das entradas
"""
import re
from collections import defaultdict

log_path = 'D:/Projeto/Modelo PPO Trader/logs/trading_session_20251031_160208_43368_8fcc7702.txt'

with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
    lines = f.readlines()

# Análise de entradas
entries_long = []
entries_short = []
trades_complete = []

current_trade = None

for i, line in enumerate(lines):
    # Detectar LONG entry
    if 'OPEN LONG |' in line:
        price_match = re.search(r'price=([\d.]+)', line)
        if price_match:
            current_trade = {
                'type': 'LONG',
                'entry_price': float(price_match.group(1)),
                'line_num': i
            }

    # Detectar SHORT entry
    elif 'OPEN SHORT |' in line:
        price_match = re.search(r'price=([\d.]+)', line)
        if price_match:
            current_trade = {
                'type': 'SHORT',
                'entry_price': float(price_match.group(1)),
                'line_num': i
            }

    # Detectar CLOSE
    elif 'CLOSE |' in line and current_trade:
        pnl_match = re.search(r'pnl=([-\d.]+)', line)
        price_match = re.search(r'price=([\d.]+)', line)

        if pnl_match and price_match:
            pnl = float(pnl_match.group(1))
            exit_price = float(price_match.group(1))

            # Calcular movimento
            if current_trade['type'] == 'LONG':
                movement = exit_price - current_trade['entry_price']
            else:
                movement = current_trade['entry_price'] - exit_price

            trade_info = {
                'type': current_trade['type'],
                'entry_price': current_trade['entry_price'],
                'exit_price': exit_price,
                'pnl': pnl,
                'movement': movement,
                'is_win': pnl > 0
            }

            trades_complete.append(trade_info)

            if current_trade['type'] == 'LONG':
                entries_long.append(trade_info)
            else:
                entries_short.append(trade_info)

            current_trade = None

# Análise
total_trades = len(trades_complete)
total_wins = sum(1 for t in trades_complete if t['is_win'])
total_losses = total_trades - total_wins
win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0

# LONG stats
long_trades = len(entries_long)
long_wins = sum(1 for t in entries_long if t['is_win'])
long_losses = long_trades - long_wins
long_wr = (long_wins / long_trades * 100) if long_trades > 0 else 0
long_pnl = sum(t['pnl'] for t in entries_long)
long_avg_pnl = long_pnl / long_trades if long_trades > 0 else 0

# SHORT stats
short_trades = len(entries_short)
short_wins = sum(1 for t in entries_short if t['is_win'])
short_losses = short_trades - short_wins
short_wr = (short_wins / short_trades * 100) if short_trades > 0 else 0
short_pnl = sum(t['pnl'] for t in entries_short)
short_avg_pnl = short_pnl / short_trades if short_trades > 0 else 0

# Análise de sequências
max_win_streak = 0
max_loss_streak = 0
current_win_streak = 0
current_loss_streak = 0

for trade in trades_complete:
    if trade['is_win']:
        current_win_streak += 1
        current_loss_streak = 0
        max_win_streak = max(max_win_streak, current_win_streak)
    else:
        current_loss_streak += 1
        current_win_streak = 0
        max_loss_streak = max(max_loss_streak, current_loss_streak)

# Análise de movimento (entry quality)
wins_by_movement = [t['movement'] for t in trades_complete if t['is_win']]
losses_by_movement = [t['movement'] for t in trades_complete if not t['is_win']]

avg_win_movement = sum(wins_by_movement) / len(wins_by_movement) if wins_by_movement else 0
avg_loss_movement = sum(losses_by_movement) / len(losses_by_movement) if losses_by_movement else 0

# Calcular métricas financeiras
total_profit = sum(t['pnl'] for t in trades_complete if t['is_win'])
total_loss = sum(abs(t['pnl']) for t in trades_complete if not t['is_win'])
net_pnl = sum(t['pnl'] for t in trades_complete)
avg_win = total_profit / total_wins if total_wins > 0 else 0
avg_loss = total_loss / total_losses if total_losses > 0 else 0
profit_factor = total_profit / total_loss if total_loss > 0 else 0

print('=' * 100)
print('🎯 ANÁLISE COMPLETA DE QUALIDADE DAS ENTRADAS - SEVENTEEN')
print('=' * 100)
print(f'Período: Semana de operação sem filtro de horário')
print(f'Total de Trades Analisados: {total_trades}')
print('')

print('=' * 100)
print('📊 PERFORMANCE GERAL')
print('=' * 100)
print(f'Win Rate Geral: {win_rate:.1f}% ({total_wins}W / {total_losses}L)')
print(f'Net PnL: ${net_pnl:.2f}')
print(f'PnL por Trade: ${net_pnl/total_trades:.2f}')
print(f'Profit Factor: {profit_factor:.2f}')
print('')
print(f'Média de Ganho: ${avg_win:.2f}')
print(f'Média de Perda: ${avg_loss:.2f}')
print(f'Risk/Reward Ratio: {avg_win/avg_loss:.2f}' if avg_loss > 0 else 'N/A')
print('')

print('=' * 100)
print('🔍 ANÁLISE POR DIREÇÃO (LONG vs SHORT)')
print('=' * 100)
print(f'{"DIREÇÃO":<10} {"TRADES":<10} {"WIN%":<10} {"NET PnL":<15} {"$/TRADE":<15}')
print('-' * 100)
print(f'{"LONG":<10} {long_trades:<10} {long_wr:<9.1f}% ${long_pnl:<14.2f} ${long_avg_pnl:<14.2f}')
print(f'{"SHORT":<10} {short_trades:<10} {short_wr:<9.1f}% ${short_pnl:<14.2f} ${short_avg_pnl:<14.2f}')
print('')

# Determinar melhor direção
if long_wr > short_wr and long_pnl > short_pnl:
    print('✅ LONG é claramente superior (maior WR e PnL)')
elif short_wr > long_wr and short_pnl > long_pnl:
    print('✅ SHORT é claramente superior (maior WR e PnL)')
elif long_pnl > short_pnl:
    print('🟡 LONG tem melhor PnL, mas SHORT tem melhor WR')
elif short_pnl > long_pnl:
    print('🟡 SHORT tem melhor PnL, mas LONG tem melhor WR')
else:
    print('⚖️  Ambas direções com performance similar')

print('')

print('=' * 100)
print('📈 ANÁLISE DE CONSISTÊNCIA')
print('=' * 100)
print(f'Maior sequência de ganhos: {max_win_streak} trades')
print(f'Maior sequência de perdas: {max_loss_streak} trades')
print('')

if max_loss_streak >= 5:
    print('🔴 ATENÇÃO: Sequências longas de perdas indicam problemas nas entradas')
elif max_loss_streak >= 3:
    print('🟡 CUIDADO: Sequências de 3+ perdas são frequentes')
else:
    print('✅ Boa consistência: Sem sequências longas de perdas')

print('')

print('=' * 100)
print('🎯 QUALIDADE DAS ENTRADAS (Movement Analysis)')
print('=' * 100)
print(f'Movimento médio em trades ganhos: {avg_win_movement:.2f} pontos')
print(f'Movimento médio em trades perdidos: {avg_loss_movement:.2f} pontos')
print('')

if avg_win_movement > 0 and avg_loss_movement < 0:
    print('✅ EXCELENTE: Entradas capturam movimento na direção correta')
elif avg_win_movement > abs(avg_loss_movement):
    print('🟢 BOM: Ganhos capturam mais movimento que perdas')
elif avg_win_movement > 0:
    print('🟡 RAZOÁVEL: Entradas têm direção correta mas magnitude similar')
else:
    print('🔴 RUIM: Entradas não capturam bem o movimento do mercado')

print('')

print('=' * 100)
print('💡 CONCLUSÃO SOBRE SEVENTEEN - ENTRY TIMING REWARDS')
print('=' * 100)
print('')

# Avaliação geral
if win_rate >= 50 and profit_factor >= 1.2:
    print('✅ SUCESSO COMPLETO: Entry Timing Rewards funcionou!')
    print('   O experimento melhorou significativamente a qualidade das entradas.')
elif win_rate >= 45 and profit_factor >= 1.0:
    print('🟢 SUCESSO PARCIAL: Entry Timing Rewards teve impacto positivo')
    print('   Há melhoria nas entradas, mas ainda pode ser otimizado.')
elif win_rate >= 40:
    print('🟡 RESULTADO MISTO: Entry Timing Rewards teve algum efeito')
    print('   As entradas são aceitáveis mas precisam de refinamento.')
elif win_rate >= 35:
    print('🟠 ABAIXO DO ESPERADO: Entry Timing Rewards não teve impacto significativo')
    print('   O experimento não melhorou substancialmente a qualidade das entradas.')
else:
    print('🔴 FALHA: Entry Timing Rewards não funcionou')
    print('   O experimento não conseguiu melhorar a qualidade das entradas.')

print('')
print('📋 ANÁLISE ESPECÍFICA:')
print(f'   • Win Rate: {win_rate:.1f}% (Meta: >=50%)')
print(f'   • Profit Factor: {profit_factor:.2f} (Meta: >=1.2)')
print(f'   • Risk/Reward: {avg_win/avg_loss:.2f}' if avg_loss > 0 else 'N/A')
print(f'   • PnL/Trade: ${net_pnl/total_trades:.2f}')

print('')
print('🔧 RECOMENDAÇÕES:')

if win_rate < 40:
    print('   1. Revisar os thresholds de entry_confidence')
    print('   2. Aumentar penalidades para entradas de baixa qualidade')
    print('   3. Considerar filtros adicionais (volatilidade, spread, etc)')

if profit_factor < 1.0:
    print('   1. Revisar SL/TP ratios')
    print('   2. Implementar trailing stops mais agressivos')
    print('   3. Melhorar exit timing')

if max_loss_streak >= 5:
    print('   1. Implementar circuit breaker após 3-4 perdas consecutivas')
    print('   2. Revisar condições de mercado durante sequências de perda')

if abs(long_wr - short_wr) > 15:
    print(f'   1. Há assimetria entre LONG ({long_wr:.1f}%) e SHORT ({short_wr:.1f}%)')
    print('   2. Considerar ajustar rewards/penalties por direção')

print('')
print('=' * 100)
