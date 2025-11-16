#!/usr/bin/env python3
"""Análise rápida dos dados JSONL de performance"""
import json
from collections import Counter

def analyze_performance_jsonl(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    data_lines = [json.loads(line) for line in lines if 'performance_metrics' in line]
    print(f'Total de entradas: {len(data_lines)}')
    
    if not data_lines:
        print('Nenhum dado de performance encontrado')
        return
    
    # Básico
    first = data_lines[0] 
    last = data_lines[-1]
    print(f'Período: Step {first["step"]} até {last["step"]}')
    print(f'Portfolio inicial: ${first["portfolio_value"]:.2f}')
    print(f'Portfolio final: ${last["portfolio_value"]:.2f}')
    print(f'Retorno: {((last["portfolio_value"]/first["portfolio_value"])-1)*100:.2f}%')
    
    # Distribuição de trades
    trade_counts = [entry['trades_count'] for entry in data_lines]
    trade_distribution = Counter(trade_counts)
    print('\nDistribuição de trades por entrada:')
    for trades, count in sorted(trade_distribution.items()):
        print(f'  {trades} trades: {count} entradas ({count/len(data_lines)*100:.1f}%)')
    
    # Portfolio range
    portfolios = [entry['portfolio_value'] for entry in data_lines]
    print(f'\nPortfolio - Min: ${min(portfolios):.2f}, Max: ${max(portfolios):.2f}')
    
    # Entradas com mais trades
    high_trades = [entry for entry in data_lines if entry['trades_count'] >= 50]
    print(f'\nEntradas com >=50 trades: {len(high_trades)}')
    if high_trades:
        print('Amostras:')
        for entry in high_trades[:5]:
            print(f'  Step {entry["step"]}: {entry["trades_count"]} trades, Portfolio: ${entry["portfolio_value"]:.2f}, Win Rate: {entry["win_rate"]*100:.1f}%')
    
    # Stats de win rate
    win_rates = [entry['win_rate'] for entry in data_lines if entry['trades_count'] > 0]
    if win_rates:
        print(f'\nWin Rate - Média: {sum(win_rates)/len(win_rates)*100:.1f}%, Min: {min(win_rates)*100:.1f}%, Max: {max(win_rates)*100:.1f}%')

if __name__ == '__main__':
    print('=== ANÁLISE PERFORMANCE JSONL ===')
    analyze_performance_jsonl('avaliacoes/performance_20250801_101805.jsonl')