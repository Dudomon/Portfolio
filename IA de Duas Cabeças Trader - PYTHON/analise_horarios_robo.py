"""
Análise de Performance dos Robôs por Horário
Extrai trades dos logs e analisa ganhos vs perdas por hora do dia
"""

import os
import re
from datetime import datetime
from collections import defaultdict
import glob

def parse_log_file(filepath):
    """Extrai trades de um arquivo de log"""
    trades = []

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Padrão para capturar fechamentos de posição com P&L
        # Exemplo: [10:00:05] 🔒 [COOLDOWN ADAPTATIVO] LOSS (1L) → 45min | P&L: $-32.94
        # ou: CLOSE | ticket=508389886 | pnl=-32.94

        # Padrão 1: Linha de COOLDOWN com P&L
        pattern1 = r'\[(\d{2}:\d{2}:\d{2})\].*?P&L:\s*\$?([-\d.]+)'
        matches1 = re.findall(pattern1, content)

        for time_str, pnl_str in matches1:
            try:
                pnl = float(pnl_str)
                hour = int(time_str.split(':')[0])
                trades.append({
                    'time': time_str,
                    'hour': hour,
                    'pnl': pnl,
                    'type': 'WIN' if pnl > 0 else 'LOSS'
                })
            except:
                continue

        # Padrão 2: Linha CLOSE direta
        pattern2 = r'CLOSE.*?pnl=([-\d.]+)'
        matches2 = re.findall(pattern2, content)

        # Tentar extrair timestamp da linha anterior
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'CLOSE' in line and 'pnl=' in line:
                match = re.search(r'pnl=([-\d.]+)', line)
                if match:
                    pnl = float(match.group(1))
                    # Procurar timestamp na mesma linha ou linha anterior
                    time_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', line)
                    if not time_match and i > 0:
                        time_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', lines[i-1])

                    if time_match:
                        time_str = time_match.group(1)
                        hour = int(time_str.split(':')[0])
                        # Evitar duplicatas
                        if not any(t['time'] == time_str and t['pnl'] == pnl for t in trades):
                            trades.append({
                                'time': time_str,
                                'hour': hour,
                                'pnl': pnl,
                                'type': 'WIN' if pnl > 0 else 'LOSS'
                            })

    except Exception as e:
        print(f"Erro ao processar {filepath}: {e}")

    return trades

def analyze_by_hour(all_trades):
    """Analisa trades por hora do dia"""
    hourly_stats = defaultdict(lambda: {
        'wins': 0,
        'losses': 0,
        'total_pnl': 0.0,
        'win_pnl': 0.0,
        'loss_pnl': 0.0,
        'trades': []
    })

    for trade in all_trades:
        hour = trade['hour']
        pnl = trade['pnl']

        hourly_stats[hour]['trades'].append(trade)
        hourly_stats[hour]['total_pnl'] += pnl

        if pnl > 0:
            hourly_stats[hour]['wins'] += 1
            hourly_stats[hour]['win_pnl'] += pnl
        else:
            hourly_stats[hour]['losses'] += 1
            hourly_stats[hour]['loss_pnl'] += pnl

    return hourly_stats

def print_analysis(hourly_stats):
    """Imprime análise formatada"""
    print("=" * 100)
    print("ANÁLISE DE PERFORMANCE DOS ROBÔS POR HORÁRIO")
    print("=" * 100)
    print()

    # Ordenar por hora
    sorted_hours = sorted(hourly_stats.keys())

    print(f"{'Hora':<6} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win%':<8} {'P&L Total':<12} {'P&L Médio':<12} {'Status':<15}")
    print("-" * 100)

    total_trades = 0
    total_wins = 0
    total_losses = 0
    total_pnl = 0.0

    for hour in sorted_hours:
        stats = hourly_stats[hour]
        total = stats['wins'] + stats['losses']
        win_rate = (stats['wins'] / total * 100) if total > 0 else 0
        avg_pnl = stats['total_pnl'] / total if total > 0 else 0

        # Determinar status
        if win_rate >= 60 and stats['total_pnl'] > 0:
            status = "✅ EXCELENTE"
        elif win_rate >= 50 and stats['total_pnl'] > 0:
            status = "🟢 BOM"
        elif stats['total_pnl'] > 0:
            status = "🟡 POSITIVO"
        elif stats['total_pnl'] == 0:
            status = "⚪ NEUTRO"
        else:
            status = "🔴 NEGATIVO"

        print(f"{hour:02d}:00  {total:<8} {stats['wins']:<6} {stats['losses']:<8} "
              f"{win_rate:>6.1f}%  ${stats['total_pnl']:>10.2f}  ${avg_pnl:>10.2f}  {status}")

        total_trades += total
        total_wins += stats['wins']
        total_losses += stats['losses']
        total_pnl += stats['total_pnl']

    print("-" * 100)
    overall_win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    overall_avg_pnl = total_pnl / total_trades if total_trades > 0 else 0
    print(f"TOTAL  {total_trades:<8} {total_wins:<6} {total_losses:<8} "
          f"{overall_win_rate:>6.1f}%  ${total_pnl:>10.2f}  ${overall_avg_pnl:>10.2f}")
    print()

    # Identificar melhores e piores horários
    print("=" * 100)
    print("🏆 TOP 5 MELHORES HORÁRIOS (por P&L Total)")
    print("=" * 100)
    best_hours = sorted(hourly_stats.items(), key=lambda x: x[1]['total_pnl'], reverse=True)[:5]
    for hour, stats in best_hours:
        total = stats['wins'] + stats['losses']
        win_rate = (stats['wins'] / total * 100) if total > 0 else 0
        print(f"{hour:02d}:00 - {total} trades | Win Rate: {win_rate:.1f}% | P&L: ${stats['total_pnl']:.2f}")

    print()
    print("=" * 100)
    print("⚠️ TOP 5 PIORES HORÁRIOS (por P&L Total)")
    print("=" * 100)
    worst_hours = sorted(hourly_stats.items(), key=lambda x: x[1]['total_pnl'])[:5]
    for hour, stats in worst_hours:
        total = stats['wins'] + stats['losses']
        win_rate = (stats['wins'] / total * 100) if total > 0 else 0
        print(f"{hour:02d}:00 - {total} trades | Win Rate: {win_rate:.1f}% | P&L: ${stats['total_pnl']:.2f}")

    print()
    print("=" * 100)
    print("📊 RESUMO GERAL")
    print("=" * 100)
    print(f"Total de Trades: {total_trades}")
    print(f"Wins: {total_wins} ({overall_win_rate:.1f}%)")
    print(f"Losses: {total_losses} ({100-overall_win_rate:.1f}%)")
    print(f"P&L Total: ${total_pnl:.2f}")
    print(f"P&L Médio por Trade: ${overall_avg_pnl:.2f}")
    print()

def main():
    # Procurar todos os logs
    log_dir = r"D:\Projeto\Modelo PPO Trader\logs"
    log_files = glob.glob(os.path.join(log_dir, "trading_session_*.txt"))

    print(f"Encontrados {len(log_files)} arquivos de log")
    print("Processando...\n")

    all_trades = []
    for log_file in log_files:
        trades = parse_log_file(log_file)
        all_trades.extend(trades)
        if trades:
            basename = os.path.basename(log_file)
            print(f"✓ {basename}: {len(trades)} trades")

    print(f"\nTotal de trades extraídos: {len(all_trades)}\n")

    if not all_trades:
        print("❌ Nenhum trade encontrado nos logs!")
        return

    # Analisar por horário
    hourly_stats = analyze_by_hour(all_trades)

    # Imprimir análise
    print_analysis(hourly_stats)

if __name__ == "__main__":
    main()
