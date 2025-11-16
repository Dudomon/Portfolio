"""
Análise Detalhada do Log de Trading - Sixteen 1.55M
Identifica todos os trades, motivos de fechamento, e padrões de perda
"""

import re
from datetime import datetime, timedelta
from collections import defaultdict

def parse_trading_log(filepath):
    """Analisa log completo e extrai informações detalhadas"""

    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    # Dados da sessão
    session_info = {
        'start_time': None,
        'symbol': None,
        'magic': None
    }

    # Trades abertos e fechados
    opened_trades = {}
    closed_trades = []

    # Estatísticas
    stats = {
        'total_opens': 0,
        'total_closes': 0,
        'longs': 0,
        'shorts': 0,
        'wins': 0,
        'losses': 0,
        'total_pnl': 0.0,
        'by_hour': defaultdict(lambda: {'opens': 0, 'closes': 0, 'pnl': 0.0}),
        'close_reasons': defaultdict(int),
        'trailing_moves': 0
    }

    current_time = None

    for i, line in enumerate(lines):
        # Capturar timestamp
        time_match = re.search(r'\[(\d{2}:\d{2}:\d{2})\]', line)
        if time_match:
            current_time = time_match.group(1)
            hour = int(current_time.split(':')[0])

        # Informações da sessão
        if 'Símbolo:' in line:
            match = re.search(r'Símbolo:\s*(\w+).*Magic:\s*(\d+)', line)
            if match:
                session_info['symbol'] = match.group(1)
                session_info['magic'] = match.group(2)

        if 'Início:' in line:
            match = re.search(r'Início:\s*(.+)', line)
            if match:
                session_info['start_time'] = match.group(1).strip()

        # ABERTURA DE TRADE
        if '[🎯 TRADE V7]' in line and 'executado' in line:
            # Exemplo: [09:53:00] [🎯 TRADE V7] 📈 LONG executado - #508388633 @ 4042.25 | SL: 4026.80 | TP: 4059.80
            match = re.search(r'([📈📉])\s*(LONG|SHORT)\s+executado.*?#(\d+)\s*@\s*([\d.]+).*?SL:\s*([\d.]+).*?TP:\s*([\d.]+)', line)
            if match:
                direction = match.group(2)
                ticket = match.group(3)
                entry = float(match.group(4))
                sl = float(match.group(5))
                tp = float(match.group(6))

                opened_trades[ticket] = {
                    'ticket': ticket,
                    'direction': direction,
                    'entry': entry,
                    'sl': sl,
                    'tp': tp,
                    'open_time': current_time,
                    'open_hour': hour,
                    'trailing_count': 0
                }

                stats['total_opens'] += 1
                stats['by_hour'][hour]['opens'] += 1
                if direction == 'LONG':
                    stats['longs'] += 1
                else:
                    stats['shorts'] += 1

        # TRAILING STOP (modificações de SL)
        if '[MODIFY SUCCESS]' in line and current_time:
            match = re.search(r'Pos #(\d+)', line)
            if match:
                ticket = match.group(1)
                if ticket in opened_trades:
                    opened_trades[ticket]['trailing_count'] += 1
                    stats['trailing_moves'] += 1

        # FECHAMENTO DE TRADE
        if 'CLOSE' in line and 'ticket=' in line:
            # Exemplo: CLOSE | ticket=508389886 | pnl=-32.94
            match = re.search(r'ticket=(\d+).*?pnl=([-\d.]+)', line)
            if match:
                ticket = match.group(1)
                pnl = float(match.group(2))

                # Buscar motivo do fechamento nas linhas anteriores
                close_reason = "UNKNOWN"
                for j in range(max(0, i-5), i):
                    prev_line = lines[j]
                    if 'COOLDOWN ADAPTATIVO' in prev_line:
                        if 'WIN' in prev_line:
                            close_reason = "TP_HIT"
                        elif 'LOSS' in prev_line:
                            close_reason = "SL_HIT"
                    elif 'TAKE PROFIT' in prev_line:
                        close_reason = "TP_HIT"
                    elif 'STOP LOSS' in prev_line:
                        close_reason = "SL_HIT"
                    elif 'TRAILING' in prev_line:
                        close_reason = "TRAILING_STOP"

                # Informações do trade aberto
                trade_info = opened_trades.get(ticket, {
                    'direction': 'UNKNOWN',
                    'entry': 0.0,
                    'open_time': '00:00:00',
                    'open_hour': 0,
                    'trailing_count': 0
                })

                closed_trade = {
                    'ticket': ticket,
                    'direction': trade_info.get('direction', 'UNKNOWN'),
                    'entry': trade_info.get('entry', 0.0),
                    'open_time': trade_info.get('open_time', '00:00:00'),
                    'close_time': current_time,
                    'open_hour': trade_info.get('open_hour', 0),
                    'close_hour': hour,
                    'pnl': pnl,
                    'close_reason': close_reason,
                    'trailing_count': trade_info.get('trailing_count', 0)
                }

                closed_trades.append(closed_trade)

                stats['total_closes'] += 1
                stats['by_hour'][hour]['closes'] += 1
                stats['by_hour'][hour]['pnl'] += pnl
                stats['total_pnl'] += pnl
                stats['close_reasons'][close_reason] += 1

                if pnl > 0:
                    stats['wins'] += 1
                else:
                    stats['losses'] += 1

                # Remover do dict de abertos
                if ticket in opened_trades:
                    del opened_trades[ticket]

    return session_info, closed_trades, stats, opened_trades

def print_detailed_analysis(session_info, closed_trades, stats, opened_trades):
    """Imprime análise detalhada formatada"""

    print("=" * 100)
    print("ANÁLISE DETALHADA - SESSÃO DE TRADING SIXTEEN 1.55M")
    print("=" * 100)
    print()

    print("📋 INFORMAÇÕES DA SESSÃO:")
    print(f"   Símbolo: {session_info['symbol']}")
    print(f"   Magic Number: {session_info['magic']}")
    print(f"   Início: {session_info['start_time']}")
    print(f"   Trades Abertos: {stats['total_opens']}")
    print(f"   Trades Fechados: {stats['total_closes']}")
    print(f"   Posições Ainda Abertas: {len(opened_trades)}")
    print()

    print("=" * 100)
    print("📊 RESUMO GERAL")
    print("=" * 100)
    print(f"Total de Trades: {stats['total_closes']}")
    print(f"   LONG: {stats['longs']} ({stats['longs']/stats['total_opens']*100:.1f}%)")
    print(f"   SHORT: {stats['shorts']} ({stats['shorts']/stats['total_opens']*100:.1f}%)")
    print()
    print(f"Resultados:")
    print(f"   WINS: {stats['wins']} ({stats['wins']/stats['total_closes']*100:.1f}%)")
    print(f"   LOSSES: {stats['losses']} ({stats['losses']/stats['total_closes']*100:.1f}%)")
    print()
    print(f"P&L Total: ${stats['total_pnl']:.2f}")
    print(f"P&L Médio: ${stats['total_pnl']/stats['total_closes']:.2f}" if stats['total_closes'] > 0 else "N/A")
    print()
    print(f"Trailing Stop Moves: {stats['trailing_moves']}")
    print()

    print("=" * 100)
    print("🎯 MOTIVOS DE FECHAMENTO")
    print("=" * 100)
    for reason, count in sorted(stats['close_reasons'].items(), key=lambda x: x[1], reverse=True):
        pct = count / stats['total_closes'] * 100 if stats['total_closes'] > 0 else 0
        print(f"   {reason}: {count} trades ({pct:.1f}%)")
    print()

    print("=" * 100)
    print("⏰ PERFORMANCE POR HORÁRIO")
    print("=" * 100)
    print(f"{'Hora':<6} {'Abertos':<10} {'Fechados':<10} {'P&L Total':<12}")
    print("-" * 100)

    for hour in sorted(stats['by_hour'].keys()):
        h_stats = stats['by_hour'][hour]
        status = "✅" if h_stats['pnl'] > 0 else "🔴" if h_stats['pnl'] < 0 else "⚪"
        print(f"{hour:02d}:00  {h_stats['opens']:<10} {h_stats['closes']:<10} ${h_stats['pnl']:>10.2f}  {status}")
    print()

    print("=" * 100)
    print("🔴 TOP 10 PIORES TRADES")
    print("=" * 100)
    worst_trades = sorted(closed_trades, key=lambda x: x['pnl'])[:10]
    print(f"{'#':<4} {'Ticket':<12} {'Dir':<6} {'Open':<10} {'Close':<10} {'P&L':<12} {'Motivo':<15} {'Trailing':<10}")
    print("-" * 100)
    for idx, trade in enumerate(worst_trades, 1):
        print(f"{idx:<4} #{trade['ticket']:<11} {trade['direction']:<6} "
              f"{trade['open_time']:<10} {trade['close_time']:<10} "
              f"${trade['pnl']:>10.2f}  {trade['close_reason']:<15} {trade['trailing_count']:<10}")
    print()

    print("=" * 100)
    print("✅ TOP 10 MELHORES TRADES")
    print("=" * 100)
    best_trades = sorted(closed_trades, key=lambda x: x['pnl'], reverse=True)[:10]
    print(f"{'#':<4} {'Ticket':<12} {'Dir':<6} {'Open':<10} {'Close':<10} {'P&L':<12} {'Motivo':<15} {'Trailing':<10}")
    print("-" * 100)
    for idx, trade in enumerate(best_trades, 1):
        print(f"{idx:<4} #{trade['ticket']:<11} {trade['direction']:<6} "
              f"{trade['open_time']:<10} {trade['close_time']:<10} "
              f"${trade['pnl']:>10.2f}  {trade['close_reason']:<15} {trade['trailing_count']:<10}")
    print()

    print("=" * 100)
    print("⚠️ POSIÇÕES AINDA ABERTAS")
    print("=" * 100)
    if opened_trades:
        print(f"{'Ticket':<12} {'Dir':<6} {'Entry':<10} {'Open Time':<12} {'Trailing':<10}")
        print("-" * 100)
        for ticket, trade in opened_trades.items():
            print(f"#{ticket:<11} {trade['direction']:<6} ${trade['entry']:<9.2f} "
                  f"{trade['open_time']:<12} {trade['trailing_count']:<10}")
    else:
        print("   Nenhuma posição aberta")
    print()

    print("=" * 100)
    print("🔍 ANÁLISE DE PADRÕES")
    print("=" * 100)

    # Análise de direção
    sl_hits = stats['close_reasons'].get('SL_HIT', 0)
    tp_hits = stats['close_reasons'].get('TP_HIT', 0)

    print(f"Win Rate: {stats['wins']/stats['total_closes']*100:.1f}%")
    print(f"Stop Loss Hit Rate: {sl_hits/stats['total_closes']*100:.1f}%")
    print(f"Take Profit Hit Rate: {tp_hits/stats['total_closes']*100:.1f}%")
    print()

    # Trades por direção
    long_trades = [t for t in closed_trades if t['direction'] == 'LONG']
    short_trades = [t for t in closed_trades if t['direction'] == 'SHORT']

    if long_trades:
        long_pnl = sum(t['pnl'] for t in long_trades)
        long_wins = sum(1 for t in long_trades if t['pnl'] > 0)
        print(f"LONG Trades: {len(long_trades)} | Win Rate: {long_wins/len(long_trades)*100:.1f}% | P&L: ${long_pnl:.2f}")

    if short_trades:
        short_pnl = sum(t['pnl'] for t in short_trades)
        short_wins = sum(1 for t in short_trades if t['pnl'] > 0)
        print(f"SHORT Trades: {len(short_trades)} | Win Rate: {short_wins/len(short_trades)*100:.1f}% | P&L: ${short_pnl:.2f}")

    print()

def main():
    # Robô com $668 balance (26/Out/2025 - Magic 777035)
    log_file = r"D:\Projeto\Modelo PPO Trader\logs\trading_session_20251026_214522_5740_af46c751.txt"

    print("Analisando log...")
    session_info, closed_trades, stats, opened_trades = parse_trading_log(log_file)

    print_detailed_analysis(session_info, closed_trades, stats, opened_trades)

if __name__ == "__main__":
    main()
