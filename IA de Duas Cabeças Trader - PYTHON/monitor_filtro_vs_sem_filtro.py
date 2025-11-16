#!/usr/bin/env python3
"""
🔥 MONITOR CONTÍNUO: Seventeen COM FILTRO vs SEM FILTRO
Atualiza automaticamente a comparação conforme novos trades acontecem
"""
import re
import os
import time
from collections import defaultdict
from datetime import datetime

# Configuração
BLOCKED_HOURS = [8, 9, 10, 11, 17, 21]

# Logs a monitorar
LOG_COM_FILTRO = "D:/Projeto/Modelo PPO Trader/logs/trading_session_20251031_160231_42780_590145c4.txt"
LOG_SEM_FILTRO = "D:/Projeto/Modelo PPO Trader/logs/trading_session_20251031_160208_43368_8fcc7702.txt"

def parse_log(log_path):
    """Parse log e retorna estatísticas"""
    if not os.path.exists(log_path):
        return None

    wins_by_hour = defaultdict(int)
    losses_by_hour = defaultdict(int)
    profit_by_hour = defaultdict(float)
    loss_by_hour = defaultdict(float)

    total_wins = 0
    total_losses = 0
    total_profit = 0.0
    total_loss = 0.0

    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()

    current_hour = None
    for i, line in enumerate(lines):
        # Extrair hora
        time_match = re.search(r'\[(\d{2}):\d{2}:\d{2}\]', line)
        if time_match:
            current_hour = int(time_match.group(1))

        # Detectar CLOSE com PnL
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
                    if pnl > 0:
                        wins_by_hour[hour_to_use] += 1
                        profit_by_hour[hour_to_use] += pnl
                        total_wins += 1
                        total_profit += pnl
                    else:
                        losses_by_hour[hour_to_use] += 1
                        loss_by_hour[hour_to_use] += abs(pnl)
                        total_losses += 1
                        total_loss += abs(pnl)

    total_trades = total_wins + total_losses
    win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
    net_pnl = total_profit - total_loss

    # Calcular estatísticas por categoria (bloqueados vs permitidos)
    blocked_wins = sum(wins_by_hour[h] for h in BLOCKED_HOURS)
    blocked_losses = sum(losses_by_hour[h] for h in BLOCKED_HOURS)
    blocked_profit = sum(profit_by_hour[h] for h in BLOCKED_HOURS)
    blocked_loss = sum(loss_by_hour[h] for h in BLOCKED_HOURS)
    blocked_trades = blocked_wins + blocked_losses
    blocked_wr = (blocked_wins / blocked_trades * 100) if blocked_trades > 0 else 0
    blocked_net = blocked_profit - blocked_loss

    allowed_wins = sum(wins_by_hour[h] for h in range(24) if h not in BLOCKED_HOURS)
    allowed_losses = sum(losses_by_hour[h] for h in range(24) if h not in BLOCKED_HOURS)
    allowed_profit = sum(profit_by_hour[h] for h in range(24) if h not in BLOCKED_HOURS)
    allowed_loss = sum(loss_by_hour[h] for h in range(24) if h not in BLOCKED_HOURS)
    allowed_trades = allowed_wins + allowed_losses
    allowed_wr = (allowed_wins / allowed_trades * 100) if allowed_trades > 0 else 0
    allowed_net = allowed_profit - allowed_loss

    return {
        'total_trades': total_trades,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': win_rate,
        'total_profit': total_profit,
        'total_loss': total_loss,
        'net_pnl': net_pnl,
        'blocked_trades': blocked_trades,
        'blocked_wins': blocked_wins,
        'blocked_losses': blocked_losses,
        'blocked_wr': blocked_wr,
        'blocked_net': blocked_net,
        'allowed_trades': allowed_trades,
        'allowed_wins': allowed_wins,
        'allowed_losses': allowed_losses,
        'allowed_wr': allowed_wr,
        'allowed_net': allowed_net,
        'wins_by_hour': wins_by_hour,
        'losses_by_hour': losses_by_hour,
        'profit_by_hour': profit_by_hour,
        'loss_by_hour': loss_by_hour
    }

def format_stats_table(stats_com, stats_sem):
    """Formata tabela comparativa"""
    lines = []
    lines.append("=" * 100)
    lines.append("📊 COMPARAÇÃO EM TEMPO REAL: COM FILTRO vs SEM FILTRO")
    lines.append("=" * 100)
    lines.append(f"Atualizado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Tabela principal
    lines.append(f"{'MÉTRICA':<30} {'COM FILTRO':<20} {'SEM FILTRO':<20} {'DIFERENÇA':<20}")
    lines.append("-" * 100)

    # Total de trades
    diff_trades = stats_com['total_trades'] - stats_sem['total_trades']
    lines.append(f"{'Total de Trades':<30} {stats_com['total_trades']:<20} {stats_sem['total_trades']:<20} {diff_trades:+<20}")

    # Win Rate
    diff_wr = stats_com['win_rate'] - stats_sem['win_rate']
    lines.append(f"{'Win Rate':<30} {stats_com['win_rate']:<19.1f}% {stats_sem['win_rate']:<19.1f}% {diff_wr:+<19.1f}%")

    # Net PnL
    diff_pnl = stats_com['net_pnl'] - stats_sem['net_pnl']
    pnl_emoji_com = "💰" if stats_com['net_pnl'] > 0 else "💸"
    pnl_emoji_sem = "💰" if stats_sem['net_pnl'] > 0 else "💸"
    diff_emoji = "🟢" if diff_pnl > 0 else "🔴"
    lines.append(f"{'Net PnL':<30} {pnl_emoji_com} ${stats_com['net_pnl']:<16.2f} {pnl_emoji_sem} ${stats_sem['net_pnl']:<16.2f} {diff_emoji} ${diff_pnl:+<16.2f}")

    # PnL por trade
    pnl_per_trade_com = stats_com['net_pnl'] / stats_com['total_trades'] if stats_com['total_trades'] > 0 else 0
    pnl_per_trade_sem = stats_sem['net_pnl'] / stats_sem['total_trades'] if stats_sem['total_trades'] > 0 else 0
    diff_ppt = pnl_per_trade_com - pnl_per_trade_sem
    lines.append(f"{'PnL por Trade':<30} ${pnl_per_trade_com:<18.2f} ${pnl_per_trade_sem:<18.2f} ${diff_ppt:+<18.2f}")

    # Separador
    lines.append("")
    lines.append("=" * 100)
    lines.append("🎯 ANÁLISE DOS HORÁRIOS BLOQUEADOS [8, 9, 10, 11, 17, 21]")
    lines.append("=" * 100)

    # Horários bloqueados (apenas SEM FILTRO tem dados relevantes)
    lines.append(f"\n📊 SEM FILTRO - Performance nos horários bloqueados:")
    lines.append(f"   • Trades: {stats_sem['blocked_trades']}")
    lines.append(f"   • Win Rate: {stats_sem['blocked_wr']:.1f}%")
    lines.append(f"   • Net PnL: ${stats_sem['blocked_net']:.2f}")

    lines.append(f"\n📊 SEM FILTRO - Performance nos horários permitidos:")
    lines.append(f"   • Trades: {stats_sem['allowed_trades']}")
    lines.append(f"   • Win Rate: {stats_sem['allowed_wr']:.1f}%")
    lines.append(f"   • Net PnL: ${stats_sem['allowed_net']:.2f}")

    # Eficácia do filtro
    lines.append("")
    lines.append("=" * 100)
    lines.append("✅ VALIDAÇÃO DO FILTRO")
    lines.append("=" * 100)

    if stats_sem['blocked_trades'] > 0 and stats_sem['allowed_trades'] > 0:
        wr_gain = stats_sem['allowed_wr'] - stats_sem['blocked_wr']
        pnl_avoided = stats_sem['blocked_net']

        lines.append(f"\n🎯 Ganho de Win Rate ao evitar bloqueados: {wr_gain:+.1f}%")
        lines.append(f"💰 PnL evitado (horários bloqueados): ${pnl_avoided:.2f}")

        # Comparação COM vs SEM filtro
        lines.append(f"\n📈 Resultado final COM FILTRO vs SEM FILTRO:")
        lines.append(f"   • Diferença de WR: {diff_wr:+.1f}%")
        lines.append(f"   • Diferença de PnL: ${diff_pnl:+.2f}")
        lines.append(f"   • Diferença %: {(diff_pnl/abs(stats_sem['net_pnl'])*100) if stats_sem['net_pnl'] != 0 else 0:+.1f}%")

        if diff_pnl > 0 and diff_wr > 0:
            lines.append(f"\n✅ FILTRO ESTÁ SENDO BENÉFICO! (+{diff_wr:.1f}% WR, +${diff_pnl:.2f} PnL)")
        elif diff_pnl < 0:
            lines.append(f"\n⚠️  FILTRO PODE ESTAR PREJUDICANDO ({diff_wr:+.1f}% WR, ${diff_pnl:+.2f} PnL)")
        else:
            lines.append(f"\n🟡 FILTRO TEM IMPACTO NEUTRO")

    lines.append("\n" + "=" * 100)

    return "\n".join(lines)

def monitor_loop(refresh_interval=10):
    """Loop de monitoramento contínuo"""
    print("🚀 Iniciando monitor contínuo...")
    print(f"📁 COM FILTRO: {os.path.basename(LOG_COM_FILTRO)}")
    print(f"📁 SEM FILTRO: {os.path.basename(LOG_SEM_FILTRO)}")
    print(f"🔄 Atualização a cada {refresh_interval} segundos")
    print(f"⏸️  Pressione Ctrl+C para parar\n")

    try:
        while True:
            # Limpar tela (Windows)
            os.system('cls' if os.name == 'nt' else 'clear')

            # Parse logs
            stats_com = parse_log(LOG_COM_FILTRO)
            stats_sem = parse_log(LOG_SEM_FILTRO)

            if stats_com and stats_sem:
                # Mostrar tabela
                print(format_stats_table(stats_com, stats_sem))
            else:
                print("⚠️  Aguardando dados dos logs...")

            # Aguardar próxima atualização
            time.sleep(refresh_interval)

    except KeyboardInterrupt:
        print("\n\n⏹️  Monitor interrompido pelo usuário.")
        print("📊 Última comparação salva acima.")

if __name__ == "__main__":
    import sys

    # Permitir passar intervalo como argumento
    refresh_interval = int(sys.argv[1]) if len(sys.argv) > 1 else 10

    monitor_loop(refresh_interval)
