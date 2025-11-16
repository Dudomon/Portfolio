#!/usr/bin/env python3
"""
🔍 ANÁLISE COMPLETA: Validação dos Horários Bloqueados
Analisa performance por horário do robô SEM filtro para validar se os bloqueios fazem sentido
"""
import re
import os
from collections import defaultdict
from datetime import datetime

# Horários atualmente bloqueados no filtro
BLOCKED_HOURS = [8, 9, 10, 11, 17, 21]

def parse_log_file(log_path):
    """Parse um arquivo de log e extrai trades com horário"""
    if not os.path.exists(log_path):
        return []

    trades = []

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

                # Pegar hora das linhas anteriores se necessário
                hour_to_use = current_hour
                for j in range(max(0, i-3), i):
                    prev_time_match = re.search(r'\[(\d{2}):\d{2}:\d{2}\]', lines[j])
                    if prev_time_match:
                        hour_to_use = int(prev_time_match.group(1))
                        break

                if hour_to_use is not None:
                    trades.append({
                        'hour': hour_to_use,
                        'pnl': pnl,
                        'is_win': pnl > 0
                    })

    return trades

def analyze_by_hour(all_trades):
    """Analisa trades por horário"""
    stats_by_hour = {}

    for hour in range(24):
        hour_trades = [t for t in all_trades if t['hour'] == hour]

        if not hour_trades:
            stats_by_hour[hour] = {
                'trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0.0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'net_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'profit_factor': 0.0,
                'pnl_per_trade': 0.0
            }
            continue

        wins = [t for t in hour_trades if t['is_win']]
        losses = [t for t in hour_trades if not t['is_win']]

        total_wins = len(wins)
        total_losses = len(losses)
        total_trades = len(hour_trades)

        total_profit = sum(t['pnl'] for t in wins)
        total_loss = sum(abs(t['pnl']) for t in losses)
        net_pnl = sum(t['pnl'] for t in hour_trades)

        win_rate = (total_wins / total_trades * 100) if total_trades > 0 else 0
        avg_win = total_profit / total_wins if total_wins > 0 else 0
        avg_loss = total_loss / total_losses if total_losses > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        pnl_per_trade = net_pnl / total_trades if total_trades > 0 else 0

        stats_by_hour[hour] = {
            'trades': total_trades,
            'wins': total_wins,
            'losses': total_losses,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'net_pnl': net_pnl,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'pnl_per_trade': pnl_per_trade
        }

    return stats_by_hour

def generate_report(stats_by_hour, all_trades):
    """Gera relatório completo de análise"""
    lines = []

    # Header
    lines.append("=" * 120)
    lines.append("🔍 ANÁLISE COMPLETA: VALIDAÇÃO DOS HORÁRIOS BLOQUEADOS")
    lines.append("=" * 120)
    lines.append(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total de trades analisados: {len(all_trades)}")
    lines.append(f"Horários atualmente bloqueados: {BLOCKED_HOURS}")
    lines.append("")

    # ============= TABELA COMPLETA POR HORÁRIO =============
    lines.append("=" * 120)
    lines.append("📊 PERFORMANCE DETALHADA POR HORÁRIO (24h)")
    lines.append("=" * 120)
    lines.append(f"{'HORA':<6} {'TRADES':<8} {'WINS':<6} {'LOSS':<6} {'WIN%':<7} {'NET PnL':<12} {'$/TRADE':<10} {'PF':<6} {'STATUS':<15}")
    lines.append("-" * 120)

    for hour in range(24):
        stats = stats_by_hour[hour]

        if stats['trades'] == 0:
            status = "❌ SEM DADOS"
            line = f"{hour:02d}:00  {'—':<8} {'—':<6} {'—':<6} {'—':<7} {'—':<12} {'—':<10} {'—':<6} {status:<15}"
        else:
            # Determinar status
            is_blocked = hour in BLOCKED_HOURS

            if is_blocked:
                if stats['net_pnl'] > 0 and stats['win_rate'] >= 50:
                    status = "⚠️  BLOQUEADO +PnL"  # Bloqueado mas é lucrativo!
                else:
                    status = "🔴 BLOQUEADO OK"  # Bloqueado corretamente
            else:
                if stats['net_pnl'] > 0:
                    status = "✅ PERMITIDO +"
                else:
                    status = "⚠️  PERMITIDO -"  # Permitido mas perde

            line = (f"{hour:02d}:00  "
                   f"{stats['trades']:<8} "
                   f"{stats['wins']:<6} "
                   f"{stats['losses']:<6} "
                   f"{stats['win_rate']:<6.1f}% "
                   f"${stats['net_pnl']:<11.2f} "
                   f"${stats['pnl_per_trade']:<9.2f} "
                   f"{stats['profit_factor']:<5.2f} "
                   f"{status:<15}")

        lines.append(line)

    lines.append("")

    # ============= ANÁLISE DOS HORÁRIOS BLOQUEADOS =============
    lines.append("=" * 120)
    lines.append("🎯 ANÁLISE DOS HORÁRIOS BLOQUEADOS [8, 9, 10, 11, 17, 21]")
    lines.append("=" * 120)

    blocked_stats = {
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'net_pnl': 0.0,
        'total_profit': 0.0,
        'total_loss': 0.0
    }

    for hour in BLOCKED_HOURS:
        stats = stats_by_hour[hour]
        blocked_stats['trades'] += stats['trades']
        blocked_stats['wins'] += stats['wins']
        blocked_stats['losses'] += stats['losses']
        blocked_stats['net_pnl'] += stats['net_pnl']
        blocked_stats['total_profit'] += stats['total_profit']
        blocked_stats['total_loss'] += stats['total_loss']

    blocked_wr = (blocked_stats['wins'] / blocked_stats['trades'] * 100) if blocked_stats['trades'] > 0 else 0
    blocked_ppt = blocked_stats['net_pnl'] / blocked_stats['trades'] if blocked_stats['trades'] > 0 else 0

    lines.append(f"\n📊 Performance agregada dos horários bloqueados:")
    lines.append(f"   • Total de trades: {blocked_stats['trades']}")
    lines.append(f"   • Wins: {blocked_stats['wins']} | Losses: {blocked_stats['losses']}")
    lines.append(f"   • Win Rate: {blocked_wr:.1f}%")
    lines.append(f"   • Net PnL: ${blocked_stats['net_pnl']:.2f}")
    lines.append(f"   • PnL por trade: ${blocked_ppt:.2f}")

    # Análise individual dos horários bloqueados
    lines.append(f"\n📋 Detalhamento por horário bloqueado:")
    for hour in BLOCKED_HOURS:
        stats = stats_by_hour[hour]
        if stats['trades'] > 0:
            emoji = "⚠️" if stats['net_pnl'] > 0 else "✅"
            lines.append(f"   {emoji} {hour:02d}:00 → {stats['trades']} trades | WR: {stats['win_rate']:.1f}% | Net: ${stats['net_pnl']:.2f} | $/trade: ${stats['pnl_per_trade']:.2f}")

    # ============= ANÁLISE DOS HORÁRIOS PERMITIDOS =============
    lines.append("")
    lines.append("=" * 120)
    lines.append("✅ ANÁLISE DOS HORÁRIOS PERMITIDOS")
    lines.append("=" * 120)

    allowed_stats = {
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'net_pnl': 0.0,
        'total_profit': 0.0,
        'total_loss': 0.0
    }

    for hour in range(24):
        if hour not in BLOCKED_HOURS:
            stats = stats_by_hour[hour]
            allowed_stats['trades'] += stats['trades']
            allowed_stats['wins'] += stats['wins']
            allowed_stats['losses'] += stats['losses']
            allowed_stats['net_pnl'] += stats['net_pnl']
            allowed_stats['total_profit'] += stats['total_profit']
            allowed_stats['total_loss'] += stats['total_loss']

    allowed_wr = (allowed_stats['wins'] / allowed_stats['trades'] * 100) if allowed_stats['trades'] > 0 else 0
    allowed_ppt = allowed_stats['net_pnl'] / allowed_stats['trades'] if allowed_stats['trades'] > 0 else 0

    lines.append(f"\n📊 Performance agregada dos horários permitidos:")
    lines.append(f"   • Total de trades: {allowed_stats['trades']}")
    lines.append(f"   • Wins: {allowed_stats['wins']} | Losses: {allowed_stats['losses']}")
    lines.append(f"   • Win Rate: {allowed_wr:.1f}%")
    lines.append(f"   • Net PnL: ${allowed_stats['net_pnl']:.2f}")
    lines.append(f"   • PnL por trade: ${allowed_ppt:.2f}")

    # ============= COMPARAÇÃO E VALIDAÇÃO =============
    lines.append("")
    lines.append("=" * 120)
    lines.append("🔬 VALIDAÇÃO DO FILTRO DE HORÁRIOS")
    lines.append("=" * 120)

    wr_diff = allowed_wr - blocked_wr
    pnl_diff = allowed_ppt - blocked_ppt

    lines.append(f"\n📈 Diferença entre horários permitidos vs bloqueados:")
    lines.append(f"   • Win Rate: {wr_diff:+.1f}% (Permitidos: {allowed_wr:.1f}% vs Bloqueados: {blocked_wr:.1f}%)")
    lines.append(f"   • PnL por trade: ${pnl_diff:+.2f} (Permitidos: ${allowed_ppt:.2f} vs Bloqueados: ${blocked_ppt:.2f})")

    # Cálculo do impacto do filtro
    if blocked_stats['trades'] > 0:
        pnl_evitado = blocked_stats['net_pnl']
        lines.append(f"\n💰 Se o filtro for aplicado:")
        lines.append(f"   • Trades evitados: {blocked_stats['trades']} ({blocked_stats['trades']/len(all_trades)*100:.1f}% do total)")
        lines.append(f"   • PnL evitado: ${pnl_evitado:.2f}")
        lines.append(f"   • Net PnL final: ${allowed_stats['net_pnl']:.2f} (apenas horários permitidos)")

    # Conclusão
    lines.append("")
    lines.append("=" * 120)
    lines.append("💡 CONCLUSÃO")
    lines.append("=" * 120)

    if wr_diff > 0 and pnl_diff > 0:
        lines.append("\n✅ FILTRO É BENÉFICO!")
        lines.append(f"   O filtro melhora tanto o Win Rate (+{wr_diff:.1f}%) quanto o PnL/trade (+${pnl_diff:.2f})")
        lines.append(f"   Bloqueando esses horários você evita ${blocked_stats['net_pnl']:.2f} de PnL negativo")
    elif blocked_stats['net_pnl'] < 0:
        lines.append("\n✅ FILTRO FAZ SENTIDO!")
        lines.append(f"   Os horários bloqueados têm Net PnL negativo (${blocked_stats['net_pnl']:.2f})")
        lines.append(f"   Evitar esses horários melhora o resultado geral")
    elif blocked_stats['net_pnl'] > 0:
        lines.append("\n⚠️  ATENÇÃO: FILTRO PODE ESTAR PREJUDICANDO!")
        lines.append(f"   Os horários bloqueados têm Net PnL POSITIVO (+${blocked_stats['net_pnl']:.2f})")
        lines.append(f"   Considere remover alguns horários do filtro")
    else:
        lines.append("\n🟡 FILTRO TEM IMPACTO NEUTRO")
        lines.append(f"   Os horários bloqueados têm impacto mínimo")

    # Recomendações específicas
    lines.append("\n📋 RECOMENDAÇÕES POR HORÁRIO:")

    # Horários bloqueados que são lucrativos
    good_blocked = []
    for hour in BLOCKED_HOURS:
        stats = stats_by_hour[hour]
        if stats['trades'] > 0 and stats['net_pnl'] > 0 and stats['win_rate'] >= 50:
            good_blocked.append((hour, stats))

    if good_blocked:
        lines.append("\n⚠️  Horários BLOQUEADOS mas LUCRATIVOS (considere desbloquear):")
        for hour, stats in sorted(good_blocked, key=lambda x: x[1]['net_pnl'], reverse=True):
            lines.append(f"   • {hour:02d}:00 → {stats['trades']} trades | WR: {stats['win_rate']:.1f}% | Net: +${stats['net_pnl']:.2f} | $/trade: +${stats['pnl_per_trade']:.2f}")

    # Horários permitidos que perdem
    bad_allowed = []
    for hour in range(24):
        if hour not in BLOCKED_HOURS:
            stats = stats_by_hour[hour]
            if stats['trades'] >= 5 and stats['net_pnl'] < -10:  # Threshold: 5+ trades e -$10
                bad_allowed.append((hour, stats))

    if bad_allowed:
        lines.append("\n🔴 Horários PERMITIDOS mas com PnL NEGATIVO (considere bloquear):")
        for hour, stats in sorted(bad_allowed, key=lambda x: x[1]['net_pnl']):
            lines.append(f"   • {hour:02d}:00 → {stats['trades']} trades | WR: {stats['win_rate']:.1f}% | Net: ${stats['net_pnl']:.2f} | $/trade: ${stats['pnl_per_trade']:.2f}")

    lines.append("\n" + "=" * 120)

    return "\n".join(lines)

def main():
    """Main function"""
    import sys

    if len(sys.argv) < 2:
        print("❌ Uso: python analise_completa_sem_filtro.py <caminho_do_log>")
        print("\nExemplo:")
        print('   python analise_completa_sem_filtro.py "D:/Projeto/Modelo PPO Trader/logs/trading_session_*.txt"')
        return

    log_path = sys.argv[1]

    print(f"📂 Analisando log: {log_path}")
    print("⏳ Processando trades...")

    # Parse log
    all_trades = parse_log_file(log_path)

    if not all_trades:
        print("❌ Nenhum trade encontrado no log!")
        return

    print(f"✅ {len(all_trades)} trades encontrados")
    print("📊 Gerando análise...")

    # Análise
    stats_by_hour = analyze_by_hour(all_trades)

    # Gerar relatório
    report = generate_report(stats_by_hour, all_trades)

    # Salvar relatório
    output_file = "analise_horarios_completa.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    # Mostrar na tela
    print("\n" + report)
    print(f"\n💾 Relatório salvo em: {output_file}")

if __name__ == "__main__":
    main()
