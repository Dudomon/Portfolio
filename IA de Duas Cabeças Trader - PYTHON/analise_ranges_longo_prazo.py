"""
Análise de Ranges para Operações de Longo Prazo (2-4 horas)
Considerando que trades podem durar várias horas
"""

import pandas as pd
import numpy as np

def analyze_long_term_swings():
    """Analisa swings em janelas de 2-4 horas"""
    print("="*100)
    print("📊 ANÁLISE DE RANGES PARA OPERAÇÕES DE LONGO PRAZO (2-4 HORAS)")
    print("="*100)

    # Carregar dados já baixados
    print("\n📂 Carregando dados...")
    df = pd.read_csv('D:/Projeto/gold_analysis_3months.csv')
    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour

    print(f"✅ {len(df):,} barras carregadas")

    # Calcular swings em janelas maiores
    windows = [60, 120, 180, 240]  # 1h, 2h, 3h, 4h

    print("\n" + "="*100)
    print("📈 SWINGS EM JANELAS LONGAS (Operações que duram horas)")
    print("="*100)

    for window in windows:
        df[f'max_{window}m'] = df['high'].rolling(window=window).max()
        df[f'min_{window}m'] = df['low'].rolling(window=window).min()
        df[f'swing_{window}m'] = df[f'max_{window}m'] - df[f'min_{window}m']

        avg = df[f'swing_{window}m'].mean()
        median = df[f'swing_{window}m'].median()
        p25 = df[f'swing_{window}m'].quantile(0.25)
        p50 = df[f'swing_{window}m'].quantile(0.50)
        p75 = df[f'swing_{window}m'].quantile(0.75)
        p90 = df[f'swing_{window}m'].quantile(0.90)
        p95 = df[f'swing_{window}m'].quantile(0.95)

        hours = window / 60
        print(f"\n⏱️  JANELA DE {hours:.1f} HORAS ({window} min):")
        print(f"   Swing Médio: ${avg:.2f}")
        print(f"   Mediana: ${median:.2f}")
        print(f"   P25: ${p25:.2f} (25% dos trades movem até isso)")
        print(f"   P50: ${p50:.2f} (50% dos trades movem até isso)")
        print(f"   P75: ${p75:.2f} (75% dos trades movem até isso)")
        print(f"   P90: ${p90:.2f} (90% dos trades movem até isso)")
        print(f"   P95: ${p95:.2f} (95% dos trades movem até isso)")

    return df

def simulate_sl_tp_long_term(df):
    """Simula SL/TP para operações que duram 2-4 horas"""
    print("\n" + "="*100)
    print("🧪 SIMULAÇÃO: SL/TP PARA OPERAÇÕES DE LONGO PRAZO")
    print("="*100)

    # Configuração atual dos robôs
    current_sl = 15.0
    current_tp = 18.0

    print(f"\n📍 CONFIGURAÇÃO ATUAL DOS ROBÔS:")
    print(f"   SL: ${current_sl}")
    print(f"   TP: ${current_tp}")
    print(f"   Ratio: {current_tp/current_sl:.2f}:1")

    # Testar diferentes configs com janela de 2-4 horas
    configs = [
        # Configs atuais e próximas
        (10, 15), (10, 20), (12, 18), (15, 18), (15, 20), (15, 25),
        # Configs mais largas
        (18, 22), (18, 25), (20, 25), (20, 30),
        (25, 30), (25, 35), (30, 35), (30, 40)
    ]

    # Sample 15% dos dados para teste (mais representativo)
    sample_size = int(len(df) * 0.15)
    sample_indices = np.random.choice(df.index[100:-240], size=sample_size, replace=False)

    print("\n🔬 Simulando (janela de até 4 horas, 15% sample)...")
    print("\nIsso vai levar ~2-3 minutos...")

    results = []

    for sl, tp in configs:
        sl_hits = 0
        tp_hits = 0
        neither_hits = 0

        for idx in sample_indices:
            entry = df.loc[idx, 'close']

            # Janela de 4 horas (240 min)
            future = df.loc[idx:idx+240]

            if len(future) < 30:
                continue

            max_p = future['high'].max()
            min_p = future['low'].min()

            # LONG test
            sl_long = entry - sl
            tp_long = entry + tp

            if min_p <= sl_long:
                sl_hits += 1
            elif max_p >= tp_long:
                tp_hits += 1
            else:
                neither_hits += 1

            # SHORT test
            sl_short = entry + sl
            tp_short = entry - tp

            if max_p >= sl_short:
                sl_hits += 1
            elif min_p <= tp_short:
                tp_hits += 1
            else:
                neither_hits += 1

        total = sl_hits + tp_hits
        if total > 0:
            sl_rate = (sl_hits / total) * 100
            tp_rate = (tp_hits / total) * 100
            ratio = tp_rate / sl_rate if sl_rate > 0 else 0

            # Status baseado em ratio e win rate absoluto
            if tp_rate > sl_rate and tp_rate >= 50:
                status = "✅ EXCELENTE"
            elif tp_rate > sl_rate and tp_rate >= 45:
                status = "🟢 BOM"
            elif abs(tp_rate - sl_rate) < 5:
                status = "🟡 NEUTRO"
            else:
                status = "🔴 RUIM"

            results.append({
                'sl': sl, 'tp': tp,
                'sl_hit': sl_rate, 'tp_hit': tp_rate,
                'ratio': ratio, 'status': status,
                'neither': neither_hits
            })

    # Ordenar por ratio
    results_sorted = sorted(results, key=lambda x: x['ratio'], reverse=True)

    print("\n" + "="*100)
    print("📊 RESULTADOS (Ordenados por Ratio TP/SL)")
    print("="*100)
    print(f"\n{'SL':<8} {'TP':<8} {'SL Hit %':<12} {'TP Hit %':<12} {'Ratio':<10} {'Status':<20}")
    print("-"*100)

    for r in results_sorted:
        print(f"${r['sl']:<7} ${r['tp']:<7} {r['sl_hit']:>6.1f}%     {r['tp_hit']:>6.1f}%     {r['ratio']:>6.2f}    {r['status']:<20}")

    # Análise da config atual
    print("\n" + "="*100)
    print("💡 ANÁLISE DA CONFIGURAÇÃO ATUAL vs MELHOR")
    print("="*100)

    current = next((r for r in results if r['sl'] == current_sl and r['tp'] == current_tp), None)
    best = results_sorted[0]

    if current:
        print(f"\n🎯 CONFIGURAÇÃO ATUAL (SL=${current_sl}, TP=${current_tp}):")
        print(f"   SL Hit Rate: {current['sl_hit']:.1f}%")
        print(f"   TP Hit Rate: {current['tp_hit']:.1f}%")
        print(f"   Ratio TP/SL: {current['ratio']:.2f}")
        print(f"   Status: {current['status']}")

    print(f"\n✨ MELHOR CONFIGURAÇÃO ENCONTRADA (SL=${best['sl']}, TP=${best['tp']}):")
    print(f"   SL Hit Rate: {best['sl_hit']:.1f}%")
    print(f"   TP Hit Rate: {best['tp_hit']:.1f}%")
    print(f"   Ratio TP/SL: {best['ratio']:.2f}")
    print(f"   Status: {best['status']}")

    if current and best['ratio'] > current['ratio']:
        improvement = ((best['ratio'] - current['ratio']) / current['ratio']) * 100
        print(f"\n🚀 MELHORIA POTENCIAL: +{improvement:.1f}% no ratio TP/SL")

    # Top 5 configs
    print("\n" + "="*100)
    print("🏆 TOP 5 MELHORES CONFIGURAÇÕES")
    print("="*100)

    for i, r in enumerate(results_sorted[:5], 1):
        print(f"\n{i}. SL=${r['sl']}, TP=${r['tp']}")
        print(f"   TP Hit: {r['tp_hit']:.1f}% | SL Hit: {r['sl_hit']:.1f}% | Ratio: {r['ratio']:.2f} | {r['status']}")

    return results_sorted

def analyze_duration_vs_outcome(df):
    """Analisa relação entre duração do trade e resultado"""
    print("\n" + "="*100)
    print("⏱️ ANÁLISE: DURAÇÃO DO TRADE vs PROBABILIDADE DE SUCESSO")
    print("="*100)

    current_sl = 15.0
    current_tp = 18.0

    # Sample para teste
    sample_size = int(len(df) * 0.05)
    sample_indices = np.random.choice(df.index[100:-240], size=sample_size, replace=False)

    durations = [30, 60, 120, 180, 240]  # 30min até 4h

    print(f"\n📊 Config: SL=${current_sl}, TP=${current_tp}")
    print(f"\n{'Duração':<15} {'TP Hit %':<12} {'SL Hit %':<12} {'Ratio':<10}")
    print("-"*100)

    for duration in durations:
        tp_hits = 0
        sl_hits = 0

        for idx in sample_indices:
            entry = df.loc[idx, 'close']
            future = df.loc[idx:idx+duration]

            if len(future) < 10:
                continue

            max_p = future['high'].max()
            min_p = future['low'].min()

            # LONG
            if min_p <= entry - current_sl:
                sl_hits += 1
            elif max_p >= entry + current_tp:
                tp_hits += 1

            # SHORT
            if max_p >= entry + current_sl:
                sl_hits += 1
            elif min_p <= entry - current_tp:
                tp_hits += 1

        total = tp_hits + sl_hits
        if total > 0:
            tp_rate = (tp_hits / total) * 100
            sl_rate = (sl_hits / total) * 100
            ratio = tp_rate / sl_rate if sl_rate > 0 else 0

            hours = duration / 60
            print(f"{hours:.1f}h ({duration}min)   {tp_rate:>6.1f}%     {sl_rate:>6.1f}%     {ratio:>6.2f}")

    print("\n💡 INSIGHT: Se TP/SL melhora com duração maior, significa que ranges estão OK mas")
    print("   o problema pode estar no timing de entrada ou na gestão do trade.")

def main():
    print("="*100)
    print("🏆 ANÁLISE DE RANGES PARA OPERAÇÕES DE LONGO PRAZO (2-4 HORAS)")
    print("="*100)

    # Analisar swings longos
    df = analyze_long_term_swings()

    # Simular SL/TP
    results = simulate_sl_tp_long_term(df)

    # Analisar duração
    analyze_duration_vs_outcome(df)

    print("\n" + "="*100)
    print("✅ ANÁLISE COMPLETA CONCLUÍDA")
    print("="*100)

if __name__ == "__main__":
    main()
