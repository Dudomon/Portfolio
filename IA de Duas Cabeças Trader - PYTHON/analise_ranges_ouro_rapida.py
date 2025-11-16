"""
Análise RÁPIDA de Ranges do Ouro - Últimos 3 Meses
Versão otimizada para análise rápida
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def initialize_mt5():
    """Inicializa conexão com MT5"""
    if not mt5.initialize():
        print("❌ Falha ao inicializar MT5")
        return False
    print("✅ MT5 inicializado")
    return True

def download_gold_data(months=3):
    """Baixa dados do ouro"""
    print(f"\n📥 Baixando GOLD últimos {months} meses...")

    end_date = datetime.now()
    start_date = end_date - timedelta(days=months*30)

    rates = mt5.copy_rates_range("GOLD", mt5.TIMEFRAME_M1, start_date, end_date)

    if rates is None or len(rates) == 0:
        print("❌ Erro ao baixar dados")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['hour'] = df['time'].dt.hour

    print(f"✅ {len(df):,} barras baixadas")
    print(f"   Período: {df['time'].min()} até {df['time'].max()}")

    return df

def analyze_ranges(df):
    """Análise rápida de ranges"""
    print("\n" + "="*80)
    print("📊 ANÁLISE DE RANGES - OURO 1MIN")
    print("="*80)

    # Range por barra
    df['bar_range'] = df['high'] - df['low']

    print("\n🎯 RANGE POR BARRA (1min):")
    print(f"   Média: ${df['bar_range'].mean():.2f}")
    print(f"   Mediana: ${df['bar_range'].median():.2f}")
    print(f"   P75: ${df['bar_range'].quantile(0.75):.2f}")
    print(f"   P90: ${df['bar_range'].quantile(0.90):.2f}")
    print(f"   P95: ${df['bar_range'].quantile(0.95):.2f}")

    # Swings em janelas
    for period in [15, 30, 60]:
        df[f'max_{period}m'] = df['high'].rolling(window=period).max()
        df[f'min_{period}m'] = df['low'].rolling(window=period).min()
        df[f'swing_{period}m'] = df[f'max_{period}m'] - df[f'min_{period}m']

        avg = df[f'swing_{period}m'].mean()
        p75 = df[f'swing_{period}m'].quantile(0.75)
        p90 = df[f'swing_{period}m'].quantile(0.90)

        print(f"\n⏱️  SWING {period} MINUTOS:")
        print(f"   Médio: ${avg:.2f}")
        print(f"   P75: ${p75:.2f}")
        print(f"   P90: ${p90:.2f}")

    return df

def analyze_atr(df):
    """Calcula ATR"""
    print("\n" + "="*80)
    print("📊 ATR (Average True Range)")
    print("="*80)

    df['high_low'] = df['high'] - df['low']
    df['high_close'] = abs(df['high'] - df['close'].shift(1))
    df['low_close'] = abs(df['low'] - df['close'].shift(1))
    df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)

    for period in [14, 20]:
        df[f'atr_{period}'] = df['true_range'].rolling(window=period).mean()
        avg = df[f'atr_{period}'].mean()
        current = df[f'atr_{period}'].iloc[-1]

        print(f"\n⏱️  ATR({period}):")
        print(f"   Médio: ${avg:.2f}")
        print(f"   Atual: ${current:.2f}")
        print(f"   Sugestão SL: ${avg * 1.5:.2f} (1.5x ATR)")
        print(f"   Sugestão TP: ${avg * 2.0:.2f} (2.0x ATR)")

    return df

def test_sl_tp_configs(df):
    """Testa configurações SL/TP de forma otimizada"""
    print("\n" + "="*80)
    print("🧪 TESTE RÁPIDO DE CONFIGURAÇÕES SL/TP")
    print("="*80)

    # Configuração atual
    current_sl = 15.0
    current_tp = 18.0

    print(f"\n📍 CONFIGURAÇÃO ATUAL:")
    print(f"   SL: ${current_sl}")
    print(f"   TP: ${current_tp}")
    print(f"   Ratio: {current_tp/current_sl:.2f}:1")

    # Teste simplificado (sample 10% dos dados para velocidade)
    sample_size = int(len(df) * 0.1)
    sample_indices = np.random.choice(df.index[100:-100], size=sample_size, replace=False)

    configs = [
        (15, 20), (15, 25), (15, 30),
        (20, 25), (20, 30), (20, 35),
        (25, 30), (25, 35), (25, 40),
        (30, 35), (30, 40), (30, 45)
    ]

    results = []

    print("\n🔬 Simulando configurações (sample 10% dos dados)...")
    print(f"\n{'SL':<8} {'TP':<8} {'SL Hit %':<12} {'TP Hit %':<12} {'Ratio':<10} {'Status':<15}")
    print("-"*80)

    for sl, tp in configs:
        sl_hits = 0
        tp_hits = 0

        for idx in sample_indices:
            entry = df.loc[idx, 'close']
            future = df.loc[idx:idx+60]  # 60min window

            if len(future) < 10:
                continue

            max_p = future['high'].max()
            min_p = future['low'].min()

            # LONG
            if min_p <= entry - sl:
                sl_hits += 1
            elif max_p >= entry + tp:
                tp_hits += 1

            # SHORT
            if max_p >= entry + sl:
                sl_hits += 1
            elif min_p <= entry - tp:
                tp_hits += 1

        total = sl_hits + tp_hits
        if total > 0:
            sl_rate = (sl_hits / total) * 100
            tp_rate = (tp_hits / total) * 100
            ratio = tp_rate / sl_rate if sl_rate > 0 else 0

            if tp_rate > sl_rate and tp_rate >= 45:
                status = "✅ EXCELENTE"
            elif tp_rate > sl_rate:
                status = "🟢 BOM"
            elif abs(tp_rate - sl_rate) < 5:
                status = "🟡 NEUTRO"
            else:
                status = "🔴 RUIM"

            results.append({
                'sl': sl, 'tp': tp, 'sl_hit': sl_rate,
                'tp_hit': tp_rate, 'ratio': ratio, 'status': status
            })

            print(f"${sl:<7} ${tp:<7} {sl_rate:>6.1f}%     {tp_rate:>6.1f}%     {ratio:>6.2f}    {status}")

    # Melhor configuração
    if results:
        best = max(results, key=lambda x: x['ratio'])

        print("\n" + "="*80)
        print("✨ MELHOR CONFIGURAÇÃO ENCONTRADA")
        print("="*80)
        print(f"   SL: ${best['sl']}")
        print(f"   TP: ${best['tp']}")
        print(f"   SL Hit Rate: {best['sl_hit']:.1f}%")
        print(f"   TP Hit Rate: {best['tp_hit']:.1f}%")
        print(f"   Ratio: {best['ratio']:.2f}")
        print(f"   Status: {best['status']}")

        # Comparar com atual
        current = next((r for r in results if r['sl'] == current_sl and r['tp'] == current_tp), None)
        if current:
            print(f"\n📊 CONFIGURAÇÃO ATUAL:")
            print(f"   SL Hit Rate: {current['sl_hit']:.1f}%")
            print(f"   TP Hit Rate: {current['tp_hit']:.1f}%")
            print(f"   Ratio: {current['ratio']:.2f}")
            print(f"   Status: {current['status']}")

            if best['ratio'] > current['ratio']:
                improvement = ((best['ratio'] - current['ratio']) / current['ratio']) * 100
                print(f"\n🚀 MELHORIA POTENCIAL: +{improvement:.1f}%")

def analyze_by_hour(df):
    """Volatilidade por horário"""
    print("\n" + "="*80)
    print("⏰ VOLATILIDADE POR HORÁRIO")
    print("="*80)

    hourly = df.groupby('hour').agg({
        'bar_range': ['mean', lambda x: x.quantile(0.75)],
        'swing_30m': ['mean', lambda x: x.quantile(0.75)]
    }).round(2)

    print(f"\n{'Hora':<8} {'Range Médio':<15} {'Range P75':<15} {'Swing 30m':<15}")
    print("-"*80)

    for hour in range(24):
        if hour in hourly.index:
            r_avg = hourly.loc[hour, ('bar_range', 'mean')]
            r_p75 = hourly.loc[hour, ('bar_range', '<lambda_0>')]
            s_avg = hourly.loc[hour, ('swing_30m', 'mean')]
            print(f"{hour:02d}:00    ${r_avg:<14.2f} ${r_p75:<14.2f} ${s_avg:<14.2f}")

def main():
    print("="*80)
    print("🏆 ANÁLISE RÁPIDA DE RANGES - OURO")
    print("="*80)

    if not initialize_mt5():
        return

    try:
        df = download_gold_data(months=3)
        if df is None:
            return

        df = analyze_ranges(df)
        df = analyze_atr(df)
        analyze_by_hour(df)
        test_sl_tp_configs(df)

        # Salvar
        print("\n💾 Salvando dados...")
        df.to_csv('D:/Projeto/gold_analysis_3months.csv', index=False)
        print("   ✅ Salvo: gold_analysis_3months.csv")

    finally:
        mt5.shutdown()
        print("\n👋 Concluído!")

if __name__ == "__main__":
    main()
