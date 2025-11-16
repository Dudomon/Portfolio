"""
Análise de Ranges do Ouro - Últimos 6 Meses
Compara movimentos reais do mercado com SL/TP configurados nos robôs
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

def initialize_mt5():
    """Inicializa conexão com MT5"""
    if not mt5.initialize():
        print("❌ Falha ao inicializar MT5")
        return False
    print("✅ MT5 inicializado com sucesso")
    return True

def download_gold_data(months=3):
    """Baixa dados do ouro dos últimos N meses"""
    print(f"\n📥 Baixando dados do GOLD (últimos {months} meses)...")

    # Calcular datas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=months*30)

    # Baixar dados 1min
    rates = mt5.copy_rates_range("GOLD", mt5.TIMEFRAME_M1, start_date, end_date)

    if rates is None or len(rates) == 0:
        print("❌ Erro ao baixar dados")
        return None

    # Converter para DataFrame
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')

    print(f"✅ {len(df)} barras baixadas")
    print(f"   Período: {df['time'].min()} até {df['time'].max()}")
    print(f"   Close médio: ${df['close'].mean():.2f}")

    return df

def analyze_price_movements(df):
    """Analisa movimentos de preço reais"""
    print("\n" + "="*100)
    print("📊 ANÁLISE DE MOVIMENTOS DE PREÇO")
    print("="*100)

    # Calcular ranges por barra
    df['bar_range'] = df['high'] - df['low']
    df['bar_range_pct'] = (df['bar_range'] / df['close']) * 100

    # Adicionar hora do dia
    df['hour'] = df['time'].dt.hour

    # Estatísticas de range por barra
    print("\n🎯 RANGE POR BARRA (1min):")
    print(f"   Média: ${df['bar_range'].mean():.2f} ({df['bar_range_pct'].mean():.4f}%)")
    print(f"   Mediana: ${df['bar_range'].median():.2f}")
    print(f"   P25: ${df['bar_range'].quantile(0.25):.2f}")
    print(f"   P75: ${df['bar_range'].quantile(0.75):.2f}")
    print(f"   P90: ${df['bar_range'].quantile(0.90):.2f}")
    print(f"   P95: ${df['bar_range'].quantile(0.95):.2f}")
    print(f"   Máximo: ${df['bar_range'].max():.2f}")

    return df

def analyze_swing_movements(df, lookback_periods=[5, 10, 15, 20, 30, 60]):
    """Analisa movimentos swing (máximos e mínimos em janelas)"""
    print("\n" + "="*100)
    print("📈 ANÁLISE DE SWINGS (Movimentos Direcionais)")
    print("="*100)

    for period in lookback_periods:
        # Calcular máximo/mínimo nos últimos N minutos
        df[f'max_{period}m'] = df['high'].rolling(window=period).max()
        df[f'min_{period}m'] = df['low'].rolling(window=period).min()

        # Range máximo nesse período
        df[f'swing_range_{period}m'] = df[f'max_{period}m'] - df[f'min_{period}m']

        # Movimento desde o open
        df[f'move_from_open_{period}m'] = df['close'] - df['open'].shift(period)

        avg_swing = df[f'swing_range_{period}m'].mean()
        avg_move = df[f'move_from_open_{period}m'].abs().mean()

        p75_swing = df[f'swing_range_{period}m'].quantile(0.75)
        p90_swing = df[f'swing_range_{period}m'].quantile(0.90)

        print(f"\n⏱️  JANELA DE {period} MINUTOS:")
        print(f"   Swing Médio: ${avg_swing:.2f}")
        print(f"   Swing P75: ${p75_swing:.2f}")
        print(f"   Swing P90: ${p90_swing:.2f}")
        print(f"   Movimento Direcional Médio: ${avg_move:.2f}")

    return df

def analyze_by_hour(df):
    """Analisa volatilidade por hora do dia"""
    print("\n" + "="*100)
    print("⏰ VOLATILIDADE POR HORÁRIO")
    print("="*100)

    hourly_stats = df.groupby('hour').agg({
        'bar_range': ['mean', 'median', 'std'],
        'swing_range_15m': ['mean', lambda x: x.quantile(0.75)],
        'swing_range_30m': ['mean', lambda x: x.quantile(0.75)],
        'close': 'count'
    }).round(2)

    print("\n📊 Range Médio e Swing por Hora:")
    print(f"{'Hora':<6} {'Barras':<8} {'Range 1m':<12} {'Swing 15m':<12} {'Swing 30m':<12}")
    print("-"*100)

    for hour in range(24):
        if hour in hourly_stats.index:
            count = int(hourly_stats.loc[hour, ('close', 'count')])
            range_1m = hourly_stats.loc[hour, ('bar_range', 'mean')]
            swing_15m = hourly_stats.loc[hour, ('swing_range_15m', 'mean')]
            swing_30m = hourly_stats.loc[hour, ('swing_range_30m', 'mean')]

            print(f"{hour:02d}:00  {count:<8} ${range_1m:<11.2f} ${swing_15m:<11.2f} ${swing_30m:<11.2f}")

    return hourly_stats

def analyze_sl_tp_hit_probability(df):
    """Simula probabilidade de SL/TP serem atingidos com diferentes ranges"""
    print("\n" + "="*100)
    print("🎯 SIMULAÇÃO: PROBABILIDADE DE HIT EM DIFERENTES RANGES")
    print("="*100)

    # Ranges atuais dos robôs (baseado nos logs)
    current_sl_range = 15.0  # ~$15 de SL típico
    current_tp_range = 18.0  # ~$18 de TP típico

    print(f"\n📍 CONFIGURAÇÃO ATUAL DOS ROBÔS:")
    print(f"   SL Range: ~${current_sl_range}")
    print(f"   TP Range: ~${current_tp_range}")
    print(f"   Ratio SL/TP: {current_tp_range/current_sl_range:.2f}:1")

    # Testar diferentes ranges
    test_ranges = [10, 15, 20, 25, 30, 40, 50, 60]

    print("\n" + "="*100)
    print("🧪 TESTE DE RANGES (Simulação Forward)")
    print("="*100)
    print(f"\n{'Range ($)':<12} {'SL Hit %':<12} {'TP Hit %':<12} {'Ratio TP/SL':<15} {'Status':<20}")
    print("-"*100)

    results = []

    for sl_range in test_ranges:
        for tp_range in test_ranges:
            if tp_range <= sl_range:
                continue  # TP deve ser >= SL

            # Simular hit rates
            sl_hits = 0
            tp_hits = 0
            total_tests = 0

            for i in range(100, len(df)-100):  # Evitar bordas
                entry_price = df.iloc[i]['close']

                # Simular próximos 100 minutos
                future_data = df.iloc[i:i+100]

                max_price = future_data['high'].max()
                min_price = future_data['low'].min()

                # Verificar hits
                sl_long = entry_price - sl_range
                tp_long = entry_price + tp_range
                sl_short = entry_price + sl_range
                tp_short = entry_price - tp_range

                # LONG
                if min_price <= sl_long:
                    sl_hits += 1
                elif max_price >= tp_long:
                    tp_hits += 1

                # SHORT
                if max_price >= sl_short:
                    sl_hits += 1
                elif min_price <= tp_short:
                    tp_hits += 1

                total_tests += 2  # LONG + SHORT

            if total_tests > 0:
                sl_hit_rate = (sl_hits / total_tests) * 100
                tp_hit_rate = (tp_hits / total_tests) * 100
                ratio = tp_hit_rate / sl_hit_rate if sl_hit_rate > 0 else 0

                # Determinar status
                if tp_hit_rate > sl_hit_rate and tp_hit_rate >= 40:
                    status = "✅ EXCELENTE"
                elif tp_hit_rate > sl_hit_rate:
                    status = "🟢 BOM"
                elif tp_hit_rate == sl_hit_rate:
                    status = "🟡 NEUTRO"
                else:
                    status = "🔴 RUIM"

                results.append({
                    'sl': sl_range,
                    'tp': tp_range,
                    'sl_hit': sl_hit_rate,
                    'tp_hit': tp_hit_rate,
                    'ratio': ratio,
                    'status': status
                })

    # Ordenar por ratio (maior primeiro)
    results_sorted = sorted(results, key=lambda x: x['ratio'], reverse=True)[:15]

    for r in results_sorted:
        print(f"SL=${r['sl']:<6.0f} TP=${r['tp']:<6.0f}  "
              f"{r['sl_hit']:>6.1f}%     {r['tp_hit']:>6.1f}%     "
              f"{r['ratio']:>6.2f}        {r['status']:<20}")

    print("\n" + "="*100)
    print("💡 COMPARAÇÃO COM CONFIGURAÇÃO ATUAL")
    print("="*100)

    # Encontrar resultado da configuração atual
    current_result = next((r for r in results if r['sl'] == current_sl_range and r['tp'] == current_tp_range), None)

    if current_result:
        print(f"\n🎯 CONFIGURAÇÃO ATUAL (SL=${current_sl_range}, TP=${current_tp_range}):")
        print(f"   SL Hit Rate: {current_result['sl_hit']:.1f}%")
        print(f"   TP Hit Rate: {current_result['tp_hit']:.1f}%")
        print(f"   Ratio: {current_result['ratio']:.2f}")
        print(f"   Status: {current_result['status']}")

    # Melhor configuração
    best = results_sorted[0]
    print(f"\n✨ MELHOR CONFIGURAÇÃO ENCONTRADA:")
    print(f"   SL: ${best['sl']}")
    print(f"   TP: ${best['tp']}")
    print(f"   SL Hit Rate: {best['sl_hit']:.1f}%")
    print(f"   TP Hit Rate: {best['tp_hit']:.1f}%")
    print(f"   Ratio: {best['ratio']:.2f}")
    print(f"   Status: {best['status']}")

    if current_result and best['ratio'] > current_result['ratio']:
        improvement = ((best['ratio'] - current_result['ratio']) / current_result['ratio']) * 100
        print(f"\n🚀 MELHORIA POTENCIAL: {improvement:.1f}% no ratio TP/SL")

    return results_sorted

def analyze_atr(df, periods=[14, 20, 50]):
    """Calcula ATR (Average True Range) para diferentes períodos"""
    print("\n" + "="*100)
    print("📊 ANÁLISE ATR (Average True Range)")
    print("="*100)

    for period in periods:
        # Calcular True Range
        df['high_low'] = df['high'] - df['low']
        df['high_close_prev'] = abs(df['high'] - df['close'].shift(1))
        df['low_close_prev'] = abs(df['low'] - df['close'].shift(1))

        df['true_range'] = df[['high_low', 'high_close_prev', 'low_close_prev']].max(axis=1)
        df[f'atr_{period}'] = df['true_range'].rolling(window=period).mean()

        avg_atr = df[f'atr_{period}'].mean()
        current_atr = df[f'atr_{period}'].iloc[-1]

        print(f"\n⏱️  ATR({period}):")
        print(f"   Médio: ${avg_atr:.2f}")
        print(f"   Atual: ${current_atr:.2f}")
        print(f"   Sugestão SL: ${avg_atr * 1.5:.2f} (1.5x ATR)")
        print(f"   Sugestão TP: ${avg_atr * 2.0:.2f} (2.0x ATR)")

    return df

def main():
    print("="*100)
    print("🏆 ANÁLISE COMPLETA DE RANGES - OURO 1MIN")
    print("="*100)

    # Inicializar MT5
    if not initialize_mt5():
        return

    try:
        # Baixar dados (3 meses para análise mais rápida)
        df = download_gold_data(months=3)

        if df is None:
            return

        # Análises
        df = analyze_price_movements(df)
        df = analyze_swing_movements(df)
        hourly_stats = analyze_by_hour(df)
        df = analyze_atr(df)
        results = analyze_sl_tp_hit_probability(df)

        print("\n" + "="*100)
        print("✅ ANÁLISE CONCLUÍDA")
        print("="*100)

        # Salvar dados para análise posterior
        print("\n💾 Salvando dados...")
        df.to_csv('D:/Projeto/gold_analysis_6months.csv', index=False)
        print("   Arquivo salvo: D:/Projeto/gold_analysis_6months.csv")

    finally:
        mt5.shutdown()
        print("\n👋 MT5 desconectado")

if __name__ == "__main__":
    main()
