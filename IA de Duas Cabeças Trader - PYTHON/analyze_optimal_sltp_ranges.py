#!/usr/bin/env python3
"""
🎯 ANÁLISE DE RANGES ÓTIMOS DE SL/TP - Baseado em Comportamento Real do Ouro

OBJETIVO:
- Analisar dataset real de 1min do ouro
- Determinar ranges ótimos de SL (prevenir hits bobos)
- Determinar ranges ótimos de TP (variações realistas por sessão)
- Fornecer dados para ajustar heurísticas do cherry.py

MÉTRICAS ANALISADAS:
1. ATR (Average True Range) - Volatilidade típica
2. Drawdowns intra-candle - Quanto o preço puxa antes de reverter
3. Swings típicos - Variações médias em N candles
4. Hit rate por range - Probabilidade de SL/TP serem atingidos
5. RR ratio ótimo - Baseado em sucesso real
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ====================================================================
# 🔧 CONFIGURAÇÕES
# ====================================================================

DATASET_PATH = "D:/Projeto/data/GC=F_3YEARS_1MIN_20250911_161600.csv"
OUTPUT_FILE = "D:/Projeto/OPTIMAL_SLTP_RANGES_ANALYSIS.md"

# Ranges a testar (em pontos)
SL_RANGES_TO_TEST = [5, 7, 10, 12, 15, 18, 20, 25, 30, 35, 40]
TP_RANGES_TO_TEST = [8, 10, 12, 15, 18, 20, 25, 30, 35, 40, 50]

# Análise por sessão (para detectar variações)
SESSIONS = {
    'asian': (0, 8),      # 00:00-08:00 UTC
    'london': (8, 16),    # 08:00-16:00 UTC
    'newyork': (14, 22),  # 14:00-22:00 UTC (overlap com London)
}

# ====================================================================
# 📊 CARREGAR DADOS
# ====================================================================

print("📊 Carregando dataset...")
df = pd.read_csv(DATASET_PATH)
print(f"✅ Dataset carregado: {len(df)} barras")

# Garantir que temos OHLC
required_cols = ['open', 'high', 'low', 'close', 'time']
for col in required_cols:
    if col not in df.columns:
        print(f"❌ Coluna '{col}' não encontrada!")
        exit(1)

# Converter time para datetime
df['time'] = pd.to_datetime(df['time'])
df['hour'] = df['time'].dt.hour

# ====================================================================
# 📈 ANÁLISE 1: ATR E VOLATILIDADE TÍPICA
# ====================================================================

print("\n" + "="*60)
print("📈 ANÁLISE 1: ATR E VOLATILIDADE TÍPICA")
print("="*60)

# Calcular true range
df['tr'] = df['high'] - df['low']
df['atr_14'] = df['tr'].rolling(14).mean()
df['atr_50'] = df['tr'].rolling(50).mean()

atr_14_mean = df['atr_14'].mean()
atr_50_mean = df['atr_50'].mean()
atr_14_std = df['atr_14'].std()

print(f"\n🎯 ATR 14 períodos:")
print(f"   Média: {atr_14_mean:.2f} pontos")
print(f"   Desvio: {atr_14_std:.2f} pontos")
print(f"   Min: {df['atr_14'].min():.2f} pontos")
print(f"   Max: {df['atr_14'].max():.2f} pontos")

print(f"\n🎯 ATR 50 períodos:")
print(f"   Média: {atr_50_mean:.2f} pontos")

# Percentis de volatilidade
percentiles = [10, 25, 50, 75, 90]
print(f"\n📊 Percentis de ATR 14:")
for p in percentiles:
    value = np.percentile(df['atr_14'].dropna(), p)
    print(f"   P{p}: {value:.2f} pontos")

# ====================================================================
# 🛡️ ANÁLISE 2: SL OPTIMAL - PREVENIR HITS BOBOS
# ====================================================================

print("\n" + "="*60)
print("🛡️ ANÁLISE 2: SL OPTIMAL - PREVENIR HITS BOBOS")
print("="*60)

sl_analysis = []

for sl_points in SL_RANGES_TO_TEST:
    print(f"\n🔍 Testando SL = {sl_points} pontos...")

    # Simular LONG trades
    long_hits = 0
    long_safe = 0
    long_hit_distances = []

    # Simular SHORT trades
    short_hits = 0
    short_safe = 0
    short_hit_distances = []

    # Para cada candle, simular se SL seria hit
    # REALISTA: Testar até 4 horas (240 candles) ou fim do dataset
    for i in range(14, len(df), 100):  # Sampling a cada 100 candles para performance
        entry_price = df['close'].iloc[i]

        # LONG: SL abaixo do entry
        sl_long = entry_price - sl_points
        # Verificar nos próximos 240 candles (4 horas) se SL hit
        for j in range(i+1, min(i+241, len(df))):
            if df['low'].iloc[j] <= sl_long:
                long_hits += 1
                # Distância que preço foi além do SL
                overshoot = sl_long - df['low'].iloc[j]
                long_hit_distances.append(overshoot)
                break
        else:
            long_safe += 1

        # SHORT: SL acima do entry
        sl_short = entry_price + sl_points
        # Verificar nos próximos 240 candles (4 horas) se SL hit
        for j in range(i+1, min(i+241, len(df))):
            if df['high'].iloc[j] >= sl_short:
                short_hits += 1
                # Distância que preço foi além do SL
                overshoot = df['high'].iloc[j] - sl_short
                short_hit_distances.append(overshoot)
                break
        else:
            short_safe += 1

    total_trades = long_hits + long_safe + short_hits + short_safe
    total_hits = long_hits + short_hits
    hit_rate = (total_hits / total_trades * 100) if total_trades > 0 else 0

    avg_overshoot = np.mean(long_hit_distances + short_hit_distances) if (long_hit_distances + short_hit_distances) else 0

    sl_analysis.append({
        'sl_points': sl_points,
        'hit_rate': hit_rate,
        'avg_overshoot': avg_overshoot,
        'long_hits': long_hits,
        'long_safe': long_safe,
        'short_hits': short_hits,
        'short_safe': short_safe,
    })

    print(f"   Hit Rate: {hit_rate:.1f}%")
    print(f"   Overshoot médio: {avg_overshoot:.2f} pontos")

# ====================================================================
# 🎯 ANÁLISE 3: TP OPTIMAL - VARIAÇÕES REALISTAS
# ====================================================================

print("\n" + "="*60)
print("🎯 ANÁLISE 3: TP OPTIMAL - VARIAÇÕES REALISTAS")
print("="*60)

tp_analysis = []

for tp_points in TP_RANGES_TO_TEST:
    print(f"\n🔍 Testando TP = {tp_points} pontos...")

    # Simular LONG trades
    long_hits = 0
    long_misses = 0
    long_hit_times = []

    # Simular SHORT trades
    short_hits = 0
    short_misses = 0
    short_hit_times = []

    # Para cada candle, simular se TP seria hit
    # REALISTA: Testar até 8 horas (480 candles) ou fim do dataset
    for i in range(14, len(df), 100):  # Sampling a cada 100 candles para performance
        entry_price = df['close'].iloc[i]

        # LONG: TP acima do entry
        tp_long = entry_price + tp_points
        # Verificar nos próximos 480 candles (8 horas) se TP hit
        hit = False
        for j in range(i+1, min(i+481, len(df))):
            if df['high'].iloc[j] >= tp_long:
                long_hits += 1
                long_hit_times.append(j - i)  # Tempo até hit
                hit = True
                break
        if not hit:
            long_misses += 1

        # SHORT: TP abaixo do entry
        tp_short = entry_price - tp_points
        # Verificar nos próximos 480 candles (8 horas) se TP hit
        hit = False
        for j in range(i+1, min(i+481, len(df))):
            if df['low'].iloc[j] <= tp_short:
                short_hits += 1
                short_hit_times.append(j - i)  # Tempo até hit
                hit = True
                break
        if not hit:
            short_misses += 1

    total_trades = long_hits + long_misses + short_hits + short_misses
    total_hits = long_hits + short_hits
    hit_rate = (total_hits / total_trades * 100) if total_trades > 0 else 0

    avg_time_to_hit = np.mean(long_hit_times + short_hit_times) if (long_hit_times + short_hit_times) else 0

    tp_analysis.append({
        'tp_points': tp_points,
        'hit_rate': hit_rate,
        'avg_time_to_hit': avg_time_to_hit,
        'long_hits': long_hits,
        'long_misses': long_misses,
        'short_hits': short_hits,
        'short_misses': short_misses,
    })

    print(f"   Hit Rate: {hit_rate:.1f}%")
    print(f"   Tempo médio até hit: {avg_time_to_hit:.1f} candles")

# ====================================================================
# 📊 ANÁLISE 4: SWINGS POR SESSÃO
# ====================================================================

print("\n" + "="*60)
print("📊 ANÁLISE 4: SWINGS POR SESSÃO")
print("="*60)

session_analysis = {}

for session_name, (start_hour, end_hour) in SESSIONS.items():
    print(f"\n🕐 Sessão: {session_name.upper()} ({start_hour}:00 - {end_hour}:00 UTC)")

    # Filtrar dados da sessão
    if start_hour < end_hour:
        session_df = df[(df['hour'] >= start_hour) & (df['hour'] < end_hour)]
    else:  # Overlap (ex: NY)
        session_df = df[(df['hour'] >= start_hour) | (df['hour'] < end_hour)]

    if len(session_df) == 0:
        print("   ⚠️ Sem dados para esta sessão")
        continue

    # Calcular swings típicos (high-low em janelas de 5, 10, 20 min)
    swings_5min = []
    swings_10min = []
    swings_20min = []

    for i in range(len(session_df) - 20):
        # 5 min window
        window_5 = session_df.iloc[i:i+5]
        swing_5 = window_5['high'].max() - window_5['low'].min()
        swings_5min.append(swing_5)

        # 10 min window
        window_10 = session_df.iloc[i:i+10]
        swing_10 = window_10['high'].max() - window_10['low'].min()
        swings_10min.append(swing_10)

        # 20 min window
        window_20 = session_df.iloc[i:i+20]
        swing_20 = window_20['high'].max() - window_20['low'].min()
        swings_20min.append(swing_20)

    session_analysis[session_name] = {
        'atr_mean': session_df['atr_14'].mean(),
        'swing_5min_mean': np.mean(swings_5min),
        'swing_5min_p75': np.percentile(swings_5min, 75),
        'swing_10min_mean': np.mean(swings_10min),
        'swing_10min_p75': np.percentile(swings_10min, 75),
        'swing_20min_mean': np.mean(swings_20min),
        'swing_20min_p75': np.percentile(swings_20min, 75),
    }

    print(f"   ATR 14: {session_analysis[session_name]['atr_mean']:.2f} pontos")
    print(f"   Swing 5min médio: {session_analysis[session_name]['swing_5min_mean']:.2f} pontos")
    print(f"   Swing 10min médio: {session_analysis[session_name]['swing_10min_mean']:.2f} pontos")
    print(f"   Swing 20min médio: {session_analysis[session_name]['swing_20min_mean']:.2f} pontos")

# ====================================================================
# 💾 GERAR RELATÓRIO
# ====================================================================

print("\n" + "="*60)
print("💾 GERANDO RELATÓRIO...")
print("="*60)

report = f"""# 🎯 ANÁLISE DE RANGES ÓTIMOS DE SL/TP - Ouro 1min

**Data:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Dataset:** {len(df)} barras de 1min
**Período:** {df['time'].min()} → {df['time'].max()}

---

## 📈 1. VOLATILIDADE TÍPICA (ATR)

### ATR 14 períodos:
- **Média:** {atr_14_mean:.2f} pontos
- **Desvio:** {atr_14_std:.2f} pontos
- **Min:** {df['atr_14'].min():.2f} pontos
- **Max:** {df['atr_14'].max():.2f} pontos

### Percentis de ATR 14:
"""

for p in percentiles:
    value = np.percentile(df['atr_14'].dropna(), p)
    report += f"- **P{p}:** {value:.2f} pontos\n"

report += f"""
**📊 Interpretação:**
- Volatilidade típica: ~{atr_14_mean:.0f} pontos por candle
- 50% das velas tem ATR abaixo de {np.percentile(df['atr_14'].dropna(), 50):.1f} pontos
- 25% das velas tem ATR acima de {np.percentile(df['atr_14'].dropna(), 75):.1f} pontos

---

## 🛡️ 2. ANÁLISE DE SL - PREVENIR HITS BOBOS

"""

# Criar tabela de SL
report += "| SL (pts) | Hit Rate | Overshoot | LONG Hits | LONG Safe | SHORT Hits | SHORT Safe |\n"
report += "|----------|----------|-----------|-----------|-----------|------------|------------|\n"

for result in sl_analysis:
    report += f"| {result['sl_points']:>8} | {result['hit_rate']:>7.1f}% | {result['avg_overshoot']:>8.2f}pt | "
    report += f"{result['long_hits']:>9} | {result['long_safe']:>9} | "
    report += f"{result['short_hits']:>10} | {result['short_safe']:>10} |\n"

# Identificar SL ótimo
sl_df = pd.DataFrame(sl_analysis)
# SL ótimo = hit rate baixo (<30%) mas não muito distante
optimal_sl_candidates = sl_df[(sl_df['hit_rate'] < 30) & (sl_df['sl_points'] <= 25)]
if len(optimal_sl_candidates) > 0:
    optimal_sl = optimal_sl_candidates.iloc[0]
    report += f"""
**✅ SL RECOMENDADO:**
- **Range: {optimal_sl['sl_points']:.0f} pontos**
- Hit rate: {optimal_sl['hit_rate']:.1f}% (bom para prevenir hits bobos)
- Overshoot médio: {optimal_sl['avg_overshoot']:.2f}pt quando hit

**🎯 SL RANGE SUGERIDO PARA MODELO:**
- **Mínimo:** {max(optimal_sl['sl_points'] - 5, 5):.0f} pontos (mais agressivo)
- **Ótimo:** {optimal_sl['sl_points']:.0f} pontos (balanço hit rate vs distância)
- **Máximo:** {optimal_sl['sl_points'] + 10:.0f} pontos (mais conservador)
"""
else:
    report += "\n⚠️ Nenhum SL candidato encontrado com hit rate < 30%\n"

report += """
---

## 🎯 3. ANÁLISE DE TP - VARIAÇÕES REALISTAS

"""

# Criar tabela de TP
report += "| TP (pts) | Hit Rate | Tempo até hit | LONG Hits | LONG Miss | SHORT Hits | SHORT Miss |\n"
report += "|----------|----------|---------------|-----------|-----------|------------|------------|\n"

for result in tp_analysis:
    report += f"| {result['tp_points']:>8} | {result['hit_rate']:>7.1f}% | {result['avg_time_to_hit']:>12.1f}min | "
    report += f"{result['long_hits']:>9} | {result['long_misses']:>9} | "
    report += f"{result['short_hits']:>10} | {result['short_misses']:>10} |\n"

# Identificar TP ótimo
tp_df = pd.DataFrame(tp_analysis)
# TP ótimo = hit rate razoável (>20%) com tempo rápido (<10 candles)
optimal_tp_candidates = tp_df[(tp_df['hit_rate'] > 20) & (tp_df['avg_time_to_hit'] < 12)]
if len(optimal_tp_candidates) > 0:
    optimal_tp = optimal_tp_candidates.iloc[-1]  # Pegar o maior TP viável
    report += f"""
**✅ TP RECOMENDADO:**
- **Range: {optimal_tp['tp_points']:.0f} pontos**
- Hit rate: {optimal_tp['hit_rate']:.1f}% (realista)
- Tempo médio: {optimal_tp['avg_time_to_hit']:.1f} candles (~{optimal_tp['avg_time_to_hit']:.0f} minutos)

**🎯 TP RANGE SUGERIDO PARA MODELO:**
- **Mínimo:** {max(optimal_tp['tp_points'] - 8, 8):.0f} pontos (mais conservador)
- **Ótimo:** {optimal_tp['tp_points']:.0f} pontos (balanço hit rate vs reward)
- **Máximo:** {min(optimal_tp['tp_points'] + 10, 50):.0f} pontos (mais ambicioso)
"""
else:
    report += "\n⚠️ Nenhum TP candidato encontrado com hit rate > 20%\n"

report += """
---

## 📊 4. SWINGS POR SESSÃO

"""

for session_name, data in session_analysis.items():
    report += f"""
### 🕐 {session_name.upper()}

- **ATR médio:** {data['atr_mean']:.2f} pontos
- **Swing 5min:**
  - Médio: {data['swing_5min_mean']:.2f} pontos
  - P75: {data['swing_5min_p75']:.2f} pontos
- **Swing 10min:**
  - Médio: {data['swing_10min_mean']:.2f} pontos
  - P75: {data['swing_10min_p75']:.2f} pontos
- **Swing 20min:**
  - Médio: {data['swing_20min_mean']:.2f} pontos
  - P75: {data['swing_20min_p75']:.2f} pontos
"""

report += """
---

## 🎯 5. RECOMENDAÇÕES FINAIS

### 📋 RANGES ATUAIS vs RECOMENDADOS

**ATUAL (cherry.py):**
- SL: 10-25 pontos
- TP: 12-25 pontos

"""

# Adicionar recomendação baseada em análise
if len(optimal_sl_candidates) > 0 and len(optimal_tp_candidates) > 0:
    rec_sl_min = max(optimal_sl['sl_points'] - 5, 5)
    rec_sl_max = optimal_sl['sl_points'] + 10
    rec_tp_min = max(optimal_tp['tp_points'] - 8, 8)
    rec_tp_max = min(optimal_tp['tp_points'] + 10, 50)

    report += f"""**RECOMENDADO (baseado em dados):**
- **SL: {rec_sl_min:.0f}-{rec_sl_max:.0f} pontos**
- **TP: {rec_tp_min:.0f}-{rec_tp_max:.0f} pontos**

### 🔧 AJUSTES SUGERIDOS PARA HEURÍSTICAS

1. **SL Zone Quality Heuristic:**
   - SL entre {rec_sl_min:.0f}-{optimal_sl['sl_points']:.0f}pt em zona segura: +0.12
   - SL > {optimal_sl['sl_points'] + 5:.0f}pt (muito largo): -0.05
   - SL < {rec_sl_min:.0f}pt (muito apertado): -0.15

2. **TP Target Zones Heuristic:**
   - TP entre {rec_tp_min:.0f}-{optimal_tp['tp_points']:.0f}pt próximo resistência: +0.10
   - TP > {optimal_tp['tp_points'] + 10:.0f}pt (muito ambicioso): -0.08
   - TP < {rec_tp_min:.0f}pt (muito conservador): +0.03

3. **RR Ratio Adjustment:**
   - RR ideal baseado em análise: {(optimal_tp['tp_points'] / optimal_sl['sl_points']):.2f}:1
   - Range aceitável: {(rec_tp_min / rec_sl_max):.2f}:1 a {(rec_tp_max / rec_sl_min):.2f}:1
"""

report += f"""
### 💡 INSIGHTS PARA TREINAMENTO

1. **Volatilidade:**
   - ATR médio é {atr_14_mean:.1f}pt → SL deve ser >= 1.5x ATR
   - 75% das velas tem ATR < {np.percentile(df['atr_14'].dropna(), 75):.1f}pt

2. **Hit Rates Realistas:**
   - SL hit rate deve ficar < 30% para ser viável
   - TP hit rate > 25% é realista para daytrading

3. **Timing:**
   - TPs devem ser hit em < 12 candles para ser viável
   - Trades devem ser rápidos (scalping 1min)

---

**Gerado em:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

# Salvar relatório
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    f.write(report)

print(f"\n✅ Relatório salvo em: {OUTPUT_FILE}")
print("\n" + "="*60)
print("✅ ANÁLISE CONCLUÍDA!")
print("="*60)
