# 🎯 ANÁLISE DE RANGES ÓTIMOS DE SL/TP - Ouro 1min

**Data:** 2025-10-06 14:55:47
**Dataset:** 22655 barras de 1min
**Período:** 2025-08-18 04:00:00 → 2025-09-11 03:58:00

---

## 📈 1. VOLATILIDADE TÍPICA (ATR)

### ATR 14 períodos:
- **Média:** 0.96 pontos
- **Desvio:** 0.57 pontos
- **Min:** 0.11 pontos
- **Max:** 6.56 pontos

### Percentis de ATR 14:
- **P10:** 0.45 pontos
- **P25:** 0.59 pontos
- **P50:** 0.81 pontos
- **P75:** 1.16 pontos
- **P90:** 1.59 pontos

**📊 Interpretação:**
- Volatilidade típica: ~1 pontos por candle
- 50% das velas tem ATR abaixo de 0.8 pontos
- 25% das velas tem ATR acima de 1.2 pontos

---

## 🛡️ 2. ANÁLISE DE SL - PREVENIR HITS BOBOS

| SL (pts) | Hit Rate | Overshoot | LONG Hits | LONG Safe | SHORT Hits | SHORT Safe |
|----------|----------|-----------|-----------|-----------|------------|------------|
|        5 |    55.9% |     0.95pt |       123 |       104 |        131 |         96 |
|        7 |    45.2% |     1.26pt |        96 |       131 |        109 |        118 |
|       10 |    28.9% |     1.65pt |        52 |       175 |         79 |        148 |
|       12 |    21.4% |     2.44pt |        37 |       190 |         60 |        167 |
|       15 |    16.7% |     2.45pt |        30 |       197 |         46 |        181 |
|       18 |    12.3% |     2.97pt |        18 |       209 |         38 |        189 |
|       20 |     8.8% |     3.67pt |        13 |       214 |         27 |        200 |
|       25 |     5.7% |     3.59pt |        10 |       217 |         16 |        211 |
|       30 |     3.3% |     4.20pt |         4 |       223 |         11 |        216 |
|       35 |     2.6% |     4.33pt |         3 |       224 |          9 |        218 |
|       40 |     2.2% |     2.99pt |         1 |       226 |          9 |        218 |

**✅ SL RECOMENDADO:**
- **Range: 10 pontos**
- Hit rate: 28.9% (bom para prevenir hits bobos)
- Overshoot médio: 1.65pt quando hit

**🎯 SL RANGE SUGERIDO PARA MODELO:**
- **Mínimo:** 5 pontos (mais agressivo)
- **Ótimo:** 10 pontos (balanço hit rate vs distância)
- **Máximo:** 20 pontos (mais conservador)

---

## 🎯 3. ANÁLISE DE TP - VARIAÇÕES REALISTAS

| TP (pts) | Hit Rate | Tempo até hit | LONG Hits | LONG Miss | SHORT Hits | SHORT Miss |
|----------|----------|---------------|-----------|-----------|------------|------------|
|        8 |    54.2% |        173.1min |       131 |        96 |        115 |        112 |
|       10 |    45.2% |        191.7min |       121 |       106 |         84 |        143 |
|       12 |    36.6% |        209.7min |       104 |       123 |         62 |        165 |
|       15 |    30.8% |        231.2min |        89 |       138 |         51 |        176 |
|       18 |    25.3% |        251.7min |        81 |       146 |         34 |        193 |
|       20 |    20.5% |        260.7min |        69 |       158 |         24 |        203 |
|       25 |    14.5% |        273.1min |        49 |       178 |         17 |        210 |
|       30 |     8.6% |        261.0min |        32 |       195 |          7 |        220 |
|       35 |     7.0% |        271.1min |        26 |       201 |          6 |        221 |
|       40 |     5.7% |        278.7min |        22 |       205 |          4 |        223 |
|       50 |     3.1% |        295.1min |        12 |       215 |          2 |        225 |

⚠️ Nenhum TP candidato encontrado com hit rate > 20%

---

## 📊 4. SWINGS POR SESSÃO


### 🕐 ASIAN

- **ATR médio:** 0.92 pontos
- **Swing 5min:**
  - Médio: 2.43 pontos
  - P75: 2.70 pontos
- **Swing 10min:**
  - Médio: 3.65 pontos
  - P75: 3.80 pontos
- **Swing 20min:**
  - Médio: 5.46 pontos
  - P75: 5.50 pontos

### 🕐 LONDON

- **ATR médio:** 1.21 pontos
- **Swing 5min:**
  - Médio: 2.97 pontos
  - P75: 3.50 pontos
- **Swing 10min:**
  - Médio: 4.32 pontos
  - P75: 5.00 pontos
- **Swing 20min:**
  - Médio: 6.28 pontos
  - P75: 7.60 pontos

### 🕐 NEWYORK

- **ATR médio:** 0.97 pontos
- **Swing 5min:**
  - Médio: 2.43 pontos
  - P75: 2.70 pontos
- **Swing 10min:**
  - Médio: 3.60 pontos
  - P75: 3.90 pontos
- **Swing 20min:**
  - Médio: 5.33 pontos
  - P75: 5.80 pontos

---

## 🎯 5. RECOMENDAÇÕES FINAIS

### 📋 RANGES ATUAIS vs RECOMENDADOS

**ATUAL (cherry.py):**
- SL: 10-25 pontos
- TP: 12-25 pontos


### 💡 INSIGHTS PARA TREINAMENTO

1. **Volatilidade:**
   - ATR médio é 1.0pt → SL deve ser >= 1.5x ATR
   - 75% das velas tem ATR < 1.2pt

2. **Hit Rates Realistas:**
   - SL hit rate deve ficar < 30% para ser viável
   - TP hit rate > 25% é realista para daytrading

3. **Timing:**
   - TPs devem ser hit em < 12 candles para ser viável
   - Trades devem ser rápidos (scalping 1min)

---

**Gerado em:** 2025-10-06 14:55:47
