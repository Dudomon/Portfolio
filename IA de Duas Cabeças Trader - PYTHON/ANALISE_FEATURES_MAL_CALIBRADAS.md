# 🔍 ANÁLISE: Features Mal Calibradas - Modelo Comprando em Queda

## Contexto
Modelo Twelveth 1.55M está comprando LONGs durante queda forte (4365 → 4130 = -235pts = -5.7%)

## Problemas Identificados

### ❌ PROBLEMA 1: Trend Strength Mal Calculado
**Localização:** `cherry.py:5850`

```python
# ❌ CÓDIGO ATUAL (ERRADO)
trend_strength = np.mean(price - sma_20) / np.std(price - sma_20)
direction = 1.0 if trend_strength > 0.5 else (-1.0 if trend_strength < -0.5 else 0.1)
```

**Por que está errado:**
- Threshold `-0.5` é MUITO ALTO para detectar quedas
- Durante queda de 5.7%, `trend_strength` pode estar em `-0.3` ou `-0.4`
- Modelo classifica como `direction = 0.1` (neutro) ao invés de `-1.0` (bearish)

**✅ CORREÇÃO:**
```python
# Calcular slope da SMA para detectar tendência
sma_diff = np.diff(sma_20[-20:])  # Últimas 20 barras
trend_slope = np.mean(sma_diff) / np.mean(sma_20[-20:])  # Normalizado

# Distance from SMA
price_vs_sma = np.mean(price - sma_20) / np.mean(sma_20)

# Combined trend strength
trend_strength = trend_slope + price_vs_sma

# THRESHOLDS AJUSTADOS (mais sensíveis)
direction = 1.0 if trend_strength > 0.002 else (-1.0 if trend_strength < -0.002 else 0.0)
```

---

### ❌ PROBLEMA 2: RSI Oversold = Compra Automática
**Localização:** `cherry.py:5948-5956`

```python
# ❌ CÓDIGO ATUAL (PERIGOSO)
if rsi < 30:
    confluence_score += 0.5
    direction_sum += 1.0  # LONG signal
```

**Por que está errado:**
- RSI < 30 durante queda forte NÃO é sinal de compra
- É sinal de "mercado despencando, fique fora"
- Modelo interpreta como "buy the dip opportunity"

**✅ CORREÇÃO:**
```python
# RSI deve ser contextualizado com TREND
if rsi < 30:
    if direction > 0:  # Se trend é bullish
        confluence_score += 0.5  # Oversold = buy opportunity
        direction_sum += 1.0
    else:  # Se trend é bearish
        confluence_score += 0.2  # Oversold em queda = stay out
        direction_sum += 0.0  # Não dá sinal de compra!
elif rsi > 70:
    if direction < 0:  # Se trend é bearish
        confluence_score += 0.5  # Overbought = short opportunity
        direction_sum -= 1.0
    else:  # Se trend é bullish
        confluence_score += 0.2
        direction_sum += 0.0
```

---

### ❌ PROBLEMA 3: Market Regime Não Detecta Crash
**Localização:** `cherry.py:5853-5858`

```python
# ❌ CÓDIGO ATUAL
if abs(trend_strength) > 1.0:
    regime = 'trending'
elif abs(trend_strength) < 0.3:
    regime = 'ranging'
else:
    regime = 'volatile'
```

**Por que está errado:**
- Durante queda forte, `trend_strength = -0.4` → classifica como `'volatile'`
- Deveria classificar como `'trending_down'` ou `'crash'`
- Modelo não diferencia entre "volátil neutro" e "queda forte"

**✅ CORREÇÃO:**
```python
# Separar regime em tipo E direção
abs_strength = abs(trend_strength)

if abs_strength > 0.005:  # Trending
    if trend_strength > 0:
        regime = 'trending_up'
    else:
        regime = 'trending_down'  # CRÍTICO para evitar compras
elif abs_strength < 0.001:
    regime = 'ranging'
else:
    regime = 'volatile'

# ADICIONAR: Detector de crash (queda > 3% em 50 barras)
price_change_pct = (price[-1] - price[0]) / price[0]
if price_change_pct < -0.03:  # Queda > 3%
    regime = 'crash'  # Modelo NUNCA deve comprar em crash
```

---

### ❌ PROBLEMA 4: Features Chegam "Diluídas" ao Modelo
**Localização:** `cherry.py:6113-6143`

```python
# Conversão de dict para array DILUI informação crítica
market_regime = np.array([
    market_regime.get('strength', 0.3),  # Valor default neutro
    market_regime.get('direction', 0.0),  # Default neutro
    1.0 if regime == 'trending' else 0.2  # Perde informação de direção!
])
```

**Por que está errado:**
- `regime == 'trending'` não diferencia UP vs DOWN
- Modelo recebe `[0.4, -0.3, 1.0]` tanto para trending_up quanto trending_down
- Feature crítica "direção da tendência" é perdida

**✅ CORREÇÃO:**
```python
# Preservar TODA a informação
regime_encoding = {
    'trending_up': 1.0,
    'trending_down': -1.0,  # CRÍTICO!
    'crash': -2.0,  # SUPER CRÍTICO! Nunca compre
    'ranging': 0.0,
    'volatile': 0.5
}

market_regime = np.array([
    market_regime.get('strength', 0.3),
    market_regime.get('direction', 0.0),
    regime_encoding.get(regime, 0.0)  # Codifica direção!
], dtype=np.float32)
```

---

## 🎯 RESUMO DA CAUSA RAIZ

O modelo está comprando em queda porque:

1. **Trend detector não é sensível** → Queda de 5.7% é classificada como "neutro"
2. **RSI oversold dispara compra** → Sem considerar contexto de trend
3. **Regime ignora direção** → 'trending' = pode ser up ou down
4. **Features perdem informação** → Codificação dilui sinal crítico

## 📋 AÇÕES RECOMENDADAS

### Imediatas (Sem retreino):
1. ✅ **Desligar Robot_cherry** até aplicar correções
2. ✅ **Adicionar regra hard-coded:** SE `price_change_50bars < -2%` ENTÃO `block_long_entries = True`

### Médio Prazo (Com retreino):
1. ✅ Aplicar todas as 4 correções acima
2. ✅ Retreinar modelo com features corrigidas
3. ✅ Testar em backtest com período de queda forte (out/2024)

## 🚨 URGÊNCIA

**CRÍTICO:** Modelo está perdendo dinheiro comprando em quedas. Aplicar pelo menos a regra hard-coded IMEDIATAMENTE.

---

Gerado por Claude Code em 2025-10-21
