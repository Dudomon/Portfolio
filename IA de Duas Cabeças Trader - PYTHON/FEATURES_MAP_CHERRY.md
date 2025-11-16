# 🗺️ MAPA COMPLETO DAS 45 FEATURES - CHERRY.PY

**Data:** 2025-10-06
**Versão:** SIMPLIFICADA E CRISTALINA

---

## 📍 FUNÇÃO MASTER

**Localização:** `cherry.py` linha 5203
**Função:** `_get_complete_observation_45_features(step)`

Esta função retorna **TODAS as 45 features de uma vez**, sem concatenações escondidas!

---

## 🎯 ESTRUTURA DAS 45 FEATURES

### **[0-15] MARKET FEATURES (16 features)**
**Fonte:** `self.processed_data[step, :16]`
**Calculadas em:** `_preprocess_data()` linha ~3900

```
[0]  close_1m          - Preço de fechamento
[1]  high_1m           - Preço máximo
[2]  low_1m            - Preço mínimo
[3]  volume_1m         - Volume
[4]  returns_1m        - Retorno percentual
[5]  sma_20_1m         - Média móvel 20
[6]  sma_50_1m         - Média móvel 50
[7]  ema_12_1m         - EMA 12
[8]  rsi_14_1m         - RSI 14
[9]  macd_12_26_9_1m   - MACD
[10] macd_signal_12_26_9_1m - MACD Signal
[11] bb_upper_20_1m    - Bollinger Upper
[12] bb_lower_20_1m    - Bollinger Lower
[13] atr_14_1m         - ATR 14
[14] momentum_5_1m     - Momentum 5
[15] session_momentum  - Momentum da sessão
```

---

### **[16-33] POSITION FEATURES (18 features = 2 positions × 9 features)**
**Fonte:** `_get_positions_observation_robot_style(step)`
**Localização:** cherry.py linha ~4996

**Position 1 (9 features):**
```
[16] active            - 1.0 se ativa, 0.0 se não
[17] entry_price       - Preço de entrada
[18] current_price     - Preço atual
[19] pnl               - PnL atual ($)
[20] duration          - Duração em candles
[21] volume            - Volume da posição
[22] sl_price          - Stop Loss price
[23] tp_price          - Take Profit price
[24] type              - 1.0=LONG, -1.0=SHORT, 0.0=NEUTRAL
```

**Position 2 (9 features):**
```
[25-33] Mesma estrutura da Position 1
```

---

### **[34-40] INTELLIGENT FEATURES (7 features) 🔥**
**Fonte:** Extraídas diretamente do `self.df`
**Calculadas em:** `_calculate_fallback_features()` linhas 4336-4451
**ALINHADAS COM:** `Robot_cherry.py` linhas 905-1010

```
[34] volume_momentum
     - Momentum de volume vs média 20
     - Fórmula: (volume - sma_20) / sma_20
     - Range: [-1, 1] normalizado para [0, 1]

[35] price_position
     - Posição do preço no range 20-bar
     - Fórmula: (close - low_20) / (high_20 - low_20)
     - Range: [0, 1]
     - 0 = no suporte, 1 = na resistência

[36] breakout_strength (TP TARGET ZONES)
     - Qualidade da zona de TP (resistência/suporte próximos)
     - Fórmula: 1 - min(dist_to_resistance, dist_to_support) / (5*ATR)
     - Range: [0, 1]
     - ALTO = alvo TP próximo, BAIXO = alvo distante

[37] trend_consistency
     - Consistência da tendência (% retornos mesma direção)
     - Fórmula: max(positive_returns, negative_returns) / 10
     - Range: [0.5, 1.0]
     - 0.5 = sem tendência, 1.0 = tendência perfeita

[38] support_resistance (SL ZONE QUALITY)
     - Qualidade da zona de SL (distância de S/R)
     - Fórmula: min(dist_to_support, dist_to_resistance) / (3*ATR)
     - Range: [0, 1]
     - ALTO = longe de S/R (seguro), BAIXO = perto (perigoso)

[39] volatility_regime
     - Regime de volatilidade atual vs histórico
     - Fórmula: (vol_20 / vol_50) / 3.0
     - Range: [0, 1]
     - ALTO = volátil, BAIXO = calmo

[40] market_structure (VOLATILITY SPIKE)
     - Detecção de picos de volatilidade recentes
     - Fórmula: (max(atr_14/atr_50 últimos 5) - 0.8) / 1.5
     - Range: [0, 1]
     - ALTO = spike recente, BAIXO = mercado calmo
```

---

### **[41-44] ORDER FLOW FEATURES (4 features)**
**Fonte:** `_generate_order_flow_features(step)`
**Localização:** cherry.py linha 5267

```
[41] spread_ratio
     - Ratio do spread bid/ask (simulado via range/mid_price)
     - Range: [0.001, 0.1]

[42] volume_imbalance
     - Desequilíbrio de volume compra/venda
     - Fórmula: tanh(price_change × volume_intensity) + 0.5
     - Range: [0, 1]
     - 0 = vendedores dominantes, 1 = compradores dominantes

[43] price_impact
     - Estimativa de impacto de preço
     - Fórmula: volume_intensity / (range + 1)
     - Range: [0.1, 0.9]

[44] market_maker_signal
     - Sinal de presença de market maker
     - Fórmula: 1 / (1 + volume × range)
     - Range: [0.1, 0.9]
     - ALTO = MM presente (consolidação), BAIXO = breakout
```

---

## ✅ VALIDAÇÕES

### **Alinhamento Cherry ↔ Robot:**
- ✅ As 7 intelligent features são IDÊNTICAS
- ✅ Calculadas com as MESMAS fórmulas
- ✅ Modelo vê as MESMAS features em treino e live

### **Rewards conectados:**
- ✅ Feature 36 (breakout_strength) → Usado em TP realism bonus
- ✅ Feature 38 (support_resistance) → Usado em SL zone quality
- ✅ Feature 37 (trend_consistency) → Usado em trend following reward

### **Simplicidade:**
- ✅ UMA função master retorna TODAS as 45 features
- ✅ SEM concatenações escondidas
- ✅ SEM confusão entre temporal e single-bar

---

## 🎯 COMO USAR

```python
# Single-bar observation (atual)
obs = self._get_complete_observation_45_features(self.current_step)
# obs.shape = (45,)

# Temporal sequence (20 barras)
temporal_seq = np.zeros((20, 45))
for i in range(20):
    temporal_seq[i] = self._get_complete_observation_45_features(start_step + i)
# temporal_seq.shape = (20, 45)
```

---

**Gerado em:** 2025-10-06
**Última atualização:** Simplificação completa - função master implementada
