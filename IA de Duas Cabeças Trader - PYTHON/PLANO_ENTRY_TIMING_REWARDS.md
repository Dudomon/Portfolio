# 🎯 PLANO: ENTRY TIMING REWARDS
## Objetivo: Reduzir SL Hit Rate de 61.5% → <45% melhorando timing de entrada

---

## 📊 ANÁLISE DAS FEATURES DISPONÍVEIS

### Intelligent Components Disponíveis (cherry.py:5798-6033):
1. **market_regime** (linha 5837-5903)
   - `regime`: 'trending_up', 'trending_down', 'ranging', 'volatile', 'crash'
   - `strength`: [0.0-2.0] força da tendência
   - `direction`: -1.0 (baixa) / 0.0 (neutro) / 1.0 (alta)

2. **volatility_context** (linha 5905-5953)
   - `level`: 'low', 'normal', 'high'
   - `percentile`: [0.0-1.0] percentil de volatilidade
   - `expanding`: True/False (volatilidade crescente)

3. **momentum_confluence** (linha 5955-6033)
   - `score`: [0.0-1.0] confluência de indicadores (RSI, MACD, MA)
   - `direction`: [-1.0 a 1.0] direção do momentum
   - `strength`: [0.0-1.0] força do momentum

4. **support_resistance** (linha 4534-4560)
   - Distância normalizada para suporte/resistência
   - Range: [0.0-1.0] (quanto menor, mais próximo de S/R)

### Features Base Disponíveis:
- `rsi_14_1m`: RSI 14 períodos
- `macd_12_26_9_1m`: MACD
- `sma_10_1m`, `sma_20_1m`: Médias móveis
- `atr_14_1m`: ATR (volatilidade)
- `hour`: Hora do dia (0-23)

---

## 🎯 PLANO DE IMPLEMENTAÇÃO: 3 COMPONENTES DE REWARD

### **COMPONENTE 1: ENTRY TIMING QUALITY (10% do shaping total)**

**Objetivo**: Recompensar entradas em momentos de alta probabilidade

#### 1.1 Market Context Alignment (40% deste componente)
**Features usadas**: `market_regime`, `momentum_confluence`

**Lógica**:
```python
# LONG entries:
if entry_decision == 1 (LONG):
    # ✅ BÔNUS: Comprar em trending_up com momentum positivo
    if regime == 'trending_up' AND momentum_direction > 0.3:
        bonus = 0.3 * momentum_strength

    # ⚠️ PENALTY: Comprar em trending_down (contra tendência)
    if regime == 'trending_down':
        penalty = -0.5 * regime_strength

    # 🚫 PENALTY SEVERA: Comprar em crash
    if regime == 'crash':
        penalty = -1.0

# SHORT entries:
if entry_decision == 2 (SHORT):
    # ✅ BÔNUS: Vender em trending_down com momentum negativo
    if regime == 'trending_down' AND momentum_direction < -0.3:
        bonus = 0.3 * momentum_strength

    # ⚠️ PENALTY: Vender em trending_up (contra tendência)
    if regime == 'trending_up':
        penalty = -0.5 * regime_strength
```

**Justificativa**:
- Análise mostrou SHORT 47% WR vs LONG 33% WR
- Entradas contra tendência são a principal causa de SL hits
- Crash detection evita compras em quedas acentuadas

#### 1.2 Volatility Timing (30% deste componente)
**Features usadas**: `volatility_context`, `atr_14_1m`

**Lógica**:
```python
if entry_decision in [1, 2]:
    # ✅ BÔNUS: Entrar em volatilidade normal/moderada
    if volatility_level == 'normal':
        bonus = 0.2

    # ⚠️ PENALTY: Entrar em volatilidade extrema (alta ou baixa)
    if volatility_level == 'high' AND volatility_percentile > 0.9:
        penalty = -0.3  # Volatilidade muito alta = stops prematuros

    if volatility_level == 'low' AND volatility_percentile < 0.1:
        penalty = -0.2  # Volatilidade muito baixa = targets demorados

    # 🎯 BÔNUS: Volatilidade expandindo em direção favorável
    if volatility_expanding AND momentum_direction * entry_direction > 0:
        bonus = 0.15
```

**Justificativa**:
- Volatilidade extrema causa SL prematuros (73% hit rate atual)
- Volatilidade muito baixa = operações longas demais (>4h)
- P75 de ATR = $2.64, ideal para SL $30

#### 1.3 Momentum Confluence (30% deste componente)
**Features usadas**: `momentum_confluence`, `rsi_14_1m`, `macd_12_26_9_1m`

**Lógica**:
```python
if entry_decision in [1, 2]:
    # ✅ BÔNUS: Alta confluência de indicadores
    if momentum_score > 0.7:
        bonus = 0.4 * momentum_strength

    # ⚠️ PENALTY: Baixa confluência (sinais mistos)
    if momentum_score < 0.3:
        penalty = -0.3

    # 🎯 BÔNUS ESPECIAL: RSI oversold em uptrend (buy) ou overbought em downtrend (short)
    if entry_decision == 1 AND rsi < 35 AND regime_direction > 0:
        bonus = 0.25  # Compra em pullback

    if entry_decision == 2 AND rsi > 65 AND regime_direction < 0:
        bonus = 0.25  # Short em rally bearish
```

**Justificativa**:
- Código já calcula momentum_confluence contextualizado (linha 5973-5993)
- RSI contextual com trend já implementado
- Confluência > 0.7 indica setup forte

---

### **COMPONENTE 2: ENTRY CONFLUENCE REWARD (5% do shaping total)**

**Objetivo**: Recompensar entradas com múltiplos sinais alinhados

#### 2.1 Multi-Indicator Confirmation (60% deste componente)
**Features usadas**: Todos os intelligent components

**Lógica**:
```python
confirmation_count = 0
max_confirmations = 5

if entry_decision in [1, 2]:
    entry_direction = 1 if entry_decision == 1 else -1

    # Check 1: Market regime favorável
    if regime in ['trending_up', 'trending_down']:
        regime_dir = 1 if 'up' in regime else -1
        if regime_dir == entry_direction:
            confirmation_count += 1

    # Check 2: Momentum alinhado
    if momentum_direction * entry_direction > 0.5:
        confirmation_count += 1

    # Check 3: RSI em zona favorável
    if entry_direction == 1 AND 30 < rsi < 50:  # LONG: oversold mas não extremo
        confirmation_count += 1
    if entry_direction == -1 AND 50 < rsi < 70:  # SHORT: overbought mas não extremo
        confirmation_count += 1

    # Check 4: MACD alinhado
    if macd > macd_signal AND entry_direction == 1:
        confirmation_count += 1
    if macd < macd_signal AND entry_direction == -1:
        confirmation_count += 1

    # Check 5: Volatilidade adequada
    if 0.3 < volatility_percentile < 0.8:
        confirmation_count += 1

    # Calcular reward baseado em confirmações
    confluence_ratio = confirmation_count / max_confirmations

    if confluence_ratio >= 0.8:  # 4+ confirmações
        bonus = 0.5
    elif confluence_ratio >= 0.6:  # 3 confirmações
        bonus = 0.2
    elif confluence_ratio >= 0.4:  # 2 confirmações
        bonus = 0.0  # Neutro
    else:  # ≤1 confirmação
        penalty = -0.4  # Entrada prematura
```

**Justificativa**:
- Análise mostrou que entradas aleatórias = 61.5% SL hit
- 4+ confirmações = setup robusto
- ≤1 confirmação = entrada sem conviction

#### 2.2 Support/Resistance Proximity (40% deste componente)
**Features usadas**: `support_resistance`

**Lógica**:
```python
if entry_decision in [1, 2]:
    # ✅ BÔNUS: Entrar próximo de suporte (LONG) ou resistência (SHORT)
    if support_resistance < 0.15:  # Muito próximo de S/R
        # Verificar se é favorável
        if entry_decision == 1:  # LONG próximo de suporte
            bonus = 0.3
        elif entry_decision == 2:  # SHORT próximo de resistência
            bonus = 0.3

    # ⚠️ PENALTY: Entrar no meio do nada
    if support_resistance > 0.7:  # Longe de qualquer S/R
        penalty = -0.2
```

**Justificativa**:
- Entradas em zonas de liquidez têm SL naturalmente melhor posicionados
- S/R são zonas de reação de preço comprovadas

---

### **COMPONENTE 3: MARKET CONTEXT REWARD (5% do shaping total)**

**Objetivo**: Recompensar entradas em horários e condições favoráveis

#### 3.1 Hour-Based Quality (70% deste componente)
**Features usadas**: `hour` (extraído do timestamp)

**Lógica**:
```python
# Baseado na análise de logs (analise_horarios_robo.py)
EXCELLENT_HOURS = [15, 12, 19, 20, 4]  # >$300 profit
GOOD_HOURS = [13, 14, 18, 22, 23, 0, 1, 2, 3, 5, 7]  # Positivo
BAD_HOURS = [6, 8, 9, 10, 11, 17, 21]  # <40% WR ou negativo

if entry_decision in [1, 2]:
    current_hour = extract_hour_from_timestamp()

    # ✅ BÔNUS: Horários excelentes
    if current_hour in EXCELLENT_HOURS:
        bonus = 0.4

    # 🟢 NEUTRO: Horários bons
    elif current_hour in GOOD_HOURS:
        bonus = 0.0

    # 🔴 PENALTY: Horários ruins
    elif current_hour in BAD_HOURS:
        penalty = -0.6  # PENALTY SEVERA
```

**Justificativa**:
- Análise empírica de 32,865 trades
- 17h: -$1315 (PIOR horário)
- 15h: +$669 (MELHOR horário)
- Diferença de >$1900 entre melhor e pior horário

#### 3.2 Intraday Position Context (30% deste componente)
**Features usadas**: `intraday_range`, posições existentes

**Lógica**:
```python
if entry_decision in [1, 2]:
    # 🎯 BÔNUS: Primeira entrada do dia em horário bom
    if len(positions) == 0 AND current_hour in EXCELLENT_HOURS:
        bonus = 0.2

    # ⚠️ PENALTY: Entrada adicional em horário ruim
    if len(positions) >= 1 AND current_hour in BAD_HOURS:
        penalty = -0.3

    # 🎯 BÔNUS: Segunda entrada para hedge/diversificação
    if len(positions) == 1:
        existing_direction = positions[0]['type']
        if existing_direction != entry_direction:
            bonus = 0.15  # Hedge inteligente
```

**Justificativa**:
- Overtrading em horários ruins agrava perdas
- Primeira entrada bem planejada tem maior WR
- Hedge pode proteger drawdown

---

## 📈 INTEGRAÇÃO NO REWARD SYSTEM

### Estrutura Final do Reward:
```
TOTAL REWARD = 70% PnL + 30% Shaping

Shaping (30%) será distribuído:
- 50% Trailing/SL/TP management (EXISTENTE)
- 25% Entry Timing Quality (NOVO - Componente 1)
- 12.5% Entry Confluence (NOVO - Componente 2)
- 12.5% Market Context (NOVO - Componente 3)
```

### Implementação:
1. Criar arquivo: `trading_framework/rewards/entry_timing_rewards.py`
2. Adicionar 3 funções principais:
   - `_calculate_entry_timing_quality(env, entry_decision, action)`
   - `_calculate_entry_confluence_reward(env, entry_decision, action)`
   - `_calculate_market_context_reward(env, entry_decision, action)`
3. Integrar em `reward_daytrade_v3_brutal.py` linha 321 (_calculate_reward_shaping)

### Pseudo-código de Integração:
```python
def _calculate_reward_shaping(self, env, action):
    # ... código existente ...

    # 🎯 NOVO: Entry timing rewards (apenas quando há entrada)
    entry_timing_reward = 0.0
    entry_decision = self._extract_entry_decision(action)

    if entry_decision in [1, 2]:  # BUY ou SELL
        # Componente 1: Entry Timing Quality (10% do shaping)
        timing_quality = self._calculate_entry_timing_quality(env, entry_decision, action)

        # Componente 2: Entry Confluence (5% do shaping)
        entry_confluence = self._calculate_entry_confluence_reward(env, entry_decision, action)

        # Componente 3: Market Context (5% do shaping)
        market_context = self._calculate_market_context_reward(env, entry_decision, action)

        # Total entry timing reward
        entry_timing_reward = timing_quality + entry_confluence + market_context

        # Adicionar ao shaping total
        shaping_reward += entry_timing_reward
        info['entry_timing_reward'] = entry_timing_reward
        info['timing_quality'] = timing_quality
        info['entry_confluence'] = entry_confluence
        info['market_context'] = market_context

    return shaping_reward, info
```

---

## 🎯 EXPECTATIVA DE RESULTADOS

### Baseline Atual:
- SL Hit Rate: 61.5%
- TP Hit Rate: 38.5%
- Win Rate: 35-40%

### Target Após Implementação (estimativa conservadora):
- SL Hit Rate: <48% (-13.5pp) ⬇️
- TP Hit Rate: >52% (+13.5pp) ⬆️
- Win Rate: 45-50% (+10pp) ⬆️

### Como Medir Sucesso:
1. **Curto prazo (primeiros 500k steps)**:
   - Redução de entradas em BAD_HOURS em >60%
   - Aumento de entradas com confluence_ratio > 0.6 em >40%
   - Redução de entradas contra tendência em >50%

2. **Médio prazo (1M-2M steps)**:
   - SL Hit Rate < 55%
   - Win Rate > 42%
   - Profit Factor > 1.2

3. **Longo prazo (3M+ steps)**:
   - SL Hit Rate < 48%
   - Win Rate > 47%
   - Profit Factor > 1.5

---

## ⚠️ CONSIDERAÇÕES E RISCOS

### Riscos Identificados:
1. **Over-guidance**: Rewards muito prescritivos podem limitar aprendizado
   - Mitigação: Usar curriculum learning (guidance decai com training_progress)

2. **Conflito com PnL**: Timing rewards podem conflitar com PnL reward
   - Mitigação: Manter timing como 20% do shaping (6% do total)

3. **Overfitting a horários**: Modelo pode ficar dependente de hour features
   - Mitigação: Balancear hour-based com technical confluence

### Testes Necessários:
1. A/B test: Modelo com vs sem entry timing rewards
2. Ablation study: Testar cada componente isoladamente
3. Generalization test: Testar em período out-of-sample

---

## 📝 PRÓXIMOS PASSOS PARA APROVAÇÃO

### Para Aprovar Este Plano, Avaliar:
1. ✅ Lógica dos componentes está correta?
2. ✅ Features escolhidas são adequadas?
3. ✅ Distribuição de pesos (25% + 12.5% + 12.5%) faz sentido?
4. ✅ Thresholds (confluence > 0.8, volatility < 0.9, etc) são realistas?
5. ✅ Implementação é viável sem quebrar sistema existente?

### Após Aprovação:
1. Implementar `entry_timing_rewards.py` completo
2. Integrar em `reward_daytrade_v3_brutal.py`
3. Adicionar logging detalhado dos componentes
4. Criar testes unitários para cada função
5. Treinar novo checkpoint com rewards atualizados
6. Comparar performance vs Sixteen 1.55M

---

**Aguardando aprovação para começar implementação.** 🚀
