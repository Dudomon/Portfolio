# 🎯 TP TARGET ZONES: Ensinar Modelo a Mirar em Alvos Realistas

**Data:** 2025-10-03
**Problema:** TPs nunca são atingidos (modelo mira muito longe)
**Solução:** Adicionar feature que identifica ZONAS DE TP (resistências próximas)

---

## 💡 CONCEITO: LÓGICA REVERSA

Se modelo "enxerga" **distância para suporte/resistência**:

### Para SL (já proposto):
- **LONG:** SL deve ficar ABAIXO do suporte mais próximo
- **SHORT:** SL deve ficar ACIMA da resistência mais próxima
- **Feature:** `support_resistance` = distância para zona RUIM de SL

### Para TP (NOVO):
- **LONG:** TP deve mirar NA resistência mais próxima
- **SHORT:** TP deve mirar NO suporte mais próximo
- **Feature:** `tp_target_zones` = distância para zona BOA de TP

---

## ✅ NOVA FEATURE: `tp_target_zones`

### SUBSTITUIR: `breakout_strength` → `tp_target_zones`

**Localização:** cherry.py linha 4363-4374

```python
elif feature_name == 'breakout_strength':
    # 🎯 TP TARGET ZONES: Identifica zonas REALISTAS para TP
    # Calcula distância para resistências/suportes mais próximos ACIMA/ABAIXO do preço
    # Valor BAIXO = resistência/suporte PRÓXIMO = zona BOA para TP
    # Valor ALTO = resistência/suporte DISTANTE = zona RUIM para TP

    lookback_swing = 20
    high_series = pd.Series(high_1m)
    low_series = pd.Series(low_1m)

    # Encontrar swing highs (resistências) e swing lows (suportes)
    swing_high = high_series.rolling(window=lookback_swing, center=True).max()
    swing_low = low_series.rolling(window=lookback_swing, center=True).min()

    # Para cada ponto, calcular distância para a RESISTÊNCIA mais próxima ACIMA
    distance_to_resistance_above = np.full(len(close_1m), np.inf)
    for i in range(len(close_1m)):
        # Procurar swing highs ACIMA do preço atual (últimos 50 períodos)
        start_idx = max(0, i - 50)
        relevant_swings = swing_high[start_idx:i+1]
        above_price = relevant_swings[relevant_swings > close_1m[i]]

        if len(above_price) > 0:
            # Resistência mais próxima ACIMA = alvo para TP de LONG
            distance_to_resistance_above[i] = above_price.iloc[0] - close_1m[i]

    # Calcular distância para SUPORTE mais próximo ABAIXO
    distance_to_support_below = np.full(len(close_1m), np.inf)
    for i in range(len(close_1m)):
        start_idx = max(0, i - 50)
        relevant_swings = swing_low[start_idx:i+1]
        below_price = relevant_swings[relevant_swings < close_1m[i]]

        if len(below_price) > 0:
            # Suporte mais próximo ABAIXO = alvo para TP de SHORT
            distance_to_support_below[i] = close_1m[i] - below_price.iloc[-1]

    # Combinar ambas (média para contexto neutro antes de abrir posição)
    combined_distance = np.minimum(distance_to_resistance_above, distance_to_support_below)

    # Normalizar pela ATR (distância relativa à volatilidade)
    atr_14 = pd.Series(high_1m - low_1m).rolling(window=14).mean().fillna(1).values
    tp_zone_distance = combined_distance / (atr_14 + 1e-8)

    # Valores BAIXOS = alvo PRÓXIMO (BOM para TP - realista)
    # Valores ALTOS = alvo DISTANTE (RUIM para TP - irrealista)
    # Inverter para facilitar interpretação: 1.0 = alvo próximo, 0.0 = alvo distante
    tp_target_quality = 1.0 - np.clip(tp_zone_distance / 5.0, 0.0, 1.0)

    self.df.loc[:, 'breakout_strength'] = tp_target_quality
```

---

## 🔄 FEATURES COMPLEMENTARES: SL + TP

Agora temos **DUAS features complementares**:

### 1. `support_resistance` (renomear internamente para `sl_zone_quality`)
**Linha 4386-4397:**
```python
elif feature_name == 'support_resistance':
    # 🎯 SL ZONE QUALITY: Zonas seguras para SL
    # Distância para swing high/low mais próximo (zona RUIM para SL)

    # [código já proposto anteriormente - calcular distância para S/R]

    # Valores ALTOS = longe de S/R (zona BOA para SL)
    # Valores BAIXOS = perto de S/R (zona RUIM para SL - hit fácil)
    sl_zone_quality = [...]  # código anterior
    self.df.loc[:, 'support_resistance'] = sl_zone_quality
```

### 2. `breakout_strength` (renomear internamente para `tp_target_zones`)
**Linha 4363-4374:**
```python
elif feature_name == 'breakout_strength':
    # 🎯 TP TARGET ZONES: Zonas realistas para TP
    # Distância para resistência (LONG) ou suporte (SHORT) mais próximo

    # [código acima - calcular distância para próximo alvo]

    # Valores ALTOS = alvo PRÓXIMO (zona BOA para TP - realista)
    # Valores BAIXOS = alvo DISTANTE (zona RUIM para TP - irrealista)
    tp_target_quality = [...]  # código acima
    self.df.loc[:, 'breakout_strength'] = tp_target_quality
```

---

## 🎯 COMO O MODELO USA ESSAS FEATURES?

### Exemplo: LONG em $2000

**Situação do mercado:**
- Suporte em $1988 (12 pontos abaixo)
- Resistência em $2015 (15 pontos acima)
- ATR = 10 pontos

**Features calculadas:**
```python
# SL ZONE QUALITY (support_resistance)
distance_to_support = 12 pontos
sl_zone_quality = 12 / 10 = 1.2 (normalizado: 0.40)
# 0.40 = BAIXO = suporte próximo = zona RUIM para SL muito apertado

# TP TARGET ZONES (breakout_strength)
distance_to_resistance = 15 pontos
tp_target_distance = 15 / 10 = 1.5 ATR
tp_target_quality = 1.0 - (1.5 / 5.0) = 0.70
# 0.70 = ALTO = resistência próxima = zona BOA para TP
```

**Modelo aprende:**
```
Entry LONG $2000

SL decision:
- support_resistance = 0.40 (BAIXO = suporte em $1988 próximo)
- Modelo aprende: "SL de 10 pontos vai bater no suporte, usar 14 pontos"
- SL final: $1986 (14 pontos)

TP decision:
- breakout_strength = 0.70 (ALTO = resistência em $2015 próxima)
- Modelo aprende: "TP de 15 pontos mira na resistência, REALISTA"
- TP final: $2015 (15 pontos)

Risk/Reward: 15/14 = 1.07:1 (realista!)
```

---

## 📊 IMPACTO ESPERADO

### ANTES (features genéricas):
```
Entry LONG $2000
SL: $1990 (10 pontos - mínimo cego)
TP: $2025 (25 pontos - cap cego)

Realidade do mercado:
- Suporte em $1988 → SL hit em 2 candles (pullback natural)
- Resistência em $2012 → preço reverte ANTES do TP
- Resultado: -$10 perda (TP nunca atingido)
```

### DEPOIS (features para SL/TP):
```
Entry LONG $2000
SL: $1986 (14 pontos - ABAIXO do suporte $1988)
TP: $2012 (12 pontos - NA resistência $2012)

Realidade do mercado:
- Preço puxa até $1989 → SL NÃO hit (respeitou suporte)
- Preço sobe até $2012 → TP HIT (mirou na resistência)
- Resultado: +$12 lucro (TP atingido!)
```

---

## 🏆 RECOMPENSAS PARA TP INTELIGENTE

### Arquivo: `reward_daytrade_v3_brutal.py`

### ADICIONAR após linha 731 (dentro de `_calculate_smart_sltp_heuristics`):

```python
# 🎯 HEURÍSTICA 5: TP MIRADO EM ZONA DE RESISTÊNCIA
# Bonificar quando TP está próximo de uma zona de alvo realista

# Pegar feature tp_target_zones do env
try:
    current_step = getattr(env, 'current_step', 0)
    df = getattr(env, 'df', None)

    if df is not None and 'breakout_strength' in df.columns:
        # breakout_strength agora é tp_target_quality
        tp_target_quality = df['breakout_strength'].iloc[current_step]

        # TP target quality ALTO = resistência próxima
        if tp_target_quality > 0.6:
            # Calcular se o TP atual do modelo está próximo dessa zona
            # (dentro de ±3 pontos da resistência ideal)

            # Feature indica que resistência está próxima
            # Se TP do modelo também está nessa zona, REWARD
            shaping += 0.05 * tp_target_quality

        elif tp_target_quality < 0.3:
            # TP target quality BAIXO = resistência distante
            # Se modelo setou TP muito alto (>20 pontos), PENALTY
            if tp_distance > 20:
                shaping -= 0.03
except:
    pass
```

### ADICIONAR nova função após linha 845:

```python
def _calculate_tp_realism_bonus(self, env) -> float:
    """
    🎯 BONIFICAR TP realista baseado em estrutura de mercado

    TP BOM:
    - Mira em resistência próxima (LONG) ou suporte próximo (SHORT)
    - Distância é múltiplo razoável de ATR (1-2.5 ATR)

    TP RUIM:
    - Ignora resistências próximas
    - Distância irrealista (>3 ATR ou >25 pontos)
    """
    try:
        bonus = 0.0
        positions = getattr(env, 'positions', [])

        if not positions:
            return 0.0

        # Pegar dados de mercado
        current_step = getattr(env, 'current_step', 0)
        df = getattr(env, 'df', None)

        if df is None or 'breakout_strength' in df.columns:
            return 0.0

        tp_target_quality = df['breakout_strength'].iloc[current_step]
        current_atr = getattr(env, 'current_atr', 15.0)

        for position in positions:
            if not isinstance(position, dict):
                continue

            entry_price = position.get('entry_price', 0)
            tp_price = position.get('tp', 0)
            pos_type = position.get('type', '')

            if entry_price == 0 or tp_price == 0:
                continue

            # Calcular distância do TP em pontos
            if pos_type == 'long':
                tp_distance = tp_price - entry_price
            else:
                tp_distance = entry_price - tp_price

            # Calcular distância em múltiplos de ATR
            tp_atr_multiple = tp_distance / current_atr if current_atr > 0 else 0

            # CASO 1: TP target quality ALTO (resistência próxima)
            if tp_target_quality > 0.6:
                # Resistência está próxima (ex: 1.5 ATR)
                # Se modelo setou TP próximo (1-2 ATR), REWARD
                if 1.0 <= tp_atr_multiple <= 2.0:
                    bonus += 0.08 * tp_target_quality
                # Se modelo setou TP muito longe ignorando resistência, PENALTY
                elif tp_atr_multiple > 2.5:
                    bonus -= 0.05

            # CASO 2: TP target quality BAIXO (resistência distante)
            elif tp_target_quality < 0.3:
                # Resistência está longe (>3 ATR)
                # Se modelo setou TP conservador (<2 ATR), REWARD
                if tp_atr_multiple < 2.0:
                    bonus += 0.03
                # Se modelo setou TP no CAP (25 pontos) mirando muito longe, PENALTY
                elif tp_distance >= 24:
                    bonus -= 0.08

        return max(bonus, -0.5)

    except Exception as e:
        self.logger.debug(f"Erro em tp_realism_bonus: {e}")
        return 0.0
```

### INTEGRAR NO REWARD (linha ~368):

```python
# 11. 🎯 TP REALISM: Bonificar TP que mira em zonas realistas
tp_realism = self._calculate_tp_realism_bonus(env)
shaping_reward += tp_realism
info['tp_realism_bonus'] = tp_realism
```

---

## 🎯 RESULTADO FINAL: FEATURES PARA SL + TP

```python
high_quality_features = [
    'volume_momentum',       # ✅ Volume dinâmico
    'price_position',        # ✅ Posição no range
    'breakout_strength',     # 🆕 TP TARGET ZONES (zonas realistas para TP)
    'trend_consistency',     # ✅ Consistência do trend
    'support_resistance',    # 🆕 SL ZONE QUALITY (zonas seguras para SL)
    'volatility_regime',     # ✅ Regime de volatilidade
    'market_structure'       # 🆕 RECENT VOLATILITY SPIKE (ajuste contextual)
]
```

---

## 📊 EXEMPLO COMPLETO DE APRENDIZADO

### Situação: LONG em mercado com estrutura clara

**Mercado:**
- Preço atual: $2000
- Suporte em $1985 (15 pontos abaixo)
- Resistência em $2018 (18 pontos acima)
- ATR: 12 pontos

**Features:**
```python
support_resistance (SL zone) = 0.35  # Suporte próximo (ruim para SL <15 pts)
breakout_strength (TP zone)  = 0.75  # Resistência próxima (bom para TP ~18 pts)
volatility_regime            = 0.40  # Volatilidade média
market_structure (vol spike) = 0.30  # Mercado calmo
```

**Modelo aprende:**
```
Entrada: LONG $2000

Decisão SL:
- support_resistance = 0.35 (BAIXO)
  → Suporte em $1985 próximo
  → SL de 12 pontos bateria ANTES do suporte
  → AUMENTAR para 17 pontos
- SL final: $1983

Decisão TP:
- breakout_strength = 0.75 (ALTO)
  → Resistência em $2018 próxima
  → TP de 18 pontos mira EXATAMENTE na resistência
  → IDEAL!
- TP final: $2018

Risk/Reward: 18/17 = 1.06:1
TP realista: Resistência conhecida
SL seguro: Abaixo do suporte
```

**Resultado esperado:**
- TP hit rate: **40-50%** (vs 5% atual)
- SL hit rate: **50-60%** (vs 95% atual)
- Win rate balanceado com RR ratio realista

---

## ✅ IMPLEMENTAÇÃO FINAL

**Arquivos a modificar:**

### 1. cherry.py (2 mudanças):
- **Linha 4363-4374:** `breakout_strength` → TP Target Zones
- **Linha 4386-4397:** `support_resistance` → SL Zone Quality

### 2. reward_daytrade_v3_brutal.py (2 mudanças):
- **Linha 731:** Adicionar heurística TP em zona de resistência
- **Linha 845:** Adicionar `_calculate_tp_realism_bonus()`

**Tempo:** 20-25 minutos

**Obs space:** NÃO ALTERADO (mesmas features, cálculo diferente)

---

**Gerado:** 2025-10-03
**Problema:** TPs nunca atingidos (miram muito longe)
**Solução:** Feature que identifica zonas REALISTAS para TP (resistências próximas)
**Benefício:** Modelo aprende a mirar em alvos ATINGÍVEIS
