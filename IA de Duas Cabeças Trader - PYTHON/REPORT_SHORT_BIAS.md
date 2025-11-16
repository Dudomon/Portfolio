# 🚨 RELATÓRIO: Viés Vendedor Dominante nos Modelos

## Sumário Executivo

Identificado **viés vendedor estrutural CRÍTICO** no action space que faz os modelos preferirem SHORT mesmo com mercado em alta.

## 🎯 Causa Raiz Identificada

### **CULPADO: ACTION SPACE ASSIMÉTRICO**

```python
# cherry.py linha 3580
self.action_space = spaces.Box(
    low=np.array([0, 0, -1, -1]),
    high=np.array([2, 1, 1, 1]),
    dtype=np.float32
)

# Mapeamento (linhas 77-78, 3868-3874)
ACTION_THRESHOLD_LONG = 0.33
ACTION_THRESHOLD_SHORT = 0.67

# action[0] em [0, 2]:
if raw_decision < 0.33:      # HOLD
    entry_decision = 0
elif raw_decision < 0.67:    # LONG
    entry_decision = 1
else:                        # SHORT (>= 0.67)
    entry_decision = 2
```

## 📊 Análise Quantitativa

### Distribuição com Ações Aleatórias Uniformes [0,2]:

| Ação  | Range         | % do Espaço | Samples (100k) | % Observado |
|-------|---------------|-------------|----------------|-------------|
| HOLD  | [0.00, 0.33]  | 16.5%      | 16,515         | 16.5%       |
| LONG  | [0.33, 0.67]  | 17.0%      | 17,110         | 17.1%       |
| SHORT | [0.67, 2.00]  | **66.5%**  | **66,375**     | **66.4%**   |

### 🚨 Viés Estrutural Detectado:

- **Range SHORT**: 1.33 unidades (66.5% do espaço total)
- **Range LONG**: 0.34 unidades (17.0% do espaço total)
- **Range HOLD**: 0.33 unidades (16.5% do espaço total)

**FATOR DE VIÉS: SHORT é 3.91x MAIOR que LONG**

## 🔍 Por Que Isso É Um Problema?

### 1. **Facilitação Estrutural**
O modelo tem **3.91x mais facilidade** para escolher SHORT do que LONG porque:
- Qualquer saída da rede neural entre [0.67, 2.0] resulta em SHORT
- Apenas saídas entre [0.33, 0.67] resultam em LONG
- Durante treinamento, gradientes naturalmente empurram para ranges maiores

### 2. **Reforço Durante Treinamento**
- Modelos exploram aleatoriamente no início
- 66% das explorações aleatórias são SHORT
- Se o mercado está em baixa em QUALQUER momento do treino:
  - SHORTs acumulam reward positivo
  - Network aprende: "SHORT = bom"
  - Bias se cristaliza nos pesos

### 3. **Impossível Recuperar**
Uma vez que o bias é aprendido:
- Network precisa "desaprender" milhões de steps de SHORT
- Enquanto isso, continua vendo 66% SHORT em novas explorações
- Feedback loop positivo mantém o viés

## 🧠 Comparação Robot_cherry.py vs cherry.py

### Robot_cherry.py (linha 385-389):
```python
self.action_space = spaces.Box(
    low=np.array([-10.0, 0.0, -3.0, -3.0]),
    high=np.array([10.0, 1.0, 3.0, 3.0]),
    dtype=np.float32
)
```

**PROBLEMA**: Action space DIFERENTE do ambiente de treino!
- Robot usa [-10, 10] mas não há mapeamento documentado
- Log mostra threshold em linha 3549-3555 (IGUAL ao cherry.py)
- **INCONSISTÊNCIA CRÍTICA**: Spaces diferentes, mas mesmo mapeamento

### Mapeamento Robot (linha 3549-3555):
```python
raw_decision = float(action[0])
if raw_decision < 0.33:      # < 0.33 = HOLD
    entry_decision = 0
elif raw_decision < 0.67:    # < 0.67 = LONG
    entry_decision = 1
else:                        # >= 0.67 = SHORT
    entry_decision = 2
```

**PROBLEMA ADICIONAL**:
- Robot espera [-10, 10] mas usa thresholds [0.33, 0.67]
- Qualquer valor negativo vira HOLD
- Qualquer valor > 0.67 vira SHORT
- Range SHORT ainda maior!

## ✅ Soluções Propostas

### Solução 1: **Action Space Balanceado (RECOMENDADA)**

```python
# cherry.py
self.action_space = spaces.Box(
    low=np.array([-1, 0, -1, -1]),  # Centrado em zero
    high=np.array([1, 1, 1, 1]),
    dtype=np.float32
)

# Novo mapeamento SIMÉTRICO
ACTION_THRESHOLD_LONG = -0.33   # [-1, -0.33] = HOLD (33%)
ACTION_THRESHOLD_SHORT = 0.33   # [-0.33, 0.33] = LONG (33%)
                                # [0.33, 1] = SHORT (33%)

raw_decision = float(action[0])
if raw_decision < -0.33:
    entry_decision = 0  # HOLD
elif raw_decision < 0.33:
    entry_decision = 1  # LONG
else:
    entry_decision = 2  # SHORT
```

**Vantagens**:
- Ranges perfeitamente balanceados (0.67 cada)
- Centrado em zero (melhor para redes neurais)
- Simétrico (LONG e SHORT equidistantes de zero)

### Solução 2: **Discrete Action Space**

```python
self.action_space = spaces.MultiDiscrete([3, 101, 201, 201])
# [0] entry: 0=HOLD, 1=LONG, 2=SHORT (discreto, sem viés)
# [1] confidence: 0-100 (mapeado para [0,1])
# [2-3] management: 0-200 (mapeado para [-1,1])
```

**Vantagens**:
- Elimina completamente viés estrutural
- Mais fácil de interpretar
- Melhor para debugging

### Solução 3: **Penalização de SHORT no Reward**

```python
# Adicionar em reward_daytrade_v3_brutal.py
if entry_decision == 2:  # SHORT
    # Penalizar SHORT para compensar viés estrutural
    base_reward *= 0.7  # Reduzir reward de SHORT em 30%
```

**Desvantagens**:
- Hack temporário, não resolve causa raiz
- Dificulta aprendizado legítimo de SHORTs
- Não recomendado

## 🎯 Plano de Ação Recomendado

### Passo 1: **Corrigir cherry.py** ✅ PRIORITÁRIO
1. Mudar action_space para [-1, 1] na dimensão [0]
2. Atualizar thresholds para [-0.33, 0.33]
3. Atualizar mapeamento em todas as funções:
   - `step()` linha 3868
   - `_process_v5_specialized_action()` linha 6560
   - `_calculate_entry_reward()` linha 6207

### Passo 2: **Alinhar Robot_cherry.py** ✅ PRIORITÁRIO
1. Corrigir action_space para [-1, 1] (linha 385)
2. Garantir mapeamento idêntico ao cherry.py
3. Testar que ranges são balanceados

### Passo 3: **Re-treinar Modelos** ⚠️ NECESSÁRIO
- Modelos atuais foram treinados com viés
- Precisam ser retreinados do zero com novo action space
- Checkpoints antigos são INCOMPATÍVEIS

### Passo 4: **Validar Distribuição**
```python
# Adicionar em cherry.py callback
if self.num_timesteps % 10000 == 0:
    print(f"Distribuição ações: HOLD={hold_pct:.1f}% LONG={long_pct:.1f}% SHORT={short_pct:.1f}%")
```

## 📊 Evidências Adicionais

### 1. Logs do Robot mostram:
```
[PREDIÇÃO] SHORT | Entry: X.XX | Confidence: X.XX
[PREDIÇÃO] SHORT | Entry: X.XX | Confidence: X.XX
[PREDIÇÃO] SHORT | Entry: X.XX | Confidence: X.XX
```

### 2. Reward System V3Brutal:
- ✅ Simétrico entre LONG e SHORT
- ✅ Sem viés no cálculo de PnL
- ✅ Pain multiplier igual para ambos

### 3. Action Space Cherry:
- ❌ Range SHORT 3.91x maior
- ❌ Exploração naturalmente viesada
- ❌ Gradientes favorecem SHORT

## 🎓 Conclusão

O **viés vendedor dominante** é causado por **design assimétrico do action space**, não por problemas de reward ou features.

**SOLUÇÃO: Balancear action space para [-1, 1] com thresholds simétricos**

**IMPACTO**: Todos os modelos precisam ser retreinados do zero

**PRIORIDADE**: 🚨 CRÍTICA - Afeta fundamentalmente o comportamento do modelo

---

**Data**: 2025-09-30
**Status**: Causa identificada, solução proposta
**Ação Necessária**: Aprovar correção e re-treino
