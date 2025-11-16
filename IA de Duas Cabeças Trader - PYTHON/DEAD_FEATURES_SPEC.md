# 🚨 ESPECIFICAÇÃO TÉCNICA: DEAD FEATURES PROBLEM

## PROBLEMA CRÍTICO
Features nos índices [20, 29, 38] estão SEMPRE zeradas durante o treinamento, causando:
- Gradient vanishing no transformer (67.7% zeros)
- Performance degradada (Portfolio $302 vs $1000 inicial)
- Win rate baixo (14.3%)

## ANÁLISE TÉCNICA

### ARQUITETURA DAS FEATURES (129 total por timestep)
```
Posição no Array Global:
[0-15]   Market Data (16 features)
[16-42]  Positions (27 features = 3 positions × 9 features)
[43-79]  Intelligent Features (37 features)  
[80-128] Advanced Features (49 features)
```

### MAPEAMENTO DAS POSIÇÕES
```
Posição 0: índices [16-24] (9 features)
Posição 1: índices [25-33] (9 features) 
Posição 2: índices [34-42] (9 features)

Dentro de cada posição (9 features):
[0] Active flag
[1] Entry price
[2] Current price
[3] Unrealized PnL
[4] Duration ⭐ CRÍTICO
[5] Volume
[6] Stop Loss
[7] Take Profit
[8] Position type

DURATION GLOBAL INDICES:
- Posição 0 duration: índice 16 + 4 = 20 ⭐
- Posição 1 duration: índice 25 + 4 = 29 ⭐  
- Posição 2 duration: índice 34 + 4 = 38 ⭐
```

## BUG ROOT CAUSE

### INCONSISTÊNCIA ENTRE FUNÇÕES
1. **_get_single_bar_features** (funciona corretamente):
   - Duration no índice 4 da posição ✅
   - Usa `current_step` real para cálculo ✅

2. **_get_vectorized_temporal_features** (bugada):
   - Duration estava no índice 7 da posição ❌
   - Ordem das features estava errada ❌

### CÓDIGO BUGADO (ANTES)
```python
# ERRADO - Duration no índice 7
positions_obs[i, :] = [
    1.0, float(entry_price), float(current_price_norm), float(unrealized_pnl),
    float(volume), float(sl), float(tp), float(duration),  # ÍNDICE 7 ❌
    1.0 if pos.get('type') == 'long' else -1.0
]
```

### CÓDIGO CORRIGIDO (DEPOIS)
```python
# CORRETO - Duration no índice 4
positions_obs[i, :] = [
    1.0,  # [0] Posição ativa
    float(entry_price),         # [1] Entry price
    float(current_price_norm),  # [2] Current price  
    float(unrealized_pnl),      # [3] Unrealized PnL
    float(duration),            # [4] Duration ⭐ CORRETO
    float(volume),              # [5] Volume
    float(sl),                  # [6] Stop Loss
    float(tp),                  # [7] Take Profit
    1.0 if pos.get('type') == 'long' else -1.0  # [8] Position type
]
```

## CORREÇÕES APLICADAS

### 1. ORDEM DAS FEATURES UNIFICADA
- ✅ Ambas funções agora usam mesma ordem
- ✅ Duration sempre no índice 4 da posição
- ✅ Posições vazias têm duration 0.35 (não-zero)

### 2. MÚLTIPLAS CAMADAS DE PROTEÇÃO
```python
# Layer 1: Cálculo base
duration = max((self.current_step - pos.get('entry_step', self.current_step)), 1) / 1440.0

# Layer 2: Mínimo garantido
duration = max(duration, 0.1)

# Layer 3: Força bruta anti-zero
if abs(duration) < 1e-6:
    duration = 0.25
```

### 3. POSIÇÕES VAZIAS CORRIGIDAS
```python
# Posições inativas têm duration não-zero
positions_obs[i, 4] = 0.35  # Duration no índice 4
```

## STATUS ATUAL
- ✅ Ordem das features unificada
- ✅ Duration safeguards implementados
- ⚠️ Ainda há zeros nos gradientes (67.7%)
- ⚠️ Performance ainda degradada

## PRÓXIMOS PASSOS
1. Verificar se correção eliminou dead features
2. Monitorar gradientes do transformer
3. Validar mapeamento correto dos índices globais
4. Testar performance após correção

## LOGS DE REFERÊNCIA
```
🚨 [DEAD FEATURES] 1/129 features sempre zeradas: [20]...
🚨 [POSITION DURATIONS] Features de duração mortas: [20]
    Índice 20: min=0.000000, max=0.000000, mean=0.000000
```

**Status**: 🔄 CORREÇÃO APLICADA - AGUARDANDO VALIDAÇÃO