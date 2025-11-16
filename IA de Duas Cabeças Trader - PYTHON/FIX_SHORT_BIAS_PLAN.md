# 🔧 PLANO DE CORREÇÃO: Viés Vendedor

## Opções de Correção (3 Abordagens)

### ⭐ **OPÇÃO 1: Correção Total (RECOMENDADA)**
**Balancear action space [-1, 1] com thresholds simétricos**

✅ **Vantagens:**
- Elimina completamente o viés estrutural
- Ranges perfeitamente balanceados (33% cada)
- Centrado em zero (melhor para redes neurais)
- Solução definitiva

❌ **Desvantagens:**
- ⚠️ **MODELOS INCOMPATÍVEIS**: Todos os checkpoints atuais ficam inválidos
- ⚠️ **RE-TREINO OBRIGATÓRIO**: Precisa treinar do zero (1M+ steps)
- Tempo: ~2-3 dias de treino

📊 **Impacto:** ALTO - Requer re-treino completo

---

### ⭐ **OPÇÃO 2: Correção Mínima (COMPATÍVEL)**
**Manter [0,2] mas ajustar thresholds para balancear**

✅ **Vantagens:**
- ✅ **MODELOS COMPATÍVEIS**: Checkpoints atuais funcionam!
- Correção imediata sem re-treino
- Apenas mudança de interpretação
- Pode continuar treinando checkpoints existentes

❌ **Desvantagens:**
- Não elimina viés estrutural na exploração
- Correção parcial (melhor que nada)
- Action space ainda não otimizado

📊 **Impacto:** BAIXO - Compatível com modelos atuais

---

### ⭐ **OPÇÃO 3: Correção Incremental (HÍBRIDA)**
**Corrigir apenas modelos novos, manter antigos compatíveis**

✅ **Vantagens:**
- Modelos antigos continuam funcionando
- Novos treinos já vêm corretos
- Migração gradual
- Permite comparação A/B

❌ **Desvantagens:**
- Dois sistemas diferentes em paralelo
- Confusão potencial
- Duplicação de código

📊 **Impacto:** MÉDIO - Requer manutenção de 2 versões

---

## 📋 Implementação Detalhada

### OPÇÃO 1: Correção Total (Balanceado)

#### Arquivos a Modificar:
1. ✅ `cherry.py` (ambiente treino)
2. ✅ `Robot_cherry.py` (produção)
3. ✅ `Old-cherry.py` (se ainda usado)

#### Mudanças no código:

**cherry.py (linha ~3580):**
```python
# ❌ ANTES (ERRADO)
self.action_space = spaces.Box(
    low=np.array([0, 0, -1, -1]),
    high=np.array([2, 1, 1, 1]),
    dtype=np.float32
)

# ✅ DEPOIS (CORRETO)
self.action_space = spaces.Box(
    low=np.array([-1, 0, -1, -1]),   # Centrado em zero
    high=np.array([1, 1, 1, 1]),
    dtype=np.float32
)
```

**cherry.py (linhas 77-78 - constantes globais):**
```python
# ❌ ANTES (ERRADO)
ACTION_THRESHOLD_LONG = 0.33   # raw_decision < 0.33 = HOLD (33% do range)
ACTION_THRESHOLD_SHORT = 0.67  # raw_decision < 0.67 = LONG, >= 0.67 = SHORT (33%/34%)

# ✅ DEPOIS (CORRETO - SIMÉTRICO)
ACTION_THRESHOLD_LONG = -0.33   # raw_decision < -0.33 = HOLD (33% do range)
ACTION_THRESHOLD_SHORT = 0.33   # raw_decision < 0.33 = LONG, >= 0.33 = SHORT (33%)
```

**cherry.py (linha ~3868 - função step):**
```python
# ❌ ANTES (ERRADO)
raw_decision = float(action[0])
if raw_decision < ACTION_THRESHOLD_LONG:      # < 0.33
    entry_decision = 0  # HOLD
elif raw_decision < ACTION_THRESHOLD_SHORT:   # < 0.67
    entry_decision = 1  # LONG
else:                                         # >= 0.67
    entry_decision = 2  # SHORT

# ✅ DEPOIS (CORRETO - SIMÉTRICO)
raw_decision = float(action[0])
if raw_decision < ACTION_THRESHOLD_LONG:      # < -0.33
    entry_decision = 0  # HOLD
elif raw_decision < ACTION_THRESHOLD_SHORT:   # < 0.33
    entry_decision = 1  # LONG
else:                                         # >= 0.33
    entry_decision = 2  # SHORT
```

**Repetir mudanças em:**
- `cherry.py` linha ~6207 (`_calculate_entry_reward`)
- `cherry.py` linha ~6560 (`_process_v5_specialized_action`)
- `Robot_cherry.py` linha ~385 (action_space)
- `Robot_cherry.py` linha ~3549 (mapeamento)

---

### OPÇÃO 2: Correção Mínima (Compatível)

#### Mudança APENAS nos thresholds (manter action_space [0,2]):

**cherry.py (linhas 77-78):**
```python
# ✅ CORREÇÃO COMPATÍVEL - Ajustar thresholds para compensar viés
ACTION_THRESHOLD_LONG = 0.67    # Trocar: HOLD agora é maior
ACTION_THRESHOLD_SHORT = 1.33   # LONG=0.67-1.33, SHORT=1.33-2.0

# Resultado:
# HOLD:  [0.00, 0.67] = 33.5%
# LONG:  [0.67, 1.33] = 33.0%
# SHORT: [1.33, 2.00] = 33.5%
```

**cherry.py e Robot_cherry.py - atualizar mapeamento:**
```python
raw_decision = float(action[0])
if raw_decision < 0.67:      # < 0.67 = HOLD (33%)
    entry_decision = 0
elif raw_decision < 1.33:    # < 1.33 = LONG (33%)
    entry_decision = 1
else:                        # >= 1.33 = SHORT (33%)
    entry_decision = 2
```

✅ **MODELOS ATUAIS CONTINUAM FUNCIONANDO!**

---

## 🧪 Validação da Correção

### Script de Teste:

```python
# test_action_space_balance.py
import numpy as np

# Testar distribuição após correção
def test_action_distribution(low, high, threshold_long, threshold_short, n_samples=100000):
    """Testa se distribuição é balanceada"""

    actions = np.random.uniform(low, high, n_samples)

    hold_count = np.sum(actions < threshold_long)
    long_count = np.sum((actions >= threshold_long) & (actions < threshold_short))
    short_count = np.sum(actions >= threshold_short)

    print(f"\n{'='*60}")
    print(f"Action Space: [{low}, {high}]")
    print(f"Thresholds: LONG={threshold_long}, SHORT={threshold_short}")
    print(f"{'='*60}")
    print(f"HOLD:  {hold_count:,} ({100*hold_count/n_samples:.1f}%)")
    print(f"LONG:  {long_count:,} ({100*long_count/n_samples:.1f}%)")
    print(f"SHORT: {short_count:,} ({100*short_count/n_samples:.1f}%)")

    # Verificar balanceamento
    expected = n_samples / 3
    tolerance = 0.02  # 2% de tolerância

    balanced = (
        abs(hold_count/expected - 1) < tolerance and
        abs(long_count/expected - 1) < tolerance and
        abs(short_count/expected - 1) < tolerance
    )

    if balanced:
        print(f"\n✅ BALANCEADO! (tolerância ±{tolerance*100}%)")
    else:
        print(f"\n❌ DESBALANCEADO!")

    return balanced

# Teste 1: Configuração atual (ERRADA)
print("\n🔴 CONFIGURAÇÃO ATUAL (COM VIÉS):")
test_action_distribution(0, 2, 0.33, 0.67)

# Teste 2: Opção 1 - Balanceado
print("\n🟢 OPÇÃO 1 (BALANCEADO [-1,1]):")
test_action_distribution(-1, 1, -0.33, 0.33)

# Teste 3: Opção 2 - Compatível
print("\n🟢 OPÇÃO 2 (COMPATÍVEL [0,2]):")
test_action_distribution(0, 2, 0.67, 1.33)
```

---

## ⚡ Procedimento de Aplicação

### Para OPÇÃO 1 (Recomendada):

1. **Backup checkpoints atuais**
   ```bash
   cp -r trading_framework/training/checkpoints/Cherry45 Cherry45_backup_OLD_ACTION_SPACE
   ```

2. **Aplicar correções**
   - Modificar `cherry.py` (3 locais)
   - Modificar `Robot_cherry.py` (2 locais)
   - Executar script de validação

3. **Limpar checkpoints antigos**
   ```bash
   rm -rf trading_framework/training/checkpoints/Cherry45/*
   ```

4. **Iniciar novo treino**
   ```bash
   python cherry.py
   ```

5. **Monitorar distribuição**
   - Verificar logs a cada 10k steps
   - Confirmar ~33% cada ação

---

### Para OPÇÃO 2 (Compatível):

1. **Aplicar correções nos thresholds**
   - Modificar `cherry.py` constantes globais (linhas 77-78)
   - Modificar `Robot_cherry.py` linha 3549-3555

2. **Executar script de validação**
   ```bash
   python test_action_space_balance.py
   ```

3. **Continuar treino normalmente**
   ```bash
   # Pode continuar dos checkpoints atuais!
   python cherry.py
   ```

4. **Atualizar robôs em produção**
   - Apenas substituir `Robot_cherry.py`
   - Modelos continuam compatíveis

---

## 📊 Comparação das Opções

| Aspecto | Opção 1 (Balanceado) | Opção 2 (Compatível) | Opção 3 (Híbrido) |
|---------|---------------------|---------------------|-------------------|
| **Elimina viés** | ✅ Total | ⚠️ Parcial | ✅/⚠️ Misto |
| **Compatibilidade** | ❌ Quebra | ✅ Mantém | ✅/❌ Ambos |
| **Re-treino** | ⚠️ Obrigatório | ✅ Opcional | ⚠️ Para novos |
| **Tempo impl** | 🕐 5-10min | 🕐 2-3min | 🕐 10-15min |
| **Tempo treino** | ⏰ 2-3 dias | ⏰ Imediato | ⏰ 2-3 dias |
| **Complexidade** | 🟢 Simples | 🟢 Simples | 🟡 Moderada |
| **Manutenção** | 🟢 Fácil | 🟢 Fácil | 🔴 Difícil |
| **Recomendação** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 💡 Recomendação Final

### 🎯 **OPÇÃO 1** se:
- ✅ Pode esperar 2-3 dias de re-treino
- ✅ Quer solução definitiva
- ✅ Não precisa dos checkpoints atuais urgentemente
- ✅ **MELHOR PARA LONGO PRAZO**

### 🎯 **OPÇÃO 2** se:
- ✅ Precisa de correção IMEDIATA
- ✅ Modelos em produção não podem parar
- ✅ Checkpoints atuais são valiosos
- ✅ **MELHOR PARA CURTO PRAZO**

### 🎯 **OPÇÃO 3** apenas se:
- ⚠️ Precisa manter dois sistemas
- ⚠️ Tem equipe grande (manutenção complexa)
- ⚠️ Quer fazer A/B testing

---

## 🚀 Pronto para Aplicar?

**Escolha uma opção e eu preparo os patches de código prontos para aplicar!**

1. Opção 1 (Balanceado) → "aplica opção 1"
2. Opção 2 (Compatível) → "aplica opção 2"
3. Opção 3 (Híbrido) → "aplica opção 3"

Ou revise o plano e me diga se precisa de ajustes.
