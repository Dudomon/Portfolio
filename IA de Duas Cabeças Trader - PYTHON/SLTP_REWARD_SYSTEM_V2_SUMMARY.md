# 🎯 SL/TP REWARD SYSTEM V2 - ABORDAGEM HÍBRIDA

**Data:** 2025-10-02
**Status:** ✅ IMPLEMENTADO

---

## 📊 PROBLEMA IDENTIFICADO

### **Comportamento Burro do Modelo Atual (1.45M steps):**
```
Live Trading Logs:
- Win rate: 15.4% (2/13 trades)
- Total PnL: -$84.24
- Comportamento: SL sempre no mínimo (5pt), TP sempre no máximo ($100)
- Ajusta SL/TP TODO minuto de forma previsível e inútil
```

### **Root Cause:**
1. **Timeout de 5h artificial** → Modelo aprende que posições fecham sozinhas
2. **Rewards de SL/TP ERRADOS** → Sistema recompensava ações, não qualidade

---

## 🔧 MUDANÇAS IMPLEMENTADAS

### **1. Timeout de 5h DESABILITADO** ✅
**Arquivo:** `D:\Projeto\cherry.py` (linha 3732)

```python
# ANTES:
self.activity_system = create_activity_enhancement_system(position_timeout=300)
print(f"[ACTIVITY SYSTEM] ✅ Timeout 5h para posições")

# DEPOIS:
self.activity_system = None  # DESABILITADO
print(f"[ACTIVITY SYSTEM] ❌ SEM TIMEOUT de posições")
print(f"[PHILOSOPHY] 🎯 Modelo aprenderá gestão natural (sem muleta)")
```

**Impacto:** Modelo não tem mais "garantia" de 5h → Precisa gerenciar SL/TP corretamente.

---

### **2. Sistema de SL/TP Rewards V2** ✅
**Arquivo:** `D:\Projeto\trading_framework\rewards\reward_daytrade_v3_brutal.py`

#### **Arquitetura Híbrida:**
```python
def _calculate_trailing_stop_rewards(env):
    # 1. Heurísticas de RR ratio e caps (70% do reward)
    heuristic_reward = _calculate_smart_sltp_heuristics(env)

    # 2. Recompensa melhorias vs estado anterior (30% do reward)
    improvement_reward = _calculate_sltp_improvement_reward(env)

    # 3. Curriculum learning (guidance decai com treino)
    guidance_weight = _get_sltp_guidance_weight(training_progress)

    # Combinação final
    return (heuristic_reward * 0.7 + improvement_reward * 0.3) * guidance_weight
```

---

## 🎯 COMPONENTES DO NOVO SISTEMA

### **A. Heurísticas Baseadas em Trading Real** (Linhas 663-735)

```python
def _calculate_smart_sltp_heuristics(env):
    """
    ✅ Risk/Reward ratio ideal: 1.5 a 2.5
    ❌ Penalty: SL muito apertado (<7pt)
    ❌ Penalty: TP muito distante (>$80)
    ❌ Penalty: RR ratio < 1.0 (risking mais que reward)
    """
```

**Heurísticas:**
1. **RR Ratio 1.5-2.5** → Reward +0.01
2. **RR Ratio < 1.0** → Penalty -0.02
3. **RR Ratio > 4.0** → Penalty -0.01 (TP irrealista)
4. **SL < 7pt** → Penalty -0.015 (hit fácil)
5. **TP PnL > $80** → Penalty -0.01 (ganância)

---

### **B. Improvement-Based Rewards** (Linhas 737-833)

```python
def _calculate_sltp_improvement_reward(env):
    """
    🎯 Compara estado atual vs anterior:
    - RR ratio melhorou? → Reward +0.005
    - SL protegeu lucro (trailing)? → Reward +0.01
    - TP no sweet spot ($40-$80)? → Reward +0.005
    """
```

**Tracking:**
- `previous_sltp_state[pos_id] = {'sl', 'tp', 'rr_ratio'}`
- Recompensa **MELHORIA**, não ação absoluta

---

### **C. Curriculum Learning** (Linhas 835-848)

```python
def _get_sltp_guidance_weight(training_progress):
    """
    0-20% treino → weight = 1.0 (guidance forte)
    20-60% treino → weight = 0.5 (guidance moderado)
    60-100% treino → weight = 0.1 (guidance mínimo)
    """
```

**Filosofia:** Guiar forte no início, deixar modelo livre no final.

---

## 📋 COMPARAÇÃO: ANTES vs DEPOIS

| Aspecto | V1 (Antigo) | V2 (Novo) |
|---------|-------------|-----------|
| **Timeout 5h** | ✅ Ativo | ❌ Desabilitado |
| **SL/TP Rewards** | Recompensa AÇÃO | Recompensa QUALIDADE |
| **Heurísticas** | Superficiais (pnl >= 0) | Baseadas em RR ratio |
| **Improvement Tracking** | ❌ Não existe | ✅ Estado anterior |
| **Curriculum Learning** | ❌ Não existe | ✅ Guidance decrescente |
| **Problema SL 5pt** | Sistema recompensava | ❌ Penalty -0.015 |
| **Problema TP $100** | Sistema recompensava | ❌ Penalty -0.01 |

---

## 🧪 COMPORTAMENTO ESPERADO NO RE-TREINO

### **Fase 1 (0-20% treino = 0-2.4M steps):**
- **Guidance forte** (weight = 1.0)
- Modelo aprende: SL < 7pt = ruim, TP > $80 = ruim
- RR ratio 1.5-2.5 = bom

### **Fase 2 (20-60% treino = 2.4M-7.2M steps):**
- **Guidance moderado** (weight = 0.5)
- Modelo refina estratégia baseado em PnL real
- Começa a aprender trailing stops inteligentes

### **Fase 3 (60-100% treino = 7.2M-12M steps):**
- **Guidance mínimo** (weight = 0.1)
- Modelo maduro, quase sem guidance artificial
- PnL real domina decisões de SL/TP

---

## 🎯 EXPECTATIVA DE RESULTADOS

### **Live Trading (após re-treino):**
```
ANTES:
- Win rate: 15.4%
- SL: Sempre 5pt (muito apertado)
- TP: Sempre $100 (muito distante)
- PnL: -$84.24

DEPOIS (expectativa):
- Win rate: 30-40% (2x melhoria)
- SL: 7-15pt (respiração adequada)
- TP: $40-$80 (realista)
- RR ratio: 1.5-2.5 (consistente)
- PnL: Positivo (objetivo)
```

---

## 🔧 COMO USAR

### **Durante Treinamento:**
```python
# O environment precisa expor training_progress
env.training_progress = current_steps / total_steps  # 0.0 a 1.0

# OU chamar manualmente:
reward_system.update_training_progress(current_steps=1500000, total_steps=12000000)
```

### **Monitoramento:**
```python
# Ver curriculum weight atual
weight = reward_system._get_sltp_guidance_weight(training_progress)

# Ver rewards breakdown
info = reward_system.calculate_reward_and_info(env, action, old_state)
print(info['trailing_reward'])  # Total de SL/TP rewards
```

---

## 📊 ARQUIVOS MODIFICADOS

1. ✅ `cherry.py` (linha 3732) - Timeout desabilitado
2. ✅ `reward_daytrade_v3_brutal.py` (linhas 48-60, 376-855) - Sistema V2
3. ✅ `cherry_avaliar.py` (linhas 54-56) - Teste apenas 1.45M checkpoint

---

## 🚀 PRÓXIMOS PASSOS

1. **Iniciar re-treino do zero** com novo sistema
2. **Monitorar convergência** (~1.5M steps esperado)
3. **Avaliar em cherry_avaliar.py** (sem timeout)
4. **Testar em live trading** se métricas >= 30% win rate

---

## 💡 INSIGHTS TÉCNICOS

### **Por que Curriculum Learning?**
- Modelo burro precisa de guidance forte (início)
- Modelo maduro precisa de liberdade (final)
- Evita overfitting em heurísticas artificiais

### **Por que Improvement-Based?**
- Resolve credit assignment problem
- Modelo aprende TENDÊNCIA, não valor absoluto
- Mais robusto a diferentes market conditions

### **Por que Heurísticas de RR Ratio?**
- Fundamentadas em trading real (não arbitrárias)
- Penalizam comportamentos extremos (5pt SL, $100 TP)
- Ensinam "range sensato" antes de otimizar

---

**Conclusão:** Sistema V2 ensina o modelo a **PENSAR** em SL/TP management, não apenas seguir regras burras.
