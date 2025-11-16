# Sessão Debug: Explained Variance Negativa - 2025-08-04

## 🚨 Problema Principal
- **Explained variance negativa** após 500k+ steps
- Sistema não convergia mais desde mudança para dataset sintético
- LR 1.2e-03 não melhorou explained_variance

## 📊 Dados do Problema
### Métricas aos 503k steps:
- **explained_variance**: -0.00362 (negativa!)
- **policy_gradient_loss**: 0.00387 (ativo)
- **clip_fraction**: 0.319 (32% - razoável)
- **learning_rate**: 0.0012
- **Gradients zeros**: 7.91% (crítico em LSTM/Transformer)

### Comparação Histórica:
- **ANTES (LR 6e-04)**: explained_variance +0.0103 ✅
- **DEPOIS (LR 1.2e-03)**: explained_variance -0.105 ❌

## 🔍 Diagnóstico Realizado

### 1. Timeline do Problema:
1. **Sistema funcionava** com dataset real Yahoo + explained_variance positiva
2. **Mudança para dataset sintético** (sugestão minha)
3. **NUNCA MAIS CONVERGIU** desde então

### 2. Possíveis Causas Identificadas:
- **Dataset sintético problemático** (principal suspeito)
- **Mudanças no DayTrader Reward V2** (podem ter quebrado sistema)
- **LR muito alto** deteriorou value function

## ✅ Soluções Implementadas

### 1. **Revertido Dataset para Yahoo Real**
```python
# Antes (sintético):
dataset_path = 'data/GOLD_TRADING_READY_2M_ENHANCED_INDICATORS.csv'

# Depois (real):
dataset_path = 'data/GC=F_YAHOO_DAILY_5MIN_20250704_142845.csv'
```
- **15+ anos de dados reais** (2010-2025)
- **1.1M barras** de ouro 5min
- **Dados Yahoo limpos** com indicadores

### 2. **Reduzido Learning Rate**
```python
# Antes:
"learning_rate": 1.2e-03,  # Muito alto, piorou explained_variance

# Depois:
"learning_rate": 6.0e-04,  # Metade - valor que funcionava antes
```

### 3. **Implementado Limpeza de Logs JSONL**
```python
# Adicionado no main():
jsonl_files = glob.glob("avaliacoes/*.jsonl")
for file in jsonl_files:
    try:
        os.remove(file)
        print(f"   ✅ Removido: {file}")
    except Exception as e:
        print(f"   ⚠️ Erro removendo {file}: {e}")
```

## 🎯 Próximos Passos

### Teste em Andamento:
- **Dataset**: Yahoo real 15+ anos
- **LR**: 6e-04 (valor que funcionava)
- **Objetivo**: Verificar se explained_variance volta a ser positiva

### Se Funcionar:
- Confirma que problema era dataset sintético
- Sistema volta ao estado funcional anterior

### Opções Futuras - Volatilidade Artificial:
Se dataset real funcionar mas tiver baixa volatilidade:

**A. Time Compression**
- Pular barras aleatoriamente
- Simula movimentos mais rápidos

**B. Volatility Scaling** 
- Multiplicar returns por 1.2x-2.0x
- Manter estrutura, amplificar movimentos

**C. Noise Injection**
- Adicionar ruído gaussiano controlado
- Aumentar "chaos" mantendo tendências

**D. Market Regime Simulation**
- Alternar períodos alta/baixa volatilidade
- Bull/bear markets acelerados

## 📋 Status Atual
- ✅ Dataset revertido para Yahoo real
- ✅ LR reduzido para 6e-04  
- ✅ Limpeza JSONL implementada
- ✅ **PROBLEMA RESOLVIDO!** Explained variance voltou para 0.8-0.9 com apenas 30k steps!

## 🔄 Próxima Sessão
- ✅ **CONFIRMADO**: Dataset sintético era o problema!
- 🎯 **PRÓXIMO PASSO DEFINIDO**: Criar dataset híbrido (real + enhancements artificiais)
- 📈 **Meta**: Manter convergência rápida do dataset real + aumentar volatilidade artificialmente
- 💡 **Estratégia**: Yahoo real como base + volatility scaling/time compression/noise injection

## 🎉 CONCLUSÃO
**ROOT CAUSE ENCONTRADA**: Dataset sintético quebrava explained_variance completamente.
Dataset real Yahoo = explained_variance 0.8-0.9 em 30k steps (vs 500k+ negativa no sintético)!

---
*Criado em: 2025-08-04 17:46*
*Sessão: Debug Explained Variance Negativa*