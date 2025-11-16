# 🔧 CORREÇÃO KL DIVERGENCE SPIKES V2

## 📊 Problema Específico:
- **KL divergence subindo MUITO** durante treinamento
- **Clip fraction estável** (descarta problema de clipping)
- **Causa**: RecurrentPPO + parâmetros mal calibrados

## 🔍 Diagnóstico Técnico:

### 1. **RecurrentPPO vs PPO Padrão**
- `sb3_contrib.RecurrentPPO` tem dinâmica KL diferente
- States temporais causam maior variação na policy

### 2. **Parâmetros Problemáticos Identificados**:
- `n_epochs=3`: Muitos updates por batch → KL cresce
- `ent_coef=0.05`: Exploração excessiva conflita com policy
- `log_std_init=-0.5`: Distribuições muito flexíveis inicialmente
- `target_kl=0.03`: Muito permissivo para RecurrentPPO
- `max_grad_norm=0.2`: Permite updates muito agressivos

## 🔧 Ajustes Aplicados:

### 1. **n_epochs**: 3 → **2**
- **Razão**: Menos epochs = menos updates = menor acúmulo de KL
- **Efeito**: Policy muda gradualmente

### 2. **ent_coef**: 0.05 → **0.02** 
- **Razão**: Menos exploração = menos divergência da policy base
- **Efeito**: Policy mais estável

### 3. **clip_range**: 0.15 → **0.12**
- **Razão**: Updates menos agressivos
- **Efeito**: Mudanças de policy mais conservadoras

### 4. **target_kl**: 0.03 → **0.01**
- **Razão**: Threshold mais restritivo para RecurrentPPO
- **Efeito**: Early stopping quando KL > 0.01

### 5. **max_grad_norm**: 0.2 → **0.1**
- **Razão**: Gradients mais conservadores
- **Efeito**: Updates menores e mais estáveis

### 6. **log_std_init**: -0.5 → **-1.0**
- **Razão**: Distribuições mais rígidas inicialmente
- **Efeito**: Menos variabilidade inicial na policy

## 📈 Resultado Esperado:
- KL divergence estável < 0.01
- Menos spikes durante treinamento
- Policy evolution mais suave
- Mantém clip fraction estável

## 🎯 Monitoramento:
Observar nas próximas 20k-50k steps:
- KL deve ficar consistentemente < 0.01
- Spikes devem desaparecer
- Training deve ser mais estável
- Performance não deve degradar

## ⚠️ Se Problema Persistir:
1. Reduzir `learning_rate`: 2e-5 → 1e-5
2. Aumentar `batch_size`: 32 → 64 (mais estabilidade)
3. Considerar `n_epochs=1` (ultra-conservador)
4. Verificar se RecurrentPPO é necessário vs PPO padrão