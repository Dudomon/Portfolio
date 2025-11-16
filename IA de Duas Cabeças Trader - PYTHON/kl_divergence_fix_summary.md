# 🔧 CORREÇÃO KL DIVERGENCE INSTÁVEL

## 📊 Problema Identificado:
- **KL divergence oscilando**: 10 → 0.1 → picos altos
- **Causa**: Parâmetros ultra-conservadores causando instabilidade paradoxal

## 🔧 Ajustes Aplicados:

### 1. **target_kl**: 0.01 → **0.03**
- **Razão**: 0.01 é excessivamente restritivo
- **Efeito**: Permite mudanças graduais na política

### 2. **clip_range**: 0.10 → **0.15** 
- **Razão**: 0.10 é muito conservador para dados complexos
- **Efeito**: Maior flexibilidade para updates de política

### 3. **n_epochs**: 4 → **3**
- **Razão**: Reduzir overfitting que causa KL spikes
- **Efeito**: Menos iterações = menos chance de overfit

### 4. **Learning Rates**: Actor 1.5e-05, Critic 3.0e-05 → **2.0e-05 (ambos)**
- **Razão**: LRs diferentes causam conflitos actor-critic
- **Efeito**: Aprendizado sincronizado e estável

## 📈 Resultado Esperado:
- KL divergence estável entre 0.01-0.05
- Menos oscilações bruscas
- Treinamento mais suave e consistente

## 🎯 Monitoramento:
Observar nas próximas 50k-100k steps:
- KL divergence deve estabilizar < 0.05
- Policy loss deve ser mais consistente
- Entropia deve manter-se estável

## ⚠️ Se Problema Persistir:
1. Aumentar `batch_size`: 32 → 64
2. Reduzir `ent_coef`: 0.05 → 0.03
3. Verificar se reward scaling está adequado