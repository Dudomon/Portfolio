# 🔥 FORCED POOLING LEARNING IMPLEMENTATION

## PROBLEMA IDENTIFICADO
Learnable pooling **NÃO ESTAVA APRENDENDO**:
- Weights permaneciam uniformes (std=0.000) por 6k+ steps
- Gradient zeros continuavam em 65-66%
- Nenhuma especialização temporal

## SOLUTION: TRIPLE ATTACK

### 1. STRONGER INITIALIZATION
```python
# ANTES: Uniform weights
weights = torch.ones(self.seq_len) / self.seq_len

# DEPOIS: Recent bias + noise
recent_boost = torch.linspace(0.8, 1.2, self.seq_len)  # Linear 0.8→1.2
weights = weights * recent_boost + noise(0, 0.01)
```

### 2. AUXILIARY LOSS (KL Divergence)
```python
# Target: Recent 3 timesteps get 60% weight
recent_target = [0.4/17, ..., 0.4/17, 0.2, 0.2, 0.2]  # Last 3 = 60%
kl_loss = F.kl_div(log_softmax(weights), recent_target)
```

### 3. MANUAL GRADIENT INJECTION
```python
# Every 100 steps: Inject gradient directly
recent_gradient = (target - current_weights) * 0.01
learnable_pooling.grad += recent_gradient
```

## EXPECTED BEHAVIOR

### Immediate Changes (Next 2k steps)
- **std > 0.01**: Weights breaking uniformity
- **recent_3_sum > 0.20**: Recent bias emerging
- **aux_loss decreasing**: Convergence to target

### Medium Term (4k-6k steps)
- **std > 0.03**: Strong specialization
- **recent_3_sum > 0.40**: Strong recent bias
- **Gradient zeros < 50%**: Better flow

### Long Term (8k+ steps)
- **std > 0.05**: Maximum specialization
- **recent_3_sum > 0.60**: Target achieved
- **Gradient zeros < 20%**: Problem solved

## MONITORING

### New Debug Output
```
🎯 [LEARNABLE POOLING] Step 20000: max=0.078, min=0.035, std=0.012, 
    recent_3_sum=0.287, aux_loss=0.0234
    Top3: [(19, '0.078'), (18, '0.071'), (17, '0.065')]
```

### Success Indicators
1. **std increasing**: 0.000 → 0.001 → 0.01+
2. **recent_3_sum increasing**: 0.15 → 0.30 → 0.60
3. **Top3 indices = [17,18,19]**: Recent timesteps dominant

## FALLBACK PLAN
Se ainda não funcionar após 4k steps:
1. Increase aux_loss weight: 0.1 → 0.5
2. Increase manual gradient: 0.01 → 0.05
3. Implement direct parameter update every 500 steps