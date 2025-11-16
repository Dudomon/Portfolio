# 🔧 TRANSFORMER TROUBLESHOOTING GUIDE

## 🚨 GRADIENT DEATH DIAGNOSTICS

### 🔍 **Sintomas de Dead Neurons**

#### ❌ **Red Flags Críticos:**
```bash
# GRADIENT ZEROS >10%:
📊 Recent avg zeros: >10% 🚨 CRÍTICO
📊 Recent avg zeros: 5-10% ⚠️ ALERTA  
📊 Recent avg zeros: <5% ✅ SAUDÁVEL

# COMPONENTES ESPECÍFICOS:
- temporal_projection: >30% zeros = DEAD NEURONS
- self_attn.in_proj_*: >20% zeros = ATTENTION PROBLEMS
- mlp_extractor.*: >15% zeros = MLP SATURATION
```

#### 📊 **Patterns a Observar:**
```bash
# PROGRESSIVE DEATH PATTERN:
Step 0-4k:   Normal (2-5% zeros)
Step 4-6k:   Escalating (10-25% zeros)  
Step 6k+:    Critical (40%+ zeros)
Step 8k+:    Death plateau (60%+ zeros)

# LAYER-SPECIFIC PATTERNS:
Linear layers: Sudden spike em zeros
Attention layers: Gradual saturation  
Bias parameters: 33% pattern consistente
```

### 🎯 **Debugging Commands**

#### 📋 **Essential Debug Checks:**
```python
# 1. CHECK GRADIENT ZEROS:
python debug_zeros_extremos.py

# 2. ANALYZE INPUT SCALES:
# Look for this in logs:
🔍 [INPUT DIAGNOSTICS] mean=X, std=Y, range=[min, max]

# 3. CHECK PROJECTION SATURATION:  
🚨 [PROJECTION SATURATION] |x|>3.0: Z% (>10% is problematic)

# 4. VERIFY LEARNABLE POOLING:
🎯 [LEARNABLE POOLING] max=X, min=Y, std=Z
```

#### 🔍 **Advanced Diagnostics:**
```python
# MANUAL FEATURE SCALE CHECK:
import torch
import torch.nn.functional as F

def diagnose_feature_scales(features):
    """Diagnose feature scale problems"""
    print(f"Mean: {features.mean():.4f}")
    print(f"Std: {features.std():.4f}")  
    print(f"Range: [{features.min():.4f}, {features.max():.4f}]")
    print(f"Out of [-3,3]: {(features.abs() > 3.0).float().mean()*100:.1f}%")
    
    # Check for dead features (always zero)
    dead_features = (features.abs() < 1e-8).all(dim=0).sum()
    print(f"Dead features: {dead_features}/{features.shape[1]}")

# USE BEFORE LINEAR LAYERS:
diagnose_feature_scales(bar_features)
```

## 🔧 **COMMON FIXES**

### ✅ **Layer Normalization Fix** (RECOMMENDED)
```python
# PROBLEMA: Feature scale mismatch
# SOLUÇÃO: Normalize antes de linear layers

# ANTES (PROBLEMÁTICO):
projected = self.linear_layer(raw_features)

# DEPOIS (CORRETO):  
normalized_features = F.layer_norm(raw_features, raw_features.shape[-1:])
projected = self.linear_layer(normalized_features)

# OPCIONAL: Small dropout
if self.training:
    projected = F.dropout(projected, p=0.1, training=True)
```

### ⚖️ **Gradient Clipping Fix**
```python
# PROBLEMA: Gradient clipping muito agressivo
# SOLUÇÃO: Ajustar max_grad_norm

# TRANSFORMER RANGES:
"max_grad_norm": 10.0,  # Para transformers (era 1.0)
"max_grad_norm": 5.0,   # Para modelos menores  
"max_grad_norm": 1.0,   # Apenas para CNNs simples
```

### 🎯 **Weight Initialization Fix**
```python
# PROBLEMA: Inicialização inadequada
# SOLUÇÃO: Xavier com gain reduzido

def init_transformer_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=0.6)  # Gain reduzido
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.MultiheadAttention):
        if hasattr(module, 'in_proj_weight'):
            nn.init.xavier_uniform_(module.in_proj_weight, gain=0.8)
        if hasattr(module, 'out_proj'):
            nn.init.xavier_uniform_(module.out_proj.weight, gain=0.8)
```

### 📊 **Learning Rate Fix**
```python  
# PROBLEMA: LR inadequada para transformers
# SOLUÇÃO: LR scheduling apropriado

# TRANSFORMER LR RANGES:
"learning_rate": 1e-4,  # Base LR para transformers
"learning_rate": 5e-4,  # Para modelos maiores
"learning_rate": 2e-4,  # Para fine-tuning

# EVITAR:
"learning_rate": 1e-3,  # Muito alto para transformers
"learning_rate": 1e-5,  # Muito baixo (treino lento)
```

## 🧪 **TESTING PROCEDURES**

### 📋 **Quick Health Check:**
```bash
# 1. RUN DEBUG:
python debug_zeros_extremos.py

# 2. CHECK RECENT LOGS:
tail -f logs/ppo_optimization_*.log | grep "ZERO DEBUG"

# 3. VERIFY METRICS:  
# Look for: Recent avg zeros: <5% ✅

# 4. VALIDATE LEARNABLE COMPONENTS:
# Look for: max≠min in pooling weights
```

### 🔬 **Deep Diagnosis:**
```python
# GRADIENT FLOW ANALYSIS:
def analyze_gradient_flow(model):
    """Analyze where gradients are dying"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            zero_percent = (param.grad.abs() < 1e-8).float().mean().item() * 100
            print(f"{name}: norm={grad_norm:.4f}, zeros={zero_percent:.1f}%")
        else:
            print(f"{name}: NO GRADIENT")

# ACTIVATION SATURATION CHECK:
def check_activation_saturation(activations, name="layer"):
    """Check for GELU/ReLU saturation"""  
    saturated = (activations.abs() > 3.0).float().mean().item() * 100
    dead = (activations.abs() < 1e-6).float().mean().item() * 100
    print(f"{name}: saturated={saturated:.1f}%, dead={dead:.1f}%")
```

## 🚨 **EMERGENCY FIXES**

### 🔥 **Critical Gradient Death (>50% zeros):**
```python
# IMMEDIATE ACTIONS:
1. Stop training immediately
2. Apply layer normalization fix  
3. Reduce learning rate by 50%
4. Check input feature scales
5. Restart from recent checkpoint

# CODE TEMPLATE:
# In _create_temporal_sequence():
bar_features_norm = F.layer_norm(bar_features, bar_features.shape[-1:])
projected_features = self.temporal_projection(bar_features_norm)
```

### ⚡ **Learnable Components Not Learning:**
```python
# SYMPTOMS: Uniform weights in pooling
# FIX: Add manual gradient injection

if self.training and self._debug_step % 100 == 0:
    # Force learning signal
    target_weights = torch.tensor([0.6 if i >= 17 else 0.4/(20-3) 
                                  for i in range(20)])
    gradient_signal = (target_weights - current_weights) * 0.01
    
    if self.learnable_pooling.grad is None:
        self.learnable_pooling.grad = gradient_signal.detach()
    else:
        self.learnable_pooling.grad += gradient_signal.detach()
```

### 🎯 **Attention Death Pattern:**
```python
# SYMPTOMS: in_proj_bias with 33% zeros consistently  
# FIX: Attention-specific initialization

def fix_attention_death(attention_layer):
    """Fix dead attention patterns"""
    with torch.no_grad():
        # Reinitialize problematic components
        if hasattr(attention_layer, 'in_proj_bias'):
            attention_layer.in_proj_bias.fill_(0.0)
        
        # Add small noise to break symmetry
        if hasattr(attention_layer, 'in_proj_weight'):
            noise = torch.randn_like(attention_layer.in_proj_weight) * 0.01
            attention_layer.in_proj_weight += noise
```

## 📚 **PREVENTION CHECKLIST**

### ✅ **Before Training:**
- [ ] Check input feature scales are reasonable [-3, 3]
- [ ] Verify layer normalization before linear layers  
- [ ] Set appropriate max_grad_norm (5-10 for transformers)
- [ ] Initialize weights with proper gain (0.6-0.8)
- [ ] Enable gradient monitoring from step 0

### ✅ **During Training:**
- [ ] Monitor gradient zeros every 2000 steps
- [ ] Check learnable component evolution  
- [ ] Validate projection saturation <10%
- [ ] Ensure position/market feature balance
- [ ] Watch for progressive death patterns

### ✅ **After Issues:**
- [ ] Document exact symptoms and steps
- [ ] Test fix on small dataset first
- [ ] Validate fix maintains performance  
- [ ] Add monitoring for similar issues
- [ ] Update troubleshooting procedures

## 🔍 **DEBUGGING TOOLS**

### 📊 **Essential Scripts:**
```bash
# GRADIENT ANALYSIS:
python debug_zeros_extremos.py

# FEATURE SCALE CHECK:  
python debug_temporal_projection.py

# SYSTEMATIC INVESTIGATION:
python systematic_gradient_investigation.py

# REAL-TIME MONITORING:
python gradient_health_monitor.py
```

### 📈 **Monitoring Commands:**
```bash
# WATCH GRADIENT HEALTH:
tail -f logs/*.log | grep -E "(ZERO DEBUG|GRADIENT|SATURATION)"

# CHECK RECENT REPORTS:
ls -la debug_zeros_report_step_*.txt | tail -5

# ANALYZE PATTERNS:
grep "Recent avg zeros" debug_zeros_report_step_*.txt | tail -10
```

## 🎓 **LESSONS LEARNED**

### ✅ **What Works:**
- **Layer normalization** before linear layers (BEST)
- **Proper gradient clipping** (5-10x for transformers)
- **Xavier initialization** with reduced gain
- **Moderate dropout** (0.1) after projection
- **Feature scale monitoring** throughout training

### ❌ **What Doesn't Work:**
- Complex architectural changes without diagnosis
- Aggressive regularization (dropout >0.3)  
- Ignoring input feature scales
- Assuming correlation = causation
- Band-aid fixes without understanding root cause

### 🧠 **Key Insights:**
- **Simple problems** often have simple solutions
- **Feature scales matter** more than architecture complexity
- **Dead neurons** are usually input-related, not weight-related  
- **Layer normalization** is transformer's best friend
- **Systematic debugging** beats random fixes

---

**🔧 REMEMBER: Always diagnose first, fix second, validate third!**

*Este guia foi criado baseado na resolução bem-sucedida do gradient death problem no transformer PPO trading system.*