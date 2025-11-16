# 🔍 ANÁLISE HYPERPARÂMETROS HeadV6.py - Causas Possíveis dos Zeros

## **Hyperparâmetros Críticos Encontrados:**

```python
BEST_PARAMS = {
    "learning_rate": 2.678385767462569e-05,  # 2.68e-5 - MUITO BAIXO!
    "n_steps": 1792,                         
    "batch_size": 64,                        
    "n_epochs": 4,                           
    "gamma": 0.99,                          
    "gae_lambda": 0.95,                     
    "clip_range": 0.0824,                    
    "ent_coef": 0.01709320402078782,         
    "vf_coef": 0.6017559963200034,           
    "max_grad_norm": 0.5,                    # GRADIENT CLIPPING RIGOROSO!
}
```

## **🚨 POSSÍVEIS CAUSAS DOS 30.9% ZEROS:**

### **1. LEARNING RATE EXTREMAMENTE BAIXO**
- **Valor**: `2.678e-05` (0.00002678)
- **Problema**: Learning rate muito baixo pode fazer gradientes ficarem próximos de zero
- **LayerNorm sensível**: LayerNorms são especialmente sensíveis a LR baixo

### **2. GRADIENT CLIPPING MUITO AGRESSIVO**  
- **Valor**: `max_grad_norm = 0.5`
- **Problema**: Clipping muito rigoroso pode zerar gradientes pequenos
- **LayerNorm vulnerável**: Gradientes de LayerNorm são tipicamente menores

### **3. SCHEDULER DINÂMICO PROBLEMÁTICO**
```python
self.lr_scheduler = DynamicLearningRateScheduler(
    initial_lr=BEST_PARAMS["learning_rate"],  # Já baixo
    patience=25000,
    factor=0.85,                              # Reduz mais ainda
    min_lr=1e-7                              # Pode chegar a quase zero!
)
```

### **4. CONFIGURAÇÃO DE DROPOUT (possível)**
- Não visto diretamente, mas pode estar em `get_v6_kwargs()`
- Dropout alto pode causar zeros artificiais

## **🎯 SOLUÇÕES RECOMENDADAS:**

### **Solução 1: Aumentar Learning Rate**
```python
"learning_rate": 1e-4,  # Ao invés de 2.68e-5
```

### **Solução 2: Relaxar Gradient Clipping**  
```python
"max_grad_norm": 1.0,  # Ao invés de 0.5
```

### **Solução 3: Desabilitar/Ajustar LR Scheduler**
```python
# Comentar ou aumentar min_lr
min_lr=1e-5  # Ao invés de 1e-7
```

### **Solução 4: LayerNorm específico**
- Usar learning rate diferente para LayerNorms
- Ou desabilitar weight decay em LayerNorms

## **🔬 TESTE IMEDIATO:**
Criar um experimento V8Heritage com:
1. `learning_rate: 1e-4` (4x maior)
2. `max_grad_norm: 1.0` (2x menos rigoroso)  
3. LR scheduler desabilitado temporariamente

Isso deve resolver os 30.9% zeros no `entry_quality_head.1.weight` (LayerNorm).