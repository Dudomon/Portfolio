# 🔬 AGGREGATION GRADIENT ANALYSIS

## PROBLEMA CONFIRMADO
Mesmo após redução drástica de 26 → 8 layers, gradientes continuam morrendo:
- Step 6000: 33.5% zeros ✅ (melhoria inicial)
- Step 8000: 65.6% zeros ❌ (degradou)  
- Step 10000: 66.6% zeros ❌ (plateau)
- Step 12000: 66.7% zeros ❌ (sem melhoria)

## ROOT CAUSE: GLOBAL AVERAGE POOLING

### Problema Matemático
```python
# ATUAL - Mata gradientes uniformemente
x_pooled = x.mean(dim=1)  # [batch, seq_len, d_model] → [batch, d_model]
```

**Por que mata gradientes:**
1. Cada timestep recebe gradient `∂L/∂x_t = (1/seq_len) * ∂L/∂x_pooled`
2. Com seq_len=20: cada timestep recebe apenas 5% do gradient signal
3. **temporal_projection** precisa aprender de TODOS timesteps mas recebe signal diluído

### Análise do Fluxo
```
temporal_projection → transformer → MEAN POOLING → aggregator → output
     ↑                                    |
     └────── gradient × 0.05 ─────────────┘
```

## SOLUÇÃO: LEARNABLE AGGREGATION

### Opção 1: Weighted Temporal Aggregation
```python
class LearnableTemporalAggregator(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        # Learnable weights para cada timestep
        self.temporal_weights = nn.Parameter(torch.ones(seq_len) / seq_len)
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # Normalizar weights (softmax)
        weights = F.softmax(self.temporal_weights, dim=0)
        
        # Weighted aggregation
        weighted_x = x * weights.unsqueeze(0).unsqueeze(-1)
        aggregated = weighted_x.sum(dim=1)
        
        # Gating mechanism
        gate_values = self.gate(aggregated)
        return aggregated * gate_values
```

### Opção 2: Attention-based Aggregation
```python
class AttentionAggregator(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, 1)
        self.scale = d_model ** -0.5
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        # Compute attention scores
        scores = self.query(x).squeeze(-1)  # [batch, seq_len]
        weights = F.softmax(scores * self.scale, dim=-1)
        
        # Weighted sum
        aggregated = torch.einsum('bs,bsd->bd', weights, x)
        return aggregated
```

### Opção 3: Learnable Pooling (MAIS SIMPLES)
```python
class LearnablePooling(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        # Start com uniform mas learnable
        self.weights = nn.Parameter(torch.ones(seq_len))
        
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        w = F.softmax(self.weights, dim=0)
        # Weighted mean preserva gradients específicos
        return torch.einsum('s,bsd->bd', w, x)
```

## BENEFÍCIOS ESPERADOS

1. **Gradient Flow**: Timesteps importantes recebem mais gradient
2. **Aprendizado**: Modelo aprende quais timesteps são relevantes
3. **Preservação**: temporal_projection recebe signal forte dos timesteps críticos

## IMPLEMENTAÇÃO PROPOSTA

1. Substituir `x.mean(dim=1)` por `LearnablePooling`
2. Inicializar com weights uniformes (não quebrar funcionalidade)
3. Monitorar gradient flow para temporal_projection

## MÉTRICAS DE SUCESSO
- Gradient zeros < 10% em temporal_projection 
- Gradient flow mais forte para timesteps recentes
- Convergência mais rápida do modelo