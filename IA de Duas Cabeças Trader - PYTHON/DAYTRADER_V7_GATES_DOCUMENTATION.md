# 🧠 DAYTRADER V7 INTUITION - DOCUMENTAÇÃO DOS GATES REAIS

## ❌ PROBLEMA IDENTIFICADO:
O daytrader.py estava usando gates **V5 QUE NÃO EXISTEM** na arquitetura V7 Intuition, causando:
- **TODOS os trades aprovados** durante o treino (gates vazios/inválidos)
- Modelo nunca aprendeu a ser seletivo
- **CONTRADIÇÃO** entre treino permissivo vs live restritivo

## 🎯 GATES REAIS DA V7 INTUITION:

### 1. **Entry Head Gates** (SpecializedEntryHead)
```python
entry_decision, entry_conf, gate_info = self.entry_head(lstm_out, lstm_out, memory_context)
```
- `entry_decision` - Decisão de entrada do modelo
- `entry_conf` - Confiança da entrada (0-1)
- `gate_info` - Dict com informações detalhadas

### 2. **Management Head Gates** (TwoHeadDecisionMaker)  
```python
mgmt_decision, mgmt_conf, mgmt_weights = self.management_head(lstm_out, lstm_out, memory_context)
```
- `mgmt_decision` - Decisão de gestão de posições
- `mgmt_conf` - Confiança da gestão (0-1)
- `mgmt_weights` - Pesos de gestão

### 3. **Unified Backbone Gates**
```python
actor_features, critic_features, regime_id, backbone_info = self.unified_backbone(features)
```
- `regime_id` - Regime de mercado detectado (0-3: bull/bear/sideways/volatile)
- `backbone_info` - Dict com:
  - `actor_attention_mean` - Atenção média do actor
  - `critic_attention_mean` - Atenção média do critic  
  - `specialization_divergence` - Divergência de especialização
  - `regime_name` - Nome do regime

### 4. **Enhanced Gate Info** (Combinado)
```python
gate_info.update(backbone_info)
gate_info['memory_regime'] = regime_id
```

## 🔧 IMPLEMENTAÇÃO CORRETA NO DAYTRADER:

### Função de Captura V7 Intuition:
```python
def _capture_v7_intuition_gates(self, gate_info):
    """
    🧠 CAPTURA GATES REAIS DA V7 INTUITION
    
    V7 Intuition usa:
    - Entry Head (SpecializedEntryHead) 
    - Management Head (TwoHeadDecisionMaker)
    - Unified Backbone com regime detection
    
    NÃO usa gates V5 (long_signal, short_signal, etc.)
    """
    
    # Gates V7 Intuition REAIS
    gates = {
        # Entry Head Gates
        'entry_decision': gate_info.get('entry_decision', 0.5),
        'entry_confidence': gate_info.get('entry_conf', 0.5),
        
        # Management Head Gates  
        'mgmt_decision': gate_info.get('mgmt_decision', 0.5),
        'mgmt_confidence': gate_info.get('mgmt_conf', 0.5),
        
        # Backbone Gates
        'regime_id': gate_info.get('regime_id', 2),  # Default: sideways
        'actor_attention': gate_info.get('actor_attention_mean', 0.5),
        'critic_attention': gate_info.get('critic_attention_mean', 0.5),
        'specialization_divergence': gate_info.get('specialization_divergence', 0.0),
        
        # Regime Info
        'regime_name': gate_info.get('regime_name', 'sideways')
    }
    
    return gates
```

### Sistema de Filtros V7:
```python
def _apply_v7_intuition_filters(self, action, gates, current_price, account_info):
    """
    🎯 FILTROS BASEADOS NOS GATES REAIS DA V7 INTUITION
    
    Usa os gates que a V7 REALMENTE produz, não gates V5 inexistentes
    """
    
    # 1. Entry Confidence Filter
    if gates['entry_confidence'] < 0.6:  # Baixa confiança
        return action * 0.5, f"Entry confidence baixa: {gates['entry_confidence']:.3f}"
    
    # 2. Management Decision Filter  
    if gates['mgmt_confidence'] < 0.4:  # Gestão insegura
        return action * 0.7, f"Management confidence baixa: {gates['mgmt_confidence']:.3f}"
    
    # 3. Regime-Based Filter
    regime_id = gates['regime_id']
    if regime_id == 3:  # Volatile market
        return action * 0.3, f"Mercado volátil (regime {regime_id})"
    elif regime_id == 1:  # Bear market
        return action * 0.8, f"Mercado baixista (regime {regime_id})"
    
    # 4. Backbone Specialization Filter
    if gates['specialization_divergence'] > 0.8:  # Alta divergência
        return action * 0.6, f"Alta divergência backbone: {gates['specialization_divergence']:.3f}"
    
    # Trade aprovado com gates V7 reais
    return action, f"✅ V7 Gates: Entry={gates['entry_confidence']:.2f}, Mgmt={gates['mgmt_confidence']:.2f}, Regime={gates['regime_name']}"
```

## 🚨 ERROS ANTERIORES:

### ❌ Código ERRADO (usando gates V5 inexistentes):
```python
# ISSO NÃO EXISTE NA V7 INTUITION!
gates = {
    'long_signal': gate_info.get('long_signal'),      # ❌ NÃO EXISTE
    'short_signal': gate_info.get('short_signal'),    # ❌ NÃO EXISTE  
    'risk_signal': gate_info.get('risk_signal'),      # ❌ NÃO EXISTE
}
```

### ✅ Código CORRETO (usando gates V7 reais):
```python
# ISSO EXISTE NA V7 INTUITION!
gates = {
    'entry_decision': gate_info.get('entry_decision'),        # ✅ EXISTE
    'entry_confidence': gate_info.get('entry_conf'),          # ✅ EXISTE
    'mgmt_decision': gate_info.get('mgmt_decision'),          # ✅ EXISTE
    'regime_id': gate_info.get('regime_id'),                  # ✅ EXISTE
}
```

## 📚 ARQUITETURA V7 INTUITION RESUMO:

```
🧠 TwoHeadV7Intuition
├── 🔗 UnifiedMarketBackbone (visão compartilhada)
│   ├── regime_detector → regime_id, regime_name
│   ├── actor_gate → actor_attention  
│   └── critic_gate → critic_attention
├── 🎯 SpecializedEntryHead  
│   ├── entry_decision
│   └── entry_conf
├── 🎯 TwoHeadDecisionMaker
│   ├── mgmt_decision
│   ├── mgmt_conf  
│   └── mgmt_weights
└── 💾 EnhancedMemoryBank
    └── memory_regime
```

## 🔥 PARA CLAUDE FUTURO:

**SEMPRE LEMBRE-SE:**
1. **V7 Intuition ≠ V5** - Arquiteturas completamente diferentes
2. **Gates V5** (long_signal, short_signal) **NÃO EXISTEM** na V7
3. **Gates V7** são: entry_decision, entry_conf, mgmt_decision, regime_id, etc.
4. **NUNCA** misture filtros V5 com policy V7 Intuition
5. **SEMPRE** verifique que arquitetura está sendo usada antes de implementar filtros

**ESTE FOI UM ERRO CRÍTICO QUE FEZ O USUÁRIO TREINAR MÚLTIPLOS MODELOS COM LÓGICA QUEBRADA!**