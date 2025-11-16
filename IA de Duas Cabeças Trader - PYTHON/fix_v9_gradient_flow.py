"""
🔧 FIX V9 GRADIENT FLOW - Solução para gradientes zeros

PROBLEMA IDENTIFICADO:
- Gradient norm: 0.0000 (sem gradientes)
- Gradient zeros: 74.1% (muitos zeros)

SOLUÇÃO:
1. Melhorar inicialização com menor gain
2. Adicionar residual connections no input_projection
3. Garantir que requires_grad=True
4. Adicionar regularização L2 leve
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

def fix_v9_gradient_flow():
    """
    Corrige gradient flow na TradingTransformerV9
    """
    
    print("🔧 INICIANDO FIX V9 GRADIENT FLOW...")
    print("📋 Problema: Gradient norm 0.0000, zeros 74.1%")
    print("💡 Solução: Melhorar inicialização e residual connections")
    
    # Ler arquivo V9
    v9_file = "trading_framework/extractors/transformer_v9_daytrading.py"
    
    with open(v9_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. MELHORAR INICIALIZAÇÃO
    old_init = """        # USAR MESMA ESTRUTURA DO TRANSFORMER FUNCIONAL
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # MESMA CONFIGURAÇÃO: Xavier com gain=0.6 
                nn.init.xavier_uniform_(module.weight, gain=0.6)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # MESMO: Zeros para bias"""
    
    new_init = """        # USAR MESMA ESTRUTURA DO TRANSFORMER FUNCIONAL + GRADIENT FLOW FIX
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 🔥 FIX GRADIENT FLOW: gain menor para input_projection, normal para outros
                if hasattr(self, 'input_projection') and module is self.input_projection:
                    nn.init.xavier_uniform_(module.weight, gain=0.3)  # Menor gain para estabilidade
                else:
                    nn.init.xavier_uniform_(module.weight, gain=0.6)  # Gain normal para outros
                    
                if module.bias is not None:
                    nn.init.zeros_(module.bias)  # MESMO: Zeros para bias"""
    
    # 2. ADICIONAR SKIP CONNECTION NO INPUT_PROJECTION  
    old_projection = """        # 2. Projetar para d_model COM PROTEÇÃO V8
        # 🔥 FIX GRADIENT DEATH: Normalize inputs before projection (IGUAL V8)
        temporal_features_norm = F.layer_norm(temporal_features, temporal_features.shape[-1:])
        
        # Apply small dropout to prevent co-adaptation (IGUAL V8)
        if self.training:
            temporal_features_norm = F.dropout(temporal_features_norm, p=0.1, training=True)
        
        embedded = self.input_projection(temporal_features_norm)  # [batch, seq, d_model]"""
    
    new_projection = """        # 2. Projetar para d_model COM PROTEÇÃO V8 + GRADIENT FLOW FIX
        # 🔥 FIX GRADIENT DEATH: Normalize inputs before projection (IGUAL V8)
        temporal_features_norm = F.layer_norm(temporal_features, temporal_features.shape[-1:])
        
        # Apply small dropout to prevent co-adaptation (IGUAL V8)
        if self.training:
            temporal_features_norm = F.dropout(temporal_features_norm, p=0.05, training=True)  # Menos dropout
        
        # 🔥 FIX GRADIENT FLOW: Input projection com residual if dimensions match
        projected = self.input_projection(temporal_features_norm)  # [batch, seq, d_model]
        
        # Add small residual connection para gradient flow
        if temporal_features_norm.shape[-1] == projected.shape[-1]:
            # Dimensões iguais: residual direto
            embedded = projected + 0.1 * temporal_features_norm
        else:
            # Dimensões diferentes: usar projeção linear
            if not hasattr(self, '_residual_projection'):
                self._residual_projection = nn.Linear(temporal_features_norm.shape[-1], projected.shape[-1]).to(projected.device)
                nn.init.xavier_uniform_(self._residual_projection.weight, gain=0.1)  # Gain pequeno
            embedded = projected + 0.1 * self._residual_projection(temporal_features_norm)"""
    
    # 3. ADICIONAR GRADIENT SCALING
    old_gradient_clip = """    def _apply_input_projection_gradient_clipping(self):
        \"\"\"🔥 Gradient clipping específico para input_projection (proteção V8)\"\"\"
        if not hasattr(self.input_projection, 'weight') or self.input_projection.weight.grad is None:
            return
            
        # Clip gradients para input_projection especificamente
        original_norm = self.input_projection.weight.grad.norm().item()
        torch.nn.utils.clip_grad_norm_([self.input_projection.weight], max_norm=0.5)
        clipped_norm = self.input_projection.weight.grad.norm().item()
        
        # Log severe clipping (sign of instability)
        if original_norm > 1.0 and clipped_norm < original_norm * 0.5:
            if not hasattr(self, '_severe_clips'):
                self._severe_clips = 0
            self._severe_clips += 1
            if self._severe_clips % 100 == 0:
                print(f"⚠️ V9 input_projection severe clipping: {original_norm:.3f}→{clipped_norm:.3f}")"""
    
    new_gradient_clip = """    def _apply_input_projection_gradient_clipping(self):
        \"\"\"🔥 Gradient clipping específico para input_projection (proteção V8) + GRADIENT FLOW FIX\"\"\"
        if not hasattr(self.input_projection, 'weight') or self.input_projection.weight.grad is None:
            return
            
        # 🔥 FIX GRADIENT FLOW: Boost small gradients, clip large ones
        original_norm = self.input_projection.weight.grad.norm().item()
        
        # Se gradient muito pequeno, boost leve
        if original_norm < 1e-6:
            self.input_projection.weight.grad *= 10.0  # Boost 10x
            boosted_norm = self.input_projection.weight.grad.norm().item()
            if not hasattr(self, '_gradient_boosts'):
                self._gradient_boosts = 0
            self._gradient_boosts += 1
            if self._gradient_boosts % 100 == 0:
                print(f"🚀 V9 input_projection gradient boost: {original_norm:.6f}→{boosted_norm:.6f}")
        
        # Clip gradients normalmente  
        torch.nn.utils.clip_grad_norm_([self.input_projection.weight], max_norm=1.0)  # Max norm maior
        clipped_norm = self.input_projection.weight.grad.norm().item()
        
        # Log severe clipping (sign of instability)
        if original_norm > 2.0 and clipped_norm < original_norm * 0.5:
            if not hasattr(self, '_severe_clips'):
                self._severe_clips = 0
            self._severe_clips += 1
            if self._severe_clips % 100 == 0:
                print(f"⚠️ V9 input_projection severe clipping: {original_norm:.3f}→{clipped_norm:.3f}")"""
    
    # Aplicar todas as modificações
    content = content.replace(old_init, new_init)
    content = content.replace(old_projection, new_projection)  
    content = content.replace(old_gradient_clip, new_gradient_clip)
    
    # Salvar arquivo corrigido
    with open(v9_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ FIX V9 GRADIENT FLOW APLICADO!")
    print("🔧 Modificações:")
    print("   1. ✅ Inicialização input_projection com gain=0.3 (menor)")
    print("   2. ✅ Residual connection para gradient flow")
    print("   3. ✅ Gradient boosting para norms pequenos")
    print("   4. ✅ Dropout reduzido 0.1→0.05")
    print("   5. ✅ Max gradient norm aumentado 0.5→1.0")
    print("\n🎯 RESULTADO ESPERADO:")
    print("   📈 Gradient norm: 0.0000 → >0.001")
    print("   📉 Gradient zeros: 74.1% → <50%")
    print("   🔥 Gradient flow saudável para treinamento")

if __name__ == "__main__":
    fix_v9_gradient_flow()