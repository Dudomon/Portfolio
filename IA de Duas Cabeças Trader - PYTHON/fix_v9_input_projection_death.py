"""
🔧 FIX V9 INPUT_PROJECTION DEATH - Solução Definitiva

PROBLEMA IDENTIFICADO:
- V8: temporal_projection com F.layer_norm ANTES da projeção
- V9: input_projection SEM normalização de entrada
- Resultado: input_projection recebe gradientes instáveis → morte de neurônios

SOLUÇÃO:
1. Adicionar F.layer_norm antes de input_projection
2. Adicionar dropout leve durante treinamento  
3. Aplicar gradient clipping específico no input_projection
4. Monitorar health dos neurônios em tempo real

ESTRATÉGIA: Replicar exatamente o que a V8 faz que funciona.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

def fix_v9_input_projection_death():
    """
    Corrige a TradingTransformerV9 para proteger input_projection 
    replicando o comportamento da V8 funcional
    """
    
    print("🔧 INICIANDO FIX V9 INPUT_PROJECTION DEATH...")
    print("📋 Problema: input_projection morrendo (91.4% → 100% zeros)")
    print("💡 Solução: Replicar normalização V8 funcional")
    
    # Ler arquivo original V9
    v9_file = "trading_framework/extractors/transformer_v9_daytrading.py"
    
    with open(v9_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. MODIFICAR FORWARD PARA INCLUIR LAYER_NORM
    old_forward_part = """        # 2. Projetar para d_model
        embedded = self.input_projection(temporal_features)  # [batch, seq, d_model]"""
    
    new_forward_part = """        # 2. Projetar para d_model COM PROTEÇÃO V8
        # 🔥 FIX GRADIENT DEATH: Normalize inputs before projection (IGUAL V8)
        temporal_features_norm = F.layer_norm(temporal_features, temporal_features.shape[-1:])
        
        # Apply small dropout to prevent co-adaptation (IGUAL V8)
        if self.training:
            temporal_features_norm = F.dropout(temporal_features_norm, p=0.1, training=True)
        
        embedded = self.input_projection(temporal_features_norm)  # [batch, seq, d_model]"""
    
    # 2. ADICIONAR GRADIENT CLIPPING ESPECÍFICO
    gradient_clipping_method = '''    
    def _apply_input_projection_gradient_clipping(self):
        """🔥 Gradient clipping específico para input_projection (proteção V8)"""
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
                print(f"⚠️ V9 input_projection severe clipping: {original_norm:.3f}→{clipped_norm:.3f}")
'''
    
    # 3. ADICIONAR MONITORAMENTO DE HEALTH
    health_monitor_method = '''    
    def _monitor_input_projection_health(self):
        """🔍 Monitor health input_projection (igual V8 temporal_projection)"""
        if not hasattr(self.input_projection, 'weight'):
            return
            
        weights = self.input_projection.weight.data
        total_params = weights.numel()
        zero_params = (weights.abs() < 1e-8).sum().item()
        zero_percentage = (zero_params / total_params) * 100
        
        # Store health metrics
        if not hasattr(self, '_health_history'):
            self._health_history = []
        
        self._health_history.append({
            'zero_percentage': zero_percentage,
            'mean_abs_weight': weights.abs().mean().item(),
            'weight_std': weights.std().item()
        })
        
        # Alert on critical health
        if zero_percentage > 50.0:
            print(f"🚨 V9 input_projection CRITICAL: {zero_percentage:.1f}% zeros!")
            return False
        elif zero_percentage > 20.0:
            print(f"⚠️ V9 input_projection WARNING: {zero_percentage:.1f}% zeros")
            
        return True
'''
    
    # 4. MODIFICAR O FORWARD PARA INCLUIR HEALTH CHECK
    old_return = """        # 6. Projeção final
        output_features = self.output_projection(pooled)  # [batch, features_dim]
        
        return output_features"""
    
    new_return = """        # 6. Projeção final
        output_features = self.output_projection(pooled)  # [batch, features_dim]
        
        # 🔥 FIX V9: Health monitoring e gradient clipping (training only)
        if self.training:
            self._apply_input_projection_gradient_clipping()
            
            # Health check periódico
            if hasattr(self, '_forward_count'):
                self._forward_count += 1
            else:
                self._forward_count = 1
                
            if self._forward_count % 1000 == 0:
                health_ok = self._monitor_input_projection_health()
                if not health_ok:
                    print("🚨 V9 input_projection health CRITICAL - aplicando emergency fix")
                    self._emergency_reinit_input_projection()
        
        return output_features"""
    
    # 5. ADICIONAR EMERGENCY REINIT
    emergency_reinit = '''
    def _emergency_reinit_input_projection(self):
        """🚨 Emergency re-initialization para input_projection"""
        print("🚨 EMERGENCY: Re-inicializando input_projection...")
        
        # Re-init com gain menor para estabilidade
        nn.init.xavier_uniform_(self.input_projection.weight, gain=0.3)  # Menor que 0.6
        if self.input_projection.bias is not None:
            nn.init.zeros_(self.input_projection.bias)
            
        print("✅ input_projection re-inicializado com gain=0.3")
'''
    
    # Aplicar todas as modificações
    content = content.replace(old_forward_part, new_forward_part)
    content = content.replace(old_return, new_return)
    
    # Adicionar métodos antes do final da classe
    insertion_point = "def create_v9_daytrading_kwargs"
    content = content.replace(insertion_point, gradient_clipping_method + health_monitor_method + emergency_reinit + insertion_point)
    
    # Salvar arquivo corrigido
    with open(v9_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ FIX V9 INPUT_PROJECTION APLICADO!")
    print("🔧 Modificações:")
    print("   1. ✅ F.layer_norm antes de input_projection (igual V8)")
    print("   2. ✅ Dropout 0.1 durante training (igual V8)")  
    print("   3. ✅ Gradient clipping específico input_projection")
    print("   4. ✅ Health monitoring em tempo real")
    print("   5. ✅ Emergency re-init se health crítica")
    print("\n🎯 RESULTADO ESPERADO:")
    print("   📉 input_projection zeros: 91.4% → <10%")
    print("   📈 Action diversity: LONG 100% → distribuição equilibrada")
    print("   🔥 Confidence range: 0.00-0.16 → valores normais")

if __name__ == "__main__":
    fix_v9_input_projection_death()