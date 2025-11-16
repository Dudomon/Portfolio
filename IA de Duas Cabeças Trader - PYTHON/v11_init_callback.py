#!/usr/bin/env python3
"""
🔧 V11 INITIALIZATION CALLBACK
Aplica inicialização V11 (xavier_uniform_) em runtime para eliminar zeros
"""

import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback

class V11InitCallback(BaseCallback):
    """
    Callback para aplicar inicialização V11 automaticamente durante o treinamento
    """
    
    def __init__(self, check_frequency: int = 100, verbose: int = 1):
        super().__init__(verbose)
        self.check_frequency = check_frequency
        self.applied = False
        self.call_count = 0
        print(f"🔧 [V11 CALLBACK] INICIALIZADO - verificando a cada {check_frequency} steps")
        
    def _on_step(self) -> bool:
        """Executado a cada step do treinamento"""
        self.call_count += 1
        
        # Verificar na frequência especificada e se ainda não aplicado
        if not self.applied and (self.call_count % self.check_frequency == 0 or self.call_count == 1):
            if self.verbose > 0:
                print(f"🔧 [V11 CALLBACK] Step {self.call_count}: Verificando zeros...")
            self._apply_v11_initialization()
        
        # DEBUG: Mostrar que callback está sendo chamado
        if self.call_count % 1000 == 0 and self.verbose > 0:
            print(f"🔧 [V11 CALLBACK] Ativo - Step {self.call_count} (aplicado: {self.applied})")
        
        return True
    
    def _apply_v11_initialization(self):
        """Aplicar inicialização V11 se necessário"""
        try:
            if not hasattr(self.model, 'policy') or not hasattr(self.model.policy, 'actor'):
                return
            
            # Encontrar primeira camada do actor
            first_layer = None
            first_layer_name = None
            
            if hasattr(self.model.policy.actor, 'latent_pi'):
                try:
                    first_layer = self.model.policy.actor.latent_pi[0]
                    first_layer_name = 'actor.latent_pi.0'
                except:
                    pass
            
            if first_layer is None:
                # Fallback: procurar primeira camada Linear
                for name, layer in self.model.policy.actor.named_modules():
                    if isinstance(layer, nn.Linear):
                        first_layer = layer
                        first_layer_name = name
                        break
            
            if first_layer is None or not isinstance(first_layer, nn.Linear):
                return
                
            # Verificar se há zeros excessivos
            zeros_pct = (first_layer.weight.data == 0).float().mean().item() * 100
            
            if zeros_pct > 50:  # Se mais de 50% zeros, aplicar fix
                if self.verbose > 0:
                    print(f"🚨 [V11 CALLBACK] {first_layer_name}: {zeros_pct:.1f}% zeros - APLICANDO V11 INIT!")
                
                # Aplicar inicialização V11 (xavier_uniform_) com retry
                for attempt in range(3):  # Tentar até 3 vezes
                    with torch.no_grad():
                        torch.nn.init.xavier_uniform_(first_layer.weight, gain=1.0)
                        if first_layer.bias is not None:
                            torch.nn.init.zeros_(first_layer.bias)
                    
                    # Verificar se funcionou
                    zeros_after = (first_layer.weight.data == 0).float().mean().item() * 100
                    
                    if zeros_after < 5:  # Sucesso
                        if self.verbose > 0:
                            print(f"✅ [V11 CALLBACK] {first_layer_name}: {zeros_pct:.1f}% → {zeros_after:.1f}% zeros (tentativa {attempt+1})")
                        
                        # Limpar optimizer state para nova inicialização
                        if hasattr(self.model.policy, 'actor_optimizer'):
                            self.model.policy.actor_optimizer.state.clear()
                        if hasattr(self.model.policy, 'critic_optimizer'):
                            self.model.policy.critic_optimizer.state.clear()
                        
                        if self.verbose > 0:
                            print("🔄 [V11 CALLBACK] Optimizer states resetados")
                        
                        self.applied = True
                        return
                    else:
                        if self.verbose > 0:
                            print(f"⚠️ [V11 CALLBACK] Tentativa {attempt+1} falhou: ainda {zeros_after:.1f}% zeros")
                
                # Se chegou aqui, todas tentativas falharam
                if self.verbose > 0:
                    print(f"❌ [V11 CALLBACK] FALHA: Não conseguiu reduzir zeros após 3 tentativas")
                self.applied = False  # Tentar novamente depois
                
            elif self.verbose > 1:
                print(f"✅ [V11 CALLBACK] {first_layer_name}: {zeros_pct:.1f}% zeros - OK")
                self.applied = True  # Não precisa mais verificar
                
        except Exception as e:
            if self.verbose > 0:
                print(f"❌ [V11 CALLBACK] Erro: {e}")

def create_v11_init_callback(check_frequency: int = 100, verbose: int = 1):
    """Factory function para criar o callback V11"""
    return V11InitCallback(check_frequency=check_frequency, verbose=verbose)

if __name__ == "__main__":
    print("🔧 V11 Initialization Callback - Elimina zeros em runtime")