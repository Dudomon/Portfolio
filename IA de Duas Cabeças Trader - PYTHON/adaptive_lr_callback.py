#!/usr/bin/env python3
"""
🚀 ADAPTIVE LEARNING RATE CALLBACK
Ajusta learning rate automaticamente baseado na saúde dos gradientes
"""

import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any

class AdaptiveLearningRateCallback(BaseCallback):
    """Callback para ajustar learning rate baseado na saúde dos gradientes"""
    
    def __init__(self, 
                 initial_lr: float = 2.68e-5,
                 min_lr: float = 1e-6,
                 max_lr: float = 1e-3,
                 adaptation_freq: int = 2000,
                 dead_gradient_threshold: float = 0.3,
                 verbose: int = 0):
        super().__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.adaptation_freq = adaptation_freq
        self.dead_gradient_threshold = dead_gradient_threshold
        
        # Tracking
        self.current_lr = initial_lr
        self.lr_history = []
        self.gradient_health_history = []
        self.step_count = 0
        self.adaptations_count = 0
        
    def _on_training_start(self) -> None:
        """Executado no início do treinamento"""
        print(f"🚀 ADAPTIVE LR CALLBACK ATIVADO - LR inicial: {self.initial_lr:.2e}")
        print(f"   📊 Range: [{self.min_lr:.2e}, {self.max_lr:.2e}] | Freq: {self.adaptation_freq} steps")
    
    def _on_step(self) -> bool:
        """Executado a cada step do treinamento"""
        self.step_count += 1
        
        # Verificar se é hora de adaptar
        if self.step_count % self.adaptation_freq == 0:
            self._adapt_learning_rate()
        
        return True
    
    def _adapt_learning_rate(self):
        """Adaptar learning rate baseado na saúde dos gradientes"""
        try:
            if not hasattr(self.model, 'policy'):
                print(f"⚠️ ADAPTIVE LR: Modelo não tem policy")
                return
            
            # Calcular saúde dos gradientes
            gradient_health = self._calculate_gradient_health()
            self.gradient_health_history.append(gradient_health)
            
            # DEBUG: Sempre mostrar o gradient health
            print(f"🔍 ADAPTIVE LR DEBUG - Step {self.num_timesteps}")
            print(f"   📊 Gradient Health: {gradient_health:.3f}")
            print(f"   📈 Current LR: {self.current_lr:.2e}")
            
            # Determinar novo learning rate
            new_lr = self._determine_new_lr(gradient_health)
            
            # Aplicar novo learning rate
            if abs(new_lr - self.current_lr) > 1e-7:  # Só atualizar se mudança significativa
                self._set_learning_rate(new_lr)
                self.adaptations_count += 1
                
                print(f"🔧 ADAPTIVE LR MUDANÇA - Step {self.num_timesteps}")
                print(f"   📈 LR: {self.current_lr:.2e} → {new_lr:.2e}")
            else:
                print(f"   ✅ LR mantido (mudança < 1e-7)")
            
        except Exception as e:
            print(f"⚠️ Erro na adaptação de LR: {e}")
            import traceback
            traceback.print_exc()
    
    def _calculate_gradient_health(self) -> float:
        """Calcular saúde geral dos gradientes"""
        total_params = 0
        healthy_gradients = 0
        
        for name, param in self.model.policy.named_parameters():
            if param.grad is not None:
                grad_array = param.grad.detach().cpu().numpy().flatten()
                total_params += len(grad_array)
                
                # Contar gradientes "saudáveis" (não zero extremo)
                healthy_mask = np.abs(grad_array) > 1e-8
                healthy_gradients += np.sum(healthy_mask)
        
        health_ratio = healthy_gradients / max(total_params, 1)
        return health_ratio
    
    def _determine_new_lr(self, gradient_health: float) -> float:
        """Determinar novo learning rate baseado na saúde"""
        # Estratégia adaptativa
        if gradient_health < 0.4:  # Gradientes muito mortos
            # Diminuir LR para estabilizar neurônios mortos
            scale_factor = 0.5
            print(f"   🔥 GRADIENTES MORTOS ({gradient_health:.1%}) - DIMINUINDO LR")
        elif gradient_health < 0.6:  # Gradientes moderadamente mortos
            # Diminuir LR moderadamente
            scale_factor = 0.8
            print(f"   ⚠️ GRADIENTES FRACOS ({gradient_health:.1%}) - DIMINUINDO LR")
        elif gradient_health > 0.85:  # Gradientes muito ativos
            # Diminuir LR para estabilizar
            scale_factor = 0.8
            print(f"   ⚡ GRADIENTES ATIVOS ({gradient_health:.1%}) - DIMINUINDO LR")
        else:  # Gradientes saudáveis
            # Manter LR próximo ao atual
            scale_factor = 1.0
        
        # Calcular novo LR
        new_lr = self.current_lr * scale_factor
        
        # Aplicar limites
        new_lr = max(self.min_lr, min(self.max_lr, new_lr))
        
        return new_lr
    
    def _set_learning_rate(self, new_lr: float):
        """Definir novo learning rate no optimizer"""
        try:
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'optimizer'):
                print(f"🔧 DEBUG: Aplicando LR {new_lr:.2e} no optimizer...")
                old_lrs = []
                for i, param_group in enumerate(self.model.policy.optimizer.param_groups):
                    old_lrs.append(param_group['lr'])
                    param_group['lr'] = new_lr
                    print(f"   Param group {i}: {old_lrs[i]:.2e} → {new_lr:.2e}")
                
                self.current_lr = new_lr
                self.lr_history.append(new_lr)
                
                # Verificar se foi aplicado
                verification_lrs = [pg['lr'] for pg in self.model.policy.optimizer.param_groups]
                print(f"🔍 VERIFICAÇÃO: LRs após aplicação: {[f'{lr:.2e}' for lr in verification_lrs]}")
                
            else:
                print(f"❌ ERRO: Optimizer não encontrado!")
                print(f"   Has policy: {hasattr(self.model, 'policy')}")
                if hasattr(self.model, 'policy'):
                    print(f"   Has optimizer: {hasattr(self.model.policy, 'optimizer')}")
                
        except Exception as e:
            print(f"❌ Erro ao definir LR: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_training_end(self) -> None:
        """Executado no final do treinamento"""
        print(f"🏁 ADAPTIVE LR CALLBACK FINALIZADO")
        print(f"   📊 Adaptações realizadas: {self.adaptations_count}")
        print(f"   📈 LR final: {self.current_lr:.2e}")
        if self.lr_history:
            print(f"   📉 LR médio: {np.mean(self.lr_history):.2e}")

def create_adaptive_lr_callback(
    initial_lr: float = 2.68e-5,
    min_lr: float = 1e-6,
    max_lr: float = 1e-3,
    adaptation_freq: int = 2000,
    verbose: int = 0
) -> AdaptiveLearningRateCallback:
    """Factory function para criar callback de LR adaptativo"""
    return AdaptiveLearningRateCallback(
        initial_lr=initial_lr,
        min_lr=min_lr,
        max_lr=max_lr,
        adaptation_freq=adaptation_freq,
        verbose=verbose
    )

if __name__ == "__main__":
    print("🚀 Adaptive Learning Rate Callback - Ajusta LR baseado na saúde dos gradientes")