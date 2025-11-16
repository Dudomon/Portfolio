#!/usr/bin/env python3
"""
🔧 CALLBACK DE REGULARIZAÇÃO DE GRADIENTES
Integra o sistema de regularização no treinamento PPO
"""

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, Any
from gradient_regularization import GradientRegularizer, ActivationHook


class GradientRegularizationCallback(BaseCallback):
    """Callback para aplicar regularização de gradientes durante treinamento"""
    
    def __init__(self, 
                 regularizer: GradientRegularizer,
                 apply_freq: int = 1,  # Aplicar a cada step
                 health_check_freq: int = 1000,
                 verbose: int = 0):
        super().__init__(verbose)
        self.regularizer = regularizer
        self.apply_freq = apply_freq
        self.health_check_freq = health_check_freq
        self.step_count = 0
        
        # Hook para ativações
        self.activation_hook = ActivationHook()
        self.hooks_registered = False
        
    def _on_training_start(self) -> None:
        """Executado no início do treinamento"""
        try:
            if hasattr(self.model, 'policy') and not self.hooks_registered:
                self.activation_hook.register_hooks(self.model.policy)
                self.hooks_registered = True
                print("🔧 Hooks de ativação registrados na policy")
        except Exception as e:
            print(f"⚠️ Erro ao registrar hooks: {e}")
    
    def _on_step(self) -> bool:
        """Executado a cada step do treinamento"""
        self.step_count += 1
        
        try:
            # 1. Aplicar regularização de gradientes a cada step
            if self.step_count % self.apply_freq == 0:
                if hasattr(self.model, 'policy'):
                    reg_stats = self.regularizer.apply_gradient_regularization(self.model.policy)
                    
                    # Log apenas se aplicou regularização
                    if reg_stats.get('regularization_applied', False) and self.verbose > 0:
                        print(f"🔧 Step {self.num_timesteps}: Regularização aplicada")
                        print(f"   Dead gradients: {reg_stats['dead_gradients']}/{reg_stats['total_params']}")
            
            # 2. Health check periódico
            if self.step_count % self.health_check_freq == 0:
                self._perform_health_check()
                
        except Exception as e:
            if self.verbose > 0:
                print(f"⚠️ Erro na regularização: {e}")
        
        return True
    
    def _perform_health_check(self):
        """Realizar verificação de saúde do modelo"""
        try:
            if hasattr(self.model, 'policy'):
                health_stats = self.regularizer.check_model_health(self.model.policy)
                
                # Log health check
                if self.verbose > 0:
                    print(f"\n🔍 HEALTH CHECK - Step {self.num_timesteps}")
                    print(f"   Health Score: {health_stats['health_score']:.3f}")
                    print(f"   Zero params: {health_stats['zero_parameters']}/{health_stats['total_parameters']} ({100*health_stats['zero_parameters']/max(health_stats['total_parameters'],1):.1f}%)")
                    print(f"   Parameter norm: {health_stats['parameter_norm']:.4f}")
                
                # Alerta se saúde baixa
                if health_stats['health_score'] < 0.8:
                    print(f"⚠️ ALERTA: Saúde do modelo baixa ({health_stats['health_score']:.3f})")
                    
                    # Aplicar correção mais agressiva
                    if health_stats['health_score'] < 0.5:
                        self._apply_emergency_regularization()
                        
        except Exception as e:
            print(f"⚠️ Erro no health check: {e}")
    
    def _apply_emergency_regularization(self):
        """Aplicar regularização emergencial para modelo em estado crítico"""
        try:
            print("🚨 APLICANDO REGULARIZAÇÃO EMERGENCIAL")
            
            if hasattr(self.model, 'policy'):
                reinitialized_count = 0
                # Re-inicializar parâmetros com muitos zeros
                for name, param in self.model.policy.named_parameters():
                    if param.data is not None:
                        zero_ratio = (torch.abs(param.data) < 1e-8).float().mean().item()
                        
                        if zero_ratio > 0.3:  # 🔥 CORREÇÃO: >30% zeros já é crítico
                            print(f"🔧 Re-inicializando {name} (zero ratio: {zero_ratio:.1%})")
                            reinitialized_count += 1
                            
                            if 'bias' in name:
                                # 🔥 CORREÇÃO: Bias mais forte para quebrar simetria
                                if 'attention' in name or 'self_attn' in name:
                                    torch.nn.init.normal_(param.data, mean=0.0, std=0.02)
                                else:
                                    torch.nn.init.uniform_(param.data, -0.05, 0.05)
                            elif 'weight' in name:
                                # 🔥 CORREÇÃO: Weight initialization mais robusta
                                if param.data.dim() >= 2:
                                    if 'attention' in name or 'transformer' in name:
                                        torch.nn.init.xavier_normal_(param.data, gain=1.0)
                                    else:
                                        torch.nn.init.kaiming_normal_(param.data, mode='fan_in', nonlinearity='relu')
                                else:
                                    torch.nn.init.normal_(param.data, 0.0, 0.05)
                                    
            print(f"✅ Regularização emergencial aplicada - {reinitialized_count} parâmetros re-inicializados")
            
        except Exception as e:
            print(f"❌ Erro na regularização emergencial: {e}")
    
    def _on_training_end(self) -> None:
        """Executado no final do treinamento"""
        try:
            # Remover hooks
            if self.hooks_registered:
                self.activation_hook.remove_hooks()
                print("🔧 Hooks de ativação removidos")
            
            # Health check final
            if hasattr(self.model, 'policy'):
                final_health = self.regularizer.check_model_health(self.model.policy)
                print(f"\n🏁 HEALTH CHECK FINAL:")
                print(f"   Health Score: {final_health['health_score']:.3f}")
                print(f"   Zero params: {100*final_health['zero_parameters']/max(final_health['total_parameters'],1):.1f}%")
                
        except Exception as e:
            print(f"⚠️ Erro ao finalizar regularização: {e}")


def create_gradient_regularization_callback(
    regularizer,
    apply_freq: int = 1,
    health_check_freq: int = 1000,
    verbose: int = 0
) -> GradientRegularizationCallback:
    """Factory function para criar callback de regularização"""
    return GradientRegularizationCallback(
        regularizer=regularizer,
        apply_freq=apply_freq,
        health_check_freq=health_check_freq,
        verbose=verbose
    )

if __name__ == "__main__":
    print("🔧 Gradient Regularization Callback - Integra regularização no treinamento PPO")