#!/usr/bin/env python3
"""
🔍 LIGHTWEIGHT ATTENTION MONITOR
Monitoramento leve para attention bias - não corrige agressivamente,
apenas monitora e alerta se ficar realmente problemático
"""

import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LightweightAttentionMonitor(BaseCallback):
    """
    🔍 Monitor leve para attention bias
    - Monitora sem corrigir agressivamente
    - Só alerta se realmente problemático (>50% zeros)
    - Não interfere no treinamento normal
    """
    
    def __init__(self, 
                 check_frequency: int = 2000,  # Verificar a cada 2000 steps (menos frequente)
                 alert_threshold: float = 0.5,  # Só alertar se >50% zeros (muito alto)
                 verbose: int = 1):
        super().__init__(verbose)
        self.check_frequency = check_frequency
        self.alert_threshold = alert_threshold
        self.attention_history = []
        self.alerts_sent = 0
        
    def _on_training_start(self) -> None:
        if self.verbose >= 1:
            print(f"🔍 Lightweight Attention Monitor ativo")
            print(f"   Check frequency: {self.check_frequency} steps")
            print(f"   Alert threshold: {self.alert_threshold:.1%}")
    
    def _on_step(self) -> bool:
        if self.num_timesteps % self.check_frequency == 0:
            self._check_attention_health()
        return True
    
    def _check_attention_health(self):
        """🔍 Verificar saúde do attention bias (sem corrigir)"""
        try:
            model = self.training_env.get_attr('model')[0] if hasattr(self.training_env, 'get_attr') else self.model
            
            if not hasattr(model, 'policy'):
                return
            
            # Verificar attention bias
            total_attention_zeros = 0
            total_attention_params = 0
            problematic_layers = []
            
            for name, param in model.policy.named_parameters():
                if 'attention' in name and 'bias' in name and param.grad is not None:
                    grad_data = param.grad.data.cpu().numpy()
                    zeros = np.sum(np.abs(grad_data) < 1e-8)
                    total = grad_data.size
                    zero_ratio = zeros / total
                    
                    total_attention_zeros += zeros
                    total_attention_params += total
                    
                    if zero_ratio > self.alert_threshold:
                        problematic_layers.append((name, zero_ratio))
            
            if total_attention_params > 0:
                overall_ratio = total_attention_zeros / total_attention_params
                self.attention_history.append(overall_ratio)
                
                # Só alertar se realmente problemático
                if overall_ratio > self.alert_threshold:
                    self.alerts_sent += 1
                    
                    if self.verbose >= 1:
                        print(f"\\n🚨 ATTENTION BIAS ALERT - Step {self.num_timesteps}")
                        print(f"   Overall attention bias zeros: {overall_ratio:.1%}")
                        print(f"   Problematic layers: {len(problematic_layers)}")
                        
                        for layer_name, ratio in problematic_layers:
                            print(f"      {layer_name}: {ratio:.1%} zeros")
                        
                        # Sugerir ação apenas se muito problemático
                        if overall_ratio > 0.7:
                            print(f"   💡 SUGESTÃO: Considerar reativar Runtime Attention Bias Fixer")
                        else:
                            print(f"   💡 MONITORANDO: Ainda dentro de limites aceitáveis")
                
                # Log silencioso para histórico
                elif self.verbose >= 2:
                    print(f"🔍 Attention Monitor - Step {self.num_timesteps}: {overall_ratio:.1%} zeros")
        
        except Exception as e:
            if self.verbose >= 1:
                print(f"⚠️ Attention Monitor error: {e}")
    
    def _on_training_end(self) -> None:
        if self.verbose >= 1 and self.attention_history:
            avg_zeros = np.mean(self.attention_history)
            max_zeros = np.max(self.attention_history)
            
            print(f"\\n🔍 ATTENTION MONITOR - RELATÓRIO FINAL")
            print(f"   Checks realizados: {len(self.attention_history)}")
            print(f"   Média de zeros: {avg_zeros:.1%}")
            print(f"   Máximo de zeros: {max_zeros:.1%}")
            print(f"   Alertas enviados: {self.alerts_sent}")
            
            if max_zeros < 0.4:
                print(f"   ✅ ATTENTION BIAS SAUDÁVEL")
            elif max_zeros < 0.6:
                print(f"   ⚠️ ATTENTION BIAS MODERADO")
            else:
                print(f"   🚨 ATTENTION BIAS PROBLEMÁTICO")

def create_lightweight_attention_monitor(check_frequency: int = 2000,
                                        alert_threshold: float = 0.5,
                                        verbose: int = 1):
    """🔍 Criar monitor leve para attention bias"""
    return LightweightAttentionMonitor(
        check_frequency=check_frequency,
        alert_threshold=alert_threshold,
        verbose=verbose
    )

if __name__ == "__main__":
    print("🔍 LIGHTWEIGHT ATTENTION MONITOR")
    print("=" * 50)
    print("💡 Monitor leve para attention bias")
    print("💡 Não corrige agressivamente")
    print("💡 Só alerta se realmente problemático (>50% zeros)")
    print("💡 Frequência baixa (2000 steps)")
    print("💡 Impacto mínimo na performance")