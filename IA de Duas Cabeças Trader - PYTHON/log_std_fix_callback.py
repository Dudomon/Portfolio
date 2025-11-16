#!/usr/bin/env python3
"""
🔧 LOG_STD FIX CALLBACK - Corrige log_std durante treinamento

Aplica fix contínuo para log_std que fica zerando
"""

from stable_baselines3.common.callbacks import BaseCallback

class LogStdFixCallback(BaseCallback):
    """🔧 Callback para corrigir log_std continuamente"""
    
    def __init__(self, fix_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.fix_freq = fix_freq
        self.step_count = 0
        
    def _on_step(self) -> bool:
        """🔧 Aplicar fix a cada step"""
        self.step_count += 1
        
        if self.step_count % self.fix_freq != 0:
            return True
            
        try:
            # Aplicar fix para log_std
            fixed_params = 0
            
            for name, param in self.model.policy.named_parameters():
                if 'log_std' in name.lower():
                    if param is not None and param.data is not None:
                        # Verificar se está zerando
                        zero_ratio = (param.data.abs() < 1e-6).float().mean().item()
                        
                        if zero_ratio > 0.8:  # >80% zeros
                            param.data.fill_(-0.5)  # Resetar para -0.5
                            fixed_params += 1
                            if self.verbose >= 1:
                                print(f"🔧 [LOG_STD FIX] {name} resetado: -0.5")
            
            # Verificar se action_dist existe e tem log_std
            if hasattr(self.model.policy, 'action_dist'):
                action_dist = self.model.policy.action_dist
                if hasattr(action_dist, 'log_std') and action_dist.log_std is not None:
                    zero_ratio = (action_dist.log_std.data.abs() < 1e-6).float().mean().item()
                    
                    if zero_ratio > 0.8:  # >80% zeros
                        action_dist.log_std.data.fill_(-0.5)
                        fixed_params += 1
                        if self.verbose >= 1:
                            print(f"🔧 [LOG_STD FIX] action_dist.log_std resetado: -0.5")
            
            if fixed_params > 0 and self.verbose >= 1:
                print(f"🔧 [LOG_STD FIX] {fixed_params} parâmetros log_std corrigidos")
                
        except Exception as e:
            if self.verbose >= 1:
                print(f"❌ ERRO LogStdFix: {e}")
        
        return True

# Uso: model.learn(callback=LogStdFixCallback())