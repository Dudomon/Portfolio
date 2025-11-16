#!/usr/bin/env python3
"""
🚨 LSTM RESCUE CALLBACK - RESGATE AUTOMÁTICO DE LSTMs COM >80% ZEROS
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class LSTMRescueCallback(BaseCallback):
    """Callback para resgatar LSTMs com >80% zeros extremos"""
    
    def __init__(self, 
                 rescue_threshold: float = 0.80,
                 check_frequency: int = 1000,
                 verbose: int = 1):
        super().__init__(verbose)
        self.rescue_threshold = rescue_threshold
        self.check_frequency = check_frequency
        self.rescues_performed = 0
        
    def _on_step(self) -> bool:
        """Verificar e resgatar LSTMs se necessário"""
        
        if self.num_timesteps % self.check_frequency == 0:
            self._check_and_rescue_lstms()
        
        return True
    
    def _check_and_rescue_lstms(self):
        """Verificar saúde das LSTMs e resgatar se necessário"""
        try:
            if not hasattr(self.model, 'policy'):
                return
            
            lstm_problems = []
            
            # Verificar todas as LSTMs
            for name, module in self.model.policy.named_modules():
                if isinstance(module, (nn.LSTM, nn.GRU)):
                    zeros_ratio = self._calculate_lstm_zeros(module)
                    
                    if zeros_ratio > self.rescue_threshold:
                        lstm_problems.append((name, module, zeros_ratio))
            
            # Resgatar LSTMs problemáticas
            if lstm_problems:
                print(f"🚨 LSTM RESCUE ATIVADO - Step {self.num_timesteps}")
                
                for name, module, zeros_ratio in lstm_problems:
                    print(f"   🔧 Resgatando {name}: {zeros_ratio:.1%} zeros")
                    self._rescue_lstm(module)
                    self.rescues_performed += 1
                
                print(f"✅ {len(lstm_problems)} LSTMs resgatadas!")
        
        except Exception as e:
            print(f"❌ Erro no LSTM rescue: {e}")
    
    def _calculate_lstm_zeros(self, module: nn.Module) -> float:
        """Calcular porcentagem de zeros extremos em LSTM"""
        total_zeros = 0
        total_params = 0
        
        for param_name, param in module.named_parameters():
            if 'bias' in param_name and param.grad is not None:
                grad_array = param.grad.detach().cpu().numpy()
                zeros = np.sum(np.abs(grad_array) < 1e-8)
                total_zeros += zeros
                total_params += len(grad_array)
        
        if total_params == 0:
            return 0.0
        
        return total_zeros / total_params
    
    def _rescue_lstm(self, module: nn.Module):
        """Resgatar LSTM aplicando nova inicialização"""
        
        for param_name, param in module.named_parameters():
            if 'bias' in param_name:
                # Aplicar nossa inicialização sem zeros
                if isinstance(module, nn.LSTM):
                    n = param.size(0)
                    # Inicializar TODOS os gates com valores não-zero
                    nn.init.uniform_(param.data[:n//4], -0.05, 0.05)      # Input gate
                    nn.init.uniform_(param.data[n//4:n//2], 0.8, 1.2)     # Forget gate
                    nn.init.uniform_(param.data[n//2:3*n//4], -0.05, 0.05) # Cell gate  
                    nn.init.uniform_(param.data[3*n//4:], -0.05, 0.05)    # Output gate
                elif isinstance(module, nn.GRU):
                    n = param.size(0)
                    # Inicializar TODOS os gates com valores não-zero
                    nn.init.uniform_(param.data[:n//3], -0.05, 0.05)      # Reset gate
                    nn.init.uniform_(param.data[n//3:2*n//3], 0.7, 1.0)   # Update gate
                    nn.init.uniform_(param.data[2*n//3:], -0.05, 0.05)    # New gate
            
            elif 'weight' in param_name:
                # Re-inicializar pesos também
                if 'weight_ih' in param_name:
                    nn.init.orthogonal_(param.data)
                elif 'weight_hh' in param_name:
                    nn.init.orthogonal_(param.data)


def create_lstm_rescue_callback(
    rescue_threshold: float = 0.80,
    check_frequency: int = 1000,
    verbose: int = 1
) -> LSTMRescueCallback:
    """Factory function para criar LSTM rescue callback"""
    return LSTMRescueCallback(
        rescue_threshold=rescue_threshold,
        check_frequency=check_frequency,
        verbose=verbose
    )

if __name__ == "__main__":
    print("🚨 LSTM Rescue Callback - Sistema de resgate automático para LSTMs")