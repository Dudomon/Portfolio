#!/usr/bin/env python3
"""
🛡️ LSTM PROTECTION CALLBACK
Protege LSTMs contra zeros críticos durante treinamento
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LSTMProtectionCallback(BaseCallback):
    """
    🛡️ Callback que monitora e protege LSTMs contra zeros críticos
    """
    
    def __init__(self, check_frequency: int = 100, verbose: int = 0):  # Mais agressivo
        super().__init__(verbose)
        self.check_frequency = check_frequency
        self.protection_count = 0
        
    def _on_step(self) -> bool:
        # Verificar a cada check_frequency steps
        if self.n_calls % self.check_frequency == 0:
            self._protect_lstms()
        return True
    
    def _protect_lstms(self):
        """Proteger todos os LSTMs contra zeros críticos"""
        
        if not hasattr(self.model, 'policy'):
            return
            
        policy = self.model.policy
        protected_count = 0
        lstm_found = 0
        
        # BUSCA ESPECÍFICA PARA V8Heritage LSTMs
        v8_neural_arch = None
        if hasattr(policy, 'neural_architecture'):
            v8_neural_arch = policy.neural_architecture
        
        if v8_neural_arch:
            v8_lstm_attrs = ['actor_lstm', 'critic_lstm']
            for lstm_name in v8_lstm_attrs:
                if hasattr(v8_neural_arch, lstm_name):
                    lstm_module = getattr(v8_neural_arch, lstm_name)
                    if isinstance(lstm_module, nn.LSTM):
                        lstm_found += 1
                        
                        # Verificar weight_hh dos LSTMs V8Heritage
                        for param_name, param in lstm_module.named_parameters():
                            if 'weight_hh' in param_name:
                                zeros_count = (param.data.abs() < 1e-8).sum().item()
                                total_params = param.data.numel()
                                zero_ratio = zeros_count / total_params
                                
                                if zero_ratio > 0.50:  # AGRESSIVO: 50% zeros = emergência
                                    with torch.no_grad():
                                        print(f"🚨 EMERGÊNCIA V8HERITAGE: {lstm_name}.{param_name} ({zero_ratio*100:.1f}% zeros)")
                                        
                                        # ESTRATÉGIA V7 COMPROVADA (não √2!)
                                        if 'weight_hh' in param_name:
                                            nn.init.orthogonal_(param, gain=1.0)  # V7 strategy
                                        elif 'weight_ih' in param_name:
                                            nn.init.xavier_uniform_(param)  # V7 strategy
                                        elif 'bias' in param_name:
                                            nn.init.zeros_(param)
                                            if param.size(0) >= 4:
                                                hidden_size = param.size(0) // 4
                                                param.data[hidden_size:2*hidden_size].fill_(1.0)  # Forget gate
                                        
                                        # Verificação final
                                        final_zeros = (param.data.abs() < 1e-8).sum().item()
                                        final_ratio = final_zeros / param.data.numel()
                                        print(f"   🔧 RESSUSCITADO: {lstm_name}.{param_name} → {final_ratio*100:.1f}% zeros")
                                        protected_count += 1
                                        self.protection_count += 1
        
        # Procurar LSTMs padrão também (fallback)
        for name, module in policy.named_modules():
            if isinstance(module, nn.LSTM) and 'neural_architecture' not in name:
                lstm_found += 1
                for param_name, param in module.named_parameters():
                    if 'weight_hh' in param_name:
                        # Verificar se há zeros críticos
                        zeros_count = (param.data.abs() < 1e-8).sum().item()
                        total_params = param.data.numel()
                        zero_ratio = zeros_count / total_params
                        
                        if zero_ratio > 0.10:  # Production threshold: 10%
                            with torch.no_grad():
                                print(f"🚨 PROTEÇÃO LSTM: {name}.{param_name} ({zero_ratio*100:.1f}% zeros)")
                                
                                # Reinicializar com proteção
                                nn.init.orthogonal_(param, gain=np.sqrt(2.0))
                                
                                # Garantir que nenhum peso seja zero
                                zero_mask = param.data.abs() < 1e-6
                                if zero_mask.any():
                                    param.data[zero_mask] = torch.randn_like(param.data[zero_mask]) * 0.01
                                
                                # Verificação final
                                final_zeros = (param.data.abs() < 1e-8).sum().item()
                                if final_zeros == 0:
                                    print(f"   ✅ {name}.{param_name}: Proteção aplicada!")
                                    protected_count += 1
                                    self.protection_count += 1
        
        if protected_count > 0:
            print(f"🛡️ PROTEÇÃO: {protected_count} LSTMs protegidos (total: {self.protection_count})")