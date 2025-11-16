#!/usr/bin/env python3
"""
🔥 LSTM Gradient Keeper - Mantém weight_hh_l0 vivo à força
"""

import torch
import torch.nn as nn

class LSTMGradientKeeper:
    """
    🔥 Classe que mantém gradientes do LSTM weight_hh_l0 sempre vivos
    """
    
    def __init__(self, lstm_module, min_grad_threshold=1e-5):
        self.lstm_module = lstm_module
        self.min_grad_threshold = min_grad_threshold
        self.gradient_history = []
        self.intervention_count = 0
        
        # Registrar hook permanente
        self._register_gradient_keeper_hook()
    
    def _register_gradient_keeper_hook(self):
        """Registra hook que intercepta todos os gradientes do weight_hh_l0"""
        
        def gradient_keeper_hook(grad):
            """Hook que NUNCA permite gradientes zero no weight_hh_l0"""
            
            if grad is None:
                return None
            
            # Análise dos gradientes
            zero_mask = torch.abs(grad) < self.min_grad_threshold
            zero_ratio = zero_mask.float().mean().item()
            
            # Se >50% são zeros, intervenção OBRIGATÓRIA
            if zero_ratio > 0.5:
                self.intervention_count += 1
                
                # INTERVENÇÃO 1: Noise injection estruturado
                hidden_size = grad.shape[0] // 4
                
                for gate_idx in range(4):
                    start_idx = gate_idx * hidden_size
                    end_idx = (gate_idx + 1) * hidden_size
                    
                    gate_mask = zero_mask[start_idx:end_idx]
                    
                    if gate_mask.sum() > 0:
                        # Noise específico por gate
                        if gate_idx == 1:  # Forget gate
                            noise_scale = 3e-4
                        elif gate_idx == 2:  # Cell gate  
                            noise_scale = 2e-4
                        else:
                            noise_scale = 1e-4
                        
                        # Aplicar noise com estrutura ortogonal
                        noise = torch.randn_like(grad[start_idx:end_idx][gate_mask]) * noise_scale
                        grad[start_idx:end_idx][gate_mask] = noise
                
                # INTERVENÇÃO 2: Guarantee minimum gradient magnitude
                remaining_small = (torch.abs(grad) > self.min_grad_threshold) & (torch.abs(grad) < 1e-4)
                if remaining_small.sum() > 0:
                    grad[remaining_small] *= 10.0
            
            # INTERVENÇÃO 3: Sempre manter pelo menos 30% dos gradientes vivos
            if zero_ratio > 0.7:
                # Forçar ressuscitação dos 30% maiores gradientes zeros
                abs_grad = torch.abs(grad)
                flat_grad = grad.flatten()
                flat_abs = abs_grad.flatten()
                
                # Encontrar threshold para manter 30% vivos
                sorted_abs, indices = torch.sort(flat_abs, descending=True)
                threshold_idx = int(0.3 * len(flat_abs))
                min_alive_threshold = sorted_abs[threshold_idx].item()
                
                # Ressuscitar gradientes abaixo do threshold
                resurrection_mask = flat_abs <= min_alive_threshold
                if resurrection_mask.sum() > 0:
                    resurrection_noise = torch.randn_like(flat_grad[resurrection_mask]) * 2e-4
                    flat_grad[resurrection_mask] = resurrection_noise
                    grad = flat_grad.reshape(grad.shape)
            
            # Log periódico
            if self.intervention_count % 100 == 0 and self.intervention_count > 0:
                print(f"🔥 LSTM Gradient Keeper - {self.intervention_count} intervenções")
                print(f"   Zeros antes: {zero_ratio*100:.1f}%")
                new_zero_ratio = (torch.abs(grad) < self.min_grad_threshold).float().mean().item()
                print(f"   Zeros depois: {new_zero_ratio*100:.1f}%")
            
            return grad
        
        # Registrar hook no weight_hh_l0
        if hasattr(self.lstm_module, 'weight_hh_l0'):
            self.lstm_module.weight_hh_l0.register_hook(gradient_keeper_hook)
            print("🔥 LSTM Gradient Keeper ATIVO - weight_hh_l0 protegido")
        else:
            print("⚠️ weight_hh_l0 não encontrado para Gradient Keeper")

def apply_lstm_gradient_keeper(policy, min_grad_threshold=1e-5):
    """Aplica LSTM Gradient Keeper na policy"""
    
    keepers = []
    
    # Aplicar no critic LSTM
    if hasattr(policy, 'v7_critic_lstm'):
        critic_keeper = LSTMGradientKeeper(policy.v7_critic_lstm, min_grad_threshold)
        keepers.append(critic_keeper)
        print("🔥 Gradient Keeper aplicado no v7_critic_lstm")
    
    # Aplicar no actor LSTM também (preventivo)  
    if hasattr(policy, 'v7_actor_lstm'):
        actor_keeper = LSTMGradientKeeper(policy.v7_actor_lstm, min_grad_threshold)
        keepers.append(actor_keeper)
        print("🔥 Gradient Keeper aplicado no v7_actor_lstm")
    
    return keepers

if __name__ == "__main__":
    print("🔥 LSTM Gradient Keeper - Mantém weight_hh_l0 vivo à força")
    print("Use: apply_lstm_gradient_keeper(policy) após criar a policy")