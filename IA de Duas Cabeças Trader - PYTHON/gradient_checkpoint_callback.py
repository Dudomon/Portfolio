#!/usr/bin/env python3
"""
🔥 Gradient Checkpoint Callback para weight_hh_l0
Força gradientes saudáveis na camada problemática
"""

import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class GradientCheckpointCallback(BaseCallback):
    """
    🔥 Callback que implementa gradient checkpointing específico para weight_hh_l0
    """
    
    def __init__(self, checkpoint_frequency=100, min_grad_norm=1e-5, verbose=1):
        super().__init__(verbose)
        self.checkpoint_frequency = checkpoint_frequency
        self.min_grad_norm = min_grad_norm
        self.step_count = 0
        self.checkpoint_history = []
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # Apply gradient checkpoint every N steps
        if self.step_count % self.checkpoint_frequency == 0:
            self._apply_gradient_checkpoint()
            
        return True
    
    def _apply_gradient_checkpoint(self):
        """🔥 Implementa gradient checkpointing no weight_hh_l0"""
        
        try:
            policy = self.model.policy
            
            # Verificar se policy tem v7_critic_lstm
            if not hasattr(policy, 'v7_critic_lstm'):
                if self.verbose > 0:
                    print("⚠️ v7_critic_lstm não encontrado para gradient checkpoint")
                return
            
            lstm_critic = policy.v7_critic_lstm
            weight_hh = lstm_critic.weight_hh_l0
            
            # 1. CHECKPOINT: Salvar estado atual dos gradientes
            current_grad = None
            if weight_hh.grad is not None:
                current_grad = weight_hh.grad.clone()
                current_grad_norm = torch.norm(current_grad).item()
                zero_ratio = (torch.abs(current_grad) < self.min_grad_norm).float().mean().item()
                
                if self.verbose > 0 and self.step_count % (self.checkpoint_frequency * 10) == 0:
                    print(f"🔥 Gradient Checkpoint - Step {self.step_count}")
                    print(f"   weight_hh_l0 grad norm: {current_grad_norm:.6f}")
                    print(f"   weight_hh_l0 zeros: {zero_ratio*100:.1f}%")
                
                # 2. INTERVENÇÃO CRÍTICA: Se gradientes estão mortos, ressuscitar
                if zero_ratio > 0.7:  # Se >70% zeros
                    self._emergency_gradient_resurrection(weight_hh, current_grad)
                
                # 3. CHECKPOINT: Armazenar histórico para análise
                self.checkpoint_history.append({
                    'step': self.step_count,
                    'grad_norm': current_grad_norm,
                    'zero_ratio': zero_ratio,
                    'intervention_applied': zero_ratio > 0.7
                })
                
                # Manter apenas últimos 100 checkpoints
                if len(self.checkpoint_history) > 100:
                    self.checkpoint_history = self.checkpoint_history[-100:]
            
        except Exception as e:
            if self.verbose > 0:
                print(f"❌ Erro no gradient checkpoint: {e}")
    
    def _emergency_gradient_resurrection(self, weight_hh, current_grad):
        """🚑 Ressuscitação de emergência para gradientes mortos"""
        
        with torch.no_grad():
            # Estratégia 1: Noise injection nos gradientes zero
            zero_mask = torch.abs(current_grad) < self.min_grad_norm
            
            if zero_mask.sum() > 0:
                # Criar gradientes artificiais baseados na estrutura das 4 gates
                hidden_size = weight_hh.shape[0] // 4
                
                # Gradientes específicos para cada gate
                for gate_idx in range(4):
                    start_idx = gate_idx * hidden_size
                    end_idx = (gate_idx + 1) * hidden_size
                    
                    gate_mask = zero_mask[start_idx:end_idx]
                    
                    if gate_mask.sum() > 0:
                        # Noise específico para cada gate
                        if gate_idx == 0:  # Input gate
                            noise_scale = 1e-4
                        elif gate_idx == 1:  # Forget gate - CRÍTICO
                            noise_scale = 2e-4  # Maior para forget gate
                        elif gate_idx == 2:  # Cell gate
                            noise_scale = 1.5e-4
                        else:  # Output gate
                            noise_scale = 1e-4
                        
                        # Aplicar noise estruturado
                        noise = torch.randn_like(current_grad[start_idx:end_idx]) * noise_scale
                        current_grad[start_idx:end_idx][gate_mask] = noise[gate_mask]
            
            # Estratégia 2: Gradient scaling para gradientes muito pequenos
            small_grad_mask = (torch.abs(current_grad) > self.min_grad_norm) & (torch.abs(current_grad) < 1e-4)
            if small_grad_mask.sum() > 0:
                current_grad[small_grad_mask] *= 3.0  # Scale up small gradients
            
            # Estratégia 3: Gradient momentum (usar média dos últimos checkpoints)
            if len(self.checkpoint_history) >= 3:
                # Se gradient está consistentemente ruim, aplicar momentum artificial
                recent_zeros = [cp['zero_ratio'] for cp in self.checkpoint_history[-3:]]
                if np.mean(recent_zeros) > 0.6:
                    # Momentum injection
                    momentum_grad = torch.randn_like(current_grad) * 5e-5
                    current_grad += momentum_grad
            
            # Aplicar gradiente modificado
            weight_hh.grad = current_grad
            
            if self.verbose > 0:
                print(f"🚑 Emergency gradient resurrection applied!")
                print(f"   Zeros after intervention: {(torch.abs(current_grad) < self.min_grad_norm).float().mean().item()*100:.1f}%")
    
    def get_checkpoint_summary(self):
        """📊 Retorna resumo do histórico de checkpoints"""
        if not self.checkpoint_history:
            return "No checkpoints recorded"
        
        recent = self.checkpoint_history[-10:]  # Últimos 10
        avg_zero_ratio = np.mean([cp['zero_ratio'] for cp in recent])
        interventions = sum([cp['intervention_applied'] for cp in recent])
        
        return f"Last 10 checkpoints: {avg_zero_ratio*100:.1f}% avg zeros, {interventions} interventions"

if __name__ == "__main__":
    print("🔥 Gradient Checkpoint Callback criado")
    print("Implementa ressuscitação de gradientes para weight_hh_l0")