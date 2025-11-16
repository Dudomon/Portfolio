#!/usr/bin/env python3
"""
🎯 CALLBACK PARA LEARNING RATES SEPARADOS - Actor e Critic
Integração com RecurrentPPO para usar optimizers independentes
"""

import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

class SeparateLRCallback(BaseCallback):
    """
    Callback para aplicar learning rates separados ao actor e critic
    """
    
    def __init__(self, verbose=1):
        super().__init__(verbose)
        self.actor_losses = []
        self.critic_losses = []
        
    def _on_training_start(self) -> None:
        """Configurar optimizers separados no início do treinamento"""
        
        # Verificar se a policy suporta optimizers separados
        if hasattr(self.model.policy, 'use_separate_optimizers'):
            print("🎯 USANDO OPTIMIZERS SEPARADOS!")
            
            # Sobrescrever o método de otimização do PPO
            self._override_ppo_optimization()
            
        else:
            print("⚠️ Policy não suporta optimizers separados, usando padrão")
            
    def _override_ppo_optimization(self):
        """Sobrescrever método de otimização do PPO para usar optimizers separados"""
        
        # Salvar método original
        original_train = self.model.train
        
        def custom_train():
            """Treinamento customizado com optimizers separados"""
            
            # Obter optimizers da policy
            actor_optimizer, critic_optimizer = self.model.policy.get_actor_critic_optimizers()
            
            # Iterar sobre o buffer
            for rollout_data in self.model.rollout_buffer.get(self.model.batch_size):
                
                # Avaliar ações com a policy atual
                values, log_prob, entropy = self.model.policy.evaluate_actions(
                    rollout_data.observations, 
                    rollout_data.actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts
                )
                
                # Calcular vantagens
                advantages = rollout_data.advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # === ACTOR LOSS ===
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * torch.clamp(
                    ratio, 1 - self.model.clip_range, 1 + self.model.clip_range
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()
                
                # Entropy loss
                entropy_loss = -torch.mean(entropy)
                
                # Total actor loss
                actor_loss = policy_loss + self.model.ent_coef * entropy_loss
                
                # === CRITIC LOSS ===
                values_pred = values.flatten()
                value_loss = torch.nn.functional.mse_loss(rollout_data.returns, values_pred)
                
                # === OTIMIZAÇÃO SEPARADA ===
                
                # Otimizar Actor
                actor_optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.policy.parameters() if p.requires_grad and 'actor' in str(p)], 
                    self.model.max_grad_norm
                )
                actor_optimizer.step()
                
                # Otimizar Critic
                critic_optimizer.zero_grad()
                (self.model.vf_coef * value_loss).backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.policy.parameters() if p.requires_grad and 'critic' in str(p)], 
                    self.model.max_grad_norm
                )
                critic_optimizer.step()
                
                # Salvar métricas
                self.actor_losses.append(actor_loss.item())
                self.critic_losses.append(value_loss.item())
        
        # Substituir método de treinamento
        self.model.train = custom_train
        
    def _on_step(self) -> bool:
        """Log das métricas separadas"""
        
        if len(self.actor_losses) > 0 and self.num_timesteps % 1000 == 0:
            avg_actor_loss = np.mean(self.actor_losses[-100:])
            avg_critic_loss = np.mean(self.critic_losses[-100:])
            
            if hasattr(self.model.policy, 'current_actor_lr'):
                actor_lr = self.model.policy.current_actor_lr
                critic_lr = self.model.policy.current_critic_lr
                
                print(f"🎯 Step {self.num_timesteps}: Actor Loss={avg_actor_loss:.6f} (LR={actor_lr:.2e}), "
                      f"Critic Loss={avg_critic_loss:.6f} (LR={critic_lr:.2e})")
                
        return True

def create_separate_lr_callback():
    """Factory function para criar o callback"""
    return SeparateLRCallback(verbose=1)