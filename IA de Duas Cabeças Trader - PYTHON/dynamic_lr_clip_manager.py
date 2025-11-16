#!/usr/bin/env python3
"""
🎯 DYNAMIC LR & CLIP MANAGER - V7 Optimizer
Sistema adaptativo para controlar learning rates e clip range
"""

import numpy as np
from typing import Dict, Tuple

class DynamicLRClipManager:
    """
    🧠 Gerenciador inteligente de LR e Clip Range
    """
    
    def __init__(self):
        # Learning Rate Base Values
        self.actor_base_lr = 1.0e-4    # 2x atual (descongelar)
        self.critic_base_lr = 3.0e-4   # 1.5x atual (manter momentum)
        
        # Warmup configs
        self.actor_warmup_steps = 10000   # Warmup mais longo (estava congelado)
        self.critic_warmup_steps = 5000   # Warmup mais rápido (já funcionando)
        
        # Decay rates
        self.actor_decay_rate = 0.98    # Decay mais agressivo (controlar clip)
        self.critic_decay_rate = 0.995  # Decay suave (preservar exp_var)
        self.decay_interval = 10000     # Aplicar decay a cada 10k steps
        
        # Clip range configs
        self.base_clip_range = 0.2
        self.target_clip_fraction_min = 0.15
        self.target_clip_fraction_max = 0.25
        self.clip_adjustment_rate = 0.02  # 2% por ajuste
        
        # History tracking
        self.clip_fraction_history = []
        self.exp_var_history = []
        
    def get_learning_rates(self, current_step: int) -> Tuple[float, float]:
        """
        Calcula LRs com warmup e decay diferenciados
        
        Returns:
            (actor_lr, critic_lr)
        """
        # ACTOR LR - Warmup longo + decay agressivo
        if current_step < self.actor_warmup_steps:
            # Linear warmup com mínimo inicial
            warmup_progress = current_step / self.actor_warmup_steps
            warmup_factor = 0.2 + 0.8 * warmup_progress  # Começa em 20% do LR
            actor_lr = self.actor_base_lr * warmup_factor
        else:
            # Exponential decay
            decay_steps = (current_step - self.actor_warmup_steps) // self.decay_interval
            actor_lr = self.actor_base_lr * (self.actor_decay_rate ** decay_steps)
        
        # CRITIC LR - Warmup rápido + decay suave
        if current_step < self.critic_warmup_steps:
            # Linear warmup com mínimo inicial
            warmup_progress = current_step / self.critic_warmup_steps
            warmup_factor = 0.3 + 0.7 * warmup_progress  # Começa em 30% do LR (critic precisa mais)
            critic_lr = self.critic_base_lr * warmup_factor
        else:
            # Exponential decay
            decay_steps = (current_step - self.critic_warmup_steps) // self.decay_interval
            critic_lr = self.critic_base_lr * (self.critic_decay_rate ** decay_steps)
        
        # Minimum LRs
        actor_lr = max(actor_lr, 1e-6)
        critic_lr = max(critic_lr, 5e-6)  # Critic nunca muito baixo
        
        return actor_lr, critic_lr
    
    def get_adaptive_clip_range(self, current_clip_fraction: float, 
                               current_exp_var: float) -> float:
        """
        Ajusta clip range baseado em métricas atuais
        """
        # Adicionar ao histórico
        self.clip_fraction_history.append(current_clip_fraction)
        self.exp_var_history.append(current_exp_var)
        
        # Manter histórico limitado
        if len(self.clip_fraction_history) > 100:
            self.clip_fraction_history.pop(0)
            self.exp_var_history.pop(0)
        
        # Calcular clip range adaptativo
        clip_range = self.base_clip_range
        
        # Ajustar baseado no clip fraction - MAIS AGRESSIVO
        if current_clip_fraction > self.target_clip_fraction_max:
            # Clip fraction muito alto - reduzir clip range agressivamente
            excess_ratio = current_clip_fraction / self.target_clip_fraction_max
            if excess_ratio > 1.5:  # Muito acima (ex: 0.45 vs 0.25)
                clip_range *= 0.8  # Redução agressiva de 20%
            else:
                clip_range *= (2.0 - excess_ratio)  # Redução proporcional
            
        elif current_clip_fraction < self.target_clip_fraction_min:
            # Clip fraction muito baixo - aumentar clip range
            if current_clip_fraction < 0.05:  # Quase zero
                clip_range *= 1.2  # Aumento agressivo de 20%
            else:
                deficit_ratio = self.target_clip_fraction_min / (current_clip_fraction + 0.01)
                clip_range *= min(deficit_ratio, 1.3)  # Aumento proporcional limitado
        
        # Ajuste adicional baseado em explained variance
        if len(self.exp_var_history) > 10:
            recent_exp_var = np.mean(self.exp_var_history[-10:])
            
            if recent_exp_var < 0:
                # Exp var negativo - ser mais conservador
                clip_range *= 0.95
            elif recent_exp_var > 0.5:
                # Exp var muito bom - pode ser mais agressivo
                clip_range *= 1.02
        
        # Limites de segurança
        clip_range = np.clip(clip_range, 0.1, 0.3)
        
        return clip_range
    
    def get_status_report(self, current_step: int) -> Dict:
        """
        Relatório do estado atual
        """
        actor_lr, critic_lr = self.get_learning_rates(current_step)
        
        # Métricas recentes
        recent_clip = np.mean(self.clip_fraction_history[-10:]) if self.clip_fraction_history else 0
        recent_exp_var = np.mean(self.exp_var_history[-10:]) if self.exp_var_history else 0
        
        return {
            'current_step': current_step,
            'actor_lr': actor_lr,
            'critic_lr': critic_lr,
            'lr_ratio': critic_lr / actor_lr if actor_lr > 0 else 0,
            'recent_clip_fraction': recent_clip,
            'recent_exp_var': recent_exp_var,
            'warmup_status': {
                'actor': 'complete' if current_step >= self.actor_warmup_steps else f'{current_step/self.actor_warmup_steps:.1%}',
                'critic': 'complete' if current_step >= self.critic_warmup_steps else f'{current_step/self.critic_warmup_steps:.1%}'
            }
        }

# Integração com daytrader.py
def create_lr_schedule_callback(manager: DynamicLRClipManager):
    """
    Callback para integrar com RecurrentPPO
    """
    def lr_schedule(progress_remaining: float) -> float:
        # Calcular step atual (aproximado)
        total_timesteps = 10_000_000  # Ajustar conforme necessário
        current_step = int((1 - progress_remaining) * total_timesteps)
        
        # Obter LRs
        actor_lr, critic_lr = manager.get_learning_rates(current_step)
        
        # Retornar actor LR (critic será ajustado separadamente)
        return actor_lr / manager.actor_base_lr  # Normalizado
    
    return lr_schedule

# Exemplo de uso no daytrader.py:
"""
# Criar manager
lr_clip_manager = DynamicLRClipManager()

# Criar modelo com LR schedule
model = RecurrentPPO(
    policy=TwoHeadV7Intuition,
    env=train_env,
    learning_rate=lr_schedule_callback(lr_clip_manager),
    clip_range=lambda p: lr_clip_manager.get_adaptive_clip_range(
        model.logger.name_to_value.get('train/clip_fraction', 0.2),
        model.logger.name_to_value.get('train/explained_variance', 0)
    ),
    **other_params
)

# Ajustar optimizer para LRs diferenciados
def setup_differential_optimizer(model, manager):
    actor_params = []
    critic_params = []
    
    for name, param in model.policy.named_parameters():
        if 'critic' in name.lower() or 'value' in name.lower():
            critic_params.append(param)
        else:
            actor_params.append(param)
    
    # Criar optimizer com grupos
    model.policy.optimizer = torch.optim.Adam([
        {'params': actor_params, 'lr': manager.actor_base_lr},
        {'params': critic_params, 'lr': manager.critic_base_lr}
    ])
"""

if __name__ == "__main__":
    # Teste do sistema
    manager = DynamicLRClipManager()
    
    print("🎯 DYNAMIC LR & CLIP MANAGER - Simulação")
    print("=" * 60)
    
    # Simular diferentes steps
    test_steps = [0, 2500, 5000, 10000, 50000, 100000, 500000]
    
    for step in test_steps:
        actor_lr, critic_lr = manager.get_learning_rates(step)
        
        print(f"\nStep {step:,}:")
        print(f"  Actor LR:  {actor_lr:.2e}")
        print(f"  Critic LR: {critic_lr:.2e}")
        print(f"  Ratio:     {critic_lr/actor_lr:.1f}x")
    
    print("\n📊 Clip Range Adaptativo:")
    
    # Simular diferentes clip fractions
    test_clips = [(0.1, 0.2), (0.2, 0.3), (0.35, 0.25), (0.45, 0.1)]
    
    for clip_frac, exp_var in test_clips:
        clip_range = manager.get_adaptive_clip_range(clip_frac, exp_var)
        print(f"  Clip={clip_frac:.2f}, ExpVar={exp_var:.2f} → Range={clip_range:.3f}")