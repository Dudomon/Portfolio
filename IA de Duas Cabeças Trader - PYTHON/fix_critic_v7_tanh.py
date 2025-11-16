#!/usr/bin/env python3
"""
🎯 FIX CRÍTICO V7 - APÓS MUDANÇAS TANH
Correções específicas para melhorar explained_variance
"""

import sys
import os
sys.path.append("D:/Projeto")

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

class CriticFixV7:
    """
    🔧 CRÍTICO V7 FIX - Correções específicas para problema pós-tanh
    """
    
    def __init__(self):
        self.fixes_applied = []
    
    def diagnose_critic_issues(self, model):
        """Diagnóstico dos problemas do crítico"""
        print("🔍 DIAGNÓSTICO CRÍTICO V7")
        print("=" * 50)
        
        issues = []
        
        # 1. Verificar gradientes do crítico
        critic_grads = []
        for name, param in model.policy.named_parameters():
            if 'critic' in name.lower() or 'value' in name.lower():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    critic_grads.append((name, grad_norm))
        
        if not critic_grads:
            issues.append("❌ Sem gradientes no crítico")
        else:
            avg_grad = np.mean([g[1] for g in critic_grads])
            print(f"📊 Gradientes crítico - Média: {avg_grad:.6f}")
            
            if avg_grad < 1e-6:
                issues.append("⚠️ Gradientes muito pequenos")
            elif avg_grad > 10:
                issues.append("⚠️ Gradientes muito grandes")
        
        # 2. Verificar ativações tanh
        tanh_saturations = 0
        total_tanh = 0
        
        def check_tanh_saturation(module):
            nonlocal tanh_saturations, total_tanh
            if isinstance(module, nn.Tanh):
                total_tanh += 1
                # Simular entrada para verificar saturação
                with torch.no_grad():
                    test_input = torch.randn(32, 128)  # Batch típico
                    output = module(test_input)
                    saturated = (torch.abs(output) > 0.95).float().mean()
                    if saturated > 0.8:  # 80% saturado
                        tanh_saturations += 1
        
        if hasattr(model.policy, 'mlp_extractor'):
            model.policy.mlp_extractor.apply(check_tanh_saturation)
        
        if total_tanh > 0:
            saturation_rate = tanh_saturations / total_tanh
            print(f"📊 Saturação Tanh: {saturation_rate:.2%} ({tanh_saturations}/{total_tanh})")
            if saturation_rate > 0.5:
                issues.append("🔥 Alta saturação de Tanh layers")
        
        # 3. Verificar learning rates
        if hasattr(model, 'lr_schedule'):
            current_lr = model.lr_schedule(model.num_timesteps)
            print(f"📊 Learning Rate atual: {current_lr}")
            if current_lr < 1e-6:
                issues.append("⚠️ Learning rate muito baixo")
        
        print(f"\n🎯 PROBLEMAS DETECTADOS:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        
        return issues
    
    def apply_critic_fixes(self):
        """Gerar código de correções para o crítico"""
        
        fixes = []
        
        # FIX 1: Learning Rate Diferenciado para Crítico
        fix_lr = """
# 🔧 FIX 1: LEARNING RATES DIFERENCIADOS
def create_optimizer_groups(policy):
    '''Criar grupos de otimização com LRs diferentes'''
    
    actor_params = []
    critic_params = []
    
    for name, param in policy.named_parameters():
        if 'critic' in name.lower() or 'value' in name.lower():
            critic_params.append(param)
        else:
            actor_params.append(param)
    
    # Crítico com LR 2-3x maior
    optimizer_groups = [
        {'params': actor_params, 'lr': 3e-4},      # LR normal para actor
        {'params': critic_params, 'lr': 1e-3}      # LR maior para crítico
    ]
    
    return optimizer_groups

# Aplicar no daytrader.py na criação do modelo:
# model.policy.optimizer = torch.optim.Adam(create_optimizer_groups(model.policy))
        """
        
        # FIX 2: Critic Architecture Improvements
        fix_arch = """
# 🔧 FIX 2: ARQUITETURA CRÍTICO MELHORADA
class ImprovedCriticHead(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        
        # Crítico mais robusto com skip connections
        self.value_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            
            nn.Linear(512, 256),
            nn.LayerNorm(256), 
            nn.LeakyReLU(0.1),
            nn.Dropout(0.05),
            
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            
            nn.Linear(128, 1)  # Output final
        )
        
        # Skip connection para estabilidade
        self.skip_connection = nn.Linear(input_dim, 1)
        
    def forward(self, features):
        main_output = self.value_net(features)
        skip_output = self.skip_connection(features)
        # Combinar com peso
        return 0.8 * main_output + 0.2 * skip_output
        """
        
        # FIX 3: Gradient Clipping Específico
        fix_grad = """
# 🔧 FIX 3: GRADIENT CLIPPING DIFERENCIADO
def apply_differential_grad_clipping(model):
    '''Aplicar clipping diferenciado por componente'''
    
    # Clipping mais conservador para crítico
    critic_params = []
    actor_params = []
    
    for name, param in model.policy.named_parameters():
        if param.grad is not None:
            if 'critic' in name.lower() or 'value' in name.lower():
                critic_params.append(param)
            else:
                actor_params.append(param)
    
    # Clipping diferenciado
    if critic_params:
        torch.nn.utils.clip_grad_norm_(critic_params, max_norm=0.5)  # Mais conservador
    if actor_params:
        torch.nn.utils.clip_grad_norm_(actor_params, max_norm=1.0)   # Normal
        
# Adicionar no callback de treinamento
        """
        
        # FIX 4: Value Loss Scaling
        fix_loss = """
# 🔧 FIX 4: VALUE LOSS SCALING ADAPTATIVO
class AdaptiveValueLossCallback:
    def __init__(self):
        self.value_loss_history = []
        self.exp_var_history = []
        self.loss_scale = 1.0
        
    def update_loss_scaling(self, value_loss, explained_variance):
        self.value_loss_history.append(value_loss)
        self.exp_var_history.append(explained_variance)
        
        # Manter histórico limitado
        if len(self.value_loss_history) > 100:
            self.value_loss_history.pop(0)
            self.exp_var_history.pop(0)
        
        # Ajustar scaling baseado na explained variance
        if len(self.exp_var_history) > 10:
            recent_exp_var = np.mean(self.exp_var_history[-10:])
            
            if recent_exp_var < -0.1:  # Muito negativo
                self.loss_scale = min(3.0, self.loss_scale * 1.1)  # Aumentar peso
            elif recent_exp_var > 0.1:  # Muito positivo
                self.loss_scale = max(0.5, self.loss_scale * 0.95)  # Reduzir peso
                
        return self.loss_scale

# Integrar no loop de treinamento
        """
        
        # FIX 5: Target Network para Crítico
        fix_target = """
# 🔧 FIX 5: TARGET NETWORK PARA ESTABILIDADE
class CriticWithTarget:
    def __init__(self, critic_network):
        self.main_critic = critic_network
        self.target_critic = copy.deepcopy(critic_network)
        self.update_freq = 1000  # Update target a cada 1000 steps
        self.tau = 0.005  # Soft update rate
        
    def update_target(self, step):
        if step % self.update_freq == 0:
            # Soft update
            for target_param, main_param in zip(
                self.target_critic.parameters(), 
                self.main_critic.parameters()
            ):
                target_param.data.copy_(
                    self.tau * main_param.data + (1.0 - self.tau) * target_param.data
                )
                
    def compute_target_values(self, next_states):
        with torch.no_grad():
            return self.target_critic(next_states)
        """
        
        fixes = [
            ("Learning Rates Diferenciados", fix_lr),
            ("Arquitetura Crítico Melhorada", fix_arch), 
            ("Gradient Clipping Diferenciado", fix_grad),
            ("Value Loss Scaling Adaptativo", fix_loss),
            ("Target Network para Estabilidade", fix_target)
        ]
        
        return fixes
    
    def generate_quick_fix_patch(self):
        """Gerar patch rápido para aplicar agora"""
        
        patch = """
# 🚀 PATCH RÁPIDO - CRÍTICO V7 FIX
# Adicionar no daytrader.py antes do model.learn()

def apply_critic_emergency_fix(model):
    '''Fix emergencial para crítico V7'''
    
    # 1. LEARNING RATE BOOST para crítico
    for name, param in model.policy.named_parameters():
        if 'critic' in name.lower() or 'value' in name.lower():
            param.requires_grad_(True)
            # Multiplier interno para gradientes do crítico
            if hasattr(param, 'grad') and param.grad is not None:
                param.grad.data *= 2.0  # Boost gradientes
    
    # 2. VALUE LOSS WEIGHT aumentado
    if hasattr(model.policy, 'vf_coef'):
        model.policy.vf_coef = 1.0  # Aumentar de 0.5 para 1.0
    
    # 3. ENTROPY REGULARIZATION reduzida para dar espaço ao crítico
    if hasattr(model.policy, 'ent_coef'):
        model.policy.ent_coef *= 0.5  # Reduzir entropia
    
    print("🔧 Critic Emergency Fix aplicado!")

# USAR ASSIM:
# apply_critic_emergency_fix(model)
# model.learn(total_timesteps=50000)  # Treinar um pouco
        """
        
        return patch

def main():
    """Gerar todas as correções"""
    print("🎯 CRÍTICO V7 FIX GENERATOR")
    print("=" * 60)
    
    fixer = CriticFixV7()
    
    # Gerar fixes
    fixes = fixer.apply_critic_fixes()
    
    print("🔧 CORREÇÕES DISPONÍVEIS:")
    for i, (name, code) in enumerate(fixes, 1):
        print(f"\n{i}. {name}")
        print("-" * 40)
        print(code[:200] + "..." if len(code) > 200 else code)
    
    # Patch rápido
    print("\n🚀 PATCH RÁPIDO PARA APLICAR AGORA:")
    print("=" * 60)
    quick_fix = fixer.generate_quick_fix_patch()
    print(quick_fix)
    
    # Salvar patches
    with open("critic_fixes_v7.py", "w") as f:
        f.write("# 🎯 CRÍTICO V7 FIXES - GENERATED\n\n")
        for name, code in fixes:
            f.write(f"# {name}\n")
            f.write(code)
            f.write("\n\n")
    
    with open("critic_quick_fix.py", "w") as f:
        f.write(quick_fix)
    
    print("\n✅ Fixes salvos em:")
    print("   - critic_fixes_v7.py (fixes completos)")
    print("   - critic_quick_fix.py (patch rápido)")

if __name__ == "__main__":
    main()