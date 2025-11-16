#!/usr/bin/env python3
"""
🎯 COMPONENT-SPECIFIC LEARNING RATES
Aplica learning rates diferenciados por tipo de componente
"""

import torch
import torch.nn as nn
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from typing import Dict, List, Tuple

class ComponentSpecificLRCallback(BaseCallback):
    """Callback para aplicar LRs específicos por tipo de componente"""
    
    def __init__(self, 
                 base_lr: float = 2.68e-5,
                 lr_multipliers: Dict[str, float] = None,
                 adaptation_freq: int = 500,
                 verbose: int = 1):
        super().__init__(verbose)
        self.base_lr = base_lr
        self.adaptation_freq = adaptation_freq
        
        # LR multipliers por tipo de componente
        self.lr_multipliers = lr_multipliers or {
            'lstm': 0.1,      # LSTMs precisam LR 10x menor
            'linear': 1.0,    # Camadas lineares usam LR base
            'conv': 0.5,      # Convoluções LR 2x menor
            'embedding': 0.2,  # Embeddings LR 5x menor
            'layernorm': 2.0,  # LayerNorm pode usar LR 2x maior
            'attention': 0.3   # Attention LR 3x menor
        }
        
        # Tracking
        self.component_groups = {}
        self.lr_history = {}
        self.initialized = False
        
    def _on_training_start(self) -> None:
        """Inicializa grupos de componentes com LRs específicos"""
        print(f"🎯 COMPONENT-SPECIFIC LR ATIVADO")
        print(f"   📊 Base LR: {self.base_lr:.2e}")
        
        self._initialize_component_groups()
        self._apply_component_lrs()
        
    def _initialize_component_groups(self):
        """Organiza parâmetros por tipo de componente"""
        if not hasattr(self.model, 'policy'):
            print("❌ Modelo não tem policy")
            return
            
        self.component_groups = {
            'lstm': [],
            'linear': [],
            'conv': [],
            'embedding': [],
            'layernorm': [],
            'attention': [],
            'other': []
        }
        
        # Classificar cada parâmetro
        print("🔍 DEBUG: Classificando parâmetros...")
        for name, param in self.model.policy.named_parameters():
            component_type = self._classify_parameter(name)
            self.component_groups[component_type].append((name, param))
            
            # Debug detalhado para LSTMs
            if 'lstm' in name.lower():
                print(f"   📍 LSTM encontrado: {name} → {component_type}")
            
        # Debug: mostrar distribuição
        print("📊 DISTRIBUIÇÃO DE COMPONENTES:")
        for comp_type, params in self.component_groups.items():
            if params:
                lr = self.base_lr * self.lr_multipliers.get(comp_type, 1.0)
                print(f"   🔧 {comp_type.upper()}: {len(params)} params → LR {lr:.2e}")
                
                # Mostrar nomes dos primeiros parâmetros de cada tipo
                if len(params) > 0:
                    print(f"      Exemplos: {[name for name, _ in params[:3]]}")
                
    def _classify_parameter(self, param_name: str) -> str:
        """Classifica parâmetro por tipo de componente"""
        name_lower = param_name.lower()
        
        # LSTM/GRU components
        if any(lstm_key in name_lower for lstm_key in ['lstm', 'gru', 'rnn']):
            return 'lstm'
            
        # Linear/Dense layers
        elif any(linear_key in name_lower for linear_key in ['linear', 'dense', 'fc', 'head']):
            return 'linear'
            
        # Convolutional layers
        elif any(conv_key in name_lower for conv_key in ['conv', 'cnn']):
            return 'conv'
            
        # Embeddings
        elif any(emb_key in name_lower for emb_key in ['embed', 'embedding']):
            return 'embedding'
            
        # Normalization layers
        elif any(norm_key in name_lower for norm_key in ['norm', 'batch', 'layer', 'group']):
            return 'layernorm'
            
        # Attention mechanisms
        elif any(attn_key in name_lower for attn_key in ['attention', 'attn', 'self_attn', 'cross_attn']):
            return 'attention'
            
        else:
            return 'other'
    
    def _apply_component_lrs(self):
        """Aplica LRs específicos reorganizando param_groups"""
        if not hasattr(self.model.policy, 'optimizer'):
            print("❌ Optimizer não encontrado")
            return
            
        # Salvar estado atual do optimizer
        old_param_groups = self.model.policy.optimizer.param_groups.copy()
        
        # Criar novos param_groups por componente
        new_param_groups = []
        
        for comp_type, params in self.component_groups.items():
            if not params:
                continue
                
            # Calcular LR para este componente
            component_lr = self.base_lr * self.lr_multipliers.get(comp_type, 1.0)
            
            # Criar param_group para este componente
            param_list = [param for _, param in params]
            
            if param_list:
                param_group = {
                    'params': param_list,
                    'lr': component_lr,
                    'component_type': comp_type
                }
                
                # Copiar outras configurações do optimizer original
                if old_param_groups:
                    for key, value in old_param_groups[0].items():
                        if key not in ['params', 'lr']:
                            param_group[key] = value
                
                new_param_groups.append(param_group)
                
        # Substituir param_groups do optimizer
        self.model.policy.optimizer.param_groups = new_param_groups
        
        print(f"✅ {len(new_param_groups)} param_groups criados com LRs específicos")
        
        # VERIFICAÇÃO IMEDIATA: Confirmar LRs aplicados
        print("🔍 VERIFICAÇÃO DE LRs APLICADOS:")
        for i, param_group in enumerate(self.model.policy.optimizer.param_groups):
            comp_type = param_group.get('component_type', f'group_{i}')
            lr = param_group['lr']
            param_count = len(param_group['params'])
            print(f"   🔧 {comp_type.upper()}: {param_count} params → LR {lr:.2e}")
        
        self.initialized = True
        
    def _on_step(self) -> bool:
        """Monitor saúde por componente"""
        if self.num_timesteps % self.adaptation_freq == 0:
            self._monitor_component_health()
        return True
        
    def _monitor_component_health(self):
        """Monitora saúde dos gradientes por componente"""
        if not self.initialized:
            return
            
        print(f"📊 COMPONENT HEALTH - Step {self.num_timesteps}")
        
        for i, param_group in enumerate(self.model.policy.optimizer.param_groups):
            comp_type = param_group.get('component_type', f'group_{i}')
            current_lr = param_group['lr']
            
            # Calcular saúde dos gradientes deste componente
            total_params = 0
            healthy_gradients = 0
            
            for param in param_group['params']:
                if param.grad is not None:
                    grad_array = param.grad.detach().cpu().numpy().flatten()
                    total_params += len(grad_array)
                    healthy_gradients += np.sum(np.abs(grad_array) > 1e-8)
            
            if total_params > 0:
                health_ratio = healthy_gradients / total_params
                print(f"   🔧 {comp_type.upper()}: {health_ratio:.1%} healthy | LR: {current_lr:.2e}")
                
                # Histórico
                if comp_type not in self.lr_history:
                    self.lr_history[comp_type] = []
                self.lr_history[comp_type].append((self.num_timesteps, health_ratio, current_lr))
        
    def _on_training_end(self) -> None:
        """Relatório final"""
        print(f"🏁 COMPONENT-SPECIFIC LR FINALIZADO")
        for comp_type, history in self.lr_history.items():
            if history:
                avg_health = np.mean([h[1] for h in history])
                final_lr = history[-1][2]
                print(f"   📊 {comp_type.upper()}: {avg_health:.1%} avg health | Final LR: {final_lr:.2e}")


def create_component_lr_callback(
    base_lr: float = 2.68e-5,
    lr_multipliers: Dict[str, float] = None,
    adaptation_freq: int = 500,
    verbose: int = 1
) -> ComponentSpecificLRCallback:
    """Factory para criar callback de LRs específicos"""
    return ComponentSpecificLRCallback(
        base_lr=base_lr,
        lr_multipliers=lr_multipliers,
        adaptation_freq=adaptation_freq,
        verbose=verbose
    )

if __name__ == "__main__":
    print("🎯 Component-Specific Learning Rates - LRs otimizados por tipo de componente")