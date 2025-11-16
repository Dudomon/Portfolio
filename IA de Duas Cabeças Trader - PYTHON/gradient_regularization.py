#!/usr/bin/env python3
"""
🔧 SISTEMA DE REGULARIZAÇÃO DE GRADIENTES
Previne vanishing gradients e dead neurons na TwoHeadV6
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional

class GradientRegularizer:
    """Sistema de regularização para prevenir gradientes mortos"""
    
    def __init__(self, 
                 gradient_clip_norm: float = 1.0,
                 min_gradient_norm: float = 1e-8,  # 🔥 CORREÇÃO: Threshold mais baixo
                 dead_neuron_threshold: float = 1e-6,  # 🔥 CORREÇÃO: Detectar zeros extremos
                 regularization_weight: float = 1e-3):  # 🔥 CORREÇÃO: Regularização mais forte
        
        self.gradient_clip_norm = gradient_clip_norm
        self.min_gradient_norm = min_gradient_norm
        self.dead_neuron_threshold = dead_neuron_threshold
        self.regularization_weight = regularization_weight
        
        # Tracking
        self.gradient_stats = {}
        self.dead_neurons_count = {}
        
    def apply_gradient_regularization(self, model: nn.Module) -> Dict:
        """Aplicar regularização de gradientes"""
        stats = {
            'total_params': 0,
            'dead_gradients': 0,
            'small_gradients': 0,
            'clipped_gradients': 0,
            'regularization_applied': False
        }
        
        # 1. Gradient Clipping
        if hasattr(model, 'parameters'):
            total_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 
                self.gradient_clip_norm
            )
            
            if total_norm > self.gradient_clip_norm:
                stats['clipped_gradients'] = 1
        
        # 2. Dead Gradient Detection and Fix
        for name, param in model.named_parameters():
            if param.grad is not None:
                stats['total_params'] += 1
                
                grad_norm = torch.norm(param.grad)
                
                # Detectar gradientes mortos (zeros extremos)
                if grad_norm < self.min_gradient_norm:
                    stats['dead_gradients'] += 1
                    
                    # 🔥 CORREÇÃO: Ruído mais forte para reativar neurônios mortos
                    noise_scale = max(1e-5, grad_norm * 100)  # Adaptive noise
                    noise = torch.randn_like(param.grad) * noise_scale
                    param.grad.add_(noise)
                    stats['regularization_applied'] = True
                
                # Detectar gradientes muito pequenos
                elif grad_norm < self.dead_neuron_threshold:
                    stats['small_gradients'] += 1
        
        return stats
    
    def add_activation_regularization(self, model: nn.Module, activations: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Adicionar regularização de ativações para prevenir neurônios mortos"""
        reg_loss = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for name, activation in activations.items():
            if activation is not None:
                # L1 regularization para esparsidade controlada
                l1_loss = torch.mean(torch.abs(activation))
                
                # Diversity regularization - prevenir que todos neurônios tenham mesma ativação
                if activation.dim() > 1 and activation.size(1) > 1:
                    # Calcular correlação entre neurônios
                    flat_act = activation.view(activation.size(0), -1)
                    if flat_act.size(1) > 1:
                        corr_matrix = torch.corrcoef(flat_act.T)
                        # Penalizar alta correlação (diversidade baixa)
                        diversity_loss = torch.mean(torch.abs(corr_matrix - torch.eye(corr_matrix.size(0), device=corr_matrix.device)))
                        reg_loss += self.regularization_weight * diversity_loss
                
                # Adicionar pequena L1 regularization
                reg_loss += self.regularization_weight * 0.1 * l1_loss
        
        return reg_loss
    
    def check_model_health(self, model: nn.Module) -> Dict:
        """Verificar saúde geral do modelo"""
        health_stats = {
            'total_parameters': 0,
            'zero_parameters': 0,
            'small_parameters': 0,
            'large_parameters': 0,
            'nan_parameters': 0,
            'parameter_norm': 0.0,
            'health_score': 1.0
        }
        
        total_norm_sq = 0.0
        
        for name, param in model.named_parameters():
            if param.data is not None:
                health_stats['total_parameters'] += param.numel()
                
                # Detectar parâmetros problemáticos
                zero_mask = torch.abs(param.data) < 1e-8
                small_mask = torch.abs(param.data) < 1e-4
                large_mask = torch.abs(param.data) > 10.0
                nan_mask = torch.isnan(param.data)
                
                health_stats['zero_parameters'] += zero_mask.sum().item()
                health_stats['small_parameters'] += small_mask.sum().item()
                health_stats['large_parameters'] += large_mask.sum().item()
                health_stats['nan_parameters'] += nan_mask.sum().item()
                
                # Calcular norma total
                total_norm_sq += torch.norm(param.data) ** 2
        
        health_stats['parameter_norm'] = torch.sqrt(total_norm_sq).item()
        
        # Calcular score de saúde
        zero_ratio = health_stats['zero_parameters'] / max(health_stats['total_parameters'], 1)
        nan_ratio = health_stats['nan_parameters'] / max(health_stats['total_parameters'], 1)
        
        health_stats['health_score'] = max(0.0, 1.0 - 2*zero_ratio - 10*nan_ratio)
        
        return health_stats

def create_gradient_regularizer(gradient_clip_norm: float = 1.0) -> GradientRegularizer:
    """Factory function para criar regularizador"""
    return GradientRegularizer(gradient_clip_norm=gradient_clip_norm)

# Hook para capturar ativações
class ActivationHook:
    """Hook para capturar ativações durante forward pass"""
    
    def __init__(self):
        self.activations = {}
        self.hooks = []
    
    def register_hooks(self, model: nn.Module):
        """Registrar hooks nos módulos"""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    self.activations[name] = output.detach()
            return hook
        
        # Registrar em camadas importantes
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.LSTM, nn.GRU, nn.MultiheadAttention)):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remover todos os hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def get_activations(self) -> Dict[str, torch.Tensor]:
        """Obter ativações capturadas"""
        return self.activations.copy()
    
    def clear_activations(self):
        """Limpar ativações"""
        self.activations.clear()

if __name__ == "__main__":
    print("🔧 Sistema de Regularização de Gradientes - Previne vanishing gradients e dead neurons")