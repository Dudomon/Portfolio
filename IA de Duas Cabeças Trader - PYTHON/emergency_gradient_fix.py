#!/usr/bin/env python3
"""
🚨 CORREÇÃO EMERGENCIAL PARA GRADIENT VANISHING
Sistema para detectar e corrigir gradientes que estão desaparecendo
"""

import torch
import torch.nn as nn
import numpy as np

def emergency_gradient_fix(model, min_grad_norm=1e-6, lstm_lr_multiplier=3.0):
    """
    Correção emergencial para gradientes vanishing
    """
    print("🚨 APLICANDO CORREÇÃO EMERGENCIAL DE GRADIENTES")
    
    fixes_applied = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            zero_ratio = (param.grad.abs() < 1e-8).float().mean().item()
            
            # Detectar gradientes vanishing críticos
            if zero_ratio > 0.7 and 'lstm' in name.lower():
                print(f"🚨 GRADIENT VANISHING CRÍTICO: {name}")
                print(f"   Zero ratio: {zero_ratio:.1%}")
                print(f"   Grad norm: {grad_norm:.2e}")
                
                # CORREÇÃO 1: Reescalar gradientes LSTM
                if 'weight_hh' in name or 'bias_hh' in name:
                    param.grad.data *= lstm_lr_multiplier
                    print(f"   ✅ Gradiente reescalado por {lstm_lr_multiplier}x")
                    fixes_applied += 1
                
                # CORREÇÃO 2: Adicionar ruído para quebrar simetria
                if zero_ratio > 0.8:
                    noise = torch.randn_like(param.grad) * 1e-6
                    param.grad.data += noise
                    print(f"   ✅ Ruído adicionado para quebrar simetria")
                    fixes_applied += 1
            
            # Detectar outros gradientes problemáticos
            elif zero_ratio > 0.5:
                print(f"⚠️  Gradiente com muitos zeros: {name} ({zero_ratio:.1%})")
    
    print(f"🔧 Total de correções aplicadas: {fixes_applied}")
    return fixes_applied

def create_emergency_gradient_callback():
    """Cria callback para correção emergencial"""
    
    class EmergencyGradientCallback:
        def __init__(self):
            self.step_count = 0
            self.last_fix_step = 0
            
        def __call__(self, model):
            self.step_count += 1
            
            # Aplicar correção a cada 1000 steps ou se detectar problema severo
            if self.step_count % 1000 == 0 or self.step_count - self.last_fix_step > 5000:
                fixes = emergency_gradient_fix(model)
                if fixes > 0:
                    self.last_fix_step = self.step_count
                    print(f"🚨 Correção emergencial aplicada no step {self.step_count}")
    
    return EmergencyGradientCallback()

if __name__ == "__main__":
    print("🚨 Sistema de Correção Emergencial de Gradientes")
    print("   - Detecta gradient vanishing")
    print("   - Reescala gradientes LSTM")
    print("   - Adiciona ruído para quebrar simetria")
    print("   - Monitora zero ratios críticos")