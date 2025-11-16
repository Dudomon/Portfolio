#!/usr/bin/env python3
"""
INTERCEPTOR DE GRADIENTES EM TEMPO REAL
Captura os dados reais que causam os 62.2% zeros
"""

import torch
import torch.nn as nn
import pickle
import os
from datetime import datetime

class LiveGradientInterceptor:
    """Intercepta gradientes em tempo real durante o treinamento"""
    
    def __init__(self, save_dir="gradient_debug_data"):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.intercept_count = 0
        self.max_intercepts = 5  # Limitar para nao encher disco
        
    def intercept_temporal_projection(self, model, step_num):
        """Intercepta especificamente a temporal_projection"""
        
        if self.intercept_count >= self.max_intercepts:
            return
            
        try:
            # Encontrar a temporal_projection
            temporal_proj = None
            for name, module in model.named_modules():
                if 'temporal_projection' in name and isinstance(module, nn.Sequential):
                    temporal_proj = module
                    break
                elif 'temporal_projection.0' in name and isinstance(module, nn.Linear):
                    temporal_proj = module
                    break
            
            if temporal_proj is None:
                print("temporal_projection nao encontrada!")
                return
            
            # Capturar estado atual
            if hasattr(temporal_proj, 'weight') and temporal_proj.weight.grad is not None:
                
                self.intercept_count += 1
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # Dados a salvar
                intercept_data = {
                    'step': step_num,
                    'timestamp': timestamp,
                    'weight_shape': temporal_proj.weight.shape,
                    'weight_data': temporal_proj.weight.detach().cpu().clone(),
                    'weight_grad': temporal_proj.weight.grad.detach().cpu().clone(),
                    'bias_data': temporal_proj.bias.detach().cpu().clone() if temporal_proj.bias is not None else None,
                    'bias_grad': temporal_proj.bias.grad.detach().cpu().clone() if temporal_proj.bias is not None and temporal_proj.bias.grad is not None else None,
                }
                
                # Calcular estatisticas
                weight_grad = intercept_data['weight_grad']
                grad_zeros = (weight_grad.abs() < 1e-8).float().mean().item()
                
                intercept_data['stats'] = {
                    'grad_zeros_percent': grad_zeros * 100,
                    'grad_mean': weight_grad.mean().item(),
                    'grad_std': weight_grad.std().item(),
                    'grad_min': weight_grad.min().item(),
                    'grad_max': weight_grad.max().item(),
                    'weight_mean': intercept_data['weight_data'].mean().item(),
                    'weight_std': intercept_data['weight_data'].std().item(),
                }
                
                # Salvar
                filename = f"temporal_proj_intercept_{timestamp}_step{step_num}.pkl"
                filepath = os.path.join(self.save_dir, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(intercept_data, f)
                
                print(f"INTERCEPTED: Step {step_num}, Grad zeros: {grad_zeros*100:.1f}%, Saved: {filename}")
                
                # Análise imediata
                self._immediate_analysis(intercept_data)
                
        except Exception as e:
            print(f"Erro no interceptor: {e}")
    
    def _immediate_analysis(self, data):
        """Análise imediata dos dados interceptados"""
        
        print(f"=== ANALISE IMEDIATA ===")
        stats = data['stats']
        print(f"Grad zeros: {stats['grad_zeros_percent']:.1f}%")
        print(f"Grad stats: mean={stats['grad_mean']:.6f}, std={stats['grad_std']:.6f}")
        print(f"Grad range: [{stats['grad_min']:.6f}, {stats['grad_max']:.6f}]")
        print(f"Weight stats: mean={stats['weight_mean']:.6f}, std={stats['weight_std']:.6f}")
        
        # Análise de padrões
        weight_grad = data['weight_grad']
        
        # Verificar se zeros estao concentrados em linhas/colunas específicas
        row_zeros = (weight_grad.abs() < 1e-8).float().mean(dim=1)  # Zeros por linha (output neurons)
        col_zeros = (weight_grad.abs() < 1e-8).float().mean(dim=0)  # Zeros por coluna (input features)
        
        # Top 10 linhas com mais zeros
        top_zero_rows = torch.topk(row_zeros, k=min(10, len(row_zeros)))
        print(f"Top neurônios de saída com mais grad zeros:")
        for i, (zero_ratio, idx) in enumerate(zip(top_zero_rows.values, top_zero_rows.indices)):
            print(f"  Neuron {idx.item()}: {zero_ratio.item()*100:.1f}% zeros")
        
        # Top 10 colunas com mais zeros  
        top_zero_cols = torch.topk(col_zeros, k=min(10, len(col_zeros)))
        print(f"Top features de entrada com mais grad zeros:")
        for i, (zero_ratio, idx) in enumerate(zip(top_zero_cols.values, top_zero_cols.indices)):
            print(f"  Feature {idx.item()}: {zero_ratio.item()*100:.1f}% zeros")
        
        # Verificar se ha padrão estrutural
        if row_zeros.max() > 0.8:
            print("ALERTA: Neurônios de saída mortos detectados!")
        
        if col_zeros.max() > 0.8:
            print("ALERTA: Features de entrada mortas detectadas!")
        
        print("========================")


def install_interceptor_hook(model):
    """Instala hooks para interceptar durante treinamento"""
    
    interceptor = LiveGradientInterceptor()
    
    def gradient_hook(step_counter=[0]):  # Usar lista para mutabilidade
        def hook_fn(grad):
            step_counter[0] += 1
            if step_counter[0] % 1000 == 0:  # A cada 1000 steps
                interceptor.intercept_temporal_projection(model, step_counter[0])
            return grad
        return hook_fn
    
    # Encontrar e instalar hook na temporal_projection
    for name, module in model.named_modules():
        if 'temporal_projection.0' in name and isinstance(module, nn.Linear):
            if module.weight.grad is not None:
                module.weight.register_hook(gradient_hook())
                print(f"Hook instalado em {name}")
                break
    
    return interceptor


if __name__ == "__main__":
    print("Live Gradient Interceptor - Use install_interceptor_hook(model) no treinamento")