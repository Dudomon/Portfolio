#!/usr/bin/env python3
"""
🔧 FIX SATURAÇÃO CRÍTICA - Reinicializar pesos saturados

Detecta e reinicializa pesos que estão 100% saturados
"""

import torch
import torch.nn as nn
import numpy as np

def fix_saturated_weights(model, threshold=0.8, verbose=True):
    """
    🔧 Corrigir pesos saturados no modelo
    
    Args:
        model: Modelo PyTorch
        threshold: Threshold para considerar saturado (0.8 = 80%)
        verbose: Imprimir detalhes
    """
    
    fixed_components = []
    
    if verbose:
        print("🔧 [SATURAÇÃO FIX] Iniciando correção de pesos saturados...")
    
    for name, param in model.named_parameters():
        if param.data is not None:
            # Verificar saturação
            data_tensor = param.data.detach()
            
            # Contar valores próximos dos extremos
            near_zero = (torch.abs(data_tensor) < 0.01).float().mean().item()
            near_one = (torch.abs(data_tensor) > 0.99).float().mean().item()
            
            total_saturation = near_zero + near_one
            
            if total_saturation > threshold:
                if verbose:
                    print(f"   🚨 FIXING: {name} - {total_saturation*100:.1f}% saturado")
                
                # CORREÇÃO ESPECÍFICA POR TIPO
                if 'log_std' in name.lower():
                    # log_std crítico: deve permitir exploração
                    # log_std = 0 → std = 1, log_std = -1 → std = 0.37
                    param.data.normal_(mean=-0.5, std=0.2)  # std média ~= 0.6
                    
                elif 'bias' in name.lower():
                    # Bias pequeno
                    param.data.normal_(mean=0.0, std=0.01)
                    
                elif 'weight' in name.lower():
                    # Reinicialização Xavier/He
                    if len(param.shape) >= 2:
                        # Linear/Conv layers
                        fan_in = param.shape[1] if len(param.shape) >= 2 else param.shape[0]
                        fan_out = param.shape[0]
                        
                        # He initialization para ReLU-like, Xavier para outros
                        if 'relu' in name.lower() or 'gelu' in name.lower():
                            std = np.sqrt(2.0 / fan_in)  # He
                        else:
                            std = np.sqrt(2.0 / (fan_in + fan_out))  # Xavier
                            
                        param.data.normal_(mean=0.0, std=std)
                    else:
                        # 1D params
                        param.data.normal_(mean=0.0, std=0.02)
                        
                elif 'pos_encoding' in name.lower() or 'embedding' in name.lower():
                    # Positional encoding - pequena variação
                    param.data.normal_(mean=0.0, std=0.02)
                    
                else:
                    # Default: small random values
                    param.data.normal_(mean=0.0, std=0.02)
                
                fixed_components.append(name)
    
    if verbose:
        if fixed_components:
            print(f"✅ [SATURAÇÃO FIX] {len(fixed_components)} componentes corrigidos:")
            for comp in fixed_components[:10]:  # Top 10
                print(f"      {comp}")
        else:
            print("✅ [SATURAÇÃO FIX] Nenhuma saturação crítica detectada")
    
    return len(fixed_components)

def apply_fix_to_policy(model, verbose=True):
    """
    🔧 Aplicar fix específico para policy PPO
    """
    
    if verbose:
        print("🔧 [POLICY FIX] Aplicando correção específica para PPO policy...")
    
    fixed_count = 0
    
    # Fix policy específica
    if hasattr(model, 'policy'):
        fixed_count += fix_saturated_weights(model.policy, threshold=0.7, verbose=verbose)
    
    # Fix adicional para componentes críticos
    for name, module in model.policy.named_modules():
        if isinstance(module, (nn.LSTM, nn.GRU)):
            if verbose:
                print(f"   🔧 Reinicializando LSTM: {name}")
            
            # Reinicializar LSTM weights
            for param_name, param in module.named_parameters():
                if 'weight' in param_name:
                    nn.init.orthogonal_(param.data, gain=1.0)
                elif 'bias' in param_name:
                    # LSTM bias: forget gate deve ser 1.0
                    if 'bias_hh' in param_name or 'bias_ih' in param_name:
                        param.data.fill_(0.0)
                        # Set forget gate bias to 1
                        hidden_size = param.size(0) // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)
                        
            fixed_count += 1
    
    if verbose:
        print(f"✅ [POLICY FIX] Correção concluída. {fixed_count} componentes afetados.")
    
    return fixed_count

if __name__ == "__main__":
    print("🔧 Fix para saturação de pesos - Pronto para uso")
    print("   Usage: fix_saturated_weights(model)")
    print("   Usage: apply_fix_to_policy(model)")