#!/usr/bin/env python3
"""
🎯 INVESTIGAÇÃO ESPECÍFICA DO MLP_EXTRACTOR
Problema identificado: mlp_extractor produz 50% zeros nas policy/value latents
"""

import torch
import torch.nn as nn
import numpy as np

def investigate_mlp_extractor():
    """🔍 Investigar o mlp_extractor especificamente"""
    print("🔍 INVESTIGAÇÃO ESPECÍFICA DO MLP_EXTRACTOR")
    print("=" * 60)
    
    try:
        from trading_framework.policies.two_head_v6_intelligent_48h import TwoHeadV6Intelligent48h
        import gym
        from gym import spaces
        
        # Criar policy
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1480,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)
        
        def lr_schedule(progress):
            return 3e-4
        
        policy = TwoHeadV6Intelligent48h(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            lstm_hidden_size=128
        )
        
        print("✅ TwoHeadV6 carregada")
        
        # Analisar mlp_extractor
        mlp_extractor = policy.mlp_extractor
        print(f"📊 MLP Extractor: {type(mlp_extractor).__name__}")
        
        # Listar componentes
        print(f"\n🔍 COMPONENTES DO MLP_EXTRACTOR:")
        for name, module in mlp_extractor.named_modules():
            if name:  # Skip root module
                print(f"   {name}: {type(module).__name__}")
        
        # Testar com input válido
        dummy_input = torch.randn(8, 1480)
        
        with torch.no_grad():
            # Passar pelo feature extractor primeiro
            features = policy.features_extractor(dummy_input)
            print(f"\n📊 Features do extractor:")
            print(f"   Shape: {features.shape}")
            print(f"   Mean: {torch.mean(features):.6f}")
            print(f"   Std: {torch.std(features):.6f}")
            print(f"   Zeros: {torch.sum(features == 0)}/{features.numel()} ({torch.sum(features == 0)/features.numel():.1%})")
            
            # Capturar ativações intermediárias do mlp_extractor
            activations = {}
            hooks = []
            
            def hook_fn(name):
                def hook(module, input, output):
                    if isinstance(output, (torch.Tensor, tuple)):
                        if isinstance(output, tuple):
                            # Para casos onde retorna múltiplos valores
                            for i, out in enumerate(output):
                                if isinstance(out, torch.Tensor):
                                    activations[f"{name}_output_{i}"] = {
                                        'tensor': out.detach(),
                                        'shape': out.shape,
                                        'mean': float(torch.mean(out)),
                                        'std': float(torch.std(out)),
                                        'zeros': int(torch.sum(out == 0)),
                                        'total': int(out.numel())
                                    }
                        else:
                            activations[name] = {
                                'tensor': output.detach(),
                                'shape': output.shape,
                                'mean': float(torch.mean(output)),
                                'std': float(torch.std(output)),
                                'zeros': int(torch.sum(output == 0)),
                                'total': int(output.numel())
                            }
                return hook
            
            # Registrar hooks
            for name, module in mlp_extractor.named_modules():
                if isinstance(module, (nn.Linear, nn.ReLU, nn.LeakyReLU)):
                    hook = module.register_forward_hook(hook_fn(name))
                    hooks.append(hook)
            
            # Forward pass através do mlp_extractor
            try:
                policy_latent, value_latent = mlp_extractor(features)
                
                print(f"\n📊 RESULTADOS DO MLP_EXTRACTOR:")
                print(f"   Policy latent shape: {policy_latent.shape}")
                print(f"   Policy latent zeros: {torch.sum(policy_latent == 0)}/{policy_latent.numel()} ({torch.sum(policy_latent == 0)/policy_latent.numel():.1%})")
                print(f"   Value latent shape: {value_latent.shape}")
                print(f"   Value latent zeros: {torch.sum(value_latent == 0)}/{value_latent.numel()} ({torch.sum(value_latent == 0)/value_latent.numel():.1%})")
                
                # Analisar ativações intermediárias
                print(f"\n🔍 ATIVAÇÕES INTERMEDIÁRIAS:")
                problematic_layers = []
                
                for name, stats in activations.items():
                    zero_ratio = stats['zeros'] / stats['total']
                    
                    print(f"📊 {name}:")
                    print(f"   Shape: {stats['shape']}")
                    print(f"   Mean: {stats['mean']:.6f}")
                    print(f"   Std: {stats['std']:.6f}")
                    print(f"   Zeros: {stats['zeros']}/{stats['total']} ({zero_ratio:.1%})")
                    
                    if zero_ratio > 0.3:
                        problematic_layers.append((name, zero_ratio))
                        print(f"   🚨 PROBLEMÁTICO: {zero_ratio:.1%} zeros")
                    elif zero_ratio > 0.1:
                        print(f"   ⚠️ SUSPEITO: {zero_ratio:.1%} zeros")
                    else:
                        print(f"   ✅ OK: {zero_ratio:.1%} zeros")
                
                # Identificar layer mais problemático
                if problematic_layers:
                    worst_layer = max(problematic_layers, key=lambda x: x[1])
                    print(f"\n🚨 LAYER MAIS PROBLEMÁTICO: {worst_layer[0]} ({worst_layer[1]:.1%} zeros)")
                    
                    # Analisar pesos deste layer
                    layer_name = worst_layer[0]
                    if layer_name in dict(mlp_extractor.named_modules()):
                        problematic_module = dict(mlp_extractor.named_modules())[layer_name]
                        
                        if isinstance(problematic_module, nn.Linear):
                            print(f"\n🔍 ANÁLISE DOS PESOS DO LAYER PROBLEMÁTICO:")
                            
                            weight_data = problematic_module.weight.data.cpu().numpy()
                            bias_data = problematic_module.bias.data.cpu().numpy() if problematic_module.bias is not None else None
                            
                            print(f"   Weight shape: {weight_data.shape}")
                            print(f"   Weight mean: {np.mean(weight_data):.6f}")
                            print(f"   Weight std: {np.std(weight_data):.6f}")
                            print(f"   Weight zeros: {np.sum(np.abs(weight_data) < 1e-8)}/{weight_data.size} ({np.sum(np.abs(weight_data) < 1e-8)/weight_data.size:.1%})")
                            
                            if bias_data is not None:
                                print(f"   Bias mean: {np.mean(bias_data):.6f}")
                                print(f"   Bias std: {np.std(bias_data):.6f}")
                                print(f"   Bias zeros: {np.sum(np.abs(bias_data) < 1e-8)}/{bias_data.size} ({np.sum(np.abs(bias_data) < 1e-8)/bias_data.size:.1%})")
                
                return {
                    'policy_zero_ratio': torch.sum(policy_latent == 0).item() / policy_latent.numel(),
                    'value_zero_ratio': torch.sum(value_latent == 0).item() / value_latent.numel(),
                    'problematic_layers': problematic_layers,
                    'activations': activations
                }
                
            finally:
                # Limpar hooks
                for hook in hooks:
                    hook.remove()
        
    except Exception as e:
        print(f"❌ Erro na investigação: {e}")
        import traceback
        traceback.print_exc()
        return None

def fix_mlp_extractor_zeros():
    """🔧 Corrigir zeros no mlp_extractor"""
    print("\n🔧 CORREÇÃO ESPECÍFICA DO MLP_EXTRACTOR")
    print("=" * 60)
    
    try:
        from trading_framework.policies.two_head_v6_intelligent_48h import TwoHeadV6Intelligent48h
        import gym
        from gym import spaces
        
        # Criar policy
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1480,), dtype=np.float32)
        action_space = spaces.Box(low=-1, high=1, shape=(11,), dtype=np.float32)
        
        def lr_schedule(progress):
            return 3e-4
        
        policy = TwoHeadV6Intelligent48h(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            lstm_hidden_size=128
        )
        
        mlp_extractor = policy.mlp_extractor
        fixes_applied = 0
        
        # CORREÇÃO 1: Reinicializar layers lineares problemáticas
        for name, module in mlp_extractor.named_modules():
            if isinstance(module, nn.Linear):
                # Verificar se pesos estão problemáticos
                weight_data = module.weight.data.cpu().numpy()
                weight_zeros = np.sum(np.abs(weight_data) < 1e-8)
                weight_total = weight_data.size
                
                if weight_zeros / weight_total > 0.05 or np.std(weight_data) < 0.01:
                    # Reinicializar com método apropriado
                    nn.init.kaiming_uniform_(module.weight, mode='fan_in', nonlinearity='relu')
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                    
                    fixes_applied += 1
                    print(f"   ✅ {name}: reinicializado (zeros: {weight_zeros}/{weight_total}, std: {np.std(weight_data):.6f})")
        
        # CORREÇÃO 2: Substituir ReLUs por LeakyReLU
        relu_fixes = 0
        for name, module in mlp_extractor.named_modules():
            if isinstance(module, nn.ReLU):
                parts = name.split('.')
                if len(parts) > 1:
                    parent_name = '.'.join(parts[:-1])
                    child_name = parts[-1]
                    
                    try:
                        parent_module = mlp_extractor
                        for part in parent_name.split('.'):
                            parent_module = getattr(parent_module, part)
                        
                        setattr(parent_module, child_name, nn.LeakyReLU(negative_slope=0.01, inplace=True))
                        relu_fixes += 1
                        print(f"   ✅ {name}: ReLU → LeakyReLU")
                    except:
                        pass
        
        print(f"\n✅ {fixes_applied} layers reinicializados")
        print(f"✅ {relu_fixes} ReLUs substituídos")
        
        # TESTE: Verificar se correção funcionou
        dummy_input = torch.randn(8, 1480)
        
        with torch.no_grad():
            features = policy.features_extractor(dummy_input)
            policy_latent, value_latent = mlp_extractor(features)
            
            policy_zero_ratio = torch.sum(policy_latent == 0).item() / policy_latent.numel()
            value_zero_ratio = torch.sum(value_latent == 0).item() / value_latent.numel()
            
            print(f"\n📊 RESULTADO APÓS CORREÇÃO:")
            print(f"   Policy latent zeros: {torch.sum(policy_latent == 0)}/{policy_latent.numel()} ({policy_zero_ratio:.1%})")
            print(f"   Value latent zeros: {torch.sum(value_latent == 0)}/{value_latent.numel()} ({value_zero_ratio:.1%})")
            
            success = policy_zero_ratio < 0.1 and value_zero_ratio < 0.1
            
            if success:
                print(f"\n🎉 MLP_EXTRACTOR CORRIGIDO COM SUCESSO!")
                return True
            else:
                print(f"\n⚠️ Correção parcial - zeros ainda altos")
                return False
    
    except Exception as e:
        print(f"❌ Erro na correção: {e}")
        return False

if __name__ == "__main__":
    # Investigar primeiro
    results = investigate_mlp_extractor()
    
    if results and (results['policy_zero_ratio'] > 0.3 or results['value_zero_ratio'] > 0.3):
        print(f"\n🚨 PROBLEMA CONFIRMADO NO MLP_EXTRACTOR!")
        print(f"   Policy zeros: {results['policy_zero_ratio']:.1%}")
        print(f"   Value zeros: {results['value_zero_ratio']:.1%}")
        
        # Aplicar correção
        success = fix_mlp_extractor_zeros()
        
        if success:
            print(f"\n🎯 CAUSA RAIZ DOS 50-53% ZEROS RESOLVIDA!")
            print(f"💡 MLP_EXTRACTOR agora produz latents válidos")
            print(f"💡 Action/Value networks receberão inputs não-zerados")
        else:
            print(f"\n⚠️ Correção insuficiente - redesign pode ser necessário")
    else:
        print(f"\n✅ MLP_EXTRACTOR funcionando adequadamente")