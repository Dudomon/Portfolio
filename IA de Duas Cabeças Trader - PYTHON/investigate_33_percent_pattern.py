#!/usr/bin/env python3
"""
🔍 INVESTIGAÇÃO: Por que exatamente 33.33% zeros?
Isso não é aleatório - indica padrão estrutural específico
"""

import torch
import torch.nn as nn
import numpy as np

def investigate_33_percent_pattern():
    """🔍 Investigar por que exatamente 33.33% zeros"""
    print("🔍 INVESTIGAÇÃO: PADRÃO 33.33% ZEROS")
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
        
        # Analisar estrutura dos attention layers
        print(f"\n🔍 ANALISANDO ESTRUTURA DOS ATTENTION LAYERS...")
        
        attention_layers = []
        for name, module in policy.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                attention_layers.append((name, module))
                
                print(f"\n📊 {name}:")
                print(f"   embed_dim: {module.embed_dim}")
                print(f"   num_heads: {module.num_heads}")
                print(f"   head_dim: {module.embed_dim // module.num_heads}")
                
                # Verificar in_proj_bias
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    bias_shape = module.in_proj_bias.shape
                    print(f"   in_proj_bias shape: {bias_shape}")
                    
                    # DESCOBERTA CHAVE: in_proj_bias tem 3 partes!
                    # Query, Key, Value - cada uma com embed_dim elementos
                    total_size = bias_shape[0]
                    part_size = total_size // 3
                    
                    print(f"   Total bias size: {total_size}")
                    print(f"   Size per part (Q/K/V): {part_size}")
                    print(f"   Ratio per part: {part_size/total_size:.1%}")
                    
                    # Analisar cada parte separadamente
                    bias_data = module.in_proj_bias.data.cpu().numpy()
                    
                    query_bias = bias_data[:part_size]
                    key_bias = bias_data[part_size:2*part_size]
                    value_bias = bias_data[2*part_size:]
                    
                    # Contar zeros em cada parte
                    query_zeros = np.sum(np.abs(query_bias) < 1e-8)
                    key_zeros = np.sum(np.abs(key_bias) < 1e-8)
                    value_zeros = np.sum(np.abs(value_bias) < 1e-8)
                    
                    print(f"   Query bias zeros: {query_zeros}/{part_size} ({query_zeros/part_size:.1%})")
                    print(f"   Key bias zeros: {key_zeros}/{part_size} ({key_zeros/part_size:.1%})")
                    print(f"   Value bias zeros: {value_zeros}/{part_size} ({value_zeros/part_size:.1%})")
                    
                    # HIPÓTESE: Uma das 3 partes está completamente zerada?
                    if query_zeros == part_size:
                        print(f"   🚨 QUERY BIAS COMPLETAMENTE ZERADO!")
                    elif key_zeros == part_size:
                        print(f"   🚨 KEY BIAS COMPLETAMENTE ZERADO!")
                    elif value_zeros == part_size:
                        print(f"   🚨 VALUE BIAS COMPLETAMENTE ZERADO!")
                    
                    # Verificar se é exatamente 1/3
                    total_zeros = query_zeros + key_zeros + value_zeros
                    zero_ratio = total_zeros / total_size
                    
                    if abs(zero_ratio - 1/3) < 0.01:
                        print(f"   🎯 CONFIRMADO: Exatamente 1/3 dos bias são zero!")
                        
                        # Identificar qual parte
                        if query_zeros > key_zeros and query_zeros > value_zeros:
                            print(f"   💡 CAUSA: Query bias dominantemente zerado")
                        elif key_zeros > query_zeros and key_zeros > value_zeros:
                            print(f"   💡 CAUSA: Key bias dominantemente zerado")
                        elif value_zeros > query_zeros and value_zeros > key_zeros:
                            print(f"   💡 CAUSA: Value bias dominantemente zerado")
        
        # Testar hipótese com gradientes
        print(f"\n🔍 TESTANDO COM GRADIENTES...")
        
        dummy_input = torch.randn(8, 1480, requires_grad=True)
        
        for param in policy.parameters():
            param.requires_grad = True
        
        # Forward + backward
        features = policy.features_extractor(dummy_input)
        policy_latent, value_latent = policy.mlp_extractor(features)
        action_logits = policy.action_net(policy_latent)
        value_pred = policy.value_net(value_latent)
        
        loss = torch.mean(action_logits ** 2) + torch.mean(value_pred ** 2)
        loss.backward()
        
        # Analisar gradientes dos attention bias
        for name, module in policy.named_modules():
            if isinstance(module, nn.MultiheadAttention) and hasattr(module, 'in_proj_bias'):
                if module.in_proj_bias.grad is not None:
                    grad_data = module.in_proj_bias.grad.data.cpu().numpy()
                    total_size = len(grad_data)
                    part_size = total_size // 3
                    
                    query_grad = grad_data[:part_size]
                    key_grad = grad_data[part_size:2*part_size]
                    value_grad = grad_data[2*part_size:]
                    
                    query_zeros = np.sum(np.abs(query_grad) < 1e-8)
                    key_zeros = np.sum(np.abs(key_grad) < 1e-8)
                    value_zeros = np.sum(np.abs(value_grad) < 1e-8)
                    
                    print(f"\n📊 {name} GRADIENTES:")
                    print(f"   Query grad zeros: {query_zeros}/{part_size} ({query_zeros/part_size:.1%})")
                    print(f"   Key grad zeros: {key_zeros}/{part_size} ({key_zeros/part_size:.1%})")
                    print(f"   Value grad zeros: {value_zeros}/{part_size} ({value_zeros/part_size:.1%})")
                    
                    total_grad_zeros = query_zeros + key_zeros + value_zeros
                    grad_zero_ratio = total_grad_zeros / total_size
                    
                    print(f"   Total grad zeros: {total_grad_zeros}/{total_size} ({grad_zero_ratio:.1%})")
                    
                    # Verificar se uma parte específica não recebe gradientes
                    if query_zeros == part_size:
                        print(f"   🚨 QUERY NÃO RECEBE GRADIENTES!")
                        return "query_unused"
                    elif key_zeros == part_size:
                        print(f"   🚨 KEY NÃO RECEBE GRADIENTES!")
                        return "key_unused"
                    elif value_zeros == part_size:
                        print(f"   🚨 VALUE NÃO RECEBE GRADIENTES!")
                        return "value_unused"
        
        return "pattern_identified"
        
    except Exception as e:
        print(f"❌ Erro na investigação: {e}")
        import traceback
        traceback.print_exc()
        return "error"

def analyze_attention_mechanism():
    """🔍 Analisar como o attention está sendo usado"""
    print(f"\n🔍 ANALISANDO USO DO ATTENTION MECHANISM...")
    
    try:
        from trading_framework.policies.two_head_v6_intelligent_48h import TwoHeadV6Intelligent48h
        import gym
        from gym import spaces
        
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
        
        # Verificar se attention está sendo usado corretamente
        dummy_input = torch.randn(8, 1480)
        
        # Capturar ativações do attention
        attention_outputs = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    # MultiheadAttention retorna (output, attention_weights)
                    attention_outputs[name] = {
                        'output': output[0].detach(),
                        'attention_weights': output[1].detach() if len(output) > 1 else None
                    }
                else:
                    attention_outputs[name] = {'output': output.detach()}
            return hook
        
        # Registrar hooks nos attention layers
        for name, module in policy.named_modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(hook_fn(name))
                hooks.append(hook)
        
        try:
            with torch.no_grad():
                _ = policy.features_extractor(dummy_input)
            
            # Analisar outputs do attention
            for name, data in attention_outputs.items():
                output = data['output']
                attention_weights = data.get('attention_weights')
                
                print(f"\n📊 {name} OUTPUT:")
                print(f"   Output shape: {output.shape}")
                print(f"   Output mean: {torch.mean(output):.6f}")
                print(f"   Output std: {torch.std(output):.6f}")
                print(f"   Output zeros: {torch.sum(output == 0)}/{output.numel()} ({torch.sum(output == 0)/output.numel():.1%})")
                
                if attention_weights is not None:
                    print(f"   Attention weights shape: {attention_weights.shape}")
                    print(f"   Attention weights mean: {torch.mean(attention_weights):.6f}")
                    print(f"   Attention weights std: {torch.std(attention_weights):.6f}")
                    
                    # Verificar se attention weights são uniformes (indicaria não uso)
                    if attention_weights.numel() > 0:
                        weights_flat = attention_weights.flatten()
                        expected_uniform = 1.0 / attention_weights.shape[-1]  # 1/seq_len
                        
                        # Se todos os weights são aproximadamente iguais, attention não está funcionando
                        weights_std = torch.std(weights_flat)
                        if weights_std < 0.01:
                            print(f"   ⚠️ ATTENTION WEIGHTS MUITO UNIFORMES (std: {weights_std:.6f})")
                            print(f"   💡 Attention pode não estar aprendendo padrões")
        
        finally:
            for hook in hooks:
                hook.remove()
        
        return True
        
    except Exception as e:
        print(f"❌ Erro na análise: {e}")
        return False

if __name__ == "__main__":
    # Investigar padrão 33.33%
    pattern_result = investigate_33_percent_pattern()
    
    # Analisar mecanismo de attention
    attention_ok = analyze_attention_mechanism()
    
    print(f"\n" + "=" * 60)
    print("🎯 DIAGNÓSTICO FINAL")
    print("=" * 60)
    
    if pattern_result == "query_unused":
        print("🚨 PROBLEMA IDENTIFICADO: QUERY BIAS NÃO USADO!")
        print("💡 1/3 do attention (Query) não recebe gradientes")
        print("💡 Isso explica os exatos 33.33% zeros")
        print("💡 SOLUÇÃO: Verificar por que Query não é usado")
        
    elif pattern_result == "key_unused":
        print("🚨 PROBLEMA IDENTIFICADO: KEY BIAS NÃO USADO!")
        print("💡 1/3 do attention (Key) não recebe gradientes")
        print("💡 Isso explica os exatos 33.33% zeros")
        print("💡 SOLUÇÃO: Verificar por que Key não é usado")
        
    elif pattern_result == "value_unused":
        print("🚨 PROBLEMA IDENTIFICADO: VALUE BIAS NÃO USADO!")
        print("💡 1/3 do attention (Value) não recebe gradientes")
        print("💡 Isso explica os exatos 33.33% zeros")
        print("💡 SOLUÇÃO: Verificar por que Value não é usado")
        
    elif pattern_result == "pattern_identified":
        print("✅ PADRÃO 33.33% CONFIRMADO!")
        print("💡 É estrutural: 1/3 do attention bias sempre zero")
        print("💡 Pode ser comportamento normal do PyTorch MultiheadAttention")
        print("💡 Ou indica uso parcial do mecanismo de attention")
        
    else:
        print("⚠️ INVESTIGAÇÃO INCONCLUSIVA")
        print("💡 Padrão 33.33% persiste mas causa não identificada")
    
    print(f"\n💡 CONCLUSÃO: 33.33% fixo NÃO é aleatório!")
    print(f"💡 É um padrão estrutural específico do attention mechanism")
    print(f"💡 Pode ser normal ou indicar uso subótimo do attention")