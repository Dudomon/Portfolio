"""
🧪 TESTE DE VALIDAÇÃO: V9 Input Projection Fix

TESTE ESPECÍFICO:
1. Criar TradingTransformerV9 com fix aplicado
2. Executar múltiplos forwards  
3. Verificar health do input_projection
4. Comparar com comportamento V8

CRITÉRIOS DE SUCESSO:
- input_projection zeros < 20%
- Gradient flow saudável
- Nenhuma saturação crítica
"""

import torch
import torch.nn as nn
import numpy as np
from trading_framework.extractors.transformer_v9_daytrading import TradingTransformerV9
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
import gym

def test_v9_input_projection_fix():
    """Teste completo da correção V9 input_projection"""
    
    print("🧪 INICIANDO TESTE V9 INPUT_PROJECTION FIX...")
    print("=" * 60)
    
    # 1. SETUP V9 CORRIGIDA
    print("📋 1. Criando TradingTransformerV9 com fix...")
    obs_space_v9 = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)  # 10*45
    v9_transformer = TradingTransformerV9(
        observation_space=obs_space_v9,
        features_dim=256,
        temporal_window=10,
        features_per_bar=45
    )
    
    # 2. SETUP V8 COMPARAÇÃO
    print("📋 2. Criando TradingTransformerV8 para comparação...")
    obs_space_v8 = gym.spaces.Box(low=-1, high=1, shape=(2580,), dtype=np.float32)  # 20*129
    v8_transformer = TradingTransformerFeatureExtractor(
        observation_space=obs_space_v8,
        features_dim=64,
        seq_len=20
    )
    
    # 3. TESTE INICIAL DE WEIGHTS
    print("\n📊 3. Análise inicial de weights...")
    
    def analyze_weights(model, projection_name):
        """Analisa weights de uma projeção específica"""
        if hasattr(model, projection_name):
            projection = getattr(model, projection_name)
            weights = projection.weight.data
            total_params = weights.numel()
            zero_params = (weights.abs() < 1e-8).sum().item()
            zero_percentage = (zero_params / total_params) * 100
            
            return {
                'zero_percentage': zero_percentage,
                'mean_abs': weights.abs().mean().item(),
                'std': weights.std().item(),
                'max_abs': weights.abs().max().item()
            }
        return None
    
    v9_initial = analyze_weights(v9_transformer, 'input_projection')
    v8_initial = analyze_weights(v8_transformer, 'temporal_projection')
    
    print(f"V9 input_projection inicial:")
    print(f"   Zeros: {v9_initial['zero_percentage']:.1f}%")
    print(f"   Mean abs: {v9_initial['mean_abs']:.4f}")
    print(f"   Std: {v9_initial['std']:.4f}")
    
    print(f"V8 temporal_projection inicial:")
    print(f"   Zeros: {v8_initial['zero_percentage']:.1f}%")
    print(f"   Mean abs: {v8_initial['mean_abs']:.4f}")
    print(f"   Std: {v8_initial['std']:.4f}")
    
    # 4. TESTE FORWARD MÚLTIPLOS
    print("\n🔄 4. Executando forwards múltiplos...")
    
    batch_size = 8
    num_forwards = 50
    
    # V9 data
    v9_obs = torch.randn(batch_size, 450)
    v9_transformer.train()  # Training mode para ativar fixes
    
    # V8 data  
    v8_obs = torch.randn(batch_size, 2580)
    v8_transformer.train()
    
    v9_health_history = []
    v8_health_history = []
    
    for i in range(num_forwards):
        # Forward V9
        v9_out = v9_transformer(v9_obs)
        
        # Forward V8
        v8_out = v8_transformer(v8_obs)
        
        # Analyze health every 10 forwards
        if i % 10 == 0:
            v9_health = analyze_weights(v9_transformer, 'input_projection')
            v8_health = analyze_weights(v8_transformer, 'temporal_projection')
            
            v9_health_history.append(v9_health)
            v8_health_history.append(v8_health)
            
            print(f"Forward {i:2d}: V9 zeros={v9_health['zero_percentage']:5.1f}% | V8 zeros={v8_health['zero_percentage']:5.1f}%")
    
    # 5. ANÁLISE FINAL
    print("\n📈 5. Análise final...")
    
    v9_final = v9_health_history[-1]
    v8_final = v8_health_history[-1]
    
    print(f"V9 input_projection final:")
    print(f"   Zeros: {v9_final['zero_percentage']:.1f}%")
    print(f"   Mean abs: {v9_final['mean_abs']:.4f}")
    print(f"   Std: {v9_final['std']:.4f}")
    
    print(f"V8 temporal_projection final:")
    print(f"   Zeros: {v8_final['zero_percentage']:.1f}%")
    print(f"   Mean abs: {v8_final['mean_abs']:.4f}")
    print(f"   Std: {v8_final['std']:.4f}")
    
    # 6. VERIFICAÇÃO DE SUCCESS CRITERIA
    print("\n✅ 6. Verificação de critérios de sucesso...")
    
    criteria_met = []
    
    # Critério 1: V9 zeros < 20%
    v9_zeros_ok = v9_final['zero_percentage'] < 20.0
    criteria_met.append(v9_zeros_ok)
    print(f"   Zeros V9 < 20%: {'✅' if v9_zeros_ok else '❌'} ({v9_final['zero_percentage']:.1f}%)")
    
    # Critério 2: V9 performance similar to V8
    zeros_diff = abs(v9_final['zero_percentage'] - v8_final['zero_percentage'])
    performance_similar = zeros_diff < 30.0  # Tolerância 30%
    criteria_met.append(performance_similar)
    print(f"   Performance similar V8: {'✅' if performance_similar else '❌'} (diff: {zeros_diff:.1f}%)")
    
    # Critério 3: Stability (não degradou ao longo dos forwards)
    v9_degradation = v9_final['zero_percentage'] - v9_initial['zero_percentage']
    stability_ok = v9_degradation < 10.0  # Máximo 10% degradação
    criteria_met.append(stability_ok)
    print(f"   Estabilidade V9: {'✅' if stability_ok else '❌'} (degradação: {v9_degradation:.1f}%)")
    
    # Critério 4: Weights magnitude saudável
    weights_healthy = v9_final['mean_abs'] > 0.01 and v9_final['std'] > 0.001
    criteria_met.append(weights_healthy)
    print(f"   Weights saudáveis: {'✅' if weights_healthy else '❌'}")
    
    # 7. RESULTADO FINAL
    print("\n" + "=" * 60)
    success_rate = sum(criteria_met) / len(criteria_met)
    print(f"🎯 RESULTADO FINAL: {sum(criteria_met)}/{len(criteria_met)} critérios atendidos ({success_rate*100:.0f}%)")
    
    if success_rate >= 0.75:
        print("🎉 SUCESSO! Fix V9 input_projection funcionando corretamente!")
        print("✅ V9 agora tem proteção similar à V8 funcional")
        return True
    else:
        print("❌ FALHA! Fix V9 precisa de mais ajustes")
        print("🔧 Revisar normalização e gradient clipping")
        return False

def test_v9_gradient_flow():
    """Teste específico de gradient flow"""
    print("\n🌊 TESTE GRADIENT FLOW V9...")
    
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    v9_transformer = TradingTransformerV9(observation_space=obs_space, features_dim=256)
    v9_transformer.train()
    
    # Forward + backward
    obs = torch.randn(4, 450, requires_grad=True)
    output = v9_transformer(obs)
    loss = output.sum()
    loss.backward()
    
    # Check gradients
    input_proj_grad = v9_transformer.input_projection.weight.grad
    if input_proj_grad is not None:
        grad_norm = input_proj_grad.norm().item()
        grad_zeros = (input_proj_grad.abs() < 1e-8).float().mean().item() * 100
        
        print(f"   Gradient norm: {grad_norm:.4f}")
        print(f"   Gradient zeros: {grad_zeros:.1f}%")
        
        if grad_norm > 0.001 and grad_zeros < 90:
            print("   ✅ Gradient flow saudável!")
            return True
        else:
            print("   ❌ Gradient flow problemático!")
            return False
    else:
        print("   ❌ Nenhum gradient encontrado!")
        return False

if __name__ == "__main__":
    print("🧪 TESTE COMPLETO V9 INPUT_PROJECTION FIX")
    print("🎯 Validando correção da morte de neurônios")
    print("=" * 60)
    
    # Teste principal
    main_success = test_v9_input_projection_fix()
    
    # Teste gradient flow
    grad_success = test_v9_gradient_flow()
    
    print("\n" + "🏁" * 20)
    if main_success and grad_success:
        print("🎉 TODOS OS TESTES PASSARAM!")
        print("✅ V9 input_projection fix validado com sucesso!")
        print("🚀 Pronto para treinamento sem morte de neurônios!")
    else:
        print("❌ ALGUNS TESTES FALHARAM!")
        print("🔧 Revisar implementação do fix")
    print("🏁" * 20)