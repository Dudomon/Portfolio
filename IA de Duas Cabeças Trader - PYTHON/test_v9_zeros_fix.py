"""
Teste para verificar se a correção dos zeros na V9 foi efetiva
"""
import torch
import numpy as np
import gym
from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs

def test_v9_zeros_fix():
    """Testa se os zeros foram corrigidos na V9"""
    
    print("🔧 Testando correção de zeros V9Optimus...")
    
    # Criar espaços dummy
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([0, 0, -1, -1]), 
        high=np.array([2, 1, 1, 1]), 
        dtype=np.float32
    )
    
    # Criar política com correções
    def lr_schedule(progress):
        return 1e-4
    
    kwargs = get_v9_optimus_kwargs()
    print(f"🔍 ortho_init: {kwargs['ortho_init']}")
    
    policy = TwoHeadV9Optimus(
        observation_space=obs_space,
        action_space=action_space,
        lr_schedule=lr_schedule,
        **kwargs
    )
    
    # Verificar pesos críticos
    def check_zeros(tensor, name):
        zeros_pct = (tensor.abs() < 1e-8).float().mean().item() * 100
        print(f"   {name}: {zeros_pct:.1f}% zeros")
        return zeros_pct
    
    print("\n🔍 ANÁLISE DE ZEROS:")
    
    # 1. Features extractor input projection
    if hasattr(policy.features_extractor, 'input_projection'):
        input_proj_zeros = check_zeros(policy.features_extractor.input_projection.weight, 
                                      "input_projection.weight")
    else:
        print("   input_projection: NÃO ENCONTRADO")
        input_proj_zeros = 0
    
    # 2. Market context encoder embedding
    embedding_zeros = check_zeros(policy.market_context_encoder.regime_embedding.weight,
                                 "regime_embedding.weight")
    
    # 3. Transformer attention (se existir)
    if hasattr(policy.features_extractor, 'transformer'):
        if hasattr(policy.features_extractor.transformer, 'layers'):
            if len(policy.features_extractor.transformer.layers) > 0:
                layer0 = policy.features_extractor.transformer.layers[0]
                if hasattr(layer0, 'self_attn') and hasattr(layer0.self_attn, 'in_proj_bias'):
                    attn_bias_zeros = check_zeros(layer0.self_attn.in_proj_bias,
                                                 "transformer.layer0.self_attn.in_proj_bias")
                else:
                    print("   transformer attention bias: NÃO ENCONTRADO")
                    attn_bias_zeros = 0
            else:
                print("   transformer layers: VAZIO")
                attn_bias_zeros = 0
        else:
            print("   transformer layers: NÃO ENCONTRADO")
            attn_bias_zeros = 0
    else:
        print("   transformer: NÃO ENCONTRADO")
        attn_bias_zeros = 0
    
    # Avaliar resultados
    print(f"\n📊 RESULTADOS:")
    print(f"   Input Projection: {input_proj_zeros:.1f}% zeros {'✅' if input_proj_zeros < 50 else '❌'}")
    print(f"   Regime Embedding: {embedding_zeros:.1f}% zeros {'✅' if embedding_zeros < 50 else '❌'}")
    print(f"   Attention Bias: {attn_bias_zeros:.1f}% zeros {'✅' if attn_bias_zeros < 40 else '❌'}")
    
    # Conclusão
    if input_proj_zeros < 50 and embedding_zeros < 50:
        print("\n🎉 SUCESSO! Zeros críticos foram corrigidos!")
        return True
    else:
        print("\n❌ FALHA! Ainda há zeros excessivos.")
        return False

if __name__ == "__main__":
    test_v9_zeros_fix()