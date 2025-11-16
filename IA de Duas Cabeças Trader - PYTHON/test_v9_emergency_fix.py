"""
TESTE FINAL DE EMERGÊNCIA - Verificar se V9 está protegida contra zeros
"""
import torch
import numpy as np
import gym
from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs, fix_v9_optimus_weights

# Simular criação igual ao 4dim.py
def simulate_4dim_model_creation():
    """Simula criação do modelo exatamente como no 4dim.py"""
    
    print("🚨 SIMULANDO CRIAÇÃO MODELO 4DIM.PY...")
    
    # Criar espaços
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([0, 0, -1, -1]), 
        high=np.array([2, 1, 1, 1]), 
        dtype=np.float32
    )
    
    # Mock PPO model
    class MockPPOModel:
        def __init__(self):
            def lr_schedule(progress):
                return 2e-5  # MESMO LR do 4dim.py
            
            kwargs = get_v9_optimus_kwargs()
            self.policy = TwoHeadV9Optimus(
                observation_space=obs_space,
                action_space=action_space,
                lr_schedule=lr_schedule,
                **kwargs
            )
    
    model = MockPPOModel()
    
    print("📊 ANTES DO FIX_V9_OPTIMUS_WEIGHTS:")
    check_zeros_detailed(model.policy, "ANTES")
    
    # Executar fix igual ao 4dim.py
    print("\n🔧 EXECUTANDO fix_v9_optimus_weights...")
    fix_v9_optimus_weights(model)
    
    print("\n📊 APÓS FIX_V9_OPTIMUS_WEIGHTS:")
    check_zeros_detailed(model.policy, "APÓS")
    
    # Simular primeiro forward (primeiro rollout)
    print("\n🚀 SIMULANDO PRIMEIRO ROLLOUT...")
    obs = torch.randn(4, 450)  # batch_size=4 como no PPO
    
    # Forward pass
    features = model.policy.features_extractor(obs)
    print(f"   Features shape: {features.shape}")
    
    print("\n📊 APÓS PRIMEIRO FORWARD:")
    check_zeros_detailed(model.policy, "PRIMEIRO_FORWARD")
    
    return model

def check_zeros_detailed(policy, stage):
    """Verifica zeros detalhadamente"""
    
    def check_layer(obj, path, tensor_attr):
        if hasattr(obj, tensor_attr):
            tensor = getattr(obj, tensor_attr)
            if tensor is not None:
                zeros_pct = (tensor.abs() < 1e-8).float().mean().item() * 100
                status = "✅" if zeros_pct < 50 else "🚨"
                print(f"   {path}.{tensor_attr}: {zeros_pct:.1f}% zeros {status}")
                return zeros_pct
        return None
    
    print(f"🔍 [{stage}] ANÁLISE DETALHADA DE ZEROS:")
    
    # Features extractor
    if hasattr(policy, 'features_extractor'):
        fe = policy.features_extractor
        check_layer(fe, "features_extractor", "input_projection.weight")
        
        if hasattr(fe, '_residual_projection'):
            check_layer(fe, "features_extractor", "_residual_projection.weight")
        else:
            print("   features_extractor._residual_projection: NÃO EXISTE AINDA")
    
    # Market context encoder
    if hasattr(policy, 'market_context_encoder'):
        mce = policy.market_context_encoder
        check_layer(mce, "market_context_encoder", "regime_embedding.weight")

def main():
    """Teste principal"""
    print("🚨 TESTE DE EMERGÊNCIA V9 ANTI-ZEROS")
    print("=" * 60)
    
    model = simulate_4dim_model_creation()
    
    print("\n" + "=" * 60)
    print("🎯 TESTE CONCLUÍDO!")
    
    # Verificação final
    final_check = True
    
    if hasattr(model.policy, 'features_extractor'):
        fe = model.policy.features_extractor
        if hasattr(fe, 'input_projection'):
            zeros = (fe.input_projection.weight.abs() < 1e-8).float().mean().item()
            if zeros > 0.5:
                final_check = False
                print("❌ input_projection ainda tem muitos zeros!")
    
    if hasattr(model.policy, 'market_context_encoder'):
        mce = model.policy.market_context_encoder
        if hasattr(mce, 'regime_embedding'):
            zeros = (mce.regime_embedding.weight.abs() < 1e-8).float().mean().item()
            if zeros > 0.5:
                final_check = False
                print("❌ regime_embedding ainda tem muitos zeros!")
    
    if final_check:
        print("🎉 SUCESSO! V9 protegida contra zeros!")
    else:
        print("❌ FALHA! Zeros persistem!")
    
    return final_check

if __name__ == "__main__":
    main()