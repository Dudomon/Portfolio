"""
🔍 INVESTIGAÇÃO PROFUNDA: Morte de neurônios na V9 durante treino
"""
import torch
import torch.nn as nn
import numpy as np
import gym
from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs, fix_v9_optimus_weights

def monitor_weights(model, step_name):
    """Monitora estado dos pesos em cada etapa"""
    print(f"\n{'='*60}")
    print(f"🔍 MONITORAMENTO: {step_name}")
    print(f"{'='*60}")
    
    results = {}
    
    # 1. Features Extractor
    if hasattr(model.policy, 'features_extractor'):
        fe = model.policy.features_extractor
        
        # input_projection
        if hasattr(fe, 'input_projection'):
            weight = fe.input_projection.weight
            zeros = (weight.abs() < 1e-8).float().mean().item() * 100
            mean_val = weight.abs().mean().item()
            std_val = weight.std().item()
            results['input_projection'] = {'zeros': zeros, 'mean': mean_val, 'std': std_val}
            
            status = "✅" if zeros < 50 else "🚨"
            print(f"   input_projection: {zeros:.1f}% zeros | mean={mean_val:.4f} | std={std_val:.4f} {status}")
            
            # Verificar gradiente se existir
            if weight.grad is not None:
                grad_zeros = (weight.grad.abs() < 1e-8).float().mean().item() * 100
                grad_mean = weight.grad.abs().mean().item()
                print(f"      → grad: {grad_zeros:.1f}% zeros | mean={grad_mean:.6f}")
        
        # _residual_projection
        if hasattr(fe, '_residual_projection'):
            weight = fe._residual_projection.weight
            zeros = (weight.abs() < 1e-8).float().mean().item() * 100
            mean_val = weight.abs().mean().item()
            std_val = weight.std().item()
            results['_residual_projection'] = {'zeros': zeros, 'mean': mean_val, 'std': std_val}
            
            status = "✅" if zeros < 50 else "🚨"
            print(f"   _residual_projection: {zeros:.1f}% zeros | mean={mean_val:.4f} | std={std_val:.4f} {status}")
            
            if weight.grad is not None:
                grad_zeros = (weight.grad.abs() < 1e-8).float().mean().item() * 100
                grad_mean = weight.grad.abs().mean().item()
                print(f"      → grad: {grad_zeros:.1f}% zeros | mean={grad_mean:.6f}")
    
    # 2. Market Context Encoder
    if hasattr(model.policy, 'market_context_encoder'):
        mce = model.policy.market_context_encoder
        
        if hasattr(mce, 'regime_embedding'):
            weight = mce.regime_embedding.weight
            zeros = (weight.abs() < 1e-8).float().mean().item() * 100
            mean_val = weight.abs().mean().item()
            std_val = weight.std().item()
            results['regime_embedding'] = {'zeros': zeros, 'mean': mean_val, 'std': std_val}
            
            status = "✅" if zeros < 50 else "🚨"
            print(f"   regime_embedding: {zeros:.1f}% zeros | mean={mean_val:.4f} | std={std_val:.4f} {status}")
            
            if weight.grad is not None:
                grad_zeros = (weight.grad.abs() < 1e-8).float().mean().item() * 100
                grad_mean = weight.grad.abs().mean().item()
                print(f"      → grad: {grad_zeros:.1f}% zeros | mean={grad_mean:.6f}")
    
    # 3. LSTM
    if hasattr(model.policy, 'lstm_actor'):
        lstm = model.policy.lstm_actor.lstm_cell if hasattr(model.policy.lstm_actor, 'lstm_cell') else model.policy.lstm_actor
        
        # Verificar weight_ih_l0
        if hasattr(lstm, 'weight_ih_l0'):
            weight = lstm.weight_ih_l0
            zeros = (weight.abs() < 1e-8).float().mean().item() * 100
            mean_val = weight.abs().mean().item()
            print(f"   lstm.weight_ih: {zeros:.1f}% zeros | mean={mean_val:.4f}")
    
    return results

def simulate_real_training():
    """Simula exatamente o que acontece no treino real do 4dim.py"""
    
    print("🚀 SIMULAÇÃO REALISTA DO TREINO V9...")
    
    # 1. Criar ambiente mock
    obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
    action_space = gym.spaces.Box(
        low=np.array([0, 0, -1, -1]), 
        high=np.array([2, 1, 1, 1]), 
        dtype=np.float32
    )
    
    class MockEnv(gym.Env):
        def __init__(self):
            self.observation_space = obs_space
            self.action_space = action_space
            
        def reset(self):
            return np.random.randn(450).astype(np.float32)
        
        def step(self, action):
            return self.reset(), 0.0, False, {}
    
    env = DummyVecEnv([lambda: MockEnv()])
    
    # 2. Criar modelo EXATAMENTE como no 4dim.py
    print("\n📊 PASSO 1: Criando RecurrentPPO...")
    
    def lr_schedule(progress):
        return 2e-5
    
    model_config = {
        "policy": TwoHeadV9Optimus,
        "env": env,
        "learning_rate": lr_schedule,
        "n_steps": 1024,
        "batch_size": 64,
        "n_epochs": 8,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.12,
        "ent_coef": 0.1,
        "vf_coef": 0.5,
        "use_sde": False,
        "policy_kwargs": get_v9_optimus_kwargs(),
    }
    
    model = RecurrentPPO(**model_config)
    
    # Monitorar após criação
    monitor_weights(model, "APÓS RecurrentPPO.__init__")
    
    # 3. Aplicar fix_v9_optimus_weights
    print("\n📊 PASSO 2: Aplicando fix_v9_optimus_weights...")
    fix_v9_optimus_weights(model)
    
    monitor_weights(model, "APÓS fix_v9_optimus_weights")
    
    # 4. Simular _fix_lstm_initialization (se não for V9)
    print("\n📊 PASSO 3: Verificando _fix_lstm_initialization...")
    policy_class_name = model.policy.__class__.__name__
    if "V9" in policy_class_name:
        print(f"   ✅ V9 detectada ({policy_class_name}) - _fix_lstm_initialization será pulado")
    else:
        print(f"   ⚠️ Aplicando _fix_lstm_initialization")
    
    # 5. Simular primeiros steps de treino
    print("\n📊 PASSO 4: Simulando primeiros steps de treino...")
    
    try:
        # Coletar rollout
        print("   Coletando rollout...")
        model.policy.train()
        
        obs = env.reset()
        for step in range(10):
            # Forward pass
            with torch.no_grad():
                actions, values, log_probs, lstm_states = model.policy.forward(
                    torch.tensor(obs).float(), 
                    None, 
                    None,
                    deterministic=False
                )
            
            # Step no ambiente
            obs, rewards, dones, infos = env.step(actions.cpu().numpy())
            
            if step % 3 == 0:
                monitor_weights(model, f"Durante rollout - step {step+1}")
        
        print("\n📊 PASSO 5: Simulando PPO update...")
        
        # Simular um mini-update
        # NOTA: Não podemos fazer model.learn() sem um ambiente completo
        # Mas podemos simular um forward/backward pass
        
        obs_tensor = torch.tensor(obs).float()
        
        # Forward com gradientes
        actions, values, log_probs, lstm_states = model.policy.forward(
            obs_tensor, None, None, deterministic=False
        )
        
        # Simular loss
        fake_loss = actions.sum() + values.sum()
        
        # Backward
        print("   Executando backward pass...")
        fake_loss.backward()
        
        monitor_weights(model, "APÓS backward pass")
        
        # Simular optimizer step
        print("\n   Executando optimizer.step()...")
        if hasattr(model.policy, 'optimizer'):
            model.policy.optimizer.step()
            model.policy.optimizer.zero_grad()
        
        monitor_weights(model, "APÓS optimizer.step()")
        
    except Exception as e:
        print(f"❌ Erro durante simulação: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Verificação final
    print("\n" + "="*60)
    print("📊 ANÁLISE FINAL")
    print("="*60)
    
    final_results = monitor_weights(model, "ESTADO FINAL")
    
    # Diagnóstico
    print("\n🔍 DIAGNÓSTICO:")
    problems = []
    
    for component, data in final_results.items():
        if data['zeros'] > 50:
            problems.append(f"   ❌ {component}: {data['zeros']:.1f}% zeros")
        elif data['zeros'] > 20:
            problems.append(f"   ⚠️ {component}: {data['zeros']:.1f}% zeros (alerta)")
        
        if data['mean'] < 1e-4:
            problems.append(f"   ❌ {component}: valores muito pequenos (mean={data['mean']:.6f})")
    
    if problems:
        print("PROBLEMAS ENCONTRADOS:")
        for p in problems:
            print(p)
    else:
        print("   ✅ Nenhum problema crítico detectado!")
    
    return final_results

if __name__ == "__main__":
    simulate_real_training()