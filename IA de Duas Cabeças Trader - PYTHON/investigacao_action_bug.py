#!/usr/bin/env python3
"""
🔍 INVESTIGAÇÃO PROFUNDA - BUG ACTION[1] ESTRUTURAL
Vamos descobrir o que está acontecendo com o action space
"""

import sys
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

projeto_path = Path("D:/Projeto")
sys.path.insert(0, str(projeto_path))

def investigar_action_bug():
    print("🔍 INVESTIGAÇÃO PROFUNDA - BUG ACTION[1]")
    print("=" * 50)
    
    # 1. Carregar modelo e analisar architecture
    checkpoint_path = projeto_path / "trading_framework/training/checkpoints/DAYTRADER/checkpoint_phase2riskmanagement_650000_steps_20250805_201935.zip"
    
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(checkpoint_path)
        print(f"✅ Modelo carregado: {model.num_timesteps:,} steps")
        print(f"🧠 Policy: {type(model.policy).__name__}")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    # 2. Análise detalhada do Action Space
    print(f"\n🎯 ANÁLISE DETALHADA DO ACTION SPACE")
    print("-" * 40)
    
    action_space = model.action_space
    print(f"📊 Action Space Type: {type(action_space)}")
    print(f"📊 Shape: {action_space.shape}")
    print(f"📊 Dimensões: {action_space.shape[0] if hasattr(action_space, 'shape') else 'N/A'}")
    
    if hasattr(action_space, 'low') and hasattr(action_space, 'high'):
        print(f"📊 Low bounds:  {action_space.low}")
        print(f"📊 High bounds: {action_space.high}")
        
        print(f"\n🔍 MAPEAMENTO DAS AÇÕES (baseado nos bounds):")
        for i in range(len(action_space.low)):
            low = action_space.low[i]
            high = action_space.high[i]
            print(f"   Action[{i}]: [{low:4.1f}, {high:4.1f}] - ", end="")
            
            # Inferir significado baseado nos bounds
            if i == 0 and low == 0 and high == 2:
                print("Tipo de ordem (0=HOLD, 1=BUY, 2=SELL)")
            elif i == 1 and low == 0 and high == 1:
                print("Quantidade (0-100%)")
            elif low == -1 and high == 1:
                print("Flag binário (-1/+1)")
            elif low == 0 and high == 1:
                print("Flag binário (0/1)")
            elif low == -3 and high == 3:
                print("Ajuste Stop Loss / Take Profit")
            else:
                print(f"Parâmetro desconhecido")
    
    # 3. Análise da Policy Architecture
    print(f"\n🧠 ANÁLISE DA POLICY ARCHITECTURE")
    print("-" * 40)
    
    policy = model.policy
    print(f"📋 Policy class: {policy.__class__.__name__}")
    
    # Verificar se tem actor/critic heads separados
    if hasattr(policy, 'action_net'):
        print(f"✅ Action network encontrado")
        action_net = policy.action_net
        print(f"   Type: {type(action_net)}")
        
        # Verificar layers da action network
        if hasattr(action_net, 'children'):
            layers = list(action_net.children())
            print(f"   Layers: {len(layers)}")
            for i, layer in enumerate(layers):
                print(f"     Layer {i}: {layer}")
    
    if hasattr(policy, 'value_net'):
        print(f"✅ Value network encontrado")
    
    # 4. Testar predições com análise detalhada
    print(f"\n🧪 TESTE DETALHADO DE PREDIÇÕES")
    print("-" * 40)
    
    # Criar diferentes tipos de observação
    test_cases = [
        ("Neutro", np.random.randn(2580).astype(np.float32) * 0.01),
        ("Bullish", np.concatenate([np.ones(100) * 2.0, np.random.randn(2480).astype(np.float32) * 0.1])),
        ("Bearish", np.concatenate([-np.ones(100) * 2.0, np.random.randn(2480).astype(np.float32) * 0.1])),
        ("High Vol", np.random.randn(2580).astype(np.float32) * 3.0),
        ("Zeros", np.zeros(2580, dtype=np.float32)),
    ]
    
    resultados_detalhados = []
    
    for nome, obs in test_cases:
        try:
            # Predição com análise de gradientes
            action, _states = model.predict(obs, deterministic=True)
            
            resultado = {
                'nome': nome,
                'action_completa': action.tolist(),
                'action_shapes': action.shape,
                'action_dtypes': action.dtype
            }
            
            print(f"🔍 {nome}:")
            print(f"   Action shape: {action.shape}")
            print(f"   Action dtype: {action.dtype}")
            print(f"   Action[0]: {action[0]:.6f}")
            print(f"   Action[1]: {action[1]:.6f}")
            
            if len(action) > 2:
                print(f"   Action[2-5]: {[f'{a:.3f}' for a in action[2:6]]}")
                print(f"   Action[6-10]: {[f'{a:.3f}' for a in action[6:11]]}")
            
            resultados_detalhados.append(resultado)
            
        except Exception as e:
            print(f"❌ Erro no teste {nome}: {e}")
    
    # 5. Análise estatística profunda
    print(f"\n📊 ANÁLISE ESTATÍSTICA PROFUNDA")
    print("-" * 40)
    
    if resultados_detalhados:
        # Coletar todas as ações
        all_actions = [r['action_completa'] for r in resultados_detalhados]
        all_actions_array = np.array(all_actions)
        
        print(f"📈 ESTATÍSTICAS POR DIMENSÃO:")
        for i in range(all_actions_array.shape[1]):
            values = all_actions_array[:, i]
            print(f"   Action[{i}]: mean={np.mean(values):.6f}, std={np.std(values):.6f}, range=[{np.min(values):.6f}, {np.max(values):.6f}]")
            
            # Verificar se está sempre zero
            if np.max(np.abs(values)) < 1e-6:
                print(f"     🔴 SEMPRE ZERO!")
            elif np.std(values) < 1e-6:
                print(f"     🟡 SEMPRE CONSTANTE: {np.mean(values):.6f}")
            else:
                print(f"     ✅ Varia normalmente")
    
    # 6. Investigar internals da policy
    print(f"\n🔬 INVESTIGAÇÃO DOS INTERNALS DA POLICY")
    print("-" * 40)
    
    try:
        # Verificar parâmetros da action network
        if hasattr(policy, 'action_net'):
            action_net = policy.action_net
            
            print(f"🔍 ACTION NETWORK PARAMETERS:")
            total_params = 0
            for name, param in action_net.named_parameters():
                print(f"   {name}: {param.shape}")
                total_params += param.numel()
                
                # Verificar se há parâmetros zerados
                if torch.all(param == 0):
                    print(f"     🔴 PARÂMETRO ZERADO!")
                elif torch.std(param) < 1e-6:
                    print(f"     🟡 PARÂMETRO CONSTANTE")
            
            print(f"   Total params: {total_params:,}")
        
        # Verificar se há problemas de inicialização
        print(f"\n🔍 VERIFICAÇÃO DE INICIALIZAÇÃO:")
        obs_test = np.random.randn(2580).astype(np.float32) * 0.1
        
        # Forward pass step by step se possível
        if hasattr(policy, 'features_extractor'):
            print(f"   ✅ Features extractor existe")
            
        if hasattr(policy, 'mlp_extractor'):  
            print(f"   ✅ MLP extractor existe")
            
        if hasattr(policy, 'action_dist'):
            print(f"   ✅ Action distribution existe")
            print(f"       Type: {type(policy.action_dist)}")
            
    except Exception as e:
        print(f"❌ Erro na investigação interna: {e}")
    
    # 7. Teste com torch direto
    print(f"\n🧪 TESTE COM PYTORCH DIRETO")
    print("-" * 40)
    
    try:
        import torch
        
        # Converter observação para tensor
        obs_tensor = torch.FloatTensor(obs_test).unsqueeze(0)  # Add batch dim
        
        print(f"📊 Obs tensor shape: {obs_tensor.shape}")
        
        # Tentar forward pass manual
        with torch.no_grad():
            if hasattr(policy, 'forward'):
                print(f"🔍 Tentando forward pass manual...")
                # result = policy.forward(obs_tensor)  # Pode dar erro
                # print(f"   Forward result: {result}")
            
            # Verificar se action_net pode ser chamado diretamente
            if hasattr(policy, 'action_net') and hasattr(policy.action_net, 'forward'):
                print(f"🔍 Testando action_net direto...")
                # Isso pode dar erro dependendo da arquitetura
    
    except Exception as e:
        print(f"❌ Erro no teste pytorch: {e}")
    
    # 8. Conclusão da investigação
    print(f"\n🏆 CONCLUSÃO DA INVESTIGAÇÃO")
    print("=" * 50)
    
    print(f"🎯 ACHADOS PRINCIPAIS:")
    print(f"   📊 Action Space: 11 dimensões")
    print(f"   🎮 Action[0]: Tipo de ordem (funciona)")
    print(f"   💰 Action[1]: Quantidade (SEMPRE ZERO)")
    print(f"   🔧 Action[2-10]: Parâmetros SL/TP")
    
    print(f"\n🔍 HIPÓTESES PARA Action[1] = 0:")
    print(f"   1. 🧠 Policy head mal treinado para Action[1]")
    print(f"   2. 🎯 Inicialização ruim dos pesos da ação 1")
    print(f"   3. 🔒 Gates/máscaras bloqueando Action[1]")
    print(f"   4. 📊 Reward function não incentivou variação de quantidade")
    print(f"   5. 🏗️ Bug na arquitetura TwoHeadV7Intuition")
    
    print(f"\n💡 PRÓXIMOS PASSOS:")
    print(f"   1. Verificar pesos específicos da Action[1] head")
    print(f"   2. Analisar logs de treinamento para Action[1]")
    print(f"   3. Testar modelo mais simples sem V7 complexity")
    print(f"   4. Verificar se reward function usa quantidade")

def main():
    import torch
    investigar_action_bug()

if __name__ == "__main__":
    main()