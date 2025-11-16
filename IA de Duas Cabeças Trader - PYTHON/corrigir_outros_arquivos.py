#!/usr/bin/env python3
"""
🔧 CORREÇÃO DOS OUTROS ARQUIVOS V7 - LSTM bias
Corrigir TwoHeadV7Simple e TwoHeadV7Unified
"""

import sys
import os
from pathlib import Path

projeto_path = Path("D:/Projeto")
sys.path.insert(0, str(projeto_path))

def corrigir_v7_simple():
    """Corrigir TwoHeadV7Simple"""
    file_path = projeto_path / "trading_framework/policies/two_head_v7_simple.py"
    
    print("🔧 CORRIGINDO TwoHeadV7Simple...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procurar padrão problemático mais específico
    patterns_to_fix = [
        "elif 'bias' in name:\n                    torch.nn.init.zeros_(param)",
        "elif 'bias' in name:\n                torch.nn.init.zeros_(param)",
        "elif 'bias_' in name:\n                    torch.nn.init.zeros_(param)",
        "elif 'bias_' in name:\n                torch.nn.init.zeros_(param)"
    ]
    
    fixed = False
    for pattern in patterns_to_fix:
        if pattern in content:
            replacement = """elif 'bias' in name:
                    # CORREÇÃO CRÍTICA: Forget gate bias = 1.0
                    torch.nn.init.zeros_(param)
                    if param.size(0) >= 4:  # LSTM bias
                        hidden_size = param.size(0) // 4
                        param.data[hidden_size:2*hidden_size].fill_(1.0)  # Forget gate bias"""
            
            content = content.replace(pattern, replacement)
            fixed = True
            break
    
    if fixed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ TwoHeadV7Simple LSTM bias corrigido")
    else:
        print("⚠️ Padrão não encontrado em TwoHeadV7Simple - verificando manualmente...")
        
        # Procurar por qualquer inicialização LSTM
        if "torch.nn.init.zeros_(param)" in content and "bias" in content:
            print("   📍 Encontrado torch.nn.init.zeros_ com bias - requer correção manual")
        else:
            print("   ✅ TwoHeadV7Simple não parece ter o problema")

def corrigir_v7_unified():
    """Corrigir TwoHeadV7Unified"""
    file_path = projeto_path / "trading_framework/policies/two_head_v7_unified.py"
    
    print("🔧 CORRIGINDO TwoHeadV7Unified...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procurar padrão problemático
    patterns_to_fix = [
        "elif 'bias' in name:\n                nn.init.zeros_(param)",
        "elif 'bias' in name:\n            nn.init.zeros_(param)",
        "elif 'bias_' in name:\n                nn.init.zeros_(param)",
        "elif 'bias_' in name:\n            nn.init.zeros_(param)"
    ]
    
    fixed = False
    for pattern in patterns_to_fix:
        if pattern in content:
            replacement = """elif 'bias' in name:
                # CORREÇÃO CRÍTICA: Forget gate bias = 1.0
                nn.init.zeros_(param)
                if param.size(0) >= 4:  # LSTM bias
                    hidden_size = param.size(0) // 4
                    param.data[hidden_size:2*hidden_size].fill_(1.0)  # Forget gate bias"""
            
            content = content.replace(pattern, replacement)
            fixed = True
            break
    
    if fixed:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print("✅ TwoHeadV7Unified LSTM bias corrigido")
    else:
        print("⚠️ Padrão não encontrado em TwoHeadV7Unified - verificando manualmente...")
        
        # Procurar por qualquer inicialização LSTM
        if "nn.init.zeros_(param)" in content and "bias" in content:
            print("   📍 Encontrado nn.init.zeros_ com bias - requer correção manual")
        else:
            print("   ✅ TwoHeadV7Unified não parece ter o problema")

def criar_teste_inicializacao():
    """Criar teste para validar as correções"""
    
    test_code = """#!/usr/bin/env python3
\"\"\"
🧪 TESTE DA INICIALIZAÇÃO CORRIGIDA
Validar se as correções funcionaram
\"\"\"

import sys
import os
import numpy as np
import torch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

projeto_path = Path("D:/Projeto")
sys.path.insert(0, str(projeto_path))

def test_corrected_initialization():
    print("🧪 TESTANDO INICIALIZAÇÃO CORRIGIDA")
    print("=" * 50)
    
    try:
        # Importar o TwoHeadV7Intuition corrigido
        from trading_framework.policies.two_head_v7_intuition import TwoHeadV7Intuition
        from gym.spaces import Box
        
        # Criar action space
        action_space = Box(
            low=np.array([0., 0., -1., 0., -1., -3., -3., -3., -3., -3., -3.]),
            high=np.array([2., 1., 1., 1., 1., 3., 3., 3., 3., 3., 3.]),
            dtype=np.float32
        )
        
        # Criar observation space
        observation_space = Box(low=-np.inf, high=np.inf, shape=(2580,), dtype=np.float32)
        
        print("📊 Criando TwoHeadV7Intuition corrigida...")
        
        # Instanciar policy (vai executar a inicialização corrigida)
        policy = TwoHeadV7Intuition(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lambda x: 3e-4
        )
        
        print("✅ Policy criada com sucesso!")
        
        # Testar Action[1] bias
        if hasattr(policy, 'actor_head'):
            last_layer = policy.actor_head[-1]
            if hasattr(last_layer, 'bias'):
                action1_bias = last_layer.bias[1].item()
                print(f"🎯 Action[1] bias: {action1_bias:.3f}")
                
                if abs(action1_bias - 2.5) < 0.1:
                    print("✅ Action[1] bias CORRIGIDO!")
                else:
                    print(f"❌ Action[1] bias incorreto: {action1_bias} (esperado ~2.5)")
        
        # Testar LSTM bias
        if hasattr(policy, 'actor_lstm'):
            for name, param in policy.actor_lstm.named_parameters():
                if 'bias' in name:
                    print(f"🧠 LSTM {name} shape: {param.shape}")
                    if param.size(0) >= 4:
                        hidden_size = param.size(0) // 4
                        forget_bias = param.data[hidden_size:2*hidden_size]
                        forget_bias_mean = forget_bias.mean().item()
                        print(f"   Forget gate bias mean: {forget_bias_mean:.3f}")
                        
                        if abs(forget_bias_mean - 1.0) < 0.1:
                            print("   ✅ Forget gate bias CORRIGIDO!")
                        else:
                            print(f"   ❌ Forget gate bias incorreto: {forget_bias_mean} (esperado ~1.0)")
        
        print("\\n🎯 TESTE DE PREDIÇÃO:")
        
        # Teste de predição para verificar se Action[1] varia
        obs = np.random.randn(2580).astype(np.float32)
        
        action1_values = []
        for i in range(5):
            obs_var = np.random.randn(2580).astype(np.float32) * (i + 1)
            
            # Usar policy diretamente (sem modelo SB3)
            obs_tensor = torch.FloatTensor(obs_var).unsqueeze(0)
            
            with torch.no_grad():
                # Processar através da policy
                features = policy.extract_features(obs_tensor)
                actions, _, _ = policy.forward(obs_tensor)
                
                action1_val = actions[0, 1].item()
                action1_values.append(action1_val)
                print(f"   Test {i+1}: Action[1] = {action1_val:.6f}")
        
        action1_array = np.array(action1_values)
        action1_std = action1_array.std()
        
        print(f"\\n📊 Action[1] Statistics:")
        print(f"   Mean: {action1_array.mean():.6f}")
        print(f"   Std:  {action1_std:.6f}")
        print(f"   Range: [{action1_array.min():.6f}, {action1_array.max():.6f}]")
        
        if action1_std > 0.01:
            print("✅ Action[1] AGORA VARIA NORMALMENTE!")
        else:
            print("❌ Action[1] ainda constante")
        
        if action1_array.mean() > 0.5:
            print("✅ Action[1] valores iniciais altos (esperado)")
        else:
            print("⚠️ Action[1] valores baixos (pode ser normal após algumas iterações)")
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_corrected_initialization()"""
    
    test_file = projeto_path / "teste_inicializacao_corrigida.py"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_code)
    
    print(f"📝 Teste criado: {test_file}")
    return test_file

def main():
    print("🔧 CORRIGINDO OUTROS ARQUIVOS V7")
    print("=" * 50)
    
    # Corrigir arquivos
    corrigir_v7_simple()
    corrigir_v7_unified()
    
    # Criar teste
    test_file = criar_teste_inicializacao()
    
    print("\n🎯 RESUMO DAS CORREÇÕES:")
    print("✅ TwoHeadV7Intuition: Já corrigido")
    print("🔧 TwoHeadV7Simple: Processado") 
    print("🔧 TwoHeadV7Unified: Processado")
    print(f"📝 Teste criado: {test_file.name}")
    
    print("\n🚀 PRÓXIMO PASSO:")
    print("   1. Execute: python teste_inicializacao_corrigida.py")
    print("   2. Se Action[1] variar: SUCESSO! Pode re-treinar")
    print("   3. Se ainda for constante: Investigar mais")

if __name__ == "__main__":
    main()