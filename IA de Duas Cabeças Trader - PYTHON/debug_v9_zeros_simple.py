"""
🔍 INVESTIGAÇÃO SIMPLIFICADA: Zeros na V9Optimus

FOCO: Encontrar a causa raiz dos zeros específicos reportados:
- features_extractor.input_projection.weight: 91.4% zeros
- market_context_encoder.regime_embedding.weight: 65.6% zeros
"""

import torch
import torch.nn as nn
import numpy as np
import os
import glob

def check_existing_v9_checkpoints():
    """Verificar zeros em checkpoints V9 existentes"""
    print("🔍 VERIFICANDO CHECKPOINTS V9 EXISTENTES")
    print("=" * 60)
    
    # Procurar por checkpoints ou modelos V9
    search_patterns = [
        "D:/Projeto/**/*v9*",
        "D:/Projeto/**/*V9*", 
        "D:/Projeto/**/*optimus*",
        "D:/Projeto/**/*4dim*"
    ]
    
    # Usar glob para encontrar arquivos
    import glob
    
    potential_files = []
    for pattern in search_patterns:
        try:
            files = glob.glob(pattern, recursive=True)
            potential_files.extend(files)
        except:
            pass
    
    # Filtrar apenas arquivos relevantes (.zip, .pkl, .pth)
    model_files = [f for f in potential_files if any(f.endswith(ext) for ext in ['.zip', '.pkl', '.pth'])]
    
    print(f"📋 Encontrados {len(model_files)} arquivos de modelo:")
    for f in model_files[:10]:  # Mostrar primeiros 10
        print(f"  - {f}")
    
    # Tentar carregar alguns
    for model_file in model_files[:3]:
        try:
            print(f"\n📊 Analisando: {os.path.basename(model_file)}")
            
            # Tentar carregar com torch primeiro
            try:
                data = torch.load(model_file, map_location='cpu')
                
                # Verificar se é um modelo SB3 ou torch puro
                if 'policy' in data:
                    # SB3 model
                    policy_state = data['policy']
                    print("  📋 Tipo: SB3 model")
                    
                elif 'model' in data:
                    # Pode ter model state
                    policy_state = data['model']
                    print("  📋 Tipo: Model state")
                    
                elif isinstance(data, dict):
                    # Torch state dict direto
                    policy_state = data
                    print("  📋 Tipo: Torch state dict")
                    
                else:
                    print("  ❌ Formato não reconhecido")
                    continue
                
                # Procurar pelas camadas específicas
                input_proj_key = None
                regime_emb_key = None
                
                for key in policy_state.keys():
                    if 'input_projection.weight' in key:
                        input_proj_key = key
                    if 'regime_embedding.weight' in key:
                        regime_emb_key = key
                
                if input_proj_key:
                    weight = policy_state[input_proj_key]
                    zeros_pct = (weight.abs() < 1e-8).float().mean().item() * 100
                    print(f"  🎯 {input_proj_key}: {zeros_pct:.1f}% zeros")
                    
                if regime_emb_key:
                    weight = policy_state[regime_emb_key]
                    zeros_pct = (weight.abs() < 1e-8).float().mean().item() * 100
                    print(f"  🎯 {regime_emb_key}: {zeros_pct:.1f}% zeros")
                
                if not input_proj_key and not regime_emb_key:
                    print("  📋 Chaves encontradas:")
                    keys = [k for k in policy_state.keys() if 'weight' in k][:5]
                    for k in keys:
                        print(f"    - {k}")
                        
            except Exception as e:
                print(f"  ❌ Erro ao carregar com torch: {e}")
                
                # Tentar como SB3
                try:
                    from sb3_contrib import RecurrentPPO
                    model = RecurrentPPO.load(model_file)
                    
                    policy = model.policy
                    
                    # Verificar se tem os atributos esperados
                    if hasattr(policy, 'features_extractor') and hasattr(policy.features_extractor, 'input_projection'):
                        input_proj = policy.features_extractor.input_projection
                        input_zeros = (input_proj.weight.abs() < 1e-8).float().mean().item() * 100
                        print(f"  🎯 input_projection.weight: {input_zeros:.1f}% zeros")
                    
                    if hasattr(policy, 'market_context_encoder') and hasattr(policy.market_context_encoder, 'regime_embedding'):
                        regime_emb = policy.market_context_encoder.regime_embedding
                        regime_zeros = (regime_emb.weight.abs() < 1e-8).float().mean().item() * 100
                        print(f"  🎯 regime_embedding.weight: {regime_zeros:.1f}% zeros")
                        
                except Exception as e2:
                    print(f"  ❌ Erro ao carregar como SB3: {e2}")
        
        except Exception as e:
            print(f"  ❌ Erro geral: {e}")

def analyze_daytrader_4dim():
    """Analisar o 4dim.py atual para ver como está configurado"""
    print("\n🔍 ANALISANDO 4DIM.PY ATUAL")
    print("=" * 60)
    
    try:
        # Verificar se 4dim.py existe e tem configuração V9
        if os.path.exists("D:/Projeto/4dim.py"):
            print("📋 4dim.py encontrado")
            
            # Ler algumas linhas para verificar qual policy está sendo usada
            with open("D:/Projeto/4dim.py", "r", encoding="utf-8") as f:
                content = f.read()
            
            # Procurar por imports e configurações V9
            if "TwoHeadV9Optimus" in content:
                print("  ✅ Configurado para TwoHeadV9Optimus")
            elif "TwoHeadV8" in content:
                print("  📋 Configurado para TwoHeadV8")
            else:
                print("  ❓ Policy não identificada claramente")
            
            # Procurar por configurações específicas
            if "ortho_init" in content:
                lines = content.split('\n')
                for line in lines:
                    if "ortho_init" in line and not line.strip().startswith('#'):
                        print(f"  📋 {line.strip()}")
            
            # Procurar por features_dim
            if "features_dim" in content:
                lines = content.split('\n')
                for line in lines:
                    if "features_dim" in line and not line.strip().startswith('#') and "=" in line:
                        print(f"  📋 {line.strip()}")
        
        else:
            print("❌ 4dim.py não encontrado")
            
    except Exception as e:
        print(f"❌ Erro ao analisar 4dim.py: {e}")

def test_immediate_zeros_after_creation():
    """Teste mais direto: criar V9 e verificar imediatamente"""
    print("\n🔍 TESTE DIRETO: V9 recém-criada")
    print("=" * 60)
    
    try:
        from trading_framework.policies.two_head_v9_optimus import TwoHeadV9Optimus, get_v9_optimus_kwargs
        import gym
        
        # Criar policy diretamente
        obs_space = gym.spaces.Box(low=-1, high=1, shape=(450,), dtype=np.float32)
        action_space = gym.spaces.Box(low=np.array([0, 0, -1, -1]), high=np.array([2, 1, 1, 1]), dtype=np.float32)
        
        def dummy_lr_schedule(progress):
            return 1e-4
        
        print("📋 Criando TwoHeadV9Optimus...")
        
        policy = TwoHeadV9Optimus(
            observation_space=obs_space,
            action_space=action_space,
            lr_schedule=dummy_lr_schedule,
            **get_v9_optimus_kwargs()
        )
        
        print("📊 Verificando zeros imediatamente após criação:")
        
        # Verificar input_projection
        if hasattr(policy.features_extractor, 'input_projection'):
            input_proj = policy.features_extractor.input_projection
            input_zeros = (input_proj.weight.abs() < 1e-8).float().mean().item() * 100
            input_std = input_proj.weight.std().item()
            print(f"  🎯 input_projection.weight: {input_zeros:.1f}% zeros (std: {input_std:.6f})")
        
        # Verificar regime_embedding
        if hasattr(policy.market_context_encoder, 'regime_embedding'):
            regime_emb = policy.market_context_encoder.regime_embedding
            regime_zeros = (regime_emb.weight.abs() < 1e-8).float().mean().item() * 100
            regime_std = regime_emb.weight.std().item()
            print(f"  🎯 regime_embedding.weight: {regime_zeros:.1f}% zeros (std: {regime_std:.6f})")
        
        # Verificar outras camadas críticas
        print("\n📋 Verificando outras camadas:")
        
        for name, module in policy.named_modules():
            if isinstance(module, nn.Linear) and hasattr(module, 'weight'):
                zeros_pct = (module.weight.abs() < 1e-8).float().mean().item() * 100
                if zeros_pct > 10:  # Só mostrar se > 10% zeros
                    print(f"  ⚠️ {name}: {zeros_pct:.1f}% zeros")
        
        return policy
        
    except Exception as e:
        print(f"❌ Erro ao criar V9: {e}")
        return None

def compare_initialization_methods():
    """Comparar diferentes métodos de inicialização"""
    print("\n🔍 COMPARANDO MÉTODOS DE INICIALIZAÇÃO")
    print("=" * 60)
    
    # Teste 1: Linear layer isolado com gain=0.6
    print("📋 Teste 1: Linear layer com Xavier gain=0.6")
    linear1 = nn.Linear(45, 128)
    nn.init.xavier_uniform_(linear1.weight, gain=0.6)
    zeros1 = (linear1.weight.abs() < 1e-8).float().mean().item() * 100
    print(f"  Zeros: {zeros1:.1f}%")
    
    # Teste 2: Embedding isolado
    print("\n📋 Teste 2: Embedding padrão")
    emb1 = nn.Embedding(4, 32)
    zeros2 = (emb1.weight.abs() < 1e-8).float().mean().item() * 100
    print(f"  Zeros: {zeros2:.1f}%")
    
    # Teste 3: Embedding com inicialização manual
    print("\n📋 Teste 3: Embedding com Xavier")
    emb2 = nn.Embedding(4, 32)
    nn.init.xavier_uniform_(emb2.weight, gain=0.8)
    zeros3 = (emb2.weight.abs() < 1e-8).float().mean().item() * 100
    print(f"  Zeros: {zeros3:.1f}%")

def main():
    """Executar investigação simplificada"""
    print("🚀 INVESTIGAÇÃO SIMPLIFICADA: Zeros V9Optimus")
    print("=" * 80)
    
    # 1. Verificar checkpoints existentes
    check_existing_v9_checkpoints()
    
    # 2. Analisar configuração atual
    analyze_daytrader_4dim()
    
    # 3. Teste direto de criação
    policy = test_immediate_zeros_after_creation()
    
    # 4. Comparar inicializações
    compare_initialization_methods()
    
    print("\n" + "="*80)
    print("🎯 RESULTADOS DA INVESTIGAÇÃO:")
    print("- Verificamos checkpoints existentes para zeros")
    print("- Analisamos configuração atual do sistema")
    print("- Testamos criação direta da V9Optimus")
    print("- Comparamos métodos de inicialização")

if __name__ == "__main__":
    main()