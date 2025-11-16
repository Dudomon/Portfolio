#!/usr/bin/env python3
"""
🔍 DIAGNÓSTICO DE AÇÕES - CHECKPOINT 9.95M STEPS
Analisa as ações que o modelo está tomando
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Adicionar projeto ao path
projeto_path = Path("D:/Projeto")
sys.path.insert(0, str(projeto_path))

def diagnosticar_modelo():
    print("🔍 DIAGNÓSTICO DE AÇÕES - CHECKPOINT 9.95M STEPS")
    print("=" * 60)
    
    # Carregar modelo
    checkpoint_path = projeto_path / "trading_framework/training/checkpoints/DAYTRADER/checkpoint_9950000_steps_20250805_120857.zip"
    
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(checkpoint_path)
        print(f"✅ Modelo carregado: {model.num_timesteps:,} steps")
        
        # Analisar action space
        print(f"\n🎯 ACTION SPACE:")
        if hasattr(model.action_space, 'low') and hasattr(model.action_space, 'high'):
            print(f"   Shape: {model.action_space.shape}")
            print(f"   Low:  {model.action_space.low}")
            print(f"   High: {model.action_space.high}")
        else:
            print(f"   Type: {type(model.action_space)}")
            print(f"   Space: {model.action_space}")
        
    except Exception as e:
        print(f"❌ Erro carregando modelo: {e}")
        return
    
    # Gerar observações de teste
    print(f"\n🧪 TESTANDO PREDIÇÕES COM DIFERENTES OBSERVAÇÕES")
    print("-" * 50)
    
    resultados_acoes = []
    
    for test_id in range(10):
        print(f"Teste {test_id+1}: ", end="")
        
        # Criar observação sintética (2580 features)
        np.random.seed(test_id)  # Diferentes seeds para variação
        
        if test_id < 5:
            # Primeiros 5: observações "normais"
            obs = np.random.randn(2580).astype(np.float32) * 0.1
        else:
            # Últimos 5: observações "extremas" para forçar ações
            obs = np.random.randn(2580).astype(np.float32) * 2.0
            obs[0:100] = 5.0  # Valores altos nas primeiras features
            obs[1000:1100] = -5.0  # Valores baixos em outras features
        
        try:
            action, _states = model.predict(obs, deterministic=True)
            
            # Analisar ação
            action_analysis = {
                'test_id': test_id + 1,
                'action': action.tolist(),
                'action_type': int(np.clip(action[0], 0, 2)),
                'quantity': float(action[1]) if len(action) > 1 else 0.0,
                'raw_action_0': float(action[0]),
                'raw_action_1': float(action[1]) if len(action) > 1 else 0.0
            }
            
            resultados_acoes.append(action_analysis)
            
            # Interpretar ação
            if action_analysis['action_type'] == 0:
                action_name = "HOLD"
            elif action_analysis['action_type'] == 1:
                action_name = "BUY"
            elif action_analysis['action_type'] == 2:
                action_name = "SELL"
            else:
                action_name = f"UNKNOWN({action_analysis['action_type']})"
            
            print(f"Ação: {action_name}, Raw: [{action[0]:.3f}, {action[1]:.3f}]")
            
        except Exception as e:
            print(f"ERRO: {e}")
    
    # Analisar padrões das ações
    print(f"\n📊 ANÁLISE DAS AÇÕES")
    print("-" * 50)
    
    if resultados_acoes:
        action_types = [r['action_type'] for r in resultados_acoes]
        quantities = [r['quantity'] for r in resultados_acoes]
        raw_actions_0 = [r['raw_action_0'] for r in resultados_acoes]
        raw_actions_1 = [r['raw_action_1'] for r in resultados_acoes]
        
        print(f"🎯 DISTRIBUIÇÃO DE TIPOS DE AÇÃO:")
        unique_types, counts = np.unique(action_types, return_counts=True)
        for action_type, count in zip(unique_types, counts):
            action_name = ['HOLD', 'BUY', 'SELL'][action_type] if action_type < 3 else f'UNKNOWN({action_type})'
            pct = count / len(action_types) * 100
            print(f"   {action_name}: {count}/{len(action_types)} ({pct:.1f}%)")
        
        print(f"\n📈 ESTATÍSTICAS DAS AÇÕES RAW:")
        print(f"   Action[0] - Médio: {np.mean(raw_actions_0):.4f}, Range: {np.min(raw_actions_0):.4f} → {np.max(raw_actions_0):.4f}")
        print(f"   Action[1] - Médio: {np.mean(raw_actions_1):.4f}, Range: {np.min(raw_actions_1):.4f} → {np.max(raw_actions_1):.4f}")
        
        print(f"\n💰 ESTATÍSTICAS DE QUANTIDADE:")
        print(f"   Quantidade Média: {np.mean(quantities):.4f}")
        print(f"   Range: {np.min(quantities):.4f} → {np.max(quantities):.4f}")
    
    # Teste com observação forçada para BUY
    print(f"\n🚀 TESTE FORÇADO PARA ESTIMULAR AÇÕES")
    print("-" * 50)
    
    try:
        # Criar observação que deveria estimular compra
        obs_buy = np.zeros(2580, dtype=np.float32)
        
        # Simular condições de alta (preços subindo, momentum positivo)
        obs_buy[0:50] = 2.0  # Features de preço/momentum altas
        obs_buy[50:100] = 1.0  # Features técnicas positivas
        obs_buy[100:150] = 0.5  # Features de volume
        
        action_buy, _ = model.predict(obs_buy, deterministic=True)
        print(f"Obs ALTA → Ação: {action_buy}, Tipo: {int(np.clip(action_buy[0], 0, 2))}")
        
        # Criar observação que deveria estimular venda
        obs_sell = np.zeros(2580, dtype=np.float32)
        obs_sell[0:50] = -2.0  # Features de preço/momentum baixas
        obs_sell[50:100] = -1.0  # Features técnicas negativas
        obs_sell[100:150] = 0.8  # Volume alto (pânico?)
        
        action_sell, _ = model.predict(obs_sell, deterministic=True)
        print(f"Obs BAIXA → Ação: {action_sell}, Tipo: {int(np.clip(action_sell[0], 0, 2))}")
        
        # Criar observação neutra
        obs_neutral = np.random.randn(2580).astype(np.float32) * 0.01  # Muito pequenos
        action_neutral, _ = model.predict(obs_neutral, deterministic=True)
        print(f"Obs NEUTRA → Ação: {action_neutral}, Tipo: {int(np.clip(action_neutral[0], 0, 2))}")
        
    except Exception as e:
        print(f"❌ Erro no teste forçado: {e}")
    
    # Salvar diagnóstico
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    diagnostico_path = projeto_path / "avaliacoes" / f"diagnostico_acoes_9950000_{timestamp}.txt"
    
    os.makedirs(projeto_path / "avaliacoes", exist_ok=True)
    
    with open(diagnostico_path, 'w', encoding='utf-8') as f:
        f.write(f"🔍 DIAGNÓSTICO DE AÇÕES - CHECKPOINT 9.95M STEPS\n")
        f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Checkpoint: {checkpoint_path.name}\n\n")
        
        f.write(f"ACTION SPACE:\n")
        if hasattr(model.action_space, 'low'):
            f.write(f"Shape: {model.action_space.shape}\n")
            f.write(f"Low: {model.action_space.low}\n")
            f.write(f"High: {model.action_space.high}\n\n")
        
        f.write(f"RESULTADOS DOS TESTES:\n")
        for r in resultados_acoes:
            f.write(f"Teste {r['test_id']}: Tipo={r['action_type']}, Raw=[{r['raw_action_0']:.4f}, {r['raw_action_1']:.4f}]\n")
        
        if resultados_acoes:
            f.write(f"\nESTATÍSTICAS:\n")
            action_types = [r['action_type'] for r in resultados_acoes]
            unique_types, counts = np.unique(action_types, return_counts=True)
            for action_type, count in zip(unique_types, counts):
                action_name = ['HOLD', 'BUY', 'SELL'][action_type] if action_type < 3 else f'UNKNOWN({action_type})'
                pct = count / len(action_types) * 100
                f.write(f"{action_name}: {count}/{len(action_types)} ({pct:.1f}%)\n")
    
    print(f"\n💾 Diagnóstico salvo: {diagnostico_path}")
    
    # Conclusão
    print(f"\n🏆 CONCLUSÃO DO DIAGNÓSTICO:")
    if resultados_acoes:
        predominant_action = max(set(action_types), key=action_types.count)
        action_names = ['HOLD', 'BUY', 'SELL']
        predominant_name = action_names[predominant_action] if predominant_action < 3 else 'UNKNOWN'
        
        if predominant_action == 0:
            print(f"   ⚠️ PROBLEMA: Modelo está predominantemente fazendo HOLD")
            print(f"   🔧 POSSÍVEIS CAUSAS:")
            print(f"      - Modelo muito conservador")
            print(f"      - Action space mal configurado")
            print(f"      - Observações não adequadas") 
            print(f"      - Necessário re-treinamento ou ajuste de reward")
        else:
            print(f"   ✅ Modelo mostra variação nas ações")
            print(f"   📊 Ação predominante: {predominant_name}")

if __name__ == "__main__":
    diagnosticar_modelo()