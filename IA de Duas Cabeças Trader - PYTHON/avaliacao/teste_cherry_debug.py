#!/usr/bin/env python3
"""
🍒 TESTE CHERRY DEBUG - INVESTIGAR AMBIENTE
==========================================

Teste direto do TradingEnv do Cherry para investigar por que 0 trades.
Vamos testar o próprio ambiente Cherry step by step.
"""

import sys
import os
import numpy as np
import pandas as pd
import torch
from datetime import datetime

sys.path.append("D:/Projeto")

def test_cherry_env_direct():
    """Testar TradingEnv do Cherry diretamente"""
    print("🍒 TESTE DIRETO DO TRADING ENV CHERRY")
    print("=" * 60)
    
    # Mudar para diretório correto
    original_cwd = os.getcwd()
    os.chdir("D:/Projeto")
    
    try:
        # Importar Cherry
        from cherry import load_optimized_data_original, TradingEnv
        
        # Carregar dados
        print("📊 Carregando dados Cherry...")
        data = load_optimized_data_original()
        print(f"✅ Dados carregados: {len(data)} barras")
        
        # Usar últimas 5000 barras (teste rápido)
        if len(data) > 5000:
            data = data.iloc[-5000:].reset_index(drop=True)
            print(f"📅 Usando dados recentes: {len(data)} barras")
        
        # Criar ambiente Cherry
        print("\n🔧 Criando TradingEnv Cherry...")
        env = TradingEnv(
            df=data,
            window_size=20,
            is_training=True,  # FORÇAR modo treinamento
            initial_balance=500.0,
            trading_params={
                'min_lot_size': 0.02,
                'max_lot_size': 0.03,
                'enable_shorts': True,
                'max_positions': 2
            }
        )
        
        print(f"✅ Ambiente criado")
        print(f"🔍 Action Space: {env.action_space}")
        print(f"🔍 Obs Space: {env.observation_space.shape}")
        
        # Reset inicial
        print("\n🔄 Reset inicial...")
        obs = env.reset()
        print(f"✅ Obs shape: {obs.shape}")
        
        # Testar actions manuais
        print("\n🎯 TESTANDO ACTIONS MANUAIS:")
        print("=" * 40)
        
        test_actions = [
            # entry_decision, confidence, pos1_mgmt, pos2_mgmt
            [1.0, 0.8, 0.0, 0.0],  # LONG com alta confiança
            [2.0, 0.9, 0.0, 0.0],  # SHORT com alta confiança
            [1.5, 0.7, 0.0, 0.0],  # Entry moderado
            [0.8, 0.6, 0.0, 0.0],  # Entry baixo mas com confiança
            [0.0, 0.0, 0.0, 0.0],  # HOLD
        ]
        
        total_trades = 0
        
        for i, action in enumerate(test_actions):
            print(f"\nStep {i+1}: Action={action}")
            
            # Converter para numpy array
            action = np.array(action, dtype=np.float32)
            
            # Executar step
            obs, reward, done, info = env.step(action)
            
            # Verificar resultado
            trades_executed = info.get('trade_executed', False)
            positions_count = len(getattr(env, 'positions', []))
            portfolio_value = env.portfolio_value
            
            print(f"  Trade executado: {trades_executed}")
            print(f"  Posições ativas: {positions_count}")
            print(f"  Portfolio: ${portfolio_value:.2f}")
            print(f"  Reward: {reward:.4f}")
            
            if trades_executed:
                total_trades += 1
                print("  ✅ TRADE EXECUTADO!")
            else:
                print("  ❌ Nenhum trade")
            
            # Info adicional
            if 'debug_info' in info:
                print(f"  Debug: {info['debug_info']}")
        
        print(f"\n📊 RESULTADO FINAL:")
        print(f"  Total trades executados: {total_trades}")
        print(f"  Portfolio final: ${env.portfolio_value:.2f}")
        
        # Teste com modelo real se disponível
        print(f"\n🤖 TESTANDO COM MODELO CHERRY:")
        print("=" * 40)
        
        model_path = "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_1000000_steps_20250905_112708.zip"
        
        if os.path.exists(model_path):
            try:
                from sb3_contrib import RecurrentPPO
                model = RecurrentPPO.load(model_path)
                model.policy.set_training_mode(False)
                print("✅ Modelo carregado")
                
                # Reset ambiente
                obs = env.reset()
                lstm_states = None
                model_trades = 0
                
                # Testar 100 steps
                print("🚀 Executando 100 steps com modelo...")
                for step in range(100):
                    # Predict
                    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                    
                    # Debug primeiros 5 steps
                    if step < 5:
                        print(f"  Step {step}: Action={action}, Entry={action[0]:.3f}, Conf={action[1]:.3f}")
                    
                    # Step
                    obs, reward, done, info = env.step(action)
                    
                    if info.get('trade_executed', False):
                        model_trades += 1
                        print(f"  Step {step}: ✅ TRADE EXECUTADO! Total: {model_trades}")
                
                print(f"\n📊 RESULTADO MODELO:")
                print(f"  Trades executados: {model_trades}")
                print(f"  Portfolio final: ${env.portfolio_value:.2f}")
                
            except Exception as e:
                print(f"❌ Erro ao testar modelo: {e}")
        else:
            print("⚠️ Modelo não encontrado")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        success = test_cherry_env_direct()
        
        if success:
            print(f"\n✅ TESTE CONCLUÍDO")
        else:
            print(f"\n❌ TESTE FALHOU")
            
    except KeyboardInterrupt:
        print(f"\n⏹️ Interrompido")
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
    
    print(f"⏰ Fim: {datetime.now().strftime('%H:%M:%S')}")