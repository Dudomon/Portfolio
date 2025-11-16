#!/usr/bin/env python3
"""
🔍 DEBUG CHERRY ACTION PROCESSING
================================

Testar ações dentro do range correto [0,1] para validar o processamento.
"""

import sys
import os
import numpy as np
import torch
from datetime import datetime

sys.path.append("D:/Projeto")

def test_cherry_correct_actions():
    """Testar ações no range correto [0,1]"""
    print("🔍 TESTE CHERRY ACTIONS CORRETAS [0,1]")
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
        
        # Usar últimas 5000 barras
        if len(data) > 5000:
            data = data.iloc[-5000:].reset_index(drop=True)
        
        # Criar ambiente
        env = TradingEnv(
            df=data,
            window_size=20,
            is_training=True,
            initial_balance=500.0,
            trading_params={
                'min_lot_size': 0.02,
                'max_lot_size': 0.03,
                'enable_shorts': True,
                'max_positions': 2
            }
        )
        
        # Reset
        obs = env.reset()
        print(f"✅ Environment reset. Obs shape: {obs.shape}")
        
        # TESTE AÇÕES CORRETAS NO RANGE [0,1]
        print("\n🎯 TESTANDO AÇÕES CORRETAS [0,1]:")
        print("=" * 40)
        
        # Ações dentro do range [0,1] correto
        test_actions = [
            [0.5, 0.8, 0.0, 0.0],   # LONG (0.33 < 0.5 < 0.67)
            [0.8, 0.9, 0.0, 0.0],   # SHORT (0.8 >= 0.67)
            [0.1, 0.7, 0.0, 0.0],   # HOLD (0.1 < 0.33)
            [0.4, 0.8, 0.0, 0.0],   # LONG (0.33 < 0.4 < 0.67)
            [0.9, 0.9, 0.0, 0.0],   # SHORT (0.9 >= 0.67)
        ]
        
        total_trades = 0
        
        for i, action in enumerate(test_actions):
            print(f"\n🎯 Step {i+1}: Action={action}")
            
            # Calcular entry_decision baseado nos thresholds
            raw_decision = action[0]
            if raw_decision < 0.33:
                expected_decision = "HOLD"
            elif raw_decision < 0.67:
                expected_decision = "LONG"
            else:
                expected_decision = "SHORT"
            
            print(f"  Raw decision: {raw_decision:.3f}")
            print(f"  Expected: {expected_decision}")
            print(f"  Confidence: {action[1]:.3f}")
            
            # Executar step
            action_array = np.array(action, dtype=np.float32)
            obs, reward, done, info = env.step(action_array)
            
            # Verificar resultado
            trade_executed = info.get('trade_executed', False)
            positions_count = len(getattr(env, 'positions', []))
            
            print(f"  ✅ Trade executed: {trade_executed}")
            print(f"  📊 Positions: {positions_count}")
            print(f"  💰 Reward: {reward:.4f}")
            
            if trade_executed:
                total_trades += 1
                print(f"  🔥 TRADE EXECUTADO! Total: {total_trades}")
            
            # Debug info adicional
            if 'debug_info' in info:
                print(f"  🔍 Debug: {info['debug_info']}")
        
        print(f"\n📊 RESULTADO FINAL:")
        print(f"  Total trades executados: {total_trades}")
        print(f"  Portfolio final: ${env.portfolio_value:.2f}")
        
        # Teste com mais steps se nenhum trade foi executado
        if total_trades == 0:
            print(f"\n🚨 NENHUM TRADE EXECUTADO - TESTANDO MÚLTIPLAS ITERAÇÕES:")
            
            # Reset ambiente
            obs = env.reset()
            
            # Testar 50 iterações com ações LONG de alta confiança
            for step in range(50):
                action = np.array([0.5, 0.8, 0.0, 0.0], dtype=np.float32)  # LONG com alta confiança
                obs, reward, done, info = env.step(action)
                
                if info.get('trade_executed', False):
                    total_trades += 1
                    print(f"  Step {step}: ✅ TRADE EXECUTADO! Total: {total_trades}")
                    break
                    
                if step < 5:
                    print(f"  Step {step}: positions={len(env.positions)}, portfolio=${env.portfolio_value:.2f}")
        
        return total_trades > 0
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
    
    success = test_cherry_correct_actions()
    
    if success:
        print(f"\n✅ TESTE BEM-SUCEDIDO - TRADES EXECUTADOS")
    else:
        print(f"\n❌ TESTE FALHOU - NENHUM TRADE EXECUTADO")
    
    print(f"⏰ Fim: {datetime.now().strftime('%H:%M:%S')}")