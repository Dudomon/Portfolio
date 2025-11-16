#!/usr/bin/env python3
"""
🔍 DEBUG CHERRY INTENSIVO - FORÇAR LOGGING
==========================================

Script que força debug output no Cherry sem editar o arquivo principal.
"""

import sys
import os
import numpy as np
import torch
from datetime import datetime

sys.path.append("D:/Projeto")

def test_cherry_intensive_debug():
    """Teste intensivo com debug forçado"""
    print("🔍 DEBUG CHERRY INTENSIVO - MONKEYPATCH")
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
        
        # Usar últimas 1000 barras (teste ultra-rápido)
        if len(data) > 1000:
            data = data.iloc[-1000:].reset_index(drop=True)
        
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
        
        print(f"✅ Environment criado")
        print(f"🔍 Action Space: {env.action_space}")
        print(f"🔍 Max Positions: {env.max_positions}")
        
        # Reset
        obs = env.reset()
        print(f"✅ Environment reset")
        print(f"🔍 Current step: {env.current_step}")
        print(f"🔍 Positions: {len(env.positions)}")
        print(f"🔍 Slot cooldowns: {dict(env.slot_cooldowns)}")
        
        # TESTE ÚNICO COM AÇÃO PERFEITA
        print("\n🎯 TESTE ÚNICO - AÇÃO PERFEITA:")
        print("=" * 40)
        
        # Ação que deveria funcionar 100%
        perfect_action = np.array([0.5, 0.9, 0.0, 0.0], dtype=np.float32)
        print(f"🎯 Perfect Action: {perfect_action}")
        print(f"  Raw decision: {perfect_action[0]:.3f} (0.33 < 0.5 < 0.67 → LONG)")
        print(f"  Confidence: {perfect_action[1]:.3f} (0.9 > 0.3 threshold)")
        
        # DEBUG MANUAL dos thresholds
        ACTION_THRESHOLD_LONG = 0.33
        ACTION_THRESHOLD_SHORT = 0.67
        raw_decision = float(perfect_action[0])
        
        if raw_decision < ACTION_THRESHOLD_LONG:
            expected_entry = "HOLD"
            entry_decision_expected = 0
        elif raw_decision < ACTION_THRESHOLD_SHORT:
            expected_entry = "LONG"
            entry_decision_expected = 1
        else:
            expected_entry = "SHORT"
            entry_decision_expected = 2
        
        print(f"  Expected entry decision: {expected_entry} ({entry_decision_expected})")
        print(f"  Thresholds: LONG={ACTION_THRESHOLD_LONG}, SHORT={ACTION_THRESHOLD_SHORT}")
        
        # MONKEYPATCH: Adicionar debug ao método step
        original_step = env.step
        def debug_step(action):
            print(f"\n🔍 [MONKEYPATCH] STEP INICIADO")
            print(f"🔍 [MONKEYPATCH] Action recebida: {action}")
            print(f"🔍 [MONKEYPATCH] Current step: {env.current_step}")
            print(f"🔍 [MONKEYPATCH] Positions: {len(env.positions)}")
            print(f"🔍 [MONKEYPATCH] Max positions: {env.max_positions}")
            print(f"🔍 [MONKEYPATCH] Slot cooldowns: {dict(env.slot_cooldowns)}")
            
            # Chamar método original
            result = original_step(action)
            
            print(f"🔍 [MONKEYPATCH] STEP FINALIZADO")
            print(f"🔍 [MONKEYPATCH] Trade executed: {result[3].get('trade_executed', False)}")
            print(f"🔍 [MONKEYPATCH] Positions after: {len(env.positions)}")
            print(f"🔍 [MONKEYPATCH] Reward: {result[1]:.4f}")
            
            return result
        
        env.step = debug_step
        
        # Executar step com debug
        obs, reward, done, info = env.step(perfect_action)
        
        # Resultado
        trade_executed = info.get('trade_executed', False)
        print(f"\n📊 RESULTADO:")
        print(f"  Trade executed: {trade_executed}")
        print(f"  Positions: {len(env.positions)}")
        print(f"  Portfolio: ${env.portfolio_value:.2f}")
        print(f"  Reward: {reward:.4f}")
        
        if not trade_executed:
            print(f"\n🚨 NENHUM TRADE - INVESTIGANDO INFO:")
            print(f"  Info completo: {info}")
            
            # Tentar várias ações em sequência
            print(f"\n🔄 TENTANDO MÚLTIPLAS AÇÕES:")
            for i in range(5):
                test_action = np.array([0.5, 0.95, 0.0, 0.0], dtype=np.float32)
                print(f"  Step {i+1}: Action {test_action}")
                
                obs, reward, done, info = env.step(test_action)
                trade_executed = info.get('trade_executed', False)
                print(f"    Trade: {trade_executed}, Positions: {len(env.positions)}")
                
                if trade_executed:
                    break
        
        return trade_executed
        
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
    
    success = test_cherry_intensive_debug()
    
    if success:
        print(f"\n✅ SUCESSO - TRADE EXECUTADO")
    else:
        print(f"\n❌ FALHA - NENHUM TRADE")
    
    print(f"⏰ Fim: {datetime.now().strftime('%H:%M:%S')}")