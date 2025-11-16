#!/usr/bin/env python3
"""
🔧 TESTE ACTION MAPPING CHERRY - VERIFICAR CORREÇÃO
===================================================
"""

import sys
import os
import numpy as np
sys.path.append("D:/Projeto")

def test_action_mapping():
    """Testar mapeamento de ações corrigido"""
    print("🔧 TESTE ACTION MAPPING CHERRY")
    print("=" * 50)
    
    # Valores de teste para action[0] em range [0,2]
    test_values = [
        0.0,   # HOLD
        0.3,   # HOLD  
        0.66,  # HOLD (limite)
        0.67,  # LONG (início)
        1.0,   # LONG
        1.32,  # LONG (limite)
        1.33,  # SHORT (início)
        1.5,   # SHORT
        2.0,   # SHORT (máximo)
    ]
    
    # Thresholds corrigidos
    ACTION_THRESHOLD_LONG = 0.67
    ACTION_THRESHOLD_SHORT = 1.33
    
    hold_count = 0
    long_count = 0
    short_count = 0
    
    print("Value  | Decision | Action")
    print("-" * 30)
    
    for value in test_values:
        if value < ACTION_THRESHOLD_LONG:
            decision = "HOLD"
            entry_decision = 0
            hold_count += 1
        elif value < ACTION_THRESHOLD_SHORT:
            decision = "LONG" 
            entry_decision = 1
            long_count += 1
        else:
            decision = "SHORT"
            entry_decision = 2
            short_count += 1
            
        print(f"{value:5.2f}  | {decision:8} | {entry_decision}")
    
    print("\n📊 DISTRIBUIÇÃO:")
    total = len(test_values)
    print(f"HOLD:  {hold_count}/{total} = {hold_count/total*100:.1f}%")
    print(f"LONG:  {long_count}/{total} = {long_count/total*100:.1f}%") 
    print(f"SHORT: {short_count}/{total} = {short_count/total*100:.1f}%")
    
    # Teste estatístico com 1000 valores aleatórios
    print("\n🎲 TESTE ESTATÍSTICO (1000 samples):")
    random_values = np.random.uniform(0, 2, 1000)
    
    stat_hold = np.sum(random_values < ACTION_THRESHOLD_LONG)
    stat_long = np.sum((random_values >= ACTION_THRESHOLD_LONG) & (random_values < ACTION_THRESHOLD_SHORT))
    stat_short = np.sum(random_values >= ACTION_THRESHOLD_SHORT)
    
    print(f"HOLD:  {stat_hold}/1000 = {stat_hold/10:.1f}%")
    print(f"LONG:  {stat_long}/1000 = {stat_long/10:.1f}%")
    print(f"SHORT: {stat_short}/1000 = {stat_short/10:.1f}%")
    
    # Verificar se está balanceado (aproximadamente 33% cada)
    if 30 <= stat_hold/10 <= 36 and 30 <= stat_long/10 <= 36 and 30 <= stat_short/10 <= 36:
        print("✅ MAPEAMENTO BALANCEADO!")
        return True
    else:
        print("❌ MAPEAMENTO DESBALANCEADO!")
        return False

def test_cherry_env_with_fixed_mapping():
    """Testar ambiente Cherry com mapeamento corrigido"""
    print("\n🍒 TESTE AMBIENTE CHERRY COM CORREÇÃO")
    print("=" * 50)
    
    original_cwd = os.getcwd()
    os.chdir("D:/Projeto")
    
    try:
        from cherry import load_optimized_data_original, TradingEnv
        
        # Carregar dados pequenos para teste rápido
        print("📊 Carregando dados...")
        data = load_optimized_data_original()
        data = data.iloc[-1000:].reset_index(drop=True)  # Últimas 1000 barras
        print(f"✅ Dados: {len(data)} barras")
        
        # Criar ambiente
        env = TradingEnv(
            df=data,
            window_size=20,
            is_training=True,
            initial_balance=500.0
        )
        
        obs = env.reset()
        print(f"✅ Ambiente criado, obs shape: {obs.shape}")
        
        # Testar actions que devem funcionar agora
        test_actions = [
            [0.5, 0.8, 0.0, 0.0],   # HOLD (0.5 < 0.67)
            [1.0, 0.9, 0.0, 0.0],   # LONG (0.67 <= 1.0 < 1.33)
            [1.5, 0.7, 0.0, 0.0],   # SHORT (1.5 >= 1.33)
            [0.8, 0.6, 0.0, 0.0],   # LONG (0.67 <= 0.8 < 1.33)
            [1.4, 0.8, 0.0, 0.0],   # SHORT (1.4 >= 1.33)
        ]
        
        trades_executed = 0
        
        print("\n🎯 TESTANDO ACTIONS:")
        print("Action Value | Expected | Result | Trades")
        print("-" * 45)
        
        for i, action in enumerate(test_actions):
            action_array = np.array(action, dtype=np.float32)
            
            # Determinar ação esperada
            val = action[0]
            if val < 0.67:
                expected = "HOLD"
            elif val < 1.33:
                expected = "LONG"
            else:
                expected = "SHORT"
            
            obs, reward, done, info = env.step(action_array)
            trade_exec = info.get('trade_executed', False)
            if trade_exec:
                trades_executed += 1
                
            positions = len(getattr(env, 'positions', []))
            result = f"Trade: {trade_exec}, Pos: {positions}"
            
            print(f"{val:11.2f} | {expected:8} | {result:15} | {trades_executed}")
        
        print(f"\n📊 RESULTADO: {trades_executed} trades executados")
        
        if trades_executed >= 2:  # Pelo menos LONG e SHORT devem funcionar
            print("✅ MAPEAMENTO FUNCIONANDO!")
            return True
        else:
            print("❌ PROBLEMA AINDA EXISTE!")
            return False
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        return False
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    print("🔧 TESTE COMPLETO ACTION MAPPING CHERRY\n")
    
    # Teste 1: Verificar mapeamento matemático
    mapping_ok = test_action_mapping()
    
    # Teste 2: Verificar ambiente real
    env_ok = test_cherry_env_with_fixed_mapping()
    
    print("\n" + "=" * 50)
    if mapping_ok and env_ok:
        print("✅ TODOS OS TESTES PASSOU! MAPEAMENTO CORRIGIDO!")
    else:
        print("❌ AINDA HÁ PROBLEMAS!")