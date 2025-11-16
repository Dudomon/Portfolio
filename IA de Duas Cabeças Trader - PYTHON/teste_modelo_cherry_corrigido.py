#!/usr/bin/env python3
"""
🍒 TESTE MODELO CHERRY COM MAPEAMENTO CORRIGIDO
==============================================
"""

import sys
import os
import numpy as np
sys.path.append("D:/Projeto")

def test_cherry_model_post_fix():
    """Testar modelo Cherry depois da correção do mapeamento"""
    print("🍒 TESTE MODELO CHERRY PÓS-CORREÇÃO")
    print("=" * 50)
    
    original_cwd = os.getcwd()
    os.chdir("D:/Projeto")
    
    try:
        from cherry import load_optimized_data_original, TradingEnv
        from sb3_contrib import RecurrentPPO
        
        # Carregar dados pequenos para teste rápido
        print("📊 Carregando dados...")
        data = load_optimized_data_original()
        data = data.iloc[-2000:].reset_index(drop=True)
        print(f"✅ Dados: {len(data)} barras")
        
        # Criar ambiente
        env = TradingEnv(
            df=data,
            window_size=20,
            is_training=False,  # Modo avaliação
            initial_balance=500.0
        )
        
        # Testar com um modelo Cherry existente
        model_path = "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_5000000_steps_20250905_173508.zip"
        
        if not os.path.exists(model_path):
            print(f"⚠️ Modelo não encontrado: {model_path}")
            return False
            
        print("🤖 Carregando modelo Cherry...")
        model = RecurrentPPO.load(model_path)
        model.policy.set_training_mode(False)
        print("✅ Modelo carregado")
        
        obs = env.reset()
        lstm_states = None
        
        trades_executed = 0
        action_distribution = {"HOLD": 0, "LONG": 0, "SHORT": 0}
        portfolio_history = [env.portfolio_value]
        
        print("\n🚀 Executando 200 steps com modelo...")
        print("Step | Action Value | Decision | Trade | Portfolio | Positions")
        print("-" * 65)
        
        for step in range(200):
            # Prever ação
            action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
            
            # Verificar mapeamento
            action_value = float(action[0])
            if action_value < 0.67:
                decision = "HOLD"
                action_distribution["HOLD"] += 1
            elif action_value < 1.33:
                decision = "LONG"
                action_distribution["LONG"] += 1
            else:
                decision = "SHORT" 
                action_distribution["SHORT"] += 1
            
            # Executar step
            obs, reward, done, info = env.step(action)
            
            trade_executed = info.get('trade_executed', False)
            if trade_executed:
                trades_executed += 1
                
            positions = len(getattr(env, 'positions', []))
            portfolio_history.append(env.portfolio_value)
            
            # Log a cada 20 steps ou se trade executado
            if step % 20 == 0 or trade_executed:
                print(f"{step:4d} | {action_value:11.3f} | {decision:8} | {trade_executed:5} | ${env.portfolio_value:8.2f} | {positions}")
            
            if done:
                break
        
        # Resultados finais
        print(f"\n📊 RESULTADOS FINAIS:")
        print(f"Steps executados: {step + 1}")
        print(f"Trades executados: {trades_executed}")
        print(f"Portfolio inicial: $500.00")
        print(f"Portfolio final: ${env.portfolio_value:.2f}")
        print(f"PnL: ${env.portfolio_value - 500:.2f}")
        
        print(f"\n📈 DISTRIBUIÇÃO DE AÇÕES:")
        total_actions = sum(action_distribution.values())
        for action_type, count in action_distribution.items():
            pct = count / total_actions * 100 if total_actions > 0 else 0
            print(f"{action_type}: {count}/{total_actions} = {pct:.1f}%")
        
        # Verificar se funciona
        trades_ok = trades_executed > 0
        distribution_ok = action_distribution["LONG"] > 0 and action_distribution["SHORT"] > 0
        
        if trades_ok and distribution_ok:
            print("\n✅ MODELO FUNCIONANDO CORRETAMENTE!")
            print("- Executa trades ✅")
            print("- Usa todas as ações ✅") 
            return True
        else:
            print("\n❌ AINDA HÁ PROBLEMAS:")
            if not trades_ok:
                print("- Não executa trades ❌")
            if not distribution_ok:
                print("- Não usa todas as ações ❌")
            return False
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_cherry_model_post_fix()
    
    print("\n" + "=" * 50)
    if success:
        print("✅ MAPEAMENTO CORRIGIDO COM SUCESSO!")
        print("Modelo Cherry agora pode executar trades corretamente.")
    else:
        print("❌ AINDA PRECISA DE AJUSTES!")