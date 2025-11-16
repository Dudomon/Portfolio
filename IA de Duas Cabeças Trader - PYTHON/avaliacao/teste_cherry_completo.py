#!/usr/bin/env python3
"""
🍒 TESTE CHERRY COMPLETO - AVALIAÇÃO CONFIÁVEL
==============================================

Teste completo e confiável para qualquer modelo Cherry.
Configuração idêntica ao ambiente de produção.
"""

import sys
import os
import traceback
from datetime import datetime
import json
import numpy as np
import pandas as pd
import torch

sys.path.append("D:/Projeto")

# CHECKPOINTS CHERRY PARA TESTAR
CHECKPOINTS_TO_TEST = [
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_450000_steps_20250905_103556.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_500000_steps_20250905_104047.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_1000000_steps_20250905_112708.zip",
    "D:/Projeto/Otimizacao/treino_principal/models/Cherry/Cherry_simpledirecttraining_2500000_steps_20250905_134304.zip"
]

# CONFIGURAÇÃO DO TESTE
INITIAL_PORTFOLIO = 500.0
TEST_STEPS = 2016  # 1 semana (7*24*12 = 2016 steps de 5min)
NUM_EPISODES = 10  # 10 semanas de teste
SEEDS = [42, 123, 456]  # Multiple seeds para robustez

def create_cherry_environment():
    """Criar ambiente Cherry idêntico à produção"""
    print("🔧 Configurando ambiente Cherry...")
    
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
        
        # Usar dados recentes (últimas 30 semanas)
        if len(data) > 30 * 2016:
            recent_data_size = 30 * 2016
            data = data.iloc[-recent_data_size:].reset_index(drop=True)
            print(f"📅 Usando dados recentes: {len(data)} steps (30 semanas)")
        
        # CRÍTICO: Usar is_training=True para ativar todos os sistemas
        env = TradingEnv(
            df=data,
            window_size=20,
            is_training=True,  # FORÇAR modo treinamento para ativar sistemas
            initial_balance=INITIAL_PORTFOLIO,
            trading_params={
                'min_lot_size': 0.02,
                'max_lot_size': 0.03,
                'enable_shorts': True,
                'max_positions': 2
            }
        )
        
        print(f"✅ Ambiente Cherry criado")
        print(f"🔍 Action Space: {env.action_space}")
        print(f"🔍 Obs Space: {env.observation_space.shape}")
        
        return env
        
    finally:
        os.chdir(original_cwd)

def test_cherry_model(model_path):
    """Testar um modelo Cherry específico"""
    print(f"\n🍒 TESTANDO: {os.path.basename(model_path)}")
    print("=" * 60)
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado: {model_path}")
        return None
    
    # Carregar modelo
    from sb3_contrib import RecurrentPPO
    try:
        model = RecurrentPPO.load(model_path)
        print(f"✅ Modelo carregado")
        
        # Modo inferência
        model.policy.set_training_mode(False)
        print(f"🧊 Modo inferência ativado")
        
    except Exception as e:
        print(f"❌ Erro ao carregar modelo: {e}")
        return None
    
    # Resultados do teste
    all_results = []
    total_trades = 0
    
    # Testar com múltiplas seeds
    for seed_idx, seed in enumerate(SEEDS):
        print(f"\n🎲 Seed {seed} ({seed_idx+1}/{len(SEEDS)})")
        
        # Set seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        seed_results = []
        
        # Múltiplos episódios por seed
        for episode in range(NUM_EPISODES // len(SEEDS)):
            print(f"  📈 Episódio {episode+1}")
            
            try:
                # Criar ambiente novo para cada episódio
                env = create_cherry_environment()
                
                # Reset
                obs = env.reset()
                lstm_states = None
                episode_return = 0.0
                episode_trades = 0
                
                # Executar episódio
                for step in range(TEST_STEPS):
                    # Predict com LSTM states
                    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                    
                    # Debug primeiros steps
                    if step < 3:
                        print(f"    Step {step}: Entry={action[0]:.3f}, Conf={action[1]:.3f}")
                    
                    # Step environment
                    obs, reward, done, info = env.step(action)
                    episode_return += reward
                    
                    # Contar trades
                    if 'trade_executed' in info and info['trade_executed']:
                        episode_trades += 1
                    
                    if done:
                        break
                
                # Resultados do episódio
                final_portfolio = env.portfolio_value
                return_pct = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
                
                episode_result = {
                    'seed': seed,
                    'episode': episode + 1,
                    'return_pct': return_pct,
                    'final_portfolio': final_portfolio,
                    'trades': episode_trades,
                    'steps': step + 1,
                    'active': episode_trades > 0
                }
                
                seed_results.append(episode_result)
                total_trades += episode_trades
                
                print(f"    💰 ${INITIAL_PORTFOLIO:.2f} → ${final_portfolio:.2f} ({return_pct:+.2f}%) | Trades: {episode_trades}")
                
            except Exception as e:
                print(f"    ❌ Erro no episódio {episode+1}: {e}")
                continue
        
        all_results.extend(seed_results)
    
    # Análise dos resultados
    if not all_results:
        print("❌ Nenhum resultado válido")
        return None
    
    returns = [r['return_pct'] for r in all_results]
    trades_list = [r['trades'] for r in all_results]
    active_episodes = [r for r in all_results if r['active']]
    
    metrics = {
        'model': os.path.basename(model_path),
        'total_episodes': len(all_results),
        'active_episodes': len(active_episodes),
        'activity_rate': (len(active_episodes) / len(all_results)) * 100,
        
        'mean_return': np.mean(returns),
        'median_return': np.median(returns),
        'std_return': np.std(returns),
        'min_return': min(returns),
        'max_return': max(returns),
        
        'total_trades': sum(trades_list),
        'avg_trades_per_episode': np.mean(trades_list),
        'avg_trades_per_week': np.mean(trades_list),  # Cada episódio = 1 semana
        
        'positive_episodes': len([r for r in returns if r > 0]),
        'win_rate_episodes': len([r for r in returns if r > 0]) / len(returns) * 100,
        
        'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8)
    }
    
    # Print resultados
    print(f"\n📊 RESULTADOS FINAIS:")
    print(f"  📈 Return médio: {metrics['mean_return']:+.2f}%")
    print(f"  📊 Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"  🎯 Taxa atividade: {metrics['activity_rate']:.1f}%")
    print(f"  📈 Trades/episódio: {metrics['avg_trades_per_episode']:.1f}")
    print(f"  🏆 Episódios positivos: {metrics['positive_episodes']}/{metrics['total_episodes']}")
    
    return {
        'metrics': metrics,
        'raw_results': all_results
    }

def run_complete_cherry_test():
    """Executar teste completo de todos os modelos Cherry"""
    print("🍒 TESTE CHERRY COMPLETO")
    print("=" * 80)
    print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
    
    results = {}
    
    for model_path in CHECKPOINTS_TO_TEST:
        if os.path.exists(model_path):
            result = test_cherry_model(model_path)
            if result:
                model_name = os.path.basename(model_path).replace('.zip', '')
                results[model_name] = result
        else:
            print(f"⚠️ Modelo não encontrado: {model_path}")
    
    # Análise comparativa
    print(f"\n🏆 COMPARAÇÃO FINAL:")
    print("=" * 80)
    
    for model_name, result in results.items():
        metrics = result['metrics']
        steps = model_name.split('_')[2] if '_' in model_name else 'unknown'
        
        print(f"\n🏷️ {steps.upper()} STEPS:")
        print(f"  📈 Return: {metrics['mean_return']:+.2f}% (±{metrics['std_return']:.2f}%)")
        print(f"  📊 Sharpe: {metrics['sharpe_ratio']:.2f}")
        print(f"  🎯 Atividade: {metrics['activity_rate']:.1f}%")
        print(f"  📈 Trades/semana: {metrics['avg_trades_per_week']:.1f}")
        print(f"  🏆 Win Rate: {metrics['win_rate_episodes']:.1f}%")
        
        # Grade
        if metrics['mean_return'] > 10 and metrics['activity_rate'] > 50:
            grade = "🟢 EXCELENTE"
        elif metrics['mean_return'] > 5 and metrics['activity_rate'] > 20:
            grade = "🟡 BOM"
        elif metrics['mean_return'] > 0:
            grade = "🟠 REGULAR"
        else:
            grade = "🔴 RUIM"
        
        print(f"  📋 Avaliação: {grade}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"D:/Projeto/avaliacoes/teste_cherry_completo_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    print(f"\n💾 Resultados salvos: {filename}")
    print(f"⏰ Fim: {datetime.now().strftime('%H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    try:
        results = run_complete_cherry_test()
        print("\n✅ TESTE CHERRY COMPLETO CONCLUÍDO!")
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        print(f"Detalhes: {traceback.format_exc()}")