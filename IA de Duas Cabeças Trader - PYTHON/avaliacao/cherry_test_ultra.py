#!/usr/bin/env python3
"""
🍒 CHERRY TEST ULTRA - MÁXIMA VELOCIDADE
=======================================

Teste ultra-otimizado: 1 ambiente por modelo, múltiplos resets.
Focado em velocidade mantendo confiabilidade.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

sys.path.append("D:/Projeto")

def get_cherry_models(limit=5):
    """Buscar modelos Cherry"""
    cherry_dir = Path("D:/Projeto/Otimizacao/treino_principal/models/Cherry")
    models = list(cherry_dir.glob("*.zip"))
    
    # Ordenar por steps
    def extract_steps(path):
        try:
            if "_steps_" in path.name:
                return int(path.name.split("_steps_")[0].split("_")[-1])
        except:
            pass
        return 0
    
    models.sort(key=extract_steps)
    if limit:
        models = models[:limit]
    
    return [str(m) for m in models]

def load_cherry_data():
    """Carregar dados Cherry uma vez"""
    original_cwd = os.getcwd()
    os.chdir("D:/Projeto")
    
    try:
        from cherry import load_optimized_data_original
        data = load_optimized_data_original()
        
        # Usar últimas 15 semanas (rápido)
        if len(data) > 15 * 2016:
            data = data.iloc[-15*2016:].reset_index(drop=True)
            
        return data
    finally:
        os.chdir(original_cwd)

def test_cherry_ultra(model_path, shared_data, shared_env=None):
    """Teste ultra-rápido de um modelo"""
    model_name = Path(model_path).stem
    
    # Extrair steps
    try:
        steps = int(model_name.split("_steps_")[0].split("_")[-1])
        steps_str = f"{steps//1000}k" if steps < 1000000 else f"{steps//1000000:.1f}M"
    except:
        steps_str = "???"
    
    print(f"🍒 {steps_str:>6} ", end="", flush=True)
    
    if not os.path.exists(model_path):
        print("❌ Não encontrado")
        return None
    
    # Carregar modelo
    try:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(model_path)
        model.policy.set_training_mode(False)
    except Exception as e:
        print(f"❌ Erro: {e}")
        return None
    
    # Criar ambiente se necessário
    if shared_env is None:
        original_cwd = os.getcwd()
        os.chdir("D:/Projeto")
        
        try:
            from cherry import TradingEnv
            shared_env = TradingEnv(
                df=shared_data,
                window_size=20,
                is_training=True,  # CRÍTICO
                initial_balance=500.0,
                trading_params={
                    'min_lot_size': 0.02,
                    'max_lot_size': 0.03,
                    'enable_shorts': True,
                    'max_positions': 2
                }
            )
        finally:
            os.chdir(original_cwd)
    
    # Configuração de teste compacta
    TEST_STEPS = 500  # Ultra-rápido
    NUM_TESTS = 5     # 5 testes por modelo
    SEEDS = [42, 123, 456, 789, 999]
    
    results = []
    
    # Executar testes
    for test_idx in range(NUM_TESTS):
        try:
            seed = SEEDS[test_idx]
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Reset ambiente
            obs = shared_env.reset()
            lstm_states = None
            episode_return = 0.0
            episode_trades = 0
            
            # Executar steps
            for step in range(TEST_STEPS):
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                obs, reward, done, info = shared_env.step(action)
                episode_return += reward
                
                if 'trade_executed' in info and info['trade_executed']:
                    episode_trades += 1
                
                if done:
                    break
            
            # Resultado
            final_portfolio = shared_env.portfolio_value
            return_pct = ((final_portfolio - 500.0) / 500.0) * 100
            
            results.append({
                'return_pct': return_pct,
                'trades': episode_trades,
                'active': episode_trades > 0
            })
            
        except Exception:
            continue
    
    if not results:
        print("❌ Falha")
        return None
    
    # Calcular métricas
    returns = [r['return_pct'] for r in results]
    trades_list = [r['trades'] for r in results]
    active_count = sum(1 for r in results if r['active'])
    
    metrics = {
        'model': model_name,
        'steps': steps_str,
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'activity_rate': (active_count / len(results)) * 100,
        'avg_trades': np.mean(trades_list),
        'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8),
        'positive_tests': sum(1 for r in returns if r > 0)
    }
    
    # Print compacto
    activity_symbol = "🟢" if metrics['activity_rate'] > 50 else "🟡" if metrics['activity_rate'] > 20 else "🔴"
    performance_symbol = "📈" if metrics['mean_return'] > 0 else "📉"
    
    print(f"{activity_symbol} {performance_symbol} Return: {metrics['mean_return']:+4.1f}% | Sharpe: {metrics['sharpe_ratio']:4.2f} | Ativ: {metrics['activity_rate']:2.0f}% | Trades: {metrics['avg_trades']:3.1f}")
    
    return metrics

def run_cherry_ultra_test(max_models=5):
    """Executar teste ultra-rápido"""
    print("🍒 CHERRY TEST ULTRA")
    print("=" * 50)
    start_time = datetime.now()
    
    # Buscar modelos
    models = get_cherry_models(limit=max_models)
    if not models:
        print("❌ Nenhum modelo encontrado")
        return
    
    print(f"🔍 {len(models)} modelos encontrados")
    
    # Carregar dados uma vez
    print("📊 Carregando dados...", end=" ")
    try:
        shared_data = load_cherry_data()
        print(f"✅ {len(shared_data)} barras")
    except Exception as e:
        print(f"❌ Erro: {e}")
        return
    
    # Testar modelos
    print(f"\n🚀 Testando modelos:")
    print("   Steps | Status | Métricas")
    print("-" * 50)
    
    results = []
    shared_env = None  # Será criado no primeiro teste
    
    for i, model_path in enumerate(models):
        result = test_cherry_ultra(model_path, shared_data, shared_env)
        if result:
            results.append(result)
    
    # Análise final
    if not results:
        print("❌ Nenhum resultado válido")
        return
    
    print("\n" + "=" * 50)
    print("🏆 RANKING FINAL:")
    print("=" * 50)
    
    # Ordenar por Sharpe ratio
    results.sort(key=lambda x: x['sharpe_ratio'], reverse=True)
    
    for rank, result in enumerate(results, 1):
        # Grade baseada em Sharpe + Atividade
        if result['sharpe_ratio'] > 1.0 and result['activity_rate'] > 40:
            grade = "🟢 EXCELENTE"
        elif result['sharpe_ratio'] > 0.5 and result['activity_rate'] > 20:
            grade = "🟡 BOM"
        elif result['sharpe_ratio'] > 0:
            grade = "🟠 REGULAR"
        else:
            grade = "🔴 RUIM"
        
        print(f"{rank:2d}. {result['steps']:>6} | Return: {result['mean_return']:+5.1f}% | Sharpe: {result['sharpe_ratio']:5.2f} | Ativ: {result['activity_rate']:3.0f}% | {grade}")
    
    # Estatísticas
    all_sharpes = [r['sharpe_ratio'] for r in results]
    all_activities = [r['activity_rate'] for r in results]
    
    print(f"\n📊 ESTATÍSTICAS:")
    print(f"  Sharpe médio: {np.mean(all_sharpes):4.2f}")
    print(f"  Atividade média: {np.mean(all_activities):4.1f}%")
    print(f"  Tempo total: {datetime.now() - start_time}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"D:/Projeto/avaliacoes/cherry_ultra_{timestamp}.json"
    
    try:
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Salvo: {filename}")
    except:
        pass
    
    return results

if __name__ == "__main__":
    try:
        # Argumento: número máximo de modelos
        max_models = 5
        if len(sys.argv) > 1:
            try:
                max_models = int(sys.argv[1])
            except:
                pass
        
        print(f"🎯 Testando até {max_models} modelos")
        
        results = run_cherry_ultra_test(max_models)
        
        if results:
            print("\n✅ TESTE CONCLUÍDO!")
            print(f"🏆 Melhor modelo: {results[0]['steps']} (Sharpe: {results[0]['sharpe_ratio']:.2f})")
        else:
            print("\n❌ Nenhum resultado")
            
    except KeyboardInterrupt:
        print("\n⏹️ Interrompido")
    except Exception as e:
        print(f"\n❌ ERRO: {e}")