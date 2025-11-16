#!/usr/bin/env python3
"""
🍒 TESTE CHERRY OTIMIZADO - AVALIAÇÃO RÁPIDA E CONFIÁVEL
========================================================

Teste otimizado com cache para avaliar 1-15 checkpoints Cherry rapidamente.
Cache inteligente para dados e ambiente, mantendo confiabilidade total.
"""

import sys
import os
import traceback
from datetime import datetime
import json
import pickle
import numpy as np
import pandas as pd
import torch
from pathlib import Path

sys.path.append("D:/Projeto")

# CONFIGURAÇÃO FLEXÍVEL - CHECKPOINTS CHERRY
def get_cherry_checkpoints(limit=None):
    """Buscar checkpoints Cherry automaticamente"""
    cherry_dir = Path("D:/Projeto/Otimizacao/treino_principal/models/Cherry")
    if not cherry_dir.exists():
        print(f"❌ Diretório Cherry não encontrado: {cherry_dir}")
        return []
    
    # Buscar todos os .zip
    checkpoints = list(cherry_dir.glob("*.zip"))
    
    # Ordenar por steps (extrair número do nome)
    def extract_steps(path):
        try:
            name = path.name
            if "_steps_" in name:
                steps_part = name.split("_steps_")[0].split("_")[-1]
                return int(steps_part)
        except:
            pass
        return 0
    
    checkpoints.sort(key=extract_steps)
    
    if limit:
        checkpoints = checkpoints[:limit]
    
    print(f"🔍 Encontrados {len(checkpoints)} checkpoints Cherry")
    for i, cp in enumerate(checkpoints):
        steps = extract_steps(cp)
        steps_str = f"{steps//1000}k" if steps < 1000000 else f"{steps//1000000:.1f}M"
        print(f"  {i+1:2d}. {steps_str:>6} - {cp.name}")
    
    return [str(cp) for cp in checkpoints]

# CONFIGURAÇÃO DO TESTE
INITIAL_PORTFOLIO = 500.0
TEST_STEPS = 1008  # 3.5 dias (mais rápido, ainda representativo)
NUM_EPISODES = 6   # 6 episódios = boa amostragem
SEEDS = [42, 123, 456]  # 3 seeds para robustez

# CACHE OTIMIZADO
CACHE_DIR = Path("D:/Projeto/avaliacao/.cache")
CACHE_DIR.mkdir(exist_ok=True)

class CherryCache:
    """Cache otimizado para dados e ambiente Cherry"""
    
    def __init__(self):
        self.cache_dir = CACHE_DIR
        self.data_cache_file = self.cache_dir / "cherry_data_cache.pkl"
        self.env_cache_file = self.cache_dir / "cherry_env_cache.pkl"
        
    def get_cached_data(self):
        """Carregar dados do cache se existir"""
        if self.data_cache_file.exists():
            try:
                print("🔄 Carregando dados do cache...")
                with open(self.data_cache_file, 'rb') as f:
                    data = pickle.load(f)
                print(f"✅ Dados carregados do cache: {len(data)} barras")
                return data
            except:
                print("⚠️ Cache de dados corrompido, recarregando...")
        
        return None
    
    def save_data_cache(self, data):
        """Salvar dados no cache"""
        try:
            with open(self.data_cache_file, 'wb') as f:
                pickle.dump(data, f)
            print("💾 Dados salvos no cache")
        except Exception as e:
            print(f"⚠️ Erro ao salvar cache: {e}")
    
    def get_environment_config(self):
        """Configuração otimizada do ambiente"""
        return {
            'df': None,  # Será preenchido
            'window_size': 20,
            'is_training': True,  # CRÍTICO: Ativar todos os sistemas
            'initial_balance': INITIAL_PORTFOLIO,
            'trading_params': {
                'min_lot_size': 0.02,
                'max_lot_size': 0.03,
                'enable_shorts': True,
                'max_positions': 2
            }
        }

def load_cherry_data_optimized():
    """Carregar dados Cherry com cache otimizado"""
    cache = CherryCache()
    
    # Tentar cache primeiro
    data = cache.get_cached_data()
    if data is not None:
        return data
    
    # Carregar dados originais
    print("📊 Carregando dados Cherry (primeira vez)...")
    
    original_cwd = os.getcwd()
    os.chdir("D:/Projeto")
    
    try:
        from cherry import load_optimized_data_original
        data = load_optimized_data_original()
        print(f"✅ Dados Cherry carregados: {len(data)} barras")
        
        # Usar apenas dados recentes (últimas 20 semanas para velocidade)
        if len(data) > 20 * 2016:
            recent_data_size = 20 * 2016
            data = data.iloc[-recent_data_size:].reset_index(drop=True)
            print(f"📅 Usando dados recentes: {len(data)} steps (20 semanas)")
        
        # Salvar no cache
        cache.save_data_cache(data)
        
        return data
        
    finally:
        os.chdir(original_cwd)

def create_cherry_environment_fast(data=None):
    """Criar ambiente Cherry otimizado"""
    if data is None:
        data = load_cherry_data_optimized()
    
    original_cwd = os.getcwd()
    os.chdir("D:/Projeto")
    
    try:
        from cherry import TradingEnv
        
        cache = CherryCache()
        config = cache.get_environment_config()
        config['df'] = data
        
        env = TradingEnv(**config)
        
        return env
        
    finally:
        os.chdir(original_cwd)

def test_cherry_model_fast(model_path, shared_data=None):
    """Teste otimizado de um modelo Cherry"""
    model_name = Path(model_path).name.replace('.zip', '')
    print(f"\n🍒 TESTANDO: {model_name}")
    print("=" * 50)
    
    if not os.path.exists(model_path):
        print(f"❌ Modelo não encontrado")
        return None
    
    # Carregar modelo
    from sb3_contrib import RecurrentPPO
    try:
        print("🔄 Carregando modelo...")
        model = RecurrentPPO.load(model_path)
        model.policy.set_training_mode(False)
        print("✅ Modelo carregado")
        
    except Exception as e:
        print(f"❌ Erro ao carregar: {e}")
        return None
    
    # Usar dados compartilhados
    if shared_data is None:
        shared_data = load_cherry_data_optimized()
    
    all_results = []
    total_trades = 0
    
    # Testar com seeds
    for seed_idx, seed in enumerate(SEEDS):
        print(f"🎲 Seed {seed} ({seed_idx+1}/{len(SEEDS)})", end=" ")
        
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Múltiplos episódios por seed
        for episode in range(NUM_EPISODES // len(SEEDS)):
            try:
                # Criar ambiente (rápido com dados cached)
                env = create_cherry_environment_fast(shared_data)
                
                # Reset
                obs = env.reset()
                lstm_states = None
                episode_return = 0.0
                episode_trades = 0
                
                # Executar episódio
                for step in range(TEST_STEPS):
                    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                    obs, reward, done, info = env.step(action)
                    episode_return += reward
                    
                    if 'trade_executed' in info and info['trade_executed']:
                        episode_trades += 1
                    
                    if done:
                        break
                
                # Resultados
                final_portfolio = env.portfolio_value
                return_pct = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
                
                all_results.append({
                    'seed': seed,
                    'episode': episode + 1,
                    'return_pct': return_pct,
                    'final_portfolio': final_portfolio,
                    'trades': episode_trades,
                    'active': episode_trades > 0
                })
                
                total_trades += episode_trades
                
            except Exception as e:
                print(f"❌", end=" ")
                continue
        
        print("✅")
    
    if not all_results:
        print("❌ Nenhum resultado válido")
        return None
    
    # Calcular métricas
    returns = [r['return_pct'] for r in all_results]
    trades_list = [r['trades'] for r in all_results]
    active_episodes = [r for r in all_results if r['active']]
    
    metrics = {
        'model': model_name,
        'total_episodes': len(all_results),
        'active_episodes': len(active_episodes),
        'activity_rate': (len(active_episodes) / len(all_results)) * 100,
        
        'mean_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': min(returns),
        'max_return': max(returns),
        
        'total_trades': sum(trades_list),
        'avg_trades_per_episode': np.mean(trades_list),
        
        'positive_episodes': len([r for r in returns if r > 0]),
        'win_rate_episodes': len([r for r in returns if r > 0]) / len(returns) * 100,
        
        'sharpe_ratio': np.mean(returns) / (np.std(returns) + 1e-8)
    }
    
    # Print resultados compactos
    print(f"📊 Return: {metrics['mean_return']:+.2f}% | Sharpe: {metrics['sharpe_ratio']:.2f} | Atividade: {metrics['activity_rate']:.0f}% | Trades/ep: {metrics['avg_trades_per_episode']:.1f}")
    
    return {
        'metrics': metrics,
        'raw_results': all_results
    }

def run_cherry_test_suite(max_checkpoints=None):
    """Executar suite de testes Cherry otimizada"""
    print("🍒 TESTE CHERRY OTIMIZADO")
    print("=" * 60)
    print(f"⏰ Início: {datetime.now().strftime('%H:%M:%S')}")
    
    # Buscar checkpoints
    checkpoints = get_cherry_checkpoints(limit=max_checkpoints)
    if not checkpoints:
        print("❌ Nenhum checkpoint encontrado")
        return
    
    # Carregar dados uma vez (cache otimizado)
    print("\n📊 Preparando dados compartilhados...")
    shared_data = load_cherry_data_optimized()
    
    results = {}
    
    # Testar cada checkpoint
    print(f"\n🚀 Testando {len(checkpoints)} modelos...")
    for i, model_path in enumerate(checkpoints):
        print(f"\n[{i+1}/{len(checkpoints)}]", end=" ")
        
        result = test_cherry_model_fast(model_path, shared_data)
        if result:
            model_name = Path(model_path).name.replace('.zip', '')
            results[model_name] = result
    
    # Análise final
    print(f"\n🏆 RANKING FINAL:")
    print("=" * 60)
    
    # Ordenar por Sharpe ratio
    sorted_results = sorted(results.items(), 
                          key=lambda x: x[1]['metrics']['sharpe_ratio'], 
                          reverse=True)
    
    for rank, (model_name, result) in enumerate(sorted_results, 1):
        metrics = result['metrics']
        
        # Extrair steps do nome
        try:
            if "_steps_" in model_name:
                steps_part = model_name.split("_steps_")[0].split("_")[-1]
                steps = int(steps_part)
                steps_str = f"{steps//1000}k" if steps < 1000000 else f"{steps//1000000:.1f}M"
            else:
                steps_str = "???"
        except:
            steps_str = "???"
        
        # Grade
        if metrics['sharpe_ratio'] > 1.0 and metrics['activity_rate'] > 30:
            grade = "🟢 EXCELENTE"
        elif metrics['sharpe_ratio'] > 0.5 and metrics['activity_rate'] > 15:
            grade = "🟡 BOM"
        elif metrics['sharpe_ratio'] > 0:
            grade = "🟠 REGULAR"
        else:
            grade = "🔴 RUIM"
        
        print(f"{rank:2d}. {steps_str:>6} | Return: {metrics['mean_return']:+5.1f}% | Sharpe: {metrics['sharpe_ratio']:4.2f} | Atividade: {metrics['activity_rate']:3.0f}% | {grade}")
    
    # Salvar resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"D:/Projeto/avaliacoes/cherry_test_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        print(f"\n💾 Resultados salvos: {filename}")
    except Exception as e:
        print(f"⚠️ Erro ao salvar: {e}")
    
    # Estatísticas finais
    if results:
        all_sharpes = [r['metrics']['sharpe_ratio'] for r in results.values()]
        all_activities = [r['metrics']['activity_rate'] for r in results.values()]
        
        print(f"\n📈 ESTATÍSTICAS:")
        print(f"  Sharpe médio: {np.mean(all_sharpes):.2f}")
        print(f"  Atividade média: {np.mean(all_activities):.1f}%")
        print(f"  Modelos testados: {len(results)}")
    
    print(f"⏰ Fim: {datetime.now().strftime('%H:%M:%S')}")
    
    return results

if __name__ == "__main__":
    try:
        # Verificar argumentos da linha de comando
        max_checkpoints = None
        if len(sys.argv) > 1:
            try:
                max_checkpoints = int(sys.argv[1])
                print(f"🎯 Limitando a {max_checkpoints} checkpoints")
            except:
                print("⚠️ Argumento inválido, testando todos os checkpoints")
        
        results = run_cherry_test_suite(max_checkpoints)
        
        if results:
            print("\n✅ TESTE CHERRY CONCLUÍDO COM SUCESSO!")
        else:
            print("\n❌ Nenhum resultado obtido")
        
    except KeyboardInterrupt:
        print("\n⏹️ Teste interrompido pelo usuário")
    except Exception as e:
        print(f"\n❌ ERRO CRÍTICO: {e}")
        print(f"Detalhes: {traceback.format_exc()}")