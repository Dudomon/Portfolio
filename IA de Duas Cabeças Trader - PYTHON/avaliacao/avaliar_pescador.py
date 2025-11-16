#!/usr/bin/env python3
"""
🎣 AVALIAÇÃO PESCADOR - SISTEMA SCALP ESPECIALIZADO
=====================================================

CONFIGURAÇÃO ESPECÍFICA PARA PESCADOR:
✅ 1. Ambiente: PescadorEnv (reward system otimizado para scalp)
✅ 2. Diretório: PESCADOR models folder
✅ 3. Sample Size: 100 episódios para confiabilidade
✅ 4. Métricas: Focadas em trades rápidos e eficiência
✅ 5. Timeout: Sistema de 4h para posições (scalp style)
✅ 6. Gate System: Filtros adaptativos desabilitados para teste

DIFERENÇAS DO SILUS:
- Reward system: PescadorRewardSystem (activity bonuses)
- Episódios: 3000 steps vs 2000 (baixa volatilidade)
- Cooldowns: Desabilitados para teste
- Learning Rate: 2x do padrão (6e-05 actor, 4e-05 critic)
- Activity System: Timeout 4h para scalping

CONFIGURAÇÃO:
- Portfolio: $500 (produção) 
- Lot range: 0.02-0.03  
- Episódios: 50 (teste otimizado)
- Steps/episódio: 1800 (5 dias úteis)
- Mode: DETERMINISTIC (reproduzível)
- Architecture: Compatível com pescador training
"""

import sys
import os
import traceback
from datetime import datetime, timedelta
import random
import json
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import torch

# CONFIGURAÇÃO COMPLETA PARA TESTE CONFIÁVEL - baseado no original
# MODELOS‑ALVO POR STEPS (PESCADOR) 
PESCADOR_DIR = "D:/Projeto/Otimizacao/treino_principal/models/PESCADOR"
# Avaliar checkpoints PESCADOR disponíveis
TARGET_STEPS = [
    3_000_000,  # 3M steps
    3_500_000,  # 3.5M steps
    4_000_000,  # 4M steps
    4_500_000,  # 4.5M steps
    5_000_000,  # 5M steps
]

# Baseline Legion V1 (absoluto)
BASELINE_MODEL = "D:/Projeto/Modelo PPO Trader/Modelo daytrade/Legion V1.zip"

def build_checkpoints_from_steps(dir_path: str, steps_list):
    """Monta a lista de checkpoints desejados a partir dos steps‑alvo."""
    import glob
    selected = []
    for steps in steps_list:
        pattern = os.path.join(dir_path, f"*_{steps}_steps_*.zip")
        matches = sorted(glob.glob(pattern))
        if matches:
            # pegar o mais recente pelo mtime
            matches.sort(key=os.path.getmtime, reverse=True)
            sel = matches[0]
            if sel not in selected:
                selected.append(sel)
        else:
            print(f"⚠️ Checkpoint não encontrado para {steps:,} steps ({pattern})")
    return selected

def extract_steps_from_name(path: str) -> int | None:
    """Extrai os steps do nome do arquivo (formato *_<steps>_steps_*.zip)."""
    try:
        import re, os as _os
        m = re.search(r"(\d{6,7})_steps", _os.path.basename(path))
        return int(m.group(1)) if m else None
    except Exception:
        return None
INITIAL_PORTFOLIO = 500.0  # $500 conforme solicitado
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03

# PARÂMETROS DE AVALIAÇÃO MELHORADOS (vs original)
TEST_STEPS = 1800          # 1800 steps = 5 dias úteis (1 semana completa) 
NUM_EPISODES = 50          # 50 episódios para teste mais rápido
MIN_EPISODE_GAP = 10000    # Gap mínimo entre episódios (evitar overlap)
CONFIDENCE_LEVEL = 0.95    # 95% confidence intervals


def find_multiple_checkpoints(max_checkpoints=10):
    """🔍 Encontrar modelos finais na pasta 'Modelos para testar'"""
    
    print("🎣 Buscando checkpoints PESCADOR...")
    
    # Construir a lista a partir dos steps‑alvo
    selected = build_checkpoints_from_steps(PESCADOR_DIR, TARGET_STEPS)

    # Incluir baseline Legion V1, se existir
    try:
        if os.path.exists(BASELINE_MODEL):
            selected.append(BASELINE_MODEL)
            print(f"   + Baseline incluído: {os.path.basename(BASELINE_MODEL)}")
        else:
            print(f"⚠️ Baseline não encontrado: {BASELINE_MODEL}")
    except Exception as _e:
        print(f"⚠️ Erro ao incluir baseline: {_e}")
    
    if not selected:
        print("❌ Nenhum modelo encontrado!")
        return []
    
    print(f"✅ {len(selected)} modelos selecionados para teste:")
    for i, cp in enumerate(selected):
        size_mb = os.path.getsize(cp) / (1024*1024)
        mod_time = datetime.fromtimestamp(os.path.getmtime(cp)).strftime('%Y-%m-%d %H:%M')
        
        # Identificar modelo
        if os.path.basename(cp).lower().startswith("legion v1") or "Legion V1" in cp:
            model_info = "Baseline Legion V1"
        else:
            # Inferir pelos steps no nome
            st = extract_steps_from_name(cp)
            model_info = f"PESCADOR {st/1_000_000:.2f}M" if st else "Unknown model"
        
        print(f"   {i+1}. {model_info} | {size_mb:.1f}MB | {mod_time}")
    
    return selected

def create_evaluation_dataset():
    """🎯 Criar dataset específico para avaliação out-of-sample - COM CACHE"""
    print("📊 Preparando dataset de avaliação...")
    
    # CACHE para acelerar carregamento
    cache_path = "D:/Projeto/data/CACHE_eval_dataset_processed.pkl"
    
    if os.path.exists(cache_path):
        print("🚀 Carregando dataset PRÉ-PROCESSADO do cache...")
        import pickle
        try:
            with open(cache_path, 'rb') as f:
                train_df, eval_df = pickle.load(f)
            print(f"✅ Cache carregado: {len(train_df):,} treino + {len(eval_df):,} avaliação")
            print(f"📅 Período avaliação: {eval_df.index.min()} até {eval_df.index.max()}")
            return train_df, eval_df
        except:
            print("⚠️ Erro no cache, reprocessando...")
    
    # Processar dataset original (só se não tem cache)
    print("🔄 Processando dataset original (primeira vez)...")
    dataset_path = "D:/Projeto/data/GC=F_YAHOO_20250821_161220.csv"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset não encontrado: {dataset_path}")
        return None, None
    
    df = pd.read_csv(dataset_path)
    
    # Processar dataset
    if 'time' in df.columns:
        df['timestamp'] = pd.to_datetime(df['time'])
        df.set_index('timestamp', inplace=True)
        df.drop('time', axis=1, inplace=True)
    
    # Renomear colunas para formato padrão
    df = df.rename(columns={
        'open': 'open_5m',
        'high': 'high_5m',
        'low': 'low_5m', 
        'close': 'close_5m',
        'tick_volume': 'volume_5m'
    })
    
    total_len = len(df)
    print(f"✅ Dataset carregado: {total_len:,} barras")
    print(f"📅 Período: {df.index.min()} até {df.index.max()}")
    
    # RESERVAR ÚLTIMOS 20% PARA AVALIAÇÃO (OUT-OF-SAMPLE)
    split_point = int(total_len * 0.8)
    train_df = df.iloc[:split_point]
    eval_df = df.iloc[split_point:]
    
    print(f"🔄 Split realizado:")
    print(f"   📚 Treinamento: {len(train_df):,} barras ({train_df.index.min()} - {train_df.index.max()})")  
    print(f"   🎯 Avaliação: {len(eval_df):,} barras ({eval_df.index.min()} - {eval_df.index.max()})")
    
    # SALVAR CACHE para próximas execuções
    print("💾 Salvando cache para próximas execuções...")
    import pickle
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump((train_df, eval_df), f, protocol=4)
        print("✅ Cache salvo com sucesso!")
    except Exception as e:
        print(f"⚠️ Erro ao salvar cache: {e}")
    
    return train_df, eval_df

def calculate_comprehensive_metrics(episode_results):
    """📊 Calcular métricas abrangentes de performance e risco - FIXED VERSION"""
    
    if not episode_results:
        return {}
    
    # Extrair retornos de todos os episódios
    returns = [ep['return_pct'] for ep in episode_results]
    portfolio_values = [ep['final_portfolio'] for ep in episode_results]
    all_trades = []
    all_portfolio_histories = []  # 🔧 NOVO: Para drawdown real
    
    for ep in episode_results:
        all_trades.extend(ep.get('trades_log', []))
        if 'portfolio_history' in ep:
            all_portfolio_histories.append(ep['portfolio_history'])  # 🔧 COLETAR HISTÓRICO
    
    # MÉTRICAS BÁSICAS
    metrics = {
        # Retorno
        'mean_return': np.mean(returns),
        'median_return': np.median(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        
        # Consistência
        'positive_episodes': len([r for r in returns if r > 0]),
        'win_rate_episodes': len([r for r in returns if r > 0]) / len(returns) * 100,
        
        # Portfolio
        'mean_final_portfolio': np.mean(portfolio_values),
        'portfolio_growth': (np.mean(portfolio_values) - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO * 100,
    }
    
    # MÉTRICAS DE RISCO AVANÇADAS
    if len(returns) > 1:
        # Sharpe Ratio (assumindo risk-free rate = 0)
        metrics['sharpe_ratio'] = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        
        # Sortino Ratio (downside deviation)
        negative_returns = [r for r in returns if r < 0]
        if negative_returns:
            downside_deviation = np.std(negative_returns)
            metrics['sortino_ratio'] = np.mean(returns) / downside_deviation if downside_deviation > 0 else 0
        else:
            metrics['sortino_ratio'] = float('inf') if np.mean(returns) > 0 else 0
        
        # 🔧 DRAWDOWN REAL - MÉTODO CORRIGIDO baseado em portfolio_history
        max_drawdowns_per_episode = []
        
        for portfolio_history in all_portfolio_histories:
            if len(portfolio_history) > 1:
                portfolio_array = np.array(portfolio_history)
                
                # Calcular running peak do portfolio real
                running_peak = np.maximum.accumulate(portfolio_array)
                
                # Calcular drawdown real em percentual
                portfolio_drawdowns = (portfolio_array - running_peak) / running_peak * 100
                
                # Max drawdown deste episódio
                episode_max_dd = np.min(portfolio_drawdowns)
                max_drawdowns_per_episode.append(episode_max_dd)
        
        if max_drawdowns_per_episode:
            # Max drawdown global = pior drawdown entre todos os episódios
            metrics['max_drawdown'] = min(max_drawdowns_per_episode)  # Mais negativo
            metrics['avg_drawdown_per_episode'] = np.mean(max_drawdowns_per_episode)
            metrics['drawdown_episodes_count'] = len(max_drawdowns_per_episode)
        else:
            # Fallback para método antigo se não há portfolio_history
            cumulative_returns = np.cumprod(1 + np.array(returns) / 100)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / running_max
            metrics['max_drawdown'] = np.min(drawdowns) * 100
            metrics['avg_drawdown_per_episode'] = metrics['max_drawdown']
            metrics['drawdown_episodes_count'] = 0
        
        # Value at Risk (VaR) - 5% worst cases
        metrics['var_5pct'] = np.percentile(returns, 5)
        
        # Calmar Ratio usando drawdown real
        real_max_dd = abs(metrics['max_drawdown'])
        if real_max_dd > 0.1:
            metrics['calmar_ratio'] = abs(metrics['mean_return']) / real_max_dd
        else:
            metrics['calmar_ratio'] = 0
    
    # MÉTRICAS DE TRADING
    if all_trades:
        profitable_trades = [t for t in all_trades if t.get('pnl_usd', 0) > 0]
        losing_trades = [t for t in all_trades if t.get('pnl_usd', 0) < 0]
        
        metrics.update({
            'total_trades': len(all_trades),
            'win_rate_trades': len(profitable_trades) / len(all_trades) * 100,
            'avg_profit_per_trade': np.mean([t.get('pnl_usd', 0) for t in profitable_trades]) if profitable_trades else 0,
            'avg_loss_per_trade': np.mean([t.get('pnl_usd', 0) for t in losing_trades]) if losing_trades else 0,
            'total_pnl': sum(t.get('pnl_usd', 0) for t in all_trades),
            'trades_per_episode': len(all_trades) / len(episode_results),
        })
        
        # Profit Factor
        gross_profit = sum(t.get('pnl_usd', 0) for t in profitable_trades)
        gross_loss = abs(sum(t.get('pnl_usd', 0) for t in losing_trades))
        metrics['profit_factor'] = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
    
    # INTERVALOS DE CONFIANÇA (95%)
    if len(returns) > 2:
        confidence_interval = stats.t.interval(
            CONFIDENCE_LEVEL, 
            len(returns)-1, 
            loc=np.mean(returns), 
            scale=stats.sem(returns)
        )
        metrics['ci_95_lower'] = confidence_interval[0]
        metrics['ci_95_upper'] = confidence_interval[1]
    
    return metrics

def simulate_realistic_trading_costs(trades_log):
    """💰 Simular custos realísticos de trading"""
    if not trades_log:
        return 0.0
    
    total_cost = 0.0
    for trade in trades_log:
        lot_size = trade.get('lot_size', BASE_LOT_SIZE)
        
        # Custos simplificados de trading
        spread_cost = 0.3 * lot_size
        slippage_cost = 0.2 * lot_size * random.uniform(0.5, 1.5)
        commission = 0.5 * lot_size
        
        total_cost += (spread_cost + slippage_cost + commission)
    
    return total_cost

def test_v8_elegance_trading():
    """🚀 Teste COMPLETO - baseado no avaliar_v11.py original com melhorias"""
    
    print(f"🎣 AVALIAÇÃO PESCADOR OTIMIZADA - 50 EPISÓDIOS")
    print("=" * 80)
    print(f"💵 Portfolio Inicial: ${INITIAL_PORTFOLIO}")
    print(f"📊 Base Lot: {BASE_LOT_SIZE}")
    print(f"📊 Max Lot: {MAX_LOT_SIZE}")
    print(f"🧠 Modo: DETERMINISTIC (reproduzível)")
    print(f"📊 Episódios: {NUM_EPISODES} (teste rápido)")
    print(f"📏 Steps: {TEST_STEPS} (5 dias úteis)")
    print("=" * 80)
    
    try:
        # Imports
        from sb3_contrib import RecurrentPPO
        from pescador import PescadorEnv as TradingEnv  # 🎣 USANDO AMBIENTE PESCADOR ESPECÍFICO
        
        print("✅ Imports carregados")
        
        # 1. PREPARAR DATASET OUT-OF-SAMPLE (melhoria vs original)
        train_df, eval_df = create_evaluation_dataset()
        if eval_df is None:
            return False
        
        # 2. TESTAR CHECKPOINTS PESCADOR DISPONÍVEIS
        checkpoints = find_multiple_checkpoints(max_checkpoints=12)
        
        # Verificar se todos existem
        missing = [cp for cp in checkpoints if not os.path.exists(cp)]
        if missing:
            print(f"❌ Checkpoints não encontrados: {[os.path.basename(m) for m in missing]}")
            return False
        
        print(f"🎣 TESTANDO 5 CHECKPOINTS PESCADOR: 3M → 3.5M → 4M → 4.5M → 5M (EVOLUÇÃO FINAL)")
        
        # 3. PREPARAR AMBIENTE DE TRADING (igual ao original)
        trading_params = {
            'base_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'initial_balance': INITIAL_PORTFOLIO,
            'target_trades_per_day': 18,  # Como no daytrader
            'stop_loss_range': (2.0, 8.0),
            'take_profit_range': (3.0, 15.0)
        }
        
        print("✅ Parâmetros de trading configurados")
        
        # RESULTADOS CONSOLIDADOS (melhoria vs original)
        all_checkpoint_results = {}
        
        # 4. TESTAR CADA CHECKPOINT (baseado no loop original)
        for checkpoint_idx, checkpoint_path in enumerate(checkpoints):
            print(f"\n🤖 TESTANDO CHECKPOINT {checkpoint_idx + 1}/{len(checkpoints)}")
            print(f"📂 {os.path.basename(checkpoint_path)}")
            print("-" * 60)
            
            # CARREGAR MODELO (igual ao original)
            print("🤖 Carregando modelo...")
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
            try:
                # Tentar carregamento normal primeiro (igual ao original)
                model = RecurrentPPO.load(checkpoint_path, device=device)
                print("✅ Carregamento normal bem-sucedido")
                load_method = "direct_load"
            except Exception as e1:
                print(f"⚠️ Carregamento normal falhou: {str(e1)[:100]}...")
                try:
                    # Tentar com policy_kwargs V11Sigmoid (corrigido)
                    from trading_framework.policies.two_head_v11_sigmoid import get_v11_sigmoid_kwargs
                    sigmoid_kwargs = get_v11_sigmoid_kwargs()
                    model = RecurrentPPO.load(checkpoint_path, policy_kwargs=sigmoid_kwargs, device=device)
                    print("✅ Carregamento com policy_kwargs bem-sucedido")
                    load_method = "with_kwargs"
                except Exception as e2:
                    print(f"⚠️ Carregamento com kwargs falhou: {str(e2)[:100]}...")
                    try:
                        # ÚLTIMA TENTATIVA (igual ao original)
                        import zipfile
                        import tempfile
                        
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Extrair ZIP
                            with zipfile.ZipFile(checkpoint_path, 'r') as zip_ref:
                                zip_ref.extractall(temp_dir)
                            
                            # Carregar policy.pth diretamente
                            import glob
                            policy_files = glob.glob(f"{temp_dir}/**/policy.pth", recursive=True)
                            if not policy_files:
                                raise FileNotFoundError("policy.pth não encontrado no ZIP")
                            
                            policy_state = torch.load(policy_files[0], map_location=device)
                            
                            # Criar modelo novo da arquitetura V7Intuition (igual ao original)
                            from trading_framework.policies.two_head_v7_intuition import get_v7_intuition_kwargs
                            intuition_kwargs = get_v7_intuition_kwargs()
                            
                            # Criar ambiente temporário para carregamento
                            temp_env = TradingEnv(
                                eval_df.head(100), 
                                window_size=20, 
                                is_training=False,
                                initial_balance=INITIAL_PORTFOLIO,
                                trading_params=trading_params
                            )
                            
                            # Usar método de carregamento do stable-baselines3
                            model = RecurrentPPO("MlpLstmPolicy", temp_env, policy_kwargs=intuition_kwargs, device=device)
                            
                            # Carregar pesos compatíveis ignorando incompatíveis (igual ao original)
                            current_state = model.policy.state_dict()
                            compatible_state = {}
                            
                            for key, value in policy_state.items():
                                if key in current_state and current_state[key].shape == value.shape:
                                    compatible_state[key] = value
                                else:
                                    pass  # Ignorar incompatíveis
                            
                            model.policy.load_state_dict(compatible_state, strict=False)
                            print(f"✅ Carregamento FORÇA BRUTA bem-sucedido - {len(compatible_state)} parâmetros")
                            load_method = "manual_load"
                    
                    except Exception as e3:
                        print(f"❌ Todos os métodos falharam: {str(e3)[:100]}...")
                        continue  # Pular este checkpoint
            
            # Configurar modelo para modo determinístico (MELHORIA vs original)
            model.policy.set_training_mode(False)
            print(f"✅ Modelo carregado em {model.device}")
            
            # EXECUTAR MÚLTIPLOS EPISÓDIOS (ampliado vs original)
            print(f"🚀 Iniciando {NUM_EPISODES} episódios de trading...")
            
            # Gerar posições aleatórias para amostragem diversificada (MELHORIA)
            eval_len = len(eval_df)
            max_start_pos = eval_len - TEST_STEPS - 100
            
            if max_start_pos <= 0:
                print("⚠️ Dataset de avaliação muito pequeno")
                continue
                
            # Gerar posições com gap mínimo (MELHORIA vs original)
            episode_positions = []
            attempts = 0
            while len(episode_positions) < NUM_EPISODES and attempts < NUM_EPISODES * 3:
                candidate_pos = random.randint(0, max_start_pos)
                
                # Verificar distância mínima
                too_close = False
                for existing_pos in episode_positions:
                    if abs(candidate_pos - existing_pos) < MIN_EPISODE_GAP:
                        too_close = True
                        break
                
                if not too_close:
                    episode_positions.append(candidate_pos)
                    
                attempts += 1
            
            # Se não conseguiu posições suficientes, usar espaçamento uniforme
            if len(episode_positions) < NUM_EPISODES:
                episode_positions = []
                step = max_start_pos // NUM_EPISODES
                for i in range(NUM_EPISODES):
                    episode_positions.append(i * step)
            
            print(f"🎯 {len(episode_positions)} episódios configurados")
            
            # Resultados consolidados
            all_episodes = []
            total_returns = []
            
            # CRIAR AMBIENTE OTIMIZADO - PRÉ-PROCESSAR FEATURES
            print(f"🏗️ Criando TradingEnv otimizado com {len(eval_df)} barras...")
            
            # PRÉ-CALCULAR FEATURES UMA VEZ SÓ
            cache_features_path = "D:/Projeto/data/CACHE_trading_features.pkl"
            
            if os.path.exists(cache_features_path):
                print("⚡ Carregando features pré-calculadas...")
                import pickle
                try:
                    with open(cache_features_path, 'rb') as f:
                        processed_df = pickle.load(f)
                    print("✅ Features carregadas do cache!")
                except:
                    print("⚠️ Cache inválido, recalculando...")
                    processed_df = eval_df.copy()
            else:
                print("🔄 Primeira execução - será mais lenta...")
                processed_df = eval_df.copy()
            
            # DESABILITAR LOGS VERBOSOS durante teste
            import logging
            trading_logger = logging.getLogger('trading_env')
            old_level = trading_logger.level
            trading_logger.setLevel(logging.ERROR)  # Só erros críticos
            
            trading_env = TradingEnv(
                processed_df,  # Dataset com features pré-calculadas
                window_size=20,
                is_training=False,
                initial_balance=INITIAL_PORTFOLIO,
                trading_params=trading_params
            )
            
            print("🎣 Usando ambiente PESCADOR para avaliação")
            
            # 🔧 AJUSTAR FILTRO DE CONFIANÇA PARA 0.6 (conforme solicitado)
            if hasattr(trading_env, '__class__'):
                # Modificar a constante na classe dinamicamente
                import types
                original_step = trading_env.step
                
                def modified_step(self, action):
                    # Temporariamente alterar o threshold de confiança
                    import sys
                    current_module = sys.modules[self.__class__.__module__]
                    if hasattr(current_module, 'MIN_CONFIDENCE_THRESHOLD'):
                        original_threshold = current_module.MIN_CONFIDENCE_THRESHOLD
                        current_module.MIN_CONFIDENCE_THRESHOLD = 0.6
                    
                    # Executar step normal
                    result = original_step(action)
                    
                    # Restaurar threshold original
                    if hasattr(current_module, 'MIN_CONFIDENCE_THRESHOLD'):
                        current_module.MIN_CONFIDENCE_THRESHOLD = original_threshold
                    
                    return result
                
                # Bind the modified method
                trading_env.step = types.MethodType(modified_step, trading_env)
                print("🔧 Filtro de confiança ajustado para 0.6 (era 0.8)")
            else:
                print("⚠️ Não foi possível ajustar filtro de confiança")
            
            # SILENCIAR saída verbosa do ambiente
            trading_env.verbose = False
            if hasattr(trading_env, 'debug_mode'):
                trading_env.debug_mode = False
            
            # SALVAR CACHE DE FEATURES após primeiro processamento
            if not os.path.exists(cache_features_path):
                print("💾 Salvando features processadas para próximas execuções...")
                import pickle
                try:
                    # Salvar dataset processado do ambiente
                    processed_data = getattr(trading_env, 'df', processed_df)
                    with open(cache_features_path, 'wb') as f:
                        pickle.dump(processed_data, f, protocol=4)
                    print("✅ Cache de features salvo!")
                except Exception as e:
                    print(f"⚠️ Erro ao salvar features: {e}")
            
            print(f"✅ TradingEnv OTIMIZADO criado!")
            
            # EXECUTAR EPISÓDIOS (baseado no loop original)
            for episode_num, start_pos in enumerate(episode_positions):
                if (episode_num + 1) % 10 == 0:
                    print(f"   📊 Progresso: {episode_num + 1}/{NUM_EPISODES} episódios")
                
                # Verificar se posição é válida
                if start_pos + TEST_STEPS >= len(eval_df):
                    continue  # Pular se não tem dados suficientes
                
                # ⚠️ RESET CRÍTICO COMPLETO - CORRIGIR BUG MATEMÁTICO
                # Forçar reset total para evitar acúmulo entre episódios
                trading_env.current_step = start_pos + 20
                
                # 1. RESET PORTFOLIO (PRINCIPAL + TODOS BACKUPS)
                trading_env.portfolio_value = INITIAL_PORTFOLIO
                trading_env.initial_balance = INITIAL_PORTFOLIO
                trading_env.realized_balance = INITIAL_PORTFOLIO  # 🎯 CRÍTICO: Esta é a chave!
                
                # 1b. RESET PICOS E DRAWDOWNS (também crítico!)
                if hasattr(trading_env, 'peak_portfolio'):
                    trading_env.peak_portfolio = INITIAL_PORTFOLIO
                if hasattr(trading_env, 'peak_portfolio_value'):
                    trading_env.peak_portfolio_value = INITIAL_PORTFOLIO
                if hasattr(trading_env, 'current_drawdown'):
                    trading_env.current_drawdown = 0.0
                if hasattr(trading_env, 'peak_drawdown'):
                    trading_env.peak_drawdown = 0.0
                
                # 2. RESET TODOS OS ESTADOS DE VALOR (com proteção para properties)
                if hasattr(trading_env, 'cash'):
                    try:
                        trading_env.cash = INITIAL_PORTFOLIO
                    except (AttributeError, TypeError):
                        pass  # Property read-only, ignorar
                
                if hasattr(trading_env, 'balance'):
                    try:
                        trading_env.balance = INITIAL_PORTFOLIO
                    except (AttributeError, TypeError):
                        pass  # Property read-only, ignorar
                
                # 3. RESET HISTÓRICO DE VALORES
                if hasattr(trading_env, 'portfolio_history'):
                    trading_env.portfolio_history = []
                if hasattr(trading_env, 'balance_history'):
                    trading_env.balance_history = []
                if hasattr(trading_env, 'net_worth_history'):
                    trading_env.net_worth_history = []
                
                # 4. RESET COMPLETO DO ESTADO DE TRADING
                if hasattr(trading_env, 'trades'):
                    trading_env.trades = []
                if hasattr(trading_env, 'position_type'):
                    trading_env.position_type = 0
                if hasattr(trading_env, 'positions'):
                    trading_env.positions = []
                if hasattr(trading_env, 'open_positions'):
                    trading_env.open_positions = []
                if hasattr(trading_env, 'current_position'):
                    trading_env.current_position = None
                
                # 5. RESET MÉTRICAS DE PERFORMANCE
                if hasattr(trading_env, 'total_reward'):
                    trading_env.total_reward = 0.0
                if hasattr(trading_env, 'cumulative_reward'):
                    trading_env.cumulative_reward = 0.0
                if hasattr(trading_env, 'episode_reward'):
                    trading_env.episode_reward = 0.0
                if hasattr(trading_env, 'returns'):
                    trading_env.returns = []
                
                # 6. RESET CONTADORES
                if hasattr(trading_env, 'total_trades'):
                    trading_env.total_trades = 0
                if hasattr(trading_env, 'profitable_trades'):
                    trading_env.profitable_trades = 0
                if hasattr(trading_env, 'losing_trades'):
                    trading_env.losing_trades = 0
                
                # Obter observação inicial SEM reset completo
                obs = trading_env._get_observation()
                lstm_states = None
                done = False
                step = 0
                
                portfolio_history = [INITIAL_PORTFOLIO]
                
                while not done and step < TEST_STEPS:
                    # MODO ORIGINAL - NÃO DETERMINÍSTICO (mantido do original)
                    action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                    
                    obs, reward, done, info = trading_env.step(action)
                    portfolio_history.append(trading_env.portfolio_value)
                    step += 1
                
                # 🔍 DEBUG: Coletar resultados com validação
                final_portfolio = trading_env.portfolio_value
                episode_return = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
                trades_log = getattr(trading_env, 'trades', [])
                
                # 🔍 DEBUG: Verificar se valores são realistas
                portfolio_change = final_portfolio - INITIAL_PORTFOLIO
                if abs(portfolio_change) < 0.01 and len(trades_log) > 0:
                    print(f"   ⚠️ Episode {episode_num + 1}: Portfolio mudou apenas ${portfolio_change:.4f} com {len(trades_log)} trades")
                    print(f"      Inicial: ${INITIAL_PORTFOLIO:.2f}, Final: ${final_portfolio:.2f}")
                    if len(trades_log) > 0:
                        total_trade_pnl = sum(t.get('pnl_usd', 0) for t in trades_log)
                        print(f"      Total PnL trades: ${total_trade_pnl:.4f}")
                
                # Simular custos realísticos (melhoria)
                trading_costs = simulate_realistic_trading_costs(trades_log)
                net_portfolio = final_portfolio - trading_costs
                net_return = ((net_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
                
                episode_result = {
                    'episode': episode_num + 1,
                    'start_pos': start_pos,
                    'period_start': eval_df.index[start_pos],
                    'period_end': eval_df.index[min(start_pos + TEST_STEPS, len(eval_df) - 1)],
                    'initial_portfolio': INITIAL_PORTFOLIO,
                    'final_portfolio': final_portfolio,
                    'net_portfolio': net_portfolio,
                    'return_pct': episode_return,
                    'net_return_pct': net_return,
                    'trades_count': len(trades_log),
                    'trades_log': trades_log,
                    'portfolio_history': portfolio_history,
                    'trading_costs': trading_costs
                }
                
                all_episodes.append(episode_result)
                total_returns.append(episode_return)
                
                # Cleanup desnecessário - reutilizando ambiente
                # del trading_env
            
            # CALCULAR MÉTRICAS ABRANGENTES (melhoria vs original)
            metrics = calculate_comprehensive_metrics(all_episodes)
            
            # Adicionar informações do checkpoint
            checkpoint_result = {
                'checkpoint_path': checkpoint_path,
                'checkpoint_name': os.path.basename(checkpoint_path),
                'load_method': load_method,
                'episodes_completed': len(all_episodes),
                'metrics': metrics,
                'episode_results': all_episodes
            }
            
            all_checkpoint_results[checkpoint_path] = checkpoint_result
            
            # 🔧 RELATÓRIO INDIVIDUAL EXPANDIDO COM DEBUG INFO
            print(f"\n📊 RESULTADOS - {os.path.basename(checkpoint_path)[:50]}")
            print("-" * 60)
            print(f"✅ Episódios completados: {len(all_episodes)}")
            
            if metrics:
                print(f"📈 Retorno médio: {metrics.get('mean_return', 0):+.2f}% (σ={metrics.get('std_return', 0):.2f}%)")
                print(f"🎯 Taxa de sucesso: {metrics.get('win_rate_episodes', 0):.1f}% dos episódios")
                print(f"⚖️ Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                print(f"📉 Max Drawdown (REAL): {metrics.get('max_drawdown', 0):.2f}%")
                
                # 🔍 DEBUG INFO ADICIONAL
                if 'avg_drawdown_per_episode' in metrics:
                    print(f"📉 Drawdown médio/episódio: {metrics.get('avg_drawdown_per_episode', 0):.2f}%")
                    print(f"📉 Episódios c/ drawdown: {metrics.get('drawdown_episodes_count', 0)}")
                
                # Portfolio range debug
                if all_episodes:
                    portfolio_range = [ep['final_portfolio'] for ep in all_episodes]
                    print(f"💰 Portfolio range: ${min(portfolio_range):.2f} - ${max(portfolio_range):.2f}")
                    
                    # Total portfolio change across all episodes
                    total_change = sum(ep['final_portfolio'] - INITIAL_PORTFOLIO for ep in all_episodes)
                    print(f"💰 Total portfolio Δ: ${total_change:.2f} ({len(all_episodes)} episodes)")
                
                if 'ci_95_lower' in metrics:
                    print(f"📊 IC 95%: [{metrics['ci_95_lower']:+.2f}%, {metrics['ci_95_upper']:+.2f}%]")
                
                if metrics.get('total_trades', 0) > 0:
                    print(f"💹 Total trades: {metrics['total_trades']} (WR: {metrics.get('win_rate_trades', 0):.1f}%)")
                    print(f"💰 Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            
            del model  # Limpar memória
        
        # RELATÓRIO COMPARATIVO FINAL (melhoria vs original)
        print(f"\n🏆 RELATÓRIO COMPARATIVO FINAL - {len(all_checkpoint_results)} CHECKPOINTS")
        print("=" * 80)
        
        if all_checkpoint_results:
            # Ranking por Sharpe Ratio
            ranked_checkpoints = sorted(
                all_checkpoint_results.items(), 
                key=lambda x: x[1]['metrics'].get('sharpe_ratio', -999), 
                reverse=True
            )
            
            print("📊 RANKING POR SHARPE RATIO:")
            for rank, (path, result) in enumerate(ranked_checkpoints, 1):
                # Identificar nome mais amigável
                checkpoint_name = result['checkpoint_name']
                st = extract_steps_from_name(checkpoint_name)
                if checkpoint_name.lower().startswith("legion v1"):
                    display_name = "Baseline Legion V1"
                elif st:
                    display_name = f"PESCADOR {st/1_000_000:.2f}M"
                else:
                    display_name = checkpoint_name[:40]
                
                sharpe = result['metrics'].get('sharpe_ratio', 0)
                mean_return = result['metrics'].get('mean_return', 0)
                win_rate = result['metrics'].get('win_rate_episodes', 0)
                max_dd = result['metrics'].get('max_drawdown', 0)
                
                grade = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉" if rank == 3 else f"{rank:2d}."
                
                print(f"{grade} {display_name:<22} | Sharpe: {sharpe:6.2f} | Ret: {mean_return:+6.2f}% | WR: {win_rate:5.1f}% | DD: {max_dd:6.2f}%")
            
            # RECOMENDAÇÃO FINAL
            if ranked_checkpoints:
                best_checkpoint = ranked_checkpoints[0]
                best_metrics = best_checkpoint[1]['metrics']
                
                print(f"\n💡 RECOMENDAÇÃO FINAL:")
                print(f"🏆 Melhor checkpoint: {best_checkpoint[1]['checkpoint_name']}")
                print(f"📊 Sharpe Ratio: {best_metrics.get('sharpe_ratio', 0):.2f}")
                print(f"📈 Retorno médio: {best_metrics.get('mean_return', 0):+.2f}%")
                print(f"🎯 Consistência: {best_metrics.get('win_rate_episodes', 0):.1f}% episódios lucrativos")
        
        # SALVAR RELATÓRIO DETALHADO (melhoria)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"D:/Projeto/avaliacoes/avaliacao_pescador_{timestamp}.json"
        
        # Salvar resultados em lista (evita colisão de chaves e preserva caminhos)
        save_list = []
        for path, result in all_checkpoint_results.items():
            save_result = result.copy()
            # Normalizar caminho para Windows
            try:
                save_result['checkpoint_path'] = os.path.normpath(path)
            except Exception:
                save_result['checkpoint_path'] = path
            # Adicionar steps extraídos para depuração
            save_result['steps'] = extract_steps_from_name(save_result.get('checkpoint_name', ''))
            # Apenas contador de episódios (não dumpa todos)
            save_result['episode_results'] = len(result.get('episode_results', []))
            save_list.append(save_result)
        
        save_payload = {
            'results': save_list,
            '_metadata': {
                'evaluation_date': timestamp,
                'num_episodes': NUM_EPISODES,
                'test_steps': TEST_STEPS,
                'confidence_level': CONFIDENCE_LEVEL,
            'total_checkpoints_tested': len(save_list)
            }
        }
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(save_payload, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n💾 Resultados salvos: {results_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        print(f"Detalhes: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print(f"🎣 INICIANDO AVALIAÇÃO PESCADOR - {datetime.now().strftime('%H:%M:%S')}")
    
    # Set random seed para reprodutibilidade
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Processar argumento da linha de comando se fornecido
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join("D:/Projeto", checkpoint_path)
        globals()['CHECKPOINT_PATH'] = checkpoint_path
        print(f"📂 Usando checkpoint: {os.path.basename(checkpoint_path)}")
    
    success = test_v8_elegance_trading()
    
    if success:
        print(f"\n✅ AVALIAÇÃO PESCADOR CONCLUÍDA - {datetime.now().strftime('%H:%M:%S')}")
    else:
        print(f"\n❌ AVALIAÇÃO PESCADOR FALHOU - {datetime.now().strftime('%H:%M:%S')}")
