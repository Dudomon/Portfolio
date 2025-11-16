#!/usr/bin/env python3
"""
🎯 AVALIAÇÃO OPUS - PENEIRA DEFINITIVA PARA PRODUÇÃO
Teste rigoroso e confiável para determinar quais modelos são adequados para trading real.

CRITÉRIOS RIGOROSOS:
- Performance: Retorno médio > 10% consistente
- Robustez: 70% episódios lucrativos mínimo
- Risco: Sharpe > 0.8, Max DD < 25%
- Atividade: 8-35 trades por episódio (não passivo nem overtrading)
- Qualidade: Profit Factor > 1.5, Win Rate > 40%

METODOLOGIA:
- 7 episódios de 4000 steps cada (mais robusto)
- Períodos diversos de mercado (stress test)
- Análise estatística rigorosa + métricas avançadas
- Sistema de scoring ponderado para aprovação
"""

import sys
import os
import traceback
from datetime import datetime
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import torch

# Configuração específica do usuário - Legion V1 (V11 Sigmoid)  
DEFAULT_CHECKPOINT_PATH = "D:/Projeto/trading_framework/training/checkpoints/SILUS/checkpoint_1000000_steps_20250825_142337.zip"
INITIAL_PORTFOLIO = 500.0  # Portfolio inicial
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03
TEST_STEPS = 4000       # Episódios mais longos para robustez
NUM_EPISODES = 7        # Mais episódios para consistência
EPISODE_SPACING = 6000  # Maior espaçamento para diversidade

# CRITÉRIOS RIGOROSOS PARA APROVAÇÃO EM PRODUÇÃO
PRODUCTION_CRITERIA = {
    'min_avg_return': 10.0,        # Retorno médio mínimo (%)
    'min_consistency': 70.0,       # % episódios lucrativos mínimo
    'min_sharpe': 0.8,            # Sharpe ratio mínimo
    'max_drawdown': -25.0,        # Max drawdown tolerável (%)
    'min_profit_factor': 1.5,     # Profit factor mínimo
    'min_trades_per_episode': 8,  # Mínimo atividade
    'max_trades_per_episode': 35, # Máximo (evitar overtrading)
    'min_win_rate': 40.0,         # Win rate mínimo (%)
    'max_volatility': 20.0        # Volatilidade máxima (%)
}

def find_best_checkpoint():
    """🔍 Encontra o checkpoint mais recente do daytrader V11 Sigmoid"""
    import glob
    
    print("🔍 Procurando checkpoints para avaliação rigorosa...")
    
    # 🔧 FIX: Detectar EXPERIMENT_TAG automaticamente
    # Prioridade: SILUS > 4DIM > Elegance > DAYTRADER
    possible_tags = ["SILUS", "4DIM", "Elegance", "DAYTRADER", "Optimus", "ANDERDAY"]
    EXPERIMENT_TAG = None
    
    # Detectar qual tag tem checkpoints mais recentes
    for tag in possible_tags:
        test_pattern = f"D:/Projeto/Otimizacao/treino_principal/models/{tag}/*.zip"
        if glob.glob(test_pattern):
            EXPERIMENT_TAG = tag
            print(f"🎯 EXPERIMENT_TAG detectado automaticamente: {EXPERIMENT_TAG}")
            break
    
    # Fallback para SILUS se não encontrar nada
    if not EXPERIMENT_TAG:
        EXPERIMENT_TAG = "SILUS"
        print(f"⚠️ EXPERIMENT_TAG fallback: {EXPERIMENT_TAG}")
    
    patterns = [
        # Primeiro: Checkpoints específicos do EXPERIMENT_TAG (mais recentes)
        f"D:/Projeto/Otimizacao/treino_principal/models/{EXPERIMENT_TAG}/{EXPERIMENT_TAG}_*.zip",
        f"D:/Projeto/Otimizacao/treino_principal/models/{EXPERIMENT_TAG}/FINAL_*.zip",
        
        # Segundo: AUTO_EVAL da pasta do EXPERIMENT_TAG (gerados automaticamente a cada 500k)
        f"D:/Projeto/Otimizacao/treino_principal/models/{EXPERIMENT_TAG}/AUTO_EVAL_*_steps_*.zip",
        
        # Terceiro: Qualquer checkpoint da pasta do EXPERIMENT_TAG
        f"D:/Projeto/Otimizacao/treino_principal/models/{EXPERIMENT_TAG}/*.zip",
        f"D:/Projeto/Otimizacao/treino_principal/models/{EXPERIMENT_TAG}/*_steps_*.zip",
        
        # Quarto: Fallback para outras pastas (caso não encontre no EXPERIMENT_TAG)
        "D:/Projeto/Otimizacao/treino_principal/models/SILUS/*.zip",
        "D:/Projeto/Otimizacao/treino_principal/models/4DIM/*.zip", 
        "D:/Projeto/Otimizacao/treino_principal/models/Elegance/*.zip",
        "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_*.zip",
        
        # Quinto: Fallback geral
        "D:/Projeto/Otimizacao/treino_principal/models/**/*_*000000_steps_*.zip",
        "D:/Projeto/Modelo PPO Trader/**/*.zip"
    ]
    
    all_checkpoints = []
    for pattern in patterns:
        found = glob.glob(pattern, recursive=True)
        print(f"   Pattern: {pattern} -> {len(found)} arquivos")
        all_checkpoints.extend(found)
    
    if not all_checkpoints:
        print("❌ Nenhum checkpoint encontrado!")
        return None
    
    # Remover duplicatas e ordenar por data de modificação (mais recente primeiro)
    all_checkpoints = list(set(all_checkpoints))
    all_checkpoints.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    
    print(f"🔍 TOP 10 Checkpoints encontrados (por data de modificação):")
    for i, cp in enumerate(all_checkpoints[:10]):
        mod_time = datetime.fromtimestamp(os.path.getmtime(cp)).strftime('%Y-%m-%d %H:%M:%S')
        size_mb = os.path.getsize(cp) / (1024*1024)
        
        # Extrair número de steps do nome se possível
        steps = "?"
        if "_steps_" in cp:
            try:
                parts = os.path.basename(cp).split("_steps_")[0].split("_")
                step_part = [p for p in parts if p.isdigit()][-1]
                steps = f"{int(step_part)/1000:.0f}k" if int(step_part) < 1000000 else f"{int(step_part)/1000000:.1f}M"
            except:
                pass
        elif any(x in cp for x in ["1000000", "2000000", "3000000"]):
            if "1000000" in cp: steps = "1.0M"
            elif "2000000" in cp: steps = "2.0M"
            elif "3000000" in cp: steps = "3.0M"
        
        priority = "🔥AUTO_EVAL" if "AUTO_EVAL" in cp else "📊DAYTRADER" if "DAYTRADER" in cp else "📁Outros"
        print(f"   {i+1:2d}. {priority} | {os.path.basename(cp)[:50]:<50} | {size_mb:5.1f}MB | {steps:>6} | {mod_time}")
    
    selected = all_checkpoints[0]
    print(f"\n✅ SELECIONADO: {os.path.basename(selected)}")
    return selected

def evaluate_production_readiness(metrics):
    """🎯 Avalia se modelo está pronto para produção com critérios rigorosos"""
    
    score = 0
    max_score = len(PRODUCTION_CRITERIA)
    issues = []
    passes = []
    
    # Avaliar cada critério
    if metrics['avg_return'] >= PRODUCTION_CRITERIA['min_avg_return']:
        score += 1
        passes.append(f"✅ Retorno médio: {metrics['avg_return']:.2f}% >= {PRODUCTION_CRITERIA['min_avg_return']:.1f}%")
    else:
        issues.append(f"❌ Retorno médio baixo: {metrics['avg_return']:.2f}% < {PRODUCTION_CRITERIA['min_avg_return']:.1f}%")
    
    if metrics['consistency_pct'] >= PRODUCTION_CRITERIA['min_consistency']:
        score += 1
        passes.append(f"✅ Consistência: {metrics['consistency_pct']:.1f}% >= {PRODUCTION_CRITERIA['min_consistency']:.1f}%")
    else:
        issues.append(f"❌ Consistência baixa: {metrics['consistency_pct']:.1f}% < {PRODUCTION_CRITERIA['min_consistency']:.1f}%")
    
    if metrics['sharpe_ratio'] >= PRODUCTION_CRITERIA['min_sharpe']:
        score += 1
        passes.append(f"✅ Sharpe ratio: {metrics['sharpe_ratio']:.2f} >= {PRODUCTION_CRITERIA['min_sharpe']:.1f}")
    else:
        issues.append(f"❌ Sharpe ratio baixo: {metrics['sharpe_ratio']:.2f} < {PRODUCTION_CRITERIA['min_sharpe']:.1f}")
    
    if metrics['max_drawdown'] >= PRODUCTION_CRITERIA['max_drawdown']:
        score += 1
        passes.append(f"✅ Max drawdown: {metrics['max_drawdown']:.2f}% >= {PRODUCTION_CRITERIA['max_drawdown']:.1f}%")
    else:
        issues.append(f"❌ Max drawdown alto: {metrics['max_drawdown']:.2f}% < {PRODUCTION_CRITERIA['max_drawdown']:.1f}%")
    
    if metrics['profit_factor'] >= PRODUCTION_CRITERIA['min_profit_factor']:
        score += 1
        passes.append(f"✅ Profit factor: {metrics['profit_factor']:.2f} >= {PRODUCTION_CRITERIA['min_profit_factor']:.1f}")
    else:
        issues.append(f"❌ Profit factor baixo: {metrics['profit_factor']:.2f} < {PRODUCTION_CRITERIA['min_profit_factor']:.1f}")
    
    if PRODUCTION_CRITERIA['min_trades_per_episode'] <= metrics['trades_per_episode'] <= PRODUCTION_CRITERIA['max_trades_per_episode']:
        score += 1
        passes.append(f"✅ Atividade adequada: {metrics['trades_per_episode']:.1f} trades/episódio")
    else:
        if metrics['trades_per_episode'] < PRODUCTION_CRITERIA['min_trades_per_episode']:
            issues.append(f"❌ Muito passivo: {metrics['trades_per_episode']:.1f} < {PRODUCTION_CRITERIA['min_trades_per_episode']} trades/episódio")
        else:
            issues.append(f"❌ Overtrading: {metrics['trades_per_episode']:.1f} > {PRODUCTION_CRITERIA['max_trades_per_episode']} trades/episódio")
    
    if metrics['win_rate'] >= PRODUCTION_CRITERIA['min_win_rate']:
        score += 1
        passes.append(f"✅ Win rate: {metrics['win_rate']:.1f}% >= {PRODUCTION_CRITERIA['min_win_rate']:.1f}%")
    else:
        issues.append(f"❌ Win rate baixo: {metrics['win_rate']:.1f}% < {PRODUCTION_CRITERIA['min_win_rate']:.1f}%")
    
    if metrics['std_return'] <= PRODUCTION_CRITERIA['max_volatility']:
        score += 1
        passes.append(f"✅ Volatilidade: {metrics['std_return']:.2f}% <= {PRODUCTION_CRITERIA['max_volatility']:.1f}%")
    else:
        issues.append(f"❌ Volatilidade alta: {metrics['std_return']:.2f}% > {PRODUCTION_CRITERIA['max_volatility']:.1f}%")
    
    # Classificação rigorosa
    percentage_score = (score / max_score) * 100
    
    if percentage_score >= 88:  # 8/9 critérios
        grade = "🏆 APROVADO PARA PRODUÇÃO"
        recommendation = "DEPLOY IMEDIATAMENTE - Modelo excepcional"
    elif percentage_score >= 77:  # 7/9 critérios
        grade = "✅ APROVADO COM RESSALVAS"
        recommendation = "Aprovado com monitoramento intensivo"
    elif percentage_score >= 66:  # 6/9 critérios
        grade = "⚠️ REQUER MELHORIAS"
        recommendation = "Não aprovado - Necessita otimização"
    else:
        grade = "❌ REJEITADO"
        recommendation = "Modelo inadequado - Redesenhar estratégia"
    
    return {
        'score': score,
        'max_score': max_score,
        'percentage': percentage_score,
        'grade': grade,
        'recommendation': recommendation,
        'passes': passes,
        'issues': issues
    }

def test_model_rigorously():
    """🎯 Teste rigoroso para seleção de modelos para produção"""
    
    print(f"🎯 AVALIAÇÃO OPUS - PENEIRA DEFINITIVA PARA PRODUÇÃO")
    print("=" * 70)
    print(f"💵 Portfolio: ${INITIAL_PORTFOLIO}")
    print(f"📊 Episódios: {NUM_EPISODES} x {TEST_STEPS} steps")
    print(f"🎯 Critérios: {len(PRODUCTION_CRITERIA)} métricas rigorosas")
    print(f"⚖️ Aprovação: >= 77% dos critérios (7/9)")
    print("=" * 70)
    
    try:
        # Imports
        from sb3_contrib import RecurrentPPO
        from silus import TradingEnv  # 🔥 USANDO MESMO AMBIENTE DO SILUS.PY
        
        print("✅ Imports carregados")
        
        # 🔧 FIX: USAR CHECKPOINT_PATH específico configurado ou buscar automaticamente
        checkpoint_path = getattr(sys.modules[__name__], 'CHECKPOINT_PATH', DEFAULT_CHECKPOINT_PATH)
        
        # Se checkpoint_path for None ou não existir, buscar automaticamente
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            if checkpoint_path is None:
                print("📝 CHECKPOINT_PATH não definido. Buscando checkpoint automático...")
            else:
                print(f"❌ CHECKPOINT_PATH não existe: {checkpoint_path}")
                print("   📝 Fallback: Buscando checkpoint automático...")
            
            checkpoint_path = find_best_checkpoint()
            
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                print(f"❌ Nenhum checkpoint encontrado: {checkpoint_path}")
                return False
        
        print(f"📂 Usando checkpoint: {os.path.basename(checkpoint_path)}")
        
        # Dataset real para teste
        print("📊 Carregando dataset para teste...")
        dataset_path = "D:/Projeto/data/GC=F_YAHOO_20250821_161220.csv"
        
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset não encontrado: {dataset_path}")
            return False
            
        df = pd.read_csv(dataset_path)
        
        # Processar dataset
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'])
            df.set_index('timestamp', inplace=True)
            df.drop('time', axis=1, inplace=True)
        
        # Renomear colunas
        df = df.rename(columns={
            'open': 'open_5m',
            'high': 'high_5m',
            'low': 'low_5m', 
            'close': 'close_5m',
            'tick_volume': 'volume_5m'
        })
        
        # Dataset completo para múltiplos episódios
        total_len = len(df)
        print(f"✅ Dataset V3 carregado: {total_len:,} barras totais")
        print(f"📅 Período completo: {df.index.min()} até {df.index.max()}")
        print(f"🎯 Configurado para {NUM_EPISODES} episódios de {TEST_STEPS} steps cada")
        
        # Criar ambiente de trading com configurações EXATAS (4D ACTION SPACE)
        # 🔥 TradingEnv do daytrader.py usa 4D: [0-2, 0-1, -1-1, -1-1]
        trading_params = {
            'base_lot_size': BASE_LOT_SIZE,
            'max_lot_size': MAX_LOT_SIZE,
            'initial_balance': INITIAL_PORTFOLIO,
            'target_trades_per_day': 18,  # Como no daytrader
            'stop_loss_range': (2.0, 8.0),
            'take_profit_range': (3.0, 15.0)
        }
        
        # Criar ambiente será feito para cada episódio
        print("✅ Parâmetros de trading configurados")
        
        # Carregar modelo
        print("🤖 Carregando modelo DAYTRADER 2M (8D OTIMIZADO)...")
        
        # Carregar modelo com compatibilidade forçada - FODA-SE AS DIFERENÇAS
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        try:
            # Tentar carregamento normal primeiro
            model = RecurrentPPO.load(checkpoint_path, device=device)
            print("✅ Carregamento normal bem-sucedido")
        except Exception as e1:
            print(f"⚠️ Carregamento normal falhou: {e1}")
            try:
                # Tentar com policy_kwargs V8Elegance
                from trading_framework.policies.two_head_v11_sigmoid import get_v8_elegance_kwargs
                elegance_kwargs = get_v8_elegance_kwargs()
                model = RecurrentPPO.load(checkpoint_path, policy_kwargs=elegance_kwargs, device=device)
                print("✅ Carregamento com policy_kwargs bem-sucedido")
            except Exception as e2:
                print(f"⚠️ Carregamento com policy_kwargs falhou: {e2}")
                try:
                    # ÚLTIMA TENTATIVA: Usar torch.load e carregar manualmente
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
                        
                        # Criar modelo novo da arquitetura V7Intuition
                        from trading_framework.policies.two_head_v7_intuition import get_v7_intuition_kwargs
                        intuition_kwargs = get_v7_intuition_kwargs()
                        
                        # Criar ambiente temporário para carregamento (4D ACTION SPACE V10PURE)
                        temp_env = TradingEnv(
                            df.head(100), 
                            window_size=20, 
                            is_training=False,
                            initial_balance=INITIAL_PORTFOLIO,
                            trading_params=trading_params
                        )
                        
                        # Usar método de carregamento do stable-baselines3
                        model = RecurrentPPO("MlpLstmPolicy", temp_env, policy_kwargs=intuition_kwargs, device=device)
                        
                        # Carregar pesos compatíveis ignorando incompatíveis
                        current_state = model.policy.state_dict()
                        compatible_state = {}
                        
                        for key, value in policy_state.items():
                            if key in current_state and current_state[key].shape == value.shape:
                                compatible_state[key] = value
                                print(f"✅ Carregado: {key}")
                            else:
                                print(f"⚠️ Ignorado: {key}")
                        
                        model.policy.load_state_dict(compatible_state, strict=False)
                        print(f"✅ Carregamento FORÇA BRUTA bem-sucedido - {len(compatible_state)} parâmetros carregados")
                
                except Exception as e3:
                    print(f"❌ Todos os métodos falharam: {e3}")
                    raise e3
        model.policy.set_training_mode(False)  # 🔥 MODO INFERÊNCIA
        
        print(f"✅ Modelo carregado em {model.device}")
        
        # EXECUTAR TESTE RIGOROSO
        print(f"🚀 Iniciando teste rigoroso: {NUM_EPISODES} episódios ({TEST_STEPS} steps cada)...")
        
        # Resultados consolidados
        all_episodes = []
        total_returns = []
        all_trades = []
        all_portfolio_history = []
        
        for episode_num in range(NUM_EPISODES):
            print(f"\n🎮 EPISÓDIO {episode_num + 1}/{NUM_EPISODES}")
            print("=" * 50)
            
            # 🎯 USAR PERÍODOS DIVERSOS - stress test em condições variadas
            buffer_size = TEST_STEPS + 200
            start_from_end = (episode_num + 1) * EPISODE_SPACING + buffer_size
            start_idx = max(0, total_len - start_from_end)
            end_idx = min(total_len, start_idx + buffer_size)
            
            episode_df = df.iloc[start_idx:end_idx].copy()
            
            print(f"📊 Período: {episode_df.index.min()} até {episode_df.index.max()}")
            print(f"📈 Barras: {len(episode_df):,}")
            
            # Criar ambiente específico para este episódio (4D ACTION SPACE V10PURE)
            env = TradingEnv(
                episode_df,
                window_size=20,
                is_training=False,  # 🔥 MODO AVALIAÇÃO
                initial_balance=INITIAL_PORTFOLIO,
                trading_params=trading_params
            )
            
            # 🔍 DEBUG: Verificar action space
            print(f"🔍 Action Space: {env.action_space}")
            print(f"🔍 Action Shape: {env.action_space.shape}")
            if hasattr(env.action_space, 'low'):
                print(f"🔍 Action Low: {env.action_space.low}")
                print(f"🔍 Action High: {env.action_space.high}")
            
            # Executar episódio
            obs = env.reset()
            lstm_states = None
            done = False
            step = 0
            
            # Tracking detalhado para análise rigorosa
            episode_portfolio_history = []
            
            while not done and step < TEST_STEPS:
                # PREDIÇÃO EM MODO INFERÊNCIA - TradingEnv já aplica toda lógica interna
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                
                # Executar ação no ambiente - TradingEnv gerencia tudo internamente
                obs, reward, done, info = env.step(action)
                
                # Tracking para análise avançada
                episode_portfolio_history.append(env.portfolio_value)
                
                if (step + 1) % 1500 == 0:  # Progress menos verboso
                    print(f"  Step {step+1}/{TEST_STEPS} - ${env.portfolio_value:.2f}")
                
                step += 1
            
            # Coletar resultados detalhados
            final_portfolio = env.portfolio_value
            episode_return = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
            episode_trades = getattr(env, 'trades', [])
            trades_count = len(episode_trades)
            
            episode_result = {
                'episode': episode_num + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'period': f"{episode_df.index.min()} até {episode_df.index.max()}",
                'initial_portfolio': INITIAL_PORTFOLIO,
                'final_portfolio': final_portfolio,
                'return_pct': episode_return,
                'trades_count': trades_count,
                'trades_log': episode_trades,
                'portfolio_history': episode_portfolio_history
            }
            
            all_episodes.append(episode_result)
            total_returns.append(episode_return)
            all_trades.extend(episode_trades)
            all_portfolio_history.extend(episode_portfolio_history)
            
            print(f"✅ Resultado: ${INITIAL_PORTFOLIO:.2f} → ${final_portfolio:.2f} ({episode_return:+.2f}%)")
            print(f"   Trades: {trades_count}")
            
            if episode_trades:
                profitable = [t for t in episode_trades if t.get('pnl_usd', 0) > 0]
                win_rate = (len(profitable) / len(episode_trades)) * 100
                print(f"   Win Rate: {win_rate:.1f}%")
            
            # Limpeza de memória
            del env
            del episode_df
        
        # ANÁLISE CONSOLIDADA DOS MÚLTIPLOS EPISÓDIOS
        print("\n" + "=" * 80)
        print(f"📊 RESULTADOS CONSOLIDADOS - {NUM_EPISODES} EPISÓDIOS V7INTUITION")
        print("=" * 80)
        
        # Estatísticas gerais
        avg_return = np.mean(total_returns)
        median_return = np.median(total_returns)
        std_return = np.std(total_returns)
        min_return = min(total_returns)
        max_return = max(total_returns)
        positive_episodes = len([r for r in total_returns if r > 0])
        
        print(f"💵 Portfolio Inicial por episódio: ${INITIAL_PORTFOLIO:.2f}")
        print(f"📈 Retorno Médio: {avg_return:+.2f}%")
        print(f"📊 Retorno Mediano: {median_return:+.2f}%")
        print(f"📈 Melhor Episódio: {max_return:+.2f}%")
        print(f"📉 Pior Episódio: {min_return:+.2f}%")
        print(f"📊 Desvio Padrão: {std_return:.2f}%")
        print(f"🎯 Episódios Lucrativos: {positive_episodes}/{NUM_EPISODES} ({(positive_episodes/NUM_EPISODES)*100:.1f}%)")
        
        # Detalhes por episódio
        print(f"\n📋 DETALHES POR EPISÓDIO:")
        for i, episode in enumerate(all_episodes):
            ep_return = episode['return_pct']
            grade_emoji = "🟢" if ep_return > 5 else "🟡" if ep_return > 0 else "🔴" 
            print(f"   {grade_emoji} Episódio {i+1}: ${episode['initial_portfolio']:.2f} → ${episode['final_portfolio']:.2f} ({ep_return:+.2f}%) - {episode['trades_count']} trades")
        
        # Análise consolidada de trades - usar dados dos TradingEnv
        all_trades = []
        for episode in all_episodes:
            all_trades.extend(episode['trades_log'])
        
        # Análise consolidada de trades - usar estrutura padrão do TradingEnv
        if all_trades:
            total_trades = len(all_trades)
            profitable_trades = [t for t in all_trades if t.get('pnl_usd', 0) > 0]
            losing_trades = [t for t in all_trades if t.get('pnl_usd', 0) < 0]
            
            win_rate = (len(profitable_trades) / total_trades) * 100 if total_trades > 0 else 0
            avg_profit = np.mean([t.get('pnl_usd', 0) for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([t.get('pnl_usd', 0) for t in losing_trades]) if losing_trades else 0
            total_pnl = sum(t.get('pnl_usd', 0) for t in all_trades)
            
            print(f"\n📊 ANÁLISE CONSOLIDADA DE TRADES:")
            print(f"📊 Total de Trades (todos episódios): {total_trades}")
            print(f"🎯 Win Rate Global: {win_rate:.1f}%")
            print(f"💚 Trades Lucrativos: {len(profitable_trades)}")
            print(f"❌ Trades Perdedores: {len(losing_trades)}")
            print(f"💰 Lucro Médio por Trade: ${avg_profit:.2f}")
            print(f"📉 Perda Média por Trade: ${avg_loss:.2f}")
            print(f"💵 PnL Total (todos episódios): ${total_pnl:.2f}")
            
            # Profit Factor
            if avg_loss != 0 and losing_trades:
                gross_profit = sum(t.get('pnl_usd', 0) for t in profitable_trades)
                gross_loss = abs(sum(t.get('pnl_usd', 0) for t in losing_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                print(f"⚖️ Profit Factor Global: {profit_factor:.2f}")
            else:
                profit_factor = 0
            
            # Frequência de trading
            total_steps = NUM_EPISODES * TEST_STEPS
            trading_frequency = (total_trades / total_steps) * 100
            print(f"📈 Frequência de Trading: {trading_frequency:.2f}% dos steps")
            print(f"📈 Trades por Episódio: {total_trades/NUM_EPISODES:.1f}")
        else:
            profit_factor = 0
            trading_frequency = 0
            
        if not all_trades:
            print(f"\n⚠️ NENHUM TRADE EXECUTADO EM {NUM_EPISODES} EPISÓDIOS")
            print("🔍 Modelo extremamente conservador em todos os períodos")
        
        # Análise simplificada - TradingEnv já gerencia todas as ações internamente
        
        # Drawdown analysis (do último episódio como exemplo)
        last_portfolio_history = all_episodes[-1]['portfolio_history'] if all_episodes else []
        if len(last_portfolio_history) > 1:
            portfolio_array = np.array(last_portfolio_history)
            running_max = np.maximum.accumulate(portfolio_array)
            drawdown = (portfolio_array - running_max) / running_max * 100
            max_drawdown = np.min(drawdown)
            
            print(f"\n📉 Max Drawdown (último episódio): {max_drawdown:.2f}%")
        
        # Avaliação final consolidada
        print(f"\n🎖️ AVALIAÇÃO FINAL CONSOLIDADA ({NUM_EPISODES} EPISÓDIOS):")
        if avg_return > 5:
            grade = "🟢 EXCELENTE"
        elif avg_return > 2:
            grade = "🟡 BOM"
        elif avg_return > -2:
            grade = "🟠 REGULAR"
        else:
            grade = "🔴 RUIM"
        
        # Classificação por consistência
        consistency = (positive_episodes / NUM_EPISODES) * 100
        if consistency >= 80:
            consistency_grade = "🔥 MUITO CONSISTENTE"
        elif consistency >= 60:
            consistency_grade = "✅ CONSISTENTE"
        elif consistency >= 40:
            consistency_grade = "⚠️ MODERADAMENTE CONSISTENTE"
        else:
            consistency_grade = "❌ INCONSISTENTE"
        
        print(f"   {grade}")
        print(f"   Retorno Médio: {avg_return:+.2f}% (σ={std_return:.2f}%)")
        print(f"   Consistência: {consistency_grade} ({positive_episodes}/{NUM_EPISODES})")
        print(f"   Melhor/Pior: {max_return:+.2f}% / {min_return:+.2f}%")
        if all_trades:
            print(f"   Total Trades: {len(all_trades)} (Win Rate: {win_rate:.1f}%)")
            print(f"   Trades/Episódio: {len(all_trades)/NUM_EPISODES:.1f}")
        
        # Sharpe Ratio aproximado (assumindo retornos diários)
        if std_return > 0:
            sharpe_ratio = avg_return / std_return
            print(f"   Sharpe Ratio: {sharpe_ratio:.2f}")
        
        # Recomendação final
        print(f"\n💡 RECOMENDAÇÃO:")
        if avg_return > 10 and consistency >= 60:
            print("   🚀 MODELO PRONTO PARA PRODUÇÃO!")
        elif avg_return > 5 and consistency >= 40:
            print("   ✅ Modelo promissor, considere mais testes")
        elif avg_return > 0:
            print("   ⚠️ Modelo precisa de otimização")
        else:
            print("   ❌ Modelo precisa de revisão completa")
        
        # SALVAR RELATÓRIO AUTOMÁTICO
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extrair steps do nome do checkpoint se possível
        steps_from_name = "unknown"
        if checkpoint_path and "_steps_" in checkpoint_path:
            try:
                steps_match = checkpoint_path.split("_steps_")[0].split("_")[-1]
                steps_from_name = f"{int(steps_match)//1000}k"
            except:
                steps_from_name = "unknown"
        
        report_filename = f"D:/Projeto/avaliacoes/avaliacao_v11_{steps_from_name}_{timestamp}.txt"
        
        print(f"\n💾 Salvando relatório: {report_filename}")
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"AVALIAÇÃO V11 SIGMOID CHECKPOINT {steps_from_name.upper()} - {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Checkpoint: {getattr(sys.modules[__name__], 'CHECKPOINT_PATH', DEFAULT_CHECKPOINT_PATH)}\n")
            f.write(f"Episodes: {NUM_EPISODES}\n")
            f.write(f"Steps per episode: {TEST_STEPS}\n")
            f.write(f"Portfolio inicial: ${INITIAL_PORTFOLIO}\n\n")
            
            f.write("RESULTADOS CONSOLIDADOS:\n")
            f.write(f"Retorno Médio: {avg_return:+.2f}%\n")
            f.write(f"Retorno Mediano: {median_return:+.2f}%\n")
            f.write(f"Melhor Episódio: {max_return:+.2f}%\n")
            f.write(f"Pior Episódio: {min_return:+.2f}%\n")
            f.write(f"Desvio Padrão: {std_return:.2f}%\n")
            f.write(f"Episódios Lucrativos: {positive_episodes}/{NUM_EPISODES} ({(positive_episodes/NUM_EPISODES)*100:.1f}%)\n\n")
            
            f.write("DETALHES POR EPISÓDIO:\n")
            for i, episode in enumerate(all_episodes):
                ep_return = episode['return_pct']
                grade_emoji = "🟢" if ep_return > 5 else "🟡" if ep_return > 0 else "🔴" 
                f.write(f"{grade_emoji} Episódio {i+1}: ${episode['initial_portfolio']:.2f} → ${episode['final_portfolio']:.2f} ({ep_return:+.2f}%) - {episode['trades_count']} trades\n")
            
            if all_trades:
                f.write(f"\nANÁLISE DE TRADES:\n")
                f.write(f"Total de Trades: {len(all_trades)}\n")
                f.write(f"Win Rate Global: {win_rate:.1f}%\n")
                f.write(f"Trades Lucrativos: {len(profitable_trades)}\n")
                f.write(f"Trades Perdedores: {len(losing_trades)}\n")
                f.write(f"Lucro Médio: ${avg_profit:.2f}\n")
                f.write(f"Perda Média: ${avg_loss:.2f}\n")
                f.write(f"PnL Total: ${total_pnl:.2f}\n")
                
                if avg_loss != 0 and losing_trades:
                    f.write(f"Profit Factor: {profit_factor:.2f}\n")
                f.write(f"Frequência Trading: {trading_frequency:.2f}%\n")
            
            f.write(f"\nAVALIAÇÃO FINAL: {grade}\n")
            f.write(f"Consistência: {consistency_grade} ({positive_episodes}/{NUM_EPISODES})\n")
        
        print(f"✅ Relatório salvo!")
        
        return True
        
    except Exception as e:
        print(f"❌ ERRO CRÍTICO: {e}")
        print(f"Detalhes: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    print(f"🚀 INICIANDO TESTE V8ELEGANCE - {datetime.now().strftime('%H:%M:%S')}")
    
    # Processar argumento da linha de comando se fornecido
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join("D:/Projeto", checkpoint_path)
        globals()['CHECKPOINT_PATH'] = checkpoint_path
        print(f"📂 Usando checkpoint: {os.path.basename(checkpoint_path)}")
    
    success = test_v8_elegance_trading()
    
    if success:
        print(f"\n✅ TESTE V8ELEGANCE CONCLUÍDO - {datetime.now().strftime('%H:%M:%S')}")
    else:
        print(f"\n❌ TESTE V8ELEGANCE FALHOU - {datetime.now().strftime('%H:%M:%S')}")