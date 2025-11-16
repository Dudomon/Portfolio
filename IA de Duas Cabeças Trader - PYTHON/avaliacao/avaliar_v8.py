#!/usr/bin/env python3
"""
🚀 AVALIAÇÃO V8ELEGANCE - MODELOS DAYTRADER
Configuração EXATA do daytrader para avaliação V8:
- Portfolio: $500 
- Base lot: 0.02
- Max lot: 0.03
- Modo inferência (deterministic=False)
- Action Space: 4D (entry[0] + confidence[1] + pos1_mgmt[2] + pos2_mgmt[3])
- Architecture: V8 Elegance - Simplicidade Focada
"""

import sys
import os
import traceback
from datetime import datetime
sys.path.append("D:/Projeto")

import numpy as np
import pandas as pd
import torch

# Configuração específica do usuário - Checkpoint 1.5M steps (PÓS-PICO DOURADO)
CHECKPOINT_PATH = "D:/Projeto/Otimizacao/treino_principal/models/Elegance/AUTO_EVAL_1000000_steps_20250822_021746.zip"
INITIAL_PORTFOLIO = 500.0  # $500 conforme solicitado
BASE_LOT_SIZE = 0.02
MAX_LOT_SIZE = 0.03
TEST_STEPS = 3000
NUM_EPISODES = 3  # Número de episódios para testar
EPISODE_SPACING = 5000  # Espaçamento entre episódios no dataset

def find_v8_checkpoint():
    """🔍 Encontra o checkpoint mais recente do daytrader V8 Elegance"""
    import glob
    
    print("🔍 Procurando checkpoints V8 Elegance...")
    
    # 🔧 FIX: Usar EXPERIMENT_TAG correto (Elegance)
    EXPERIMENT_TAG = "Elegance"
    
    patterns = [
        # Primeiro: AUTO_EVAL da pasta Elegance (gerados automaticamente a cada 500k)
        f"D:/Projeto/Otimizacao/treino_principal/models/{EXPERIMENT_TAG}/AUTO_EVAL_*_steps_*.zip",
        f"D:/Projeto/Otimizacao/treino_principal/checkpoints/{EXPERIMENT_TAG}/*.zip",
        f"D:/Projeto/trading_framework/training/checkpoints/{EXPERIMENT_TAG}/*.zip",
        
        # Segundo: Qualquer checkpoint da pasta Elegance
        f"D:/Projeto/Otimizacao/treino_principal/models/{EXPERIMENT_TAG}/*.zip",
        f"D:/Projeto/Otimizacao/treino_principal/models/{EXPERIMENT_TAG}/*_steps_*.zip",
        
        # Terceiro: Fallback para DAYTRADER
        "D:/Projeto/Otimizacao/treino_principal/models/DAYTRADER/DAYTRADER_*.zip",
        "D:/Projeto/Otimizacao/treino_principal/models/**/DAYTRADER_*.zip",
        
        # Quarto: Fallback geral
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

def test_v8_elegance_trading():
    """🚀 Teste V8Elegance com configurações exatas do daytrader"""
    
    print(f"🚀 TESTE V8ELEGANCE - MODELOS DAYTRADER")
    print("=" * 60)
    print(f"💵 Portfolio Inicial: ${INITIAL_PORTFOLIO}")
    print(f"📊 Base Lot: {BASE_LOT_SIZE}")
    print(f"📊 Max Lot: {MAX_LOT_SIZE}")
    print(f"🧠 Modo: INFERÊNCIA (não-determinístico)")
    print("=" * 60)
    
    try:
        # Imports
        from sb3_contrib import RecurrentPPO
        from daytrader8dim import TradingEnv  # 🔥 USANDO DAYTRADER8DIM para compatibilidade 450D
        
        print("✅ Imports carregados")
        
        # 🔧 FIX: SEMPRE usar busca automática para checkpoint mais recente
        checkpoint_path = find_v8_checkpoint()
        if not checkpoint_path:
            print("❌ Nenhum checkpoint encontrado!")
            print(f"   📝 Fallback: Tentando usar CHECKPOINT_PATH hardcodado...")
            checkpoint_path = CHECKPOINT_PATH
            if not os.path.exists(checkpoint_path):
                print(f"❌ CHECKPOINT_PATH também não existe: {checkpoint_path}")
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
                from trading_framework.policies.two_head_v8_elegance import get_v8_elegance_kwargs
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
        
        # EXECUTAR MÚLTIPLOS EPISÓDIOS
        print(f"🚀 Iniciando {NUM_EPISODES} episódios de trading ({TEST_STEPS} steps cada)...")
        
        # Resultados consolidados
        all_episodes = []
        total_returns = []
        
        for episode_num in range(NUM_EPISODES):
            print(f"\n🎮 EPISÓDIO {episode_num + 1}/{NUM_EPISODES}")
            print("=" * 50)
            
            # 🔥 USAR DADOS MAIS RECENTES - trabalhar de trás para frente
            # Começar do final do dataset e ir para trás
            buffer_size = TEST_STEPS + 100
            start_from_end = (episode_num + 1) * buffer_size
            start_idx = total_len - start_from_end
            
            # Garantir que não vai antes do início
            if start_idx < 0:
                start_idx = 0
                print(f"⚠️ Ajustando início para {start_idx} (início do dataset)")
            else:
                print(f"🔥 Usando dados recentes: posição {start_idx} (últimos {start_from_end} registros)")
            
            end_idx = start_idx + TEST_STEPS + 100  # +100 buffer para janela
            episode_df = df.iloc[start_idx:end_idx].copy()
            
            print(f"📊 Dataset episódio: {len(episode_df):,} barras")
            print(f"📅 Período: {episode_df.index.min()} até {episode_df.index.max()}")
            
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
            
            # Variáveis de tracking do episódio
            portfolio_history = [INITIAL_PORTFOLIO]
            trades_log = []
            actions_log = []
            
            # 🚨 SISTEMA DE COOLDOWN - Mesmo do daytrader.py
            cooldown_counter = 0
            COOLDOWN_STEPS = 15  # Mesmo valor do daytrader.py
            
            # 🚨 CORREÇÃO: Track trades diretamente do environment
            initial_trades_count = len(getattr(env, 'trades', []))
            
            while not done and step < TEST_STEPS:
                # PREDIÇÃO EM MODO INFERÊNCIA (não-determinístico)
                action, lstm_states = model.predict(obs, state=lstm_states, deterministic=False)
                
                # 🚨 DEBUG: Log da decisão antes do step (4D ACTION SPACE V10PURE)
                if len(action) >= 2:
                    # 🔧 USAR MESMO MAPEAMENTO DO DAYTRADER.PY
                    raw_decision = float(action[0])
                    ACTION_THRESHOLD_LONG = 0.33   # Mesmo valor do daytrader.py
                    ACTION_THRESHOLD_SHORT = 0.67  # Mesmo valor do daytrader.py
                    
                    if raw_decision < ACTION_THRESHOLD_LONG:
                        entry_decision = 0  # HOLD
                    elif raw_decision < ACTION_THRESHOLD_SHORT:
                        entry_decision = 1  # LONG
                    else:
                        entry_decision = 2  # SHORT
                        
                    entry_confidence = float(action[1])  # FUSÃO quality+risk
                    
                    # 🎯 APLICAR MESMO FILTRO DO DAYTRADER.PY
                    MIN_CONFIDENCE_THRESHOLD = 0.8  # 🚨 ANTI-OVERTRADING: Mesmo valor do daytrader.py
                    
                    # 🚨 APLICAR COOLDOWN - Mesmo sistema do daytrader.py
                    if cooldown_counter > 0:
                        entry_decision = 0  # FORÇA HOLD durante cooldown
                        cooldown_counter -= 1
                        if step % 1000 == 0:  # Log a cada 1000 steps como daytrader.py
                            print(f"    [COOLDOWN] Forçando HOLD - {cooldown_counter} steps restantes")
                    
                    # 🚨 FILTRO DE CONFIANÇA - EXATAMENTE COMO DAYTRADER.PY
                    if entry_decision > 0 and entry_confidence < MIN_CONFIDENCE_THRESHOLD:
                        entry_decision = 0  # REJEITAR entrada
                        if step % 1000 == 0:  # Log a cada 1000 steps como daytrader.py
                            print(f"    [CONFIDENCE FILTER] Entry rejected: confidence={entry_confidence:.2f} < {MIN_CONFIDENCE_THRESHOLD}")
                    elif entry_decision > 0 and entry_confidence >= MIN_CONFIDENCE_THRESHOLD:
                        if step % 1000 == 0:  # Log de entradas aprovadas
                            print(f"    [ENTRY APPROVED] Decision={entry_decision}, Confidence={entry_confidence:.2f} >= {MIN_CONFIDENCE_THRESHOLD}")
                
                # Executar ação no ambiente (que deve aplicar a mesma lógica internamente)
                obs, reward, done, info = env.step(action)
                
                # Log da ação (4D action space V10PURE)
                actions_log.append({
                    'step': step,
                    'entry_decision': entry_decision,
                    'entry_confidence': float(action[1]),  # Entry confidence
                    'pos1_management': float(action[2]),   # Position 1 management (bidirecional)
                    'pos2_management': float(action[3]),   # Position 2 management (bidirecional)
                    'portfolio_value': env.portfolio_value,
                    'current_price': getattr(env, 'current_price', 0)
                })
                
                # 🚨 DEBUG: Log mais detalhado do environment response
                if step % 500 == 0:
                    current_positions = len(getattr(env, 'positions', []))
                    print(f"    [ENV STATE] Positions: {current_positions}, Portfolio: ${env.portfolio_value:.2f}")
                    if hasattr(env, 'last_action_debug'):
                        print(f"    [LAST ACTION] {env.last_action_debug}")
                
                # 🚨 CORREÇÃO: Log trades diretamente do environment
                current_trades_count = len(getattr(env, 'trades', []))
                if current_trades_count > len(trades_log) + initial_trades_count:
                    # Novos trades foram completados
                    new_trades = env.trades[-(current_trades_count - len(trades_log) - initial_trades_count):]
                    for trade in new_trades:
                        trade_info = {
                            'step': step,
                            'type': trade.get('type', 'unknown'),
                            'entry_price': trade.get('entry_price', 0),
                            'exit_price': trade.get('exit_price', 0),
                            'pnl': trade.get('pnl_usd', 0),
                            'lot_size': trade.get('volume', 0),
                            'duration': trade.get('duration', 0)
                        }
                        trades_log.append(trade_info)
                        print(f"  💼 Trade #{len(trades_log)}: {trade_info['type']} PnL=${trade_info['pnl']:.2f}")
                        
                        # 🚨 ATIVAR COOLDOWN após fechamento de trade
                        cooldown_counter = COOLDOWN_STEPS
                        print(f"  🕐 COOLDOWN ATIVADO: {COOLDOWN_STEPS} steps")
                
                # 🚨 DEBUG: Check for new positions opened
                current_positions = len(getattr(env, 'positions', []))
                if step > 0 and current_positions > getattr(env, 'prev_positions_count', 0):
                    print(f"  🟢 NEW POSITION OPENED at step {step}! Total positions: {current_positions}")
                env.prev_positions_count = current_positions
                
                portfolio_history.append(env.portfolio_value)
                
                if (step + 1) % 1000 == 0:  # Reduzir frequência de logs para múltiplos episódios
                    print(f"  Step {step+1}/{TEST_STEPS} - Portfolio: ${env.portfolio_value:.2f}")
                    print(f"    Positions: {len(env.positions)} | Realized: ${env.realized_balance:.2f} | Unrealized: ${env._get_unrealized_pnl():.2f}")
                
                step += 1
            
            # Análise do episódio
            final_portfolio = env.portfolio_value
            episode_return = ((final_portfolio - INITIAL_PORTFOLIO) / INITIAL_PORTFOLIO) * 100
            
            episode_result = {
                'episode': episode_num + 1,
                'start_idx': start_idx,
                'end_idx': end_idx,
                'period': f"{episode_df.index.min()} até {episode_df.index.max()}",
                'initial_portfolio': INITIAL_PORTFOLIO,
                'final_portfolio': final_portfolio,
                'return_pct': episode_return,
                'trades_count': len(trades_log),
                'actions_log': actions_log,
                'trades_log': trades_log,
                'portfolio_history': portfolio_history
            }
            
            all_episodes.append(episode_result)
            total_returns.append(episode_return)
            
            print(f"✅ Episódio {episode_num + 1}: ${INITIAL_PORTFOLIO:.2f} → ${final_portfolio:.2f} ({episode_return:+.2f}%)")
            print(f"   Trades executados: {len(trades_log)}")
            
            if len(trades_log) > 0:
                profitable_trades = [t for t in trades_log if t['pnl'] > 0]
                win_rate = (len(profitable_trades) / len(trades_log)) * 100
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
        
        # Análise consolidada de trades
        all_trades = []
        for episode in all_episodes:
            all_trades.extend(episode['trades_log'])
        
        # Usar o último episódio para análise de ações (representativo)
        last_episode = all_episodes[-1] if all_episodes else {'actions_log': []}
        actions_log = last_episode['actions_log']
        
        # Análise consolidada de trades
        if all_trades:
            total_trades = len(all_trades)
            profitable_trades = [t for t in all_trades if t['pnl'] > 0]
            losing_trades = [t for t in all_trades if t['pnl'] < 0]
            
            win_rate = (len(profitable_trades) / total_trades) * 100 if total_trades > 0 else 0
            avg_profit = np.mean([t['pnl'] for t in profitable_trades]) if profitable_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            total_pnl = sum(t['pnl'] for t in all_trades)
            
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
                gross_profit = sum(t['pnl'] for t in profitable_trades)
                gross_loss = abs(sum(t['pnl'] for t in losing_trades))
                profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
                print(f"⚖️ Profit Factor Global: {profit_factor:.2f}")
            
            # Frequência de trading
            total_steps = NUM_EPISODES * TEST_STEPS
            trading_frequency = (total_trades / total_steps) * 100
            print(f"📈 Frequência de Trading: {trading_frequency:.2f}% dos steps")
            print(f"📈 Trades por Episódio: {total_trades/NUM_EPISODES:.1f}")
            
        else:
            print(f"\n⚠️ NENHUM TRADE EXECUTADO EM {NUM_EPISODES} EPISÓDIOS")
            print("🔍 Modelo extremamente conservador em todos os períodos")
        
        # Análise de ações - V8Elegance (4D V10PURE - bidirecional)
        if actions_log:
            entry_decisions = [a['entry_decision'] for a in actions_log]
            entry_confidences = [a['entry_confidence'] for a in actions_log]
            pos1_mgmts = [a['pos1_management'] for a in actions_log]
            pos2_mgmts = [a['pos2_management'] for a in actions_log]
            
            hold_pct = (sum(1 for d in entry_decisions if d == 0) / len(entry_decisions)) * 100
            long_pct = (sum(1 for d in entry_decisions if d == 1) / len(entry_decisions)) * 100
            short_pct = (sum(1 for d in entry_decisions if d == 2) / len(entry_decisions)) * 100
            
            avg_confidence = np.mean(entry_confidences)
            avg_pos1_mgmt = np.mean(pos1_mgmts)
            avg_pos2_mgmt = np.mean(pos2_mgmts)
            
            print(f"\n🎮 ANÁLISE DAS AÇÕES V8ELEGANCE (4D V10PURE):")
            print(f"   📊 ENTRY DECISIONS:")
            print(f"   ⚪ HOLD: {hold_pct:.1f}%")
            print(f"   🟢 LONG: {long_pct:.1f}%") 
            print(f"   🔴 SHORT: {short_pct:.1f}%")
            print(f"   ⭐ Entry Confidence Média: {avg_confidence:.3f}")
            print(f"   📊 POSITION MANAGEMENT (bidirecional):")
            print(f"   🎯 [2] Pos1 Management: {avg_pos1_mgmt:+.2f} (negativo=SL, positivo=TP)")
            print(f"   🎯 [3] Pos2 Management: {avg_pos2_mgmt:+.2f} (negativo=SL, positivo=TP)")
        
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
        
        report_filename = f"D:/Projeto/avaliacoes/avaliacao_v8_{steps_from_name}_{timestamp}.txt"
        
        print(f"\n💾 Salvando relatório: {report_filename}")
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(f"AVALIAÇÃO V8 ELEGANCE CHECKPOINT {steps_from_name.upper()} - {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Checkpoint: {CHECKPOINT_PATH}\n")
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
    
    success = test_v8_elegance_trading()
    
    if success:
        print(f"\n✅ TESTE V8ELEGANCE CONCLUÍDO - {datetime.now().strftime('%H:%M:%S')}")
    else:
        print(f"\n❌ TESTE V8ELEGANCE FALHOU - {datetime.now().strftime('%H:%M:%S')}")