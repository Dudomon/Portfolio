#!/usr/bin/env python3
"""
🏗️ AMBIENTE DE TRADING MODULAR
Ambiente unificado para ser usado por massive.py, mainppo1.py, ppo.py, etc.
Garante 100% de compatibilidade entre todos os scripts.
"""

import gym
import numpy as np
import pandas as pd
from gym import spaces
from sklearn.impute import KNNImputer
import time

# Importar sistema de recompensa modular
try:
    from trading_framework.rewards import create_reward_system, CLEAN_REWARD_CONFIG
except ImportError:
    print("[WARNING] Não foi possível importar reward_system. Usando recompensa básica.")
    def create_reward_system(*args, **kwargs):
        return None
    CLEAN_REWARD_CONFIG = {}

class TradingEnv(gym.Env):
    """
    🎯 AMBIENTE DE TRADING UNIFICADO
    - Usado por massive.py, mainppo1.py, ppo.py
    - Garante 100% compatibilidade
    - Sistema de recompensas GENTLE_GUIDANCE
    """
    
    MAX_STEPS = 50000  # 🔥 PADRÃO: 50k steps por episódio
    
    def __init__(self, df, window_size=20, is_training=True, initial_balance=1000, reward_system_type="gentle_guidance"):
        super(TradingEnv, self).__init__()
        
        # 🚨 VERIFICAÇÃO DE SEGURANÇA: Dataset não pode estar vazio
        if df is None or len(df) == 0:
            raise ValueError("❌ Dataset está vazio ou None! Verifique o arquivo de dados.")
        
        # 🎯 SPLIT PERSONALIZADO: Usar dataset completo para treinamento contínuo
        if is_training:
            # Para treinamento: usar dataset completo com loop infinito
            self.df = df.copy()
            print(f"[TRADING ENV] Modo treinamento: {len(self.df):,} barras (loop infinito)")
        else:
            # Para avaliação: usar últimos 20% 
            train_size = int(len(df) * 0.8)
            self.df = df.iloc[train_size:].copy()
            print(f"[TRADING ENV] Modo avaliação: {len(self.df):,} barras")
            
        self.window_size = window_size
        self.current_step = window_size
        self.initial_balance = initial_balance
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.positions = []
        self.returns = []
        self.trades = []  # Garantir que seja uma lista
        self.start_date = pd.to_datetime(self.df.index[0])
        self.end_date = pd.to_datetime(self.df.index[-1])
        self.current_drawdown = 0.0
        self.peak_drawdown = 0.0
        self.max_lot_size = 0.08  # Ajustado conforme solicitado
        self.max_positions = 3
        self.current_positions = 0
        
        # 🔥 TEMPORAL BUFFER - Para visão temporal real
        self.temporal_buffer_size = 8  # Histórico de 8 barras consecutivas
        self.temporal_buffer = []  # Buffer circular para dados temporais
        
        # 📊 SISTEMA DE MÉTRICAS MODULAR - Reutilizável em outros scripts
        self.global_trades_history = []  # Histórico global de trades
        self.global_portfolio_peak = initial_balance
        self.recent_portfolios = []  # Deque para drawdown
        self.recent_pnls = []  # Deque para PnL médio
        self.steps_offset = 0  # Offset para steps acumulados
        self.global_step_offset = 0  # Offset global para steps
        
        # 🔥 ACTION SPACE PADRÃO: 100% compatível
        # estratégica: 0=hold, 1=long, 2=short
        # tática: 0=hold, 1=close, 2=adjust
        # sltp: valores ampliados [-3,3] para SL/TP mais significativos
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -3, -3, -3, -3, -3, -3], dtype=np.float32),  # estratégica, táticas, sltp
            high=np.array([2, 2, 2, 2, 3, 3, 3, 3, 3, 3], dtype=np.float32),  # estratégica, táticas, sltp
            dtype=np.float32
        )
        
        self.imputer = KNNImputer(n_neighbors=5)
        base_features = [
            'returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 'stoch_k', 'volume_ratio', 'var_99', 'atr_14'
        ]
        self.feature_columns = []
        for tf in ['5m', '15m', '4h']:
            self.feature_columns.extend([f"{f}_{tf}" for f in base_features])
        self._prepare_data()
        
        # 🔥 OBSERVATION SPACE PADRÃO: 100% compatível
        n_features = len(self.feature_columns) + self.max_positions * 7
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size * n_features,), dtype=np.float32
        )
        
        self.win_streak = 0
        self.episode_steps = 0
        self.episode_start_time = None
        self.partial_reward_alpha = 0.2   # Fator de escala para recompensa parcial (ajustado para melhor equilíbrio)
        # Garantir compatibilidade com reward
        self.realized_balance = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.last_trade_pnl = 0.0
        self.HOLDING_PENALTY_THRESHOLD = 60
        self.base_tf = '5m'
        self.lot_size = 0.05
        self.steps_since_last_trade = 0
        self.INACTIVITY_THRESHOLD = 24  # ~2h em 5m
        self.last_action = None
        self.hold_count = 0
        
        # 🚀 SISTEMA DE RECOMPENSAS MODULAR
        self.reward_system = create_reward_system(reward_system_type, initial_balance, CLEAN_REWARD_CONFIG)
        if self.reward_system:
            print(f"[TRADING ENV] ✅ Sistema {reward_system_type.upper()} ativado!")
        else:
            print(f"[TRADING ENV] ⚠️ Sistema de recompensas não disponível - usando básico")
        
        # Cache min/max do close_5m para performance
        self.close_5m_min = float(self.df['close_5m'].min())
        self.close_5m_max = float(self.df['close_5m'].max())

    def reset(self, **kwargs):
        """Reset do ambiente para um novo episódio."""
        # Log de debug para monitorar resets
        if hasattr(self, 'episode_steps') and self.episode_steps > 0:
            print(f"[TRADING ENV] RESET - Episódio anterior: {self.episode_steps:,} steps, Portfolio: ${self.portfolio_value:.2f}")
        
        # Reset robusto de todos os contadores e do pico
        self.current_step = self.window_size
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.peak_portfolio_value = self.initial_balance  # Zera o pico só no início do episódio
        self.realized_balance = self.initial_balance  # 🔥 FIX CRÍTICO: Resetar o realized_balance!
        self.positions = []
        self.returns = []
        self.trades = []  # Garantir que seja uma lista
        self.current_drawdown = 0.0
        self.peak_drawdown = 0.0
        self.current_positions = 0
        self.win_streak = 0
        self.episode_steps = 0
        self.episode_start_time = time.time()
        self.steps_since_last_trade = 0
        self.hold_count = 0
        self.last_action = None
        
        # 🔥 RESET BUFFER TEMPORAL
        self.temporal_buffer = []
        
        if hasattr(self, 'low_balance_steps'):
            self.low_balance_steps = 0
        if hasattr(self, 'high_drawdown_steps'):
            self.high_drawdown_steps = 0
        obs = self._get_observation()
        
        return obs

    def step(self, action):
        """Executa um passo no ambiente."""
        done = False
        
        # 🔥 LÓGICA DE EPISÓDIOS: Episódios mais longos para aprender consequências de longo prazo
        # Isso permite que o PPO entenda melhor os efeitos do overtrading
        if self.episode_steps >= 5000:  # Aumentado de 2048 para 5000
            done = True
            print(f"[TRADING ENV] Episódio completo: {self.episode_steps} steps - aprendizado de longo prazo")
        
        # 🔥 DATASET LOOP: Permitir loop infinito no dataset para treinamento contínuo
        # NÃO terminar quando acabam os dados - fazer loop
        if self.current_step >= len(self.df) - 1:
            self.current_step = self.window_size  # Reset para início
            print(f"[TRADING ENV] Dataset loop - resetando step para {self.window_size}")
            # NÃO definir done = True - continuar treinamento
            
        old_state = {
            "portfolio_total_value": self.realized_balance + sum(self._get_position_pnl(pos, self.df[f'close_{self.base_tf}'].iloc[self.current_step]) for pos in self.positions),
            "current_drawdown": self.current_drawdown
        }
        
        # 🔥 SISTEMA DE RECOMPENSAS: Usar sistema modular se disponível
        reward, info, done_from_reward = self._calculate_reward_and_info(action, old_state)
        # Ignorar done_from_reward - nunca terminar por recompensa
        # done = done or done_from_reward  # DESABILITADO
        
        # 📊 ATUALIZAR MÉTRICAS MODULARES
        self.update_metrics()
        
        # Portfolio já foi atualizado corretamente em _calculate_reward_and_info()
        # Apenas incrementar step e episode
        self.current_step += 1
        self.episode_steps += 1
        
        # Atualizar pico do portfólio e drawdown antes de calcular reward
        self.peak_portfolio = max(self.peak_portfolio, self.portfolio_value)
        self.current_drawdown = (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio if self.peak_portfolio > 0 else 0.0
        self.peak_drawdown = max(self.peak_drawdown, self.current_drawdown)
        
        obs = self._get_observation()
        if not isinstance(obs, np.ndarray):
            pass
        elif obs.dtype != np.float32:
            obs = obs.astype(np.float32)
            
        if done:
            # Fechar todas as posições abertas no final do episódio
            final_price = self.df[f'close_{self.base_tf}'].iloc[min(self.current_step, len(self.df)-1)]
            for pos in self.positions[:]:
                pnl = self._get_position_pnl(pos, final_price)
                self.realized_balance += pnl
                trade_info = {
                    'type': pos['type'],
                    'entry_price': pos['entry_price'],
                    'exit_price': final_price,
                    'lot_size': pos['lot_size'],
                    'entry_step': pos['entry_step'],
                    'exit_step': self.current_step,
                    'pnl_usd': pnl,
                    'duration': self.current_step - pos['entry_step']
                }
                self.trades.append(trade_info)
            self.positions = []
            
            # Atualizar portfolio final
            self.portfolio_value = self.realized_balance
            info["peak_drawdown_episode"] = self.current_drawdown
            info["final_balance"] = self.portfolio_value
            info["peak_portfolio"] = self.peak_portfolio_value
            info["total_trades"] = len(self.trades)
            info["win_rate"] = len([t for t in self.trades if t.get('pnl_usd', 0) > 0]) / len(self.trades) if self.trades else 0.0
            
        return obs, reward, done, info

    def _prepare_data(self):
        """Preparar dados técnicos para todos os timeframes."""
        # Renomear colunas close duplicadas para cada timeframe, se necessário
        if 'close' in self.df.columns:
            close_cols = [col for col in self.df.columns if col == 'close']
            if len(close_cols) > 1:
                # Se houver múltiplas colunas 'close', renomear para close_5m, close_15m, close_4h
                close_names = ['close_5m', 'close_15m', 'close_4h']
                for i, col in enumerate(close_cols):
                    self.df.rename(columns={col: close_names[i]}, inplace=True, level=0)
        # Se só existe uma coluna 'close', renomear para 'close_5m'
        if 'close' in self.df.columns and 'close_5m' not in self.df.columns:
            self.df.rename(columns={'close': 'close_5m'}, inplace=True)
        
        # Calcular features técnicas básicas para cada timeframe
        for tf in ['5m', '15m', '4h']:
            close_col = f'close_{tf}'
            if close_col in self.df.columns:
                # returns
                self.df.loc[:, f'returns_{tf}'] = self.df[close_col].pct_change().fillna(0)
                # volatility_20
                self.df.loc[:, f'volatility_20_{tf}'] = self.df[close_col].rolling(window=20).std().fillna(0)
                # sma_20
                self.df.loc[:, f'sma_20_{tf}'] = self.df[close_col].rolling(window=20).mean().bfill().fillna(0)
                # sma_50
                self.df.loc[:, f'sma_50_{tf}'] = self.df[close_col].rolling(window=50).mean().bfill().fillna(0)
                # rsi_14
                try:
                    import ta
                    self.df.loc[:, f'rsi_14_{tf}'] = ta.momentum.RSIIndicator(self.df[close_col], window=14).rsi().fillna(0)
                except Exception:
                    self.df.loc[:, f'rsi_14_{tf}'] = 0
                # stoch_k
                try:
                    self.df.loc[:, f'stoch_k_{tf}'] = ta.momentum.StochasticOscillator(self.df[close_col], self.df[close_col], self.df[close_col], window=14).stoch().fillna(0)
                except Exception:
                    self.df.loc[:, f'stoch_k_{tf}'] = 0
                # volume_ratio (não há volume, então zero)
                self.df.loc[:, f'volume_ratio_{tf}'] = 0
                # var_99
                self.df.loc[:, f'var_99_{tf}'] = self.df[close_col].rolling(window=20).quantile(0.01).fillna(0)
                # atr_14
                try:
                    self.df.loc[:, f'atr_14_{tf}'] = ta.volatility.AverageTrueRange(self.df[close_col], self.df[close_col], self.df[close_col], window=14).average_true_range().fillna(0)
                except Exception:
                    self.df.loc[:, f'atr_14_{tf}'] = 0
                # sma_cross
                self.df.loc[:, f'sma_cross_{tf}'] = (self.df[f'sma_20_{tf}'] > self.df[f'sma_50_{tf}']).astype(float) - (self.df[f'sma_20_{tf}'] < self.df[f'sma_50_{tf}']).astype(float)
                # momentum_5
                self.df.loc[:, f'momentum_5_{tf}'] = self.df[close_col].pct_change(periods=5).fillna(0)
        
        # Criar colunas ausentes como zero
        for col in self.feature_columns:
            if col not in self.df.columns:
                self.df[col] = 0
        
        # Limpeza e normalização de dados
        for col in self.feature_columns:
            self.df[col] = self.df[col].replace([np.inf, -np.inf], np.nan)
            Q1 = self.df[col].quantile(0.25)
            Q3 = self.df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            self.df.loc[:, col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
        
        # Imputação de valores faltantes
        base_imputer = KNNImputer(n_neighbors=5)
        base_imputed = base_imputer.fit_transform(self.df[self.feature_columns])
        
        if base_imputed.shape[1] != len(self.feature_columns):
            raise ValueError('Shape mismatch entre base_imputed e feature_columns')
        
        self.df.loc[:, self.feature_columns] = pd.DataFrame(base_imputed, index=self.df.index, columns=self.feature_columns)
        self.processed_data = self.df[self.feature_columns].values
        
        if np.any(np.isnan(self.processed_data)) or np.any(np.isinf(self.processed_data)):
            self.processed_data = np.nan_to_num(self.processed_data, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Feature binária de oportunidade (apenas para 5m)
        self.df['opportunity'] = 0
        if 'sma_cross_5m' in self.df.columns:
            cross = self.df['sma_cross_5m']
            self.df['opportunity'] = ((cross.shift(1) != cross) & (cross != 0)).astype(int)

    def _get_observation(self):
        """Obter observação do estado atual com visão temporal real."""
        # Verificar limites
        if self.current_step < self.window_size:
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.current_step >= len(self.df):
            return np.zeros(self.observation_space.shape, dtype=np.float32)
        
        # 🎯 OBSERVAÇÃO TEMPORAL REAL - Usar buffer temporal
        current_features = self.processed_data[self.current_step]
        self._update_temporal_buffer(current_features)
        
        # Obter sequência temporal real das últimas barras
        temporal_sequence = self._get_temporal_sequence()
        
        # 🎯 OBSERVAÇÃO DAS POSIÇÕES
        positions_obs = np.zeros((self.max_positions, 7))
        current_price = self.df['close_5m'].iloc[self.current_step]
        
        for i in range(self.max_positions):
            if i < len(self.positions):
                pos = self.positions[i]
                positions_obs[i, 0] = 1
                positions_obs[i, 1] = 0 if pos['type'] == 'long' else 1
                positions_obs[i, 2] = (pos['entry_price'] - self.close_5m_min) / (self.close_5m_max - self.close_5m_min)
                if pos['type'] == 'long':
                    pnl = (current_price - pos['entry_price']) * pos['lot_size']
                else:
                    pnl = (pos['entry_price'] - current_price) * pos['lot_size']
                positions_obs[i, 3] = pnl
                positions_obs[i, 4] = pos.get('sl', 0)
                positions_obs[i, 5] = pos.get('tp', 0)
                positions_obs[i, 6] = (self.current_step - pos['entry_step']) / len(self.df)
            else:
                positions_obs[i, :] = 0
        
        # 🎯 COMBINAR DADOS TEMPORAIS COM POSIÇÕES
        n_features = len(self.feature_columns)
        expanded_obs = []
        
        for i in range(self.window_size):
            if i < len(temporal_sequence):
                temporal_features = temporal_sequence[i]
            else:
                temporal_features = np.zeros(n_features)
            
            bar_obs = np.concatenate([temporal_features, positions_obs.flatten()])
            expanded_obs.append(bar_obs)
        
        obs = np.array(expanded_obs)
        flat_obs = obs.flatten().astype(np.float32)
        
        return flat_obs

    def _calculate_reward_and_info(self, action, old_state):
        """🚀 SISTEMA DE EXECUÇÃO DE POSIÇÕES REAIS + RECOMPENSA"""
        
        # 🎯 EXTRAIR AÇÕES DO MODELO
        if len(action) >= 4:
            entry_decision = int(action[0])  # 0=hold, 1=long, 2=short
            entry_confidence = float(action[1])  # Confiança 0-1
            position_size = float(action[2])  # Tamanho 0-1
            mgmt_action = int(action[3])  # 0=hold, 1=close, 2=adjust
        else:
            entry_decision = int(action[0]) if len(action) > 0 else 0
            entry_confidence = 0.5
            position_size = 0.5
            mgmt_action = 0
        
        current_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step]
        
        # 🔥 1. GERENCIAR POSIÇÕES EXISTENTES
        positions_to_remove = []
        for i, pos in enumerate(self.positions):
            # Atualizar PnL atual da posição
            current_pnl = self._get_position_pnl(pos, current_price)
            pos['current_pnl'] = current_pnl
            pos['duration'] = self.current_step - pos['entry_step']
            
            # Verificar SL/TP automáticos
            should_close = False
            close_reason = ""
            
            if pos['type'] == 'long':
                if current_price <= pos.get('sl_price', 0):
                    should_close = True
                    close_reason = "SL hit"
                elif current_price >= pos.get('tp_price', float('inf')):
                    should_close = True  
                    close_reason = "TP hit"
            else:  # short
                if current_price >= pos.get('sl_price', float('inf')):
                    should_close = True
                    close_reason = "SL hit" 
                elif current_price <= pos.get('tp_price', 0):
                    should_close = True
                    close_reason = "TP hit"
            
            # Fechar por gestão ou automático
            if should_close or mgmt_action == 1:  # Close all
                self._close_position(i, current_price, close_reason or "Manual close")
                positions_to_remove.append(i)
        
        # Remover posições fechadas
        for i in reversed(positions_to_remove):
            self.positions.pop(i)
            self.current_positions -= 1
        
        # 🔥 2. ABRIR NOVA POSIÇÃO SE DECISÃO DE ENTRADA
        if entry_decision > 0 and self.current_positions < self.max_positions:
            self._open_position(entry_decision, entry_confidence, position_size, current_price)
        
        # 🔥 3. CALCULAR RECOMPENSA COM SISTEMA MODULAR
        if self.reward_system is not None:
            try:
                reward, info, done = self.reward_system.calculate_reward_and_info(self, action, old_state)
                return reward, info, done
            except Exception as e:
                print(f"[WARNING] Erro no sistema de recompensas: {e}")
        
        # Fallback: recompensa básica
        current_portfolio = self.realized_balance + self._get_unrealized_pnl()
        old_portfolio = old_state.get("portfolio_total_value", self.initial_balance)
        reward = (current_portfolio - old_portfolio) / self.initial_balance
        
        return reward, {}, False

    def _get_position_pnl(self, pos, current_price):
        """🔥 COMPATIBILIDADE 100%: Cálculo PnL padrão"""
        if pos['type'] == 'long':
            return (current_price - pos['entry_price']) * pos['lot_size'] * 100
        else:
            return (pos['entry_price'] - current_price) * pos['lot_size'] * 100
    
    def _get_unrealized_pnl(self):
        """🔥 COMPATIBILIDADE 100%: Calcula PnL não realizado de todas as posições abertas."""
        if not self.positions:
            return 0.0
        current_price = self.df[f'close_{self.base_tf}'].iloc[self.current_step]
        return sum(self._get_position_pnl(pos, current_price) for pos in self.positions)

    def _update_temporal_buffer(self, current_features):
        """
        Atualiza o buffer temporal com as features da barra atual
        
        Args:
            current_features: Features da barra atual
        """
        # Adiciona features atuais ao buffer
        self.temporal_buffer.append(current_features.copy())
        
        # Mantém apenas as últimas temporal_buffer_size barras
        if len(self.temporal_buffer) > self.temporal_buffer_size:
            self.temporal_buffer.pop(0)
    
    def _get_temporal_sequence(self):
        """
        Retorna sequência temporal real das últimas barras
        
        Returns:
            temporal_sequence: Array com features das últimas barras
        """
        if len(self.temporal_buffer) < self.temporal_buffer_size:
            # Se não temos barras suficientes, preenche com zeros
            missing_bars = self.temporal_buffer_size - len(self.temporal_buffer)
            padding = [np.zeros_like(self.temporal_buffer[0]) for _ in range(missing_bars)]
            return np.array(padding + self.temporal_buffer)
        
        return np.array(self.temporal_buffer)
    
    def _open_position(self, entry_decision, confidence, position_size, current_price):
        """🚀 ABRIR POSIÇÃO REAL COM GESTÃO COMPLETA"""
        
        # Calcular lot size baseado em confidence e position_size
        base_lot = 0.01  # Lot mínimo
        max_lot = 0.08   # Lot máximo
        lot_size = base_lot + (max_lot - base_lot) * confidence * position_size
        lot_size = max(base_lot, min(lot_size, max_lot))
        
        # Calcular SL/TP adaptativos
        if entry_decision == 1:  # Long
            sl_points = 15 + (35 * (1 - confidence))  # 15-50 pontos baseado em confiança
            tp_points = 20 + (60 * confidence)        # 20-80 pontos baseado em confiança
            sl_price = current_price - sl_points
            tp_price = current_price + tp_points
            position_type = 'long'
        else:  # Short
            sl_points = 15 + (35 * (1 - confidence))
            tp_points = 20 + (60 * confidence)
            sl_price = current_price + sl_points
            tp_price = current_price - tp_points
            position_type = 'short'
        
        # Criar posição
        position = {
            'type': position_type,
            'entry_price': current_price,
            'entry_step': self.current_step,
            'lot_size': lot_size,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'sl_points': sl_points,
            'tp_points': tp_points,
            'confidence': confidence,
            'position_size': position_size,
            'current_pnl': 0.0,
            'duration': 0
        }
        
        self.positions.append(position)
        self.current_positions += 1
        
        # Debug
        if self.current_step % 1000 == 0:
            print(f"🔥 POSIÇÃO ABERTA: {position_type.upper()} @ {current_price:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}, Lot: {lot_size:.3f}")
    
    def _close_position(self, pos_index, current_price, reason):
        """🚀 FECHAR POSIÇÃO REAL COM REGISTRO COMPLETO"""
        
        if pos_index >= len(self.positions):
            return
        
        pos = self.positions[pos_index]
        
        # Calcular PnL final
        final_pnl = self._get_position_pnl(pos, current_price)
        duration = self.current_step - pos['entry_step']
        
        # Atualizar portfolio realizado
        self.realized_balance += final_pnl
        self.portfolio_value = self.realized_balance + self._get_unrealized_pnl()
        
        # Registrar trade no histórico
        trade = {
            'type': pos['type'],
            'entry_price': pos['entry_price'],
            'exit_price': current_price,
            'entry_step': pos['entry_step'],
            'exit_step': self.current_step,
            'lot_size': pos['lot_size'],
            'pnl': final_pnl,
            'pnl_usd': final_pnl,  # Compatibilidade
            'duration': duration,
            'sl_points': pos['sl_points'],
            'tp_points': pos['tp_points'],
            'confidence': pos['confidence'],
            'close_reason': reason,
            'step': self.current_step  # Para compatibilidade com reward system
        }
        
        self.trades.append(trade)
        
        # 📊 ADICIONAR AO HISTÓRICO GLOBAL MODULAR
        self.add_trade_to_global_history(trade)
        
        # Debug
        if self.current_step % 1000 == 0:
            print(f"💰 POSIÇÃO FECHADA: {pos['type'].upper()} PnL: ${final_pnl:.2f}, Duração: {duration} steps, Razão: {reason}")
        
        return trade
    
    # 📊 SISTEMA DE MÉTRICAS MODULAR - Reutilizável em outros scripts
    def get_metrics_summary(self, total_steps=None):
        """📊 Obter resumo completo das métricas - MODULAR"""
        if total_steps is None:
            total_steps = self.current_step
        
        # Atualizar pico global
        self.global_portfolio_peak = max(self.global_portfolio_peak, self.portfolio_value)
        
        # Calcular trades/dia total
        steps_elapsed = total_steps
        days_elapsed = steps_elapsed / 288  # 288 barras = 1 dia
        total_trades_accumulated = len(self.global_trades_history)
        trades_per_day = total_trades_accumulated / days_elapsed if days_elapsed > 0 else 0
        
        # 🎯 TRADES/DIA GLOBAL - ÚLTIMOS 100 DIAS EXATOS
        days_100_steps = 100 * 288  # Exatamente 100 dias = 28,800 steps
        threshold_100_days = total_steps - days_100_steps  # Últimos 100 dias EXATOS
        
        # 🚨 CORREÇÃO: Filtrar trades de sessões anteriores
        min_valid_step = total_steps - 10000  # Últimos 10k steps (sessão atual)
        current_session_threshold = max(min_valid_step, threshold_100_days)
        
        recent_trades = 0
        for trade in reversed(self.global_trades_history):
            trade_step = trade.get('step', 0)
            if trade_step >= current_session_threshold:
                recent_trades += 1
            else:
                break
        
        # Trades/dia dos últimos 100 dias
        global_trades_per_day = recent_trades / 100  # Sempre dividir por 100 dias exatos
        
        # Win rate
        if self.global_trades_history:
            wins = sum(1 for trade in self.global_trades_history if trade.get('pnl', 0) > 0)
            total_trades = len(self.global_trades_history)
            avg_win_rate = wins / total_trades if total_trades > 0 else 0
        else:
            avg_win_rate = 0
        
        # PnL médio
        if self.recent_pnls:
            avg_pnl = sum(list(self.recent_pnls)[-20:]) / min(20, len(self.recent_pnls))
        else:
            avg_pnl = 0
        
        # Drawdown médio
        if self.recent_portfolios:
            recent_drawdowns = []
            portfolios_list = list(self.recent_portfolios)
            for i in range(max(0, len(portfolios_list) - 100), len(portfolios_list)):
                if i > 0:
                    peak = max(portfolios_list[:i+1])
                    current = portfolios_list[i]
                    dd = (peak - current) / peak if peak > 0 else 0
                    recent_drawdowns.append(dd)
            avg_drawdown = sum(recent_drawdowns) / len(recent_drawdowns) if recent_drawdowns else 0
        else:
            avg_drawdown = 0
        
        # Duração média
        recent_durations = []
        for trade in reversed(self.global_trades_history):
            if len(recent_durations) >= 100:
                break
            trade_step = trade.get('step', 0)
            if trade_step >= current_session_threshold:
                duration = trade.get('duration', 0)
                if duration > 0:
                    recent_durations.append(duration)
        
        trades_count_for_duration = len(recent_durations)
        avg_duration_hours = sum(recent_durations) * 5 / 60 / len(recent_durations) if recent_durations else 0
        
        return {
            'portfolio_value': self.portfolio_value,
            'portfolio_peak': self.global_portfolio_peak,
            'trades_per_day': trades_per_day,
            'trades_per_day_100_days': global_trades_per_day,
            'win_rate': avg_win_rate,
            'avg_pnl': avg_pnl,
            'avg_drawdown': avg_drawdown,
            'avg_duration_hours': avg_duration_hours,
            'total_trades': total_trades_accumulated,
            'days_elapsed': days_elapsed,
            'trades_count_for_duration': trades_count_for_duration,
            'current_step': self.current_step,
            'total_steps': total_steps
        }
    
    def update_metrics(self):
        """📊 Atualizar métricas a cada step - MODULAR"""
        # Atualizar portfolio recente
        self.recent_portfolios.append(self.portfolio_value)
        if len(self.recent_portfolios) > 1000:
            self.recent_portfolios.pop(0)
        
        # Atualizar PnL recente se há trades
        if self.trades:
            latest_trade = self.trades[-1]
            if 'pnl' in latest_trade:
                self.recent_pnls.append(latest_trade['pnl'])
                if len(self.recent_pnls) > 100:
                    self.recent_pnls.pop(0)
    
    def add_trade_to_global_history(self, trade):
        """📊 Adicionar trade ao histórico global - MODULAR"""
        # Adicionar step global
        trade['step'] = self.current_step + self.global_step_offset
        self.global_trades_history.append(trade)
    
    def set_steps_offset(self, offset):
        """📊 Definir offset de steps - MODULAR"""
        self.steps_offset = offset
        self.global_step_offset = offset
    
    def get_debug_metrics(self, total_steps):
        """🔍 Debug detalhado das métricas - MODULAR"""
        days_100_steps = 100 * 288
        threshold_100_days = total_steps - days_100_steps
        min_valid_step = total_steps - 10000
        current_session_threshold = max(min_valid_step, threshold_100_days)
        
        recent_trades = 0
        for trade in reversed(self.global_trades_history):
            trade_step = trade.get('step', 0)
            if trade_step >= current_session_threshold:
                recent_trades += 1
            else:
                break
        
        debug_info = {
            'total_steps': total_steps,
            'steps_offset': self.steps_offset,
            'days_100_steps': days_100_steps,
            'old_threshold': threshold_100_days,
            'min_valid_step': min_valid_step,
            'new_threshold': current_session_threshold,
            'total_trades_history': len(self.global_trades_history),
            'recent_trades_found': recent_trades
        }
        
        if self.global_trades_history:
            latest_trade_step = self.global_trades_history[-1].get('step', 0)
            oldest_trade_step = self.global_trades_history[0].get('step', 0)
            debug_info.update({
                'latest_trade_step': latest_trade_step,
                'oldest_trade_step': oldest_trade_step,
                'trades_range': f"{oldest_trade_step:,} - {latest_trade_step:,}"
            })
        
        return debug_info 