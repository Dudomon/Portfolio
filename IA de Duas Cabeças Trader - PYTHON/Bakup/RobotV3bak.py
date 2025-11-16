# -*- coding: utf-8 -*-
"""
⚔️ Legion AI Trader V1 - Trading Robot PPO v3.0 Enhanced
🔥 ATUALIZADO: Proteção anti-flip-flop avançada + GUI informativa
🎯 CONFIGURADO: TwoHeadV3HybridPolicy (arquitetura híbrida LSTM+GRU avançada)

COMPATIBILIDADE:
- Modelo ANTIGO: SL/TP range [-1,1] → valores ~0.005 (pequenos)
- Modelo NOVO: SL/TP range [-3,3] → valores ~1.5-2.8 (significativos)
- Multiplicadores ajustados proporcionalmente para nova escala
- Thresholds ajustados de 0.01 para 0.3

Compativel com modelos treinados usando TwoHeadV3HybridPolicy (HÍBRIDA) e TradingTransformerFeatureExtractor
"""

import gym
import numpy as np
import pandas as pd
import time
import tkinter as tk
from tkinter import scrolledtext, ttk
from threading import Thread, Event
from sb3_contrib import RecurrentPPO
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import MetaTrader5 as mt5
import os
import sys
import warnings
import torch
from datetime import datetime, timedelta
from collections import deque, Counter
import statistics

# Configuracoes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# 🔥 FUNÇÃO AUXILIAR PARA MT5 - CORREÇÃO DOS ERROS DE CHART_OBJECT_DELETE
def safe_mt5_object_delete(obj_name):
    """🔧 Função segura para deletar objetos do MT5"""
    try:
        # Tentar diferentes métodos de deleção do MT5
        if hasattr(mt5, 'chart_objects_delete'):
            mt5.chart_objects_delete(0, obj_name)
        elif hasattr(mt5, 'chart_object_delete'):
            safe_mt5_object_delete(obj_name)
        else:
            # Fallback: tentar deletar por tipo
            mt5.chart_objects_delete_all(0, -1, mt5.OBJ_ARROW_BUY)
            mt5.chart_objects_delete_all(0, -1, mt5.OBJ_ARROW_SELL)
            mt5.chart_objects_delete_all(0, -1, mt5.OBJ_TEXT)
            mt5.chart_objects_delete_all(0, -1, mt5.OBJ_HLINE)
    except Exception as e:
        # Silencioso - não é crítico se não conseguir deletar
        pass

# 🎨 IMPORTAÇÕES PARA VISUALIZAÇÃO AVANÇADA DO MODELO
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from collections import deque
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime
import json

# Configurar matplotlib para modo não-bloqueante
plt.ion()
plt.style.use('dark_background')  # Tema escuro para melhor visualização

# Paths para imports - CORRIGIR PARA ENCONTRAR TREINODIFERENCIADOPPO.PY
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Pasta pai (Otimizacao)
sys.path.insert(0, parent_dir)  # Adicionar no início para prioridade
sys.path.append(current_dir)

# 🔥 IMPORTAR SISTEMA DE REWARDS MODULAR
try:
    from trading_framework.rewards import create_reward_system, CLEAN_REWARD_CONFIG, ADAPTIVE_ANTI_OVERTRADING_CONFIG
    REWARD_SYSTEM_AVAILABLE = True
    print("[INFO] ✅ Sistema de rewards modular importado com sucesso!")
except ImportError as e:
    print(f"[WARNING] ❌ Não foi possível importar reward_system: {e}")
    REWARD_SYSTEM_AVAILABLE = False
    def create_reward_system(*args, **kwargs):
        return None
    CLEAN_REWARD_CONFIG = {}
    ADAPTIVE_ANTI_OVERTRADING_CONFIG = {
        "enable_regime_detection": False,
        "enable_portfolio_scaling": False,
        "enable_timing_evaluation": True,
        "enable_volatility_adjustment": False,
        "max_trades_per_day": 5,
        "min_trade_duration": 10,
        "quality_over_quantity": True,
        "hold_tolerance": 100,
        "scalping_penalty": 5.0
    }

# Classes de fallback
class BaseTradingEnv:
    def __init__(self, df, config=None, is_training=False):
        self.df = df
        self.config = config or type('Config', (), {
            'WINDOW_SIZE': 20,
            'MAX_POSITIONS': 3,
            'POSITION_SIZE': 0.02,  # 🔥 ATUALIZADO: Base lot 0.02
            'INITIAL_BALANCE': 500.0
        })()
        self.current_step = 20
        self.positions_tracker = []

class Config:
    def __init__(self):
        self.WINDOW_SIZE = 20
        self.MAX_POSITIONS = 3  
        self.POSITION_SIZE = 0.02  # 🔥 ATUALIZADO: Base lot 0.02
        self.INITIAL_BALANCE = 500.0

# Importar classes customizadas
TWOPOLICY_AVAILABLE = False
TRANSFORMER_AVAILABLE = False

# 🔥 IMPORTAR TWOHEADV3HYBRID ENHANCED
try:
    from trading_framework.policies.two_head_v3_hybrid_enhanced import TwoHeadV3HybridEnhanced
    print("[INFO] ✅ TwoHeadV3HybridEnhanced importada com sucesso!")
    TWOPOLICY_AVAILABLE = True
    TwoHeadPolicy = TwoHeadV3HybridEnhanced  # Alias para compatibilidade
except ImportError as e:
    print(f"[ERROR] ❌ Não conseguiu importar TwoHeadV3HybridEnhanced: {e}")
    try:
        from trading_framework.policies.two_head_v3_hybrid import TwoHeadV3HybridPolicy
        print("[INFO] ✅ TwoHeadV3HybridPolicy importada como fallback!")
        TWOPOLICY_AVAILABLE = True
        TwoHeadPolicy = TwoHeadV3HybridPolicy
    except ImportError as e2:
        print(f"[ERROR] ❌ Não conseguiu importar TwoHeadV3HybridPolicy: {e2}")
        try:
            from trading_framework.policies import TwoHeadPolicy
            print("[INFO] ✅ TwoHeadPolicy ORIGINAL importada como fallback!")
            TWOPOLICY_AVAILABLE = True
        except ImportError as e3:
            print(f"[ERROR] ❌ Não conseguiu importar TwoHeadPolicy original: {e3}")
            print("[WARN] ⚠️ RobotV3 NÃO SERÁ COMPATÍVEL com modelos treinados!")
            TWOPOLICY_AVAILABLE = False
            TwoHeadPolicy = "MlpPolicy"

# 🔥 CORREÇÃO CRÍTICA: Importar EXATAMENTE o mesmo extractor do treinodiferenciadoPPO.py
try:
    from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
    # Alias para compatibilidade com código existente
    TransformerFeatureExtractor = TradingTransformerFeatureExtractor
    print("[INFO] ✅ TradingTransformerFeatureExtractor importado do framework (COMPATÍVEL COM TREINAMENTO)!")
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] ❌ TradingTransformerFeatureExtractor não disponível: {e}")
    try:
        # Fallback para o import antigo
        from trading_framework.extractors import TransformerFeatureExtractor
        print("[INFO] ✅ TransformerFeatureExtractor (fallback) importado do framework!")
        TRANSFORMER_AVAILABLE = True
    except ImportError as e2:
        print(f"[WARN] ❌ TransformerFeatureExtractor fallback não disponível: {e2}")
        try:
            # Fallback para BaseFeaturesExtractor
            from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
            TransformerFeatureExtractor = BaseFeaturesExtractor
            print("[WARN] ⚠️ Usando BaseFeaturesExtractor como fallback")
        except ImportError:
            # Fallback final
            TransformerFeatureExtractor = None
            print("[WARN] ❌ Usando fallback None para TransformerFeatureExtractor")
        TRANSFORMER_AVAILABLE = False

# 🔥 IMPORTAR SISTEMA DE REWARDS DIFERENCIADO
# Sistema de rewards diferenciado removido - não necessário para o robot
# DIFF_REWARD_AVAILABLE = False

# 🔥 SISTEMA DE ESTATÍSTICAS DE SESSÃO
class SessionStats:
    def __init__(self):
        self.session_start = datetime.now()
        self.initial_balance = 0.0
        self.current_balance = 0.0
        self.peak_balance = 0.0
        self.total_buys = 0
        self.total_sells = 0
        self.successful_trades = 0
        self.failed_trades = 0
        self.total_profit = 0.0
        self.total_loss = 0.0
        self.max_drawdown = 0.0
        self.current_drawdown = 0.0
        self.positions_opened = 0
        self.positions_closed = 0
        self.avg_trade_duration = 0.0
        self.trade_durations = []
        
        # 🔥 ESTATÍSTICAS DO MODELO IA
        self.model_decisions = 0
        self.model_confidence_sum = 0.0
        self.avg_confidence = 0.0
        self.blocked_actions = 0
        self.protections_triggered = 0
        self.last_action = "HOLD"  # 🔥 ADICIONAR ATRIBUTO FALTANTE
        
    def add_model_decision(self, confidence=0.5):
        """Adiciona uma decisão do modelo"""
        self.model_decisions += 1
        self.model_confidence_sum += confidence
        self.avg_confidence = self.model_confidence_sum / self.model_decisions
        
    def add_blocked_action(self):
        """Adiciona uma ação bloqueada pelo anti-flip-flop"""
        self.blocked_actions += 1
        
    def update_balance(self, new_balance):
        """Atualiza balance e calcula drawdown"""
        self.current_balance = new_balance
        if self.initial_balance == 0.0:
            self.initial_balance = new_balance
            self.peak_balance = new_balance
            
        # Atualizar pico
        if new_balance > self.peak_balance:
            self.peak_balance = new_balance
            self.current_drawdown = 0.0
        else:
            # Calcular drawdown atual
            self.current_drawdown = (self.peak_balance - new_balance) / self.peak_balance * 100
            if self.current_drawdown > self.max_drawdown:
                self.max_drawdown = self.current_drawdown
    
    def add_trade(self, trade_type, profit, duration_seconds=None):
        """Adiciona um trade às estatísticas"""
        if trade_type.upper() == 'BUY':
            self.total_buys += 1
        elif trade_type.upper() == 'SELL':
            self.total_sells += 1
            
        if profit > 0:
            self.successful_trades += 1
            self.total_profit += profit
        else:
            self.failed_trades += 1
            self.total_loss += abs(profit)
            
        if duration_seconds:
            self.trade_durations.append(duration_seconds)
            self.avg_trade_duration = sum(self.trade_durations) / len(self.trade_durations)
    
    def get_session_profit(self):
        """Retorna lucro da sessão"""
        return self.current_balance - self.initial_balance
    
    def get_win_rate(self):
        """Retorna taxa de acerto"""
        total_trades = self.successful_trades + self.failed_trades
        return (self.successful_trades / total_trades * 100) if total_trades > 0 else 0.0
    
    def get_session_duration(self):
        """Retorna duração da sessão"""
        return datetime.now() - self.session_start

    def get_avg_confidence(self):
        """Retorna confiança média do modelo"""
        return self.avg_confidence if self.model_decisions > 0 else 0.0
    
    def update_last_action(self, action_name):
        """Atualiza a última ação executada"""
        self.last_action = action_name

# 🗑️ SISTEMA DE VISUALIZAÇÃO REMOVIDO
# MOTIVO: MetaTrader5 Python API não suporta ObjectCreate/ObjectDelete
# Essas funções existem apenas no MQL5, não no Python API

#  CLASSE ModelVisualizationSystem REMOVIDA COMPLETAMENTE
# MOTIVO: MetaTrader5 Python API não suporta ObjectCreate/ObjectDelete
# Essas funções existem apenas no MQL5 (Expert Advisors), não no Python

class IntelligentAntiFlipFlop:
    def __init__(self):
        # Sistema completamente desabilitado para avaliar modelo puro
        self.total_blocks = 0
        self.false_positives = 0
        self.pattern_matches = 0
        self.behavior_score = 100.0
        self.current_cooldown = 0.0
        self.last_action_time = 0
        
    def add_market_context(self, price, volatility, trend):
        """Sistema desabilitado - contexto ignorado"""
        pass
    
    def analyze_action_pattern(self, action_signature):
        """Sistema desabilitado - sempre aprova"""
        return True, 1.0, "Sistema anti-flip-flop DESABILITADO"
    
    def _detect_oscillation(self, actions):
        """Sistema desabilitado - nunca detecta oscilação"""
        return 0.0
    
    def _analyze_market_context(self, action_signature):
        """Sistema desabilitado - sempre aprova contexto"""
        return 1.0
    
    def _analyze_timing(self):
        """Sistema desabilitado - sempre aprova timing"""
        return 1.0
    
    def should_block_action(self, action_signature, current_time):
        """NUNCA bloqueia ações - sistema desabilitado para comportamento puro"""
        return False, "Sistema anti-flip-flop DESABILITADO - Modelo puro ativo"
    
    def update_action_executed(self, action_signature, current_time):
        """Sistema desabilitado - apenas atualiza timestamp"""
        self.last_action_time = current_time
    
    def _cleanup_old_patterns(self):
        """Sistema desabilitado - não faz limpeza"""
        pass
    
    def get_system_status(self):
        """Status: sistema completamente desabilitado"""
        return {
            'behavior_score': self.behavior_score,
            'current_cooldown': 0.0,
            'total_blocks': 0,
            'pattern_count': 0,
            'status': 'SISTEMA DESABILITADO - MODELO PURO ATIVO',
            'flip_flop_protection': 'OFF',
            'microtrade_protection': 'OFF'
        }

# 🗑️ CLASSE RISKMANAGEMENT REMOVIDA - ERA CÓDIGO MORTO INÚTIL
# Toda funcionalidade de gerenciamento de risco já está implementada na classe TradingEnv

class TradingEnv(gym.Env):
    """Ambiente completo de trading com MT5 - IDÊNTICO AO MAINPPO1.PY"""
    
    def __init__(self, log_widget=None):
        super().__init__()
        self.log_widget = log_widget  # Opcional para compatibilidade
        self.symbol = "GOLD"
        
        # 🔥 CONFIGURAÇÕES IDÊNTICAS AO MAINPPO1.PY
        self.window_size = 20
        self.initial_balance = 500.0  # ✅ Portfolio inicial $500
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.positions = []
        self.returns = []
        self.trades = []
        self.current_drawdown = 0.0
        self.peak_drawdown = 0.0
        self.max_lot_size = 0.02  # 🔥 ATUALIZADO: Max lot 0.02
        self.max_positions = 3  # MÁXIMO 3 POSIÇÕES SIMULTÂNEAS
        self.current_positions = 0
        self.current_step = 0
        self.done = False
        self.last_order_time = 0
        
        # 🛡️ TRACKER DE POSIÇÕES: Para detectar novas posições manuais
        self.known_positions = set()  # Set com tickets de posições conhecidas
        
        # 🔥 ACTION SPACE ATUALIZADO PARA MODELO CORRIGIDO
        # [estratégica, tática_0, tática_1, tática_2, sltp_0, sltp_1, sltp_2, sltp_3, sltp_4, sltp_5]
        # SL/TP agora usa range [-3, 3] para valores significativos
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, -3, -3, -3, -3, -3, -3]),
            high=np.array([2, 2, 2, 2, 3, 3, 3, 3, 3, 3]),
            dtype=np.float32
        )
        
        # 🔥 OBSERVATION SPACE PPOV1: 960 dimensões = 20 window × 48 features per step
        # Features alinhadas com ppov1.py: 5m+15m + features de alta qualidade
        base_features_5m_15m = [
            'returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 
            'stoch_k', 'bb_position', 'trend_strength', 'atr_14'
        ]
        
        # 🎯 FEATURES DE ALTA QUALIDADE (substituem 4h inúteis)
        high_quality_features = [
            'volume_momentum', 'price_position', 'volatility_ratio', 
            'intraday_range', 'market_regime', 'spread_pressure',
            'session_momentum', 'time_of_day', 'tick_momentum'
        ]
        
        self.feature_columns = []
        # Adicionar 5m e 15m (funcionam perfeitamente)
        for tf in ['5m', '15m']:
            self.feature_columns.extend([f"{f}_{tf}" for f in base_features_5m_15m])
        
        # Substituir 4h inúteis por features de alta qualidade
        self.feature_columns.extend(high_quality_features)
        
        # 🔥 CORREÇÃO CRÍTICA: Calcular n_features dinamicamente igual ao treinamento
        n_features = len(self.feature_columns) + self.max_positions * 7  # DINÂMICO como treinodiferenciadoPPO.py
        window_size = 20  # Igual ao treinamento
        total_obs_size = window_size * n_features
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(total_obs_size,), dtype=np.float32
        )
        
        self._log(f"[OBS SPACE] 🔥 ALINHADO COM TREINAMENTO: {len(self.feature_columns)} features + {self.max_positions}×7 positions = {n_features} × {window_size} = {total_obs_size}")
        
        # Variáveis de controle idênticas ao mainppo1.py
        self.realized_balance = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.last_trade_pnl = 0.0
        self.steps_since_last_trade = 0
        self.last_action = None
        self.hold_count = 0
        self.base_tf = '5m'
        
        # 🚀 POSITION SIZING ALINHADO COM TREINAMENTO (treinodiferenciadoPPO.py)
        self.base_lot_size = 0.02  # 🔥 ALINHADO: Base lot 0.02
        self.max_lot_size = 0.03   # 🔥 ALINHADO: Max lot 0.03 (IGUAL AO TREINAMENTO)
        self.lot_size = self.base_lot_size  # Será calculado dinamicamente
        
        # 🔥 SISTEMA DE REWARDS BÁSICO - SEM DEPENDÊNCIAS EXTERNAS
        self.reward_system = None  # Sistema de rewards desabilitado para o robot
        self._log("[INFO] ✅ Sistema de rewards básico ativado - Robot mode!")
        
        # Inicialização do MT5 com tratamento de erro
        try:
            if not mt5.initialize():
                self._log(f"[WARNING] ⚠️ Falha ao inicializar MetaTrader5. Erro: {mt5.last_error()}")
                self.mt5_connected = False
            else:
                self.mt5_connected = True
                
                if not mt5.symbol_select(self.symbol, True):
                    self._log(f"[WARNING] ⚠️ Símbolo {self.symbol} não disponível no Market Watch")
                    self.mt5_connected = False
        except Exception as e:
            self._log(f"[WARNING] ⚠️ Erro na inicialização do MT5: {e}")
            self.mt5_connected = False
        
        # Configurar MT5 filling mode apenas se conectado
        if self.mt5_connected:
            try:
                symbol_info = mt5.symbol_info(self.symbol)
                if symbol_info:
                    filling_mode = symbol_info.filling_mode
                    if filling_mode & 1:
                        self.filling_mode = mt5.ORDER_FILLING_FOK
                    elif filling_mode & 2:
                        self.filling_mode = mt5.ORDER_FILLING_IOC
                    elif filling_mode & 4:
                        self.filling_mode = mt5.ORDER_FILLING_RETURN
                    else:
                        self.filling_mode = mt5.ORDER_FILLING_FOK  # Default
                else:
                    self.filling_mode = mt5.ORDER_FILLING_FOK  # Default
            except Exception as e:
                self._log(f"[WARNING] ⚠️ Erro ao configurar filling mode: {e}")
                self.filling_mode = mt5.ORDER_FILLING_FOK  # Default
        else:
            self.filling_mode = None
        
        # Inicializar dados históricos para observações
        self._initialize_historical_data()
        
        # Log de status de conexão e configuração
        if self.mt5_connected:
            try:
                account_info = mt5.account_info()
                server_info = mt5.terminal_info()
                
                self._log(f"[🔌 MT5] Conectado - Conta: {account_info.login if account_info else 'N/A'}")
                self._log(f"[💰 SALDO] ${account_info.balance:.2f}" if account_info else "[💰 SALDO] N/A")
            except Exception as e:
                self._log(f"[WARNING] ⚠️ Erro ao obter informações da conta: {e}")
        else:
            self._log("[WARNING] ⚠️ MT5 não conectado - funcionando em modo limitado")
            
        self._log(f"[📊 SÍMBOLO] {self.symbol} - Max posições: {self.max_positions}")
        self._log(f"[⚙️ CONFIG] Lot size: {self.lot_size}, Balance inicial: ${self.initial_balance}")
    
    def _initialize_historical_data(self):
        """Inicializa dados históricos necessários para as observações"""
        try:
            # Carregar dados dos últimos 1000 bars de M5 para ter histórico suficiente
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 1000)
            if rates is None or len(rates) == 0:
                self._log("[WARNING] Não foi possível carregar dados históricos, usando dados vazios")
                # Criar dataframe vazio com colunas necessárias
                self.historical_df = pd.DataFrame()
                for col in self.feature_columns:
                    self.historical_df[col] = [0.0] * 100  # 100 linhas de dados vazios
                return
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            
            # Criar múltiplos timeframes simulados (baseado no M5)
            # 5m = dados originais, 15m = resample, 4h = resample
            df_5m = df.copy()
            df_15m = df.resample('15T').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
            }).dropna()
            df_4h = df.resample('4H').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
            }).dropna()
            
            # Calcular features para cada timeframe
            self.historical_df = pd.DataFrame(index=df_5m.index)
            
            # Processar apenas 5m e 15m (como no ppov1.py)
            for tf_name, tf_df in [('5m', df_5m), ('15m', df_15m)]:
                # Interpolar dados para o índice principal se necessário
                if len(tf_df) != len(df_5m):
                    tf_df = tf_df.reindex(df_5m.index, method='ffill')
                
                close_col = tf_df['close']
                high_col = tf_df['high']
                low_col = tf_df['low']
                
                # Calcular features técnicas básicas
                self.historical_df[f'returns_{tf_name}'] = close_col.pct_change().fillna(0)
                self.historical_df[f'volatility_20_{tf_name}'] = close_col.rolling(20).std().fillna(0)
                self.historical_df[f'sma_20_{tf_name}'] = close_col.rolling(20).mean().fillna(close_col)
                self.historical_df[f'sma_50_{tf_name}'] = close_col.rolling(50).mean().fillna(close_col)
                self.historical_df[f'rsi_14_{tf_name}'] = self._calculate_rsi(close_col, 14)
                self.historical_df[f'stoch_k_{tf_name}'] = 50.0  # Simplificado
                
                # Bollinger Band Position (0-1)
                bb_sma = close_col.rolling(20).mean().fillna(close_col)
                bb_std = close_col.rolling(20).std().fillna(0.01)
                bb_upper = bb_sma + (bb_std * 2)
                bb_lower = bb_sma - (bb_std * 2)
                self.historical_df[f'bb_position_{tf_name}'] = ((close_col - bb_lower) / (bb_upper - bb_lower)).fillna(0.5).clip(0, 1)
                
                # Trend Strength (força de tendência rolling)
                returns = close_col.pct_change().fillna(0)
                self.historical_df[f'trend_strength_{tf_name}'] = returns.rolling(10).mean().fillna(0)
                
                self.historical_df[f'atr_14_{tf_name}'] = self._calculate_atr(tf_df, 14)
            
            # 🎯 CALCULAR FEATURES DE ALTA QUALIDADE (baseadas em 5m)
            close_5m = df_5m['close']
            high_5m = df_5m['high']
            low_5m = df_5m['low']
            volume_5m = df_5m['tick_volume']
            
            # Volume momentum
            volume_sma = volume_5m.rolling(20).mean().fillna(1)
            self.historical_df['volume_momentum'] = (volume_5m / volume_sma).fillna(1.0)
            
            # Price position (posição do preço no range recente)
            high_20 = high_5m.rolling(20).max()
            low_20 = low_5m.rolling(20).min()
            self.historical_df['price_position'] = ((close_5m - low_20) / (high_20 - low_20).replace(0, 1)).fillna(0.5)
            
            # Volatility ratio
            vol_short = close_5m.rolling(5).std().fillna(0.01)
            vol_long = close_5m.rolling(20).std().fillna(0.01)
            self.historical_df['volatility_ratio'] = (vol_short / vol_long).fillna(1.0)
            
            # Intraday range
            self.historical_df['intraday_range'] = ((high_5m - low_5m) / close_5m.replace(0, 1)).fillna(0)
            
            # Market regime (trending vs ranging)
            sma_20 = close_5m.rolling(20).mean()
            atr_14 = (high_5m - low_5m).rolling(14).mean()
            self.historical_df['market_regime'] = (abs(close_5m - sma_20) / atr_14.replace(0, 1)).fillna(0.5)
            
            # Spread pressure (corrigido como no ppov1.py)
            intraday_range = high_5m - low_5m
            volatility_avg = intraday_range.rolling(20).mean()
            spread_pressure = (intraday_range / close_5m.replace(0, 1)) / (volatility_avg / close_5m.replace(0, 1)).replace(0, 1)
            self.historical_df['spread_pressure'] = spread_pressure.clip(0, 5).fillna(1.0)
            
            # Session momentum (48 barras = 4h)
            self.historical_df['session_momentum'] = close_5m.pct_change(periods=48).fillna(0)
            
            # Time of day (encoding circular)
            hours = pd.to_datetime(df_5m.index).hour
            self.historical_df['time_of_day'] = np.sin(2 * np.pi * hours / 24)
            
            # Tick momentum (direção dos ticks recentes)
            price_changes = close_5m.diff()
            tick_momentum = price_changes.rolling(5).apply(lambda x: (x > 0).sum() - (x < 0).sum()).fillna(0)
            self.historical_df['tick_momentum'] = (tick_momentum / 5.0).fillna(0)  # Normalizar -1 a 1
            
            # 🔥 NORMALIZAR E LIMPAR DADOS COMPLETAMENTE
            for col in self.feature_columns:
                if col in self.historical_df.columns:
                    # Limpar inf e nan
                    self.historical_df[col] = self.historical_df[col].replace([np.inf, -np.inf], np.nan)
                    self.historical_df[col] = self.historical_df[col].fillna(0.0)
                    # Garantir que são float32 válidos
                    self.historical_df[col] = self.historical_df[col].astype(np.float32)
                    # Clip para evitar valores extremos
                    self.historical_df[col] = np.clip(self.historical_df[col], -1000, 1000)
                else:
                    self.historical_df[col] = 0.0
                        
            self._log(f"[INFO] ✅ Dados históricos carregados: {len(self.historical_df)} registros")
            
        except Exception as e:
            self._log(f"[ERROR] Erro ao inicializar dados históricos: {e}")
            # Fallback: criar dataframe vazio
            self.historical_df = pd.DataFrame()
            for col in self.feature_columns:
                self.historical_df[col] = [0.0] * 100
    
    def _calculate_rsi(self, prices, window=14):
        """Calcula RSI para numpy array"""
        try:
            if len(prices) < window + 1:
                return 50.0
            
            # Calcular deltas
            deltas = np.diff(prices)
            
            # Separar ganhos e perdas
            gains = np.where(deltas > 0, deltas, 0)
            losses = np.where(deltas < 0, -deltas, 0)
            
            # Calcular médias móveis
            avg_gain = np.mean(gains[-window:]) if len(gains) >= window else 0
            avg_loss = np.mean(losses[-window:]) if len(losses) >= window else 1e-8
            
            # Evitar divisão por zero
            if avg_loss == 0:
                avg_loss = 1e-8
            
            # Calcular RSI
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return np.clip(rsi, 0, 100)
            
        except Exception as e:
            self._log(f"[⚠️ RSI] Erro no cálculo: {e}")
            return 50.0
    
    def _calculate_atr(self, df, window=14):
        """Calcula ATR sem NaN"""
        try:
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            
            # Garantir que não há NaN
            high_low = high_low.fillna(0.001)
            high_close = high_close.fillna(0.001)
            low_close = low_close.fillna(0.001)
            
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = true_range.rolling(window).mean().fillna(0.001)
            
            # Clip para evitar valores extremos
            atr = np.clip(atr, 0.0001, 1000)
            
            return atr.astype(np.float32)
        except Exception as e:
            self._log(f"[WARNING] Erro no cálculo ATR: {e}")
            return pd.Series([0.001] * len(df), index=df.index, dtype=np.float32)
    

    
    def _log(self, message):
        """Log com widget"""
        if self.log_widget:
            timestamp = time.strftime("%H:%M:%S")
            formatted_message = f"[{timestamp}] {message}\n"
            self.log_widget.insert(tk.END, formatted_message)
            self.log_widget.see(tk.END)
        print(message)
    
    def _get_observation(self):
        """Obtém observação IDÊNTICA ao mainppo1.py"""
        try:
            # Atualizar dados históricos com tick mais recente
            self._update_historical_data()
            
            # 🔥 OBSERVAÇÃO IDÊNTICA AO MAINPPO1.PY
            if len(self.historical_df) < self.window_size:
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            
            # Obter preço atual para cálculos de posições
            tick = mt5.symbol_info_tick(self.symbol)
            current_price = tick.bid if tick else 2000.0  # Fallback
            
            # 🔥 POSIÇÕES EXATAMENTE COMO NO MAINPPO1.PY
            positions_obs = np.zeros((self.max_positions, 7))
            
            # Converter posições MT5 para formato do ambiente de treinamento
            mt5_positions = mt5.positions_get(symbol=self.symbol) or []
            
            for i in range(self.max_positions):
                if i < len(mt5_positions):
                    pos = mt5_positions[i]
                    # Converter posição MT5 para formato de treinamento
                    positions_obs[i, 0] = 1  # Status aberta
                    positions_obs[i, 1] = 0 if pos.type == 0 else 1  # 0=long, 1=short
                    
                    # 🚀 SPEEDUP: Usar cache de min/max igual ao treinamento
                    if not hasattr(self, '_price_min_max_cache'):
                        # Calcular min/max baseado em dados históricos
                        if len(self.historical_df) > 0 and 'close_5m' in self.historical_df.columns:
                            close_values = self.historical_df['close_5m'].values
                            self._price_min_max_cache = {
                                'min': np.min(close_values),
                                'max': np.max(close_values), 
                                'range': np.max(close_values) - np.min(close_values)
                            }
                        else:
                            # Fallback se não há dados históricos
                            self._price_min_max_cache = {
                                'min': current_price - 100,
                                'max': current_price + 100,
                                'range': 200
                            }
                    
                    # Normalizar preço de entrada usando cache igual ao treinamento
                    positions_obs[i, 2] = (pos.price_open - self._price_min_max_cache['min']) / self._price_min_max_cache['range']
                    
                    # PnL atual (normalizado para observação - escala corrigida para eval)
                    pnl = self._get_position_pnl(pos, current_price) / 1000  # Normalizar para observação
                    positions_obs[i, 3] = pnl
                    
                    # SL e TP (valores diretos como no treinamento)
                    positions_obs[i, 4] = pos.sl if pos.sl > 0 else 0
                    positions_obs[i, 5] = pos.tp if pos.tp > 0 else 0
                    
                    # Position age igual ao treinamento: (current_step - entry_step) / total_steps
                    # Simular entry_step baseado no tempo da posição
                    if hasattr(pos, 'time'):
                        # Converter tempo da posição para steps simulados
                        position_age_seconds = time.time() - pos.time
                        position_age_steps = position_age_seconds / 300  # 5 minutos por step
                        total_steps = len(self.historical_df) if len(self.historical_df) > 0 else 1000
                        positions_obs[i, 6] = position_age_steps / total_steps
                    else:
                        positions_obs[i, 6] = 0.1  # Valor padrão
                else:
                    positions_obs[i, :] = 0  # Slot vazio
            
            # 🔥 FEATURES DINÂMICAS IGUAL AO TREINAMENTO
            # N features de mercado + max_positions×7 features de posições
            
            if len(self.historical_df) > 0 and len(self.feature_columns) > 0:
                recent_data = self.historical_df[self.feature_columns].tail(self.window_size).values
                
                # Se não temos dados suficientes, preencher com zeros
                if len(recent_data) < self.window_size:
                    padding = np.zeros((self.window_size - len(recent_data), len(self.feature_columns)))
                    recent_data = np.vstack([padding, recent_data])
            else:
                recent_data = np.zeros((self.window_size, len(self.feature_columns)))  # Features de mercado dinâmicas
            
            # Tile das posições para cada timestep (max_positions×7 features)
            tile_positions = np.tile(positions_obs.flatten(), (self.window_size, 1))
            
            # Concatenar mercado + posições = len(feature_columns) + max_positions×7 features
            obs = np.concatenate([recent_data, tile_positions], axis=1)
            
            # Flatten para formato final
            flat_obs = obs.flatten().astype(np.float32)
            
            # 🔥 LIMPAR NaN E INF ANTES DE RETORNAR
            flat_obs = np.nan_to_num(flat_obs, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # Verificações de integridade
            assert flat_obs.shape == self.observation_space.shape, f"Obs shape {flat_obs.shape} != expected {self.observation_space.shape}"
            assert not np.any(np.isnan(flat_obs)), f"Observação ainda contém NaN após limpeza"
            assert not np.any(np.isinf(flat_obs)), f"Observação ainda contém Inf após limpeza"
            
            return flat_obs
            
        except Exception as e:
            self._log(f"[ERROR] Erro ao obter observação: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _update_historical_data(self):
        """🔥 OBTER DADOS REAIS DO MT5 - NÃO SIMULADOS"""
        try:
            # Obter dados REAIS do MT5 para cada timeframe
            timeframes = {
                '5m': mt5.TIMEFRAME_M5,
                '15m': mt5.TIMEFRAME_M15,  
                '4h': mt5.TIMEFRAME_H4
            }
            
            new_time = pd.Timestamp.now()
            new_row = {}
            
            for tf_name, tf_mt5 in timeframes.items():
                # Obter barras históricas REAIS do MT5
                rates = mt5.copy_rates_from_pos(self.symbol, tf_mt5, 0, 100)
                
                if rates is not None and len(rates) > 50:
                    # Converter para DataFrame para cálculos
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    
                    # Calcular features REAIS
                    prices = df['close'].values
                    current_price = prices[-1]
                    
                    # Returns reais
                    returns = (current_price - prices[-2]) / prices[-2] if len(prices) > 1 else 0.0
                    
                    # SMAs reais
                    sma_20 = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
                    sma_50 = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
                    
                    # RSI real
                    rsi = self._calculate_rsi(prices[-15:], 14) if len(prices) >= 15 else 50.0
                    
                    # Volatilidade real
                    returns_array = np.diff(prices[-21:]) / prices[-21:-1] if len(prices) > 21 else [0]
                    volatility = np.std(returns_array) if len(returns_array) > 1 else 0.0
                    
                    # ATR real  
                    atr = self._calculate_atr_simple(df.iloc[-15:]) if len(df) >= 15 else abs(returns)
                    
                    # Stochastic real
                    if len(prices) >= 14:
                        high_14 = np.max(df['high'].values[-14:])
                        low_14 = np.min(df['low'].values[-14:])
                        stoch_k = ((current_price - low_14) / (high_14 - low_14)) * 100 if high_14 > low_14 else 50.0
                    else:
                        stoch_k = 50.0
                    
                    # Volume ratio real - com verificação de coluna
                    if 'tick_volume' in df.columns:
                        volumes = df['tick_volume'].values
                        avg_volume = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
                        volume_ratio = volumes[-1] / avg_volume if avg_volume > 0 else 1.0
                    else:
                        # Fallback se tick_volume não existir
                        volume_ratio = 1.0
                    
                    # VaR 99% real
                    var_99 = np.percentile(prices[-20:], 1) if len(prices) >= 20 else current_price * 0.95
                    
                    # Aplicar valores REAIS
                    new_row[f'returns_{tf_name}'] = np.float32(np.clip(returns, -0.1, 0.1))
                    new_row[f'volatility_20_{tf_name}'] = np.float32(volatility * 100)
                    new_row[f'sma_20_{tf_name}'] = np.float32(sma_20)
                    new_row[f'sma_50_{tf_name}'] = np.float32(sma_50)
                    new_row[f'rsi_14_{tf_name}'] = np.float32(np.clip(rsi, 0, 100))
                    new_row[f'stoch_k_{tf_name}'] = np.float32(np.clip(stoch_k, 0, 100))
                    new_row[f'volume_ratio_{tf_name}'] = np.float32(np.clip(volume_ratio, 0.1, 5.0))
                    new_row[f'var_99_{tf_name}'] = np.float32(var_99)
                    new_row[f'atr_14_{tf_name}'] = np.float32(atr)
                    new_row[f'close_{tf_name}'] = np.float32(current_price)
                    
                    # Log para confirmar dados reais (primeira vez)
                    if tf_name == '5m' and not hasattr(self, '_data_confirmed'):
                        self._log(f"[✅ DADOS] MT5 REAL: RSI={rsi:.1f}, SMA20={sma_20:.2f}, Preço={current_price:.2f}")
                        self._data_confirmed = True
                    
                else:
                    # Fallback com dados do tick se MT5 falhar
                    tick = mt5.symbol_info_tick(self.symbol)
                    current_price = tick.bid if tick else 2000.0
                    
                    new_row[f'returns_{tf_name}'] = np.float32(0.0)
                    new_row[f'volatility_20_{tf_name}'] = np.float32(0.01)
                    new_row[f'sma_20_{tf_name}'] = np.float32(current_price)
                    new_row[f'sma_50_{tf_name}'] = np.float32(current_price)
                    new_row[f'rsi_14_{tf_name}'] = np.float32(50.0)
                    new_row[f'stoch_k_{tf_name}'] = np.float32(50.0)
                    new_row[f'volume_ratio_{tf_name}'] = np.float32(1.0)
                    new_row[f'var_99_{tf_name}'] = np.float32(current_price * 0.95)
                    new_row[f'atr_14_{tf_name}'] = np.float32(current_price * 0.001)
                    new_row[f'close_{tf_name}'] = np.float32(current_price)
            
            # Adicionar nova linha com dados REAIS
            if new_row:
                self.historical_df = pd.concat([
                    self.historical_df,
                    pd.DataFrame([new_row], index=[new_time])
                ])
                
                # Manter apenas últimos 1000 registros
                if len(self.historical_df) > 1000:
                    self.historical_df = self.historical_df.tail(1000)
            
        except Exception as e:
            self._log(f"[⚠️ DADOS] Erro ao obter dados reais: {e}")
    
    def _calculate_atr_simple(self, df):
        """Calcula ATR simples"""
        try:
            if len(df) < 2:
                return 0.001
            tr_values = []
            for i in range(1, len(df)):
                high = df.iloc[i]['high']
                low = df.iloc[i]['low'] 
                prev_close = df.iloc[i-1]['close']
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
                tr_values.append(tr)
            return np.mean(tr_values) if tr_values else 0.001
        except:
            return 0.001
    
    def _execute_order(self, order_type: int, volume: float, sl_price: float = None, tp_price: float = None) -> str:
        """Executa ordem com SL/TP opcionais - conforme ação do agente"""
        try:
            current_time = time.time()
            if current_time - self.last_order_time < 1:
                return "ERROR_COOLDOWN"
            
            self.last_order_time = current_time
            
            # Verificar se mercado está aberto
            from datetime import datetime
            now = datetime.now()
            weekday = now.weekday()  # 0=Monday, 6=Sunday
            hour = now.hour
            
            # GOLD abre domingo às 19:00 BRT e fecha sexta às 21:00 BRT
            market_closed = False
            
            if weekday == 5:  # Saturday - sempre fechado
                market_closed = True
            elif weekday == 6 and hour < 19:  # Sunday before 19:00 BRT
                market_closed = True
            elif weekday == 4 and hour >= 21:  # Friday after 21:00 BRT
                market_closed = True
            
            if market_closed:
                self._log(f"[⚠️ MERCADO] Mercado fechado - {['Seg','Ter','Qua','Qui','Sex','Sáb','Dom'][weekday]} {hour:02d}:00")
                return "ERROR_MARKET_CLOSED"
            
            # Obter preço atual
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                self._log("[❌ ERRO] Não foi possível obter preço atual")
                return "ERROR_NO_PRICE"
            
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            
            # Preparar requisição com SL/TP opcionais
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "magic": 123456,
                "comment": "PPO Robot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.filling_mode
            }

            # Adicionar SL/TP se o agente especificou
            if sl_price is not None and sl_price > 0:
                request["sl"] = sl_price
            if tp_price is not None and tp_price > 0:
                request["tp"] = tp_price
            
            # Verificar ordem antes de executar
            check_result = mt5.order_check(request)
            if not check_result:
                last_error = mt5.last_error()
                self._log(f"[❌ ERRO] Ordem inválida: {last_error}")
                return f"ERROR_INVALID_ORDER|{last_error}"
            
            # TRADE_RETCODE_DONE = 10009
            # Retcode 0 também indica sucesso em order_check
            if check_result.retcode != 0 and check_result.retcode != mt5.TRADE_RETCODE_DONE:
                self._log(f"[❌ ERRO] Ordem seria rejeitada: {check_result.retcode} - {check_result.comment}")
                return f"ERROR_ORDER_CHECK|{check_result.retcode}"
            
            # Executar ordem
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                action_type = "📈 LONG" if order_type == mt5.ORDER_TYPE_BUY else "📉 SHORT"
                sl_info = f" | SL: {sl_price:.2f}" if sl_price else ""
                tp_info = f" | TP: {tp_price:.2f}" if tp_price else ""
                self._log(f"[🎯 TRADE] {action_type} executado - #{result.order} @ {price:.2f}{sl_info}{tp_info}")
                return f"SUCCESS|{result.order}|{price}|{action_type}|{sl_price or 0}|{tp_price or 0}"
            else:
                error_code = result.retcode if result else "None"
                last_error = mt5.last_error()
                self._log(f"[❌ ERRO] Falha na ordem: {error_code} | MT5 Error: {last_error}")
                
                # Diagnóstico adicional
                if error_code == "None":
                    self._log("[🔍 DEBUG] mt5.order_send() retornou None - possível mercado fechado")
                
                return f"ERROR_MT5|{error_code}"
                
        except Exception as e:
            self._log(f"[ERROR] ❌ Erro ao executar ordem: {e}")
            return "ERROR"

    def _auto_protect_manual_positions(self, model=None, vec_env=None):
        """🛡️ PROTEÇÃO AUTOMÁTICA: Aplica SL/TP em posições manuais sem proteção"""
        try:
            positions = mt5.positions_get(symbol=self.symbol) or []
            current_tickets = {pos.ticket for pos in positions}
            
            # Detectar novas posições (tickets que não conhecemos)
            new_positions = current_tickets - self.known_positions
            
            for position in positions:
                try:
                    # Debug: verificar atributos da posição
                    # self._log(f"🔍 DEBUG: Posição #{position.ticket} - Atributos disponíveis")
                    
                    # Verificar se é nova posição ou posição sem proteção
                    is_new = position.ticket in new_positions
                    needs_protection = (position.sl == 0.0 or position.tp == 0.0)
                    
                except Exception as pos_error:
                    self._log(f"❌ ERRO ao acessar atributos da posição: {pos_error}")
                    continue
                
                if is_new or needs_protection:
                    if is_new:
                        position_type = "LONG" if position.type == 0 else "SHORT"
                        try:
                            # MT5 TradePosition atributos: price_open, price_current, etc.
                            open_price = getattr(position, 'price_open', 'N/A')
                            self._log(f"🔍 NOVA POSIÇÃO DETECTADA: {position_type} #{position.ticket} @ {open_price}")
                        except Exception as attr_error:
                            self._log(f"🔍 NOVA POSIÇÃO DETECTADA: {position_type} #{position.ticket} (preço: erro {attr_error})")
                        self.known_positions.add(position.ticket)  # Adicionar ao tracker
                    
                    # Obter análise atual do modelo para definir SL/TP inteligente
                    obs = self._get_observation()
                    
                    # Verificar se temos modelo carregado
                    if model is None:
                        # Usar valores de segurança padrão sem modelo
                        sl_value = 0.3  # Valor médio de proteção
                        tp_value = 0.5  # Valor médio de lucro
                        self._log(f"⚠️ Modelo não disponível, usando valores de segurança padrão")
                    else:
                        try:
                            # Obter análise atual do modelo para definir SL/TP inteligente
                            obs = self._get_observation()
                            
                            # Verificar se precisamos de normalização
                            if vec_env is not None:
                                obs_reshaped = obs.reshape(1, -1)
                                normalized_obs = vec_env.normalize_obs(obs_reshaped)
                                model_obs = normalized_obs.flatten()
                            else:
                                model_obs = obs
                                
                            action, _states = model.predict(model_obs, deterministic=True)
                            
                            # Extrair valores SL/TP da ação do modelo
                            if len(action) >= 6:
                                sl_value = action[4] if len(action) > 4 else 0.3
                                tp_value = action[5] if len(action) > 5 else 0.5  # 🔥 CORREÇÃO: Definir tp_value
                            else:
                                sl_value = 0.3
                                tp_value = 0.5
                        except Exception as e:
                            self._log(f"⚠️ Erro na predição do modelo: {e}")
                            sl_value = 0.3
                            tp_value = 0.5
                    
                    current_price = mt5.symbol_info_tick(self.symbol)
                    if not current_price:
                        continue
                        
                    # Calcular SL/TP inteligente baseado no modelo + regras de segurança
                    new_sl = None
                    new_tp = None
                    
                    if position.type == 0:  # LONG
                        # SL: Modelo + regra mínima de 50 pontos
                        # 🔥 SL SEMPRE APLICADO: Modelo decide valor
                        model_sl = current_price.bid - abs(sl_value * 50)
                        safety_sl = current_price.bid - 100  # SL de segurança mínimo
                        new_sl = max(model_sl, safety_sl)  # Usar o mais conservador
                        
                        # TP: Modelo + regra mínima de 100 pontos  
                        # 🔥 TP SEMPRE APLICADO: Modelo decide valor
                        model_tp = current_price.ask + abs(tp_value * 100)
                        new_tp = model_tp
                            
                    else:  # SHORT
                        # SL: Modelo + regra mínima de 50 pontos
                        # 🔥 SL SEMPRE APLICADO: Modelo decide valor
                        model_sl = current_price.ask + abs(sl_value * 50)
                        safety_sl = current_price.ask + 100  # SL de segurança mínimo
                        new_sl = min(model_sl, safety_sl)  # Usar o mais conservador
                        
                        # TP: Modelo + regra mínima de 100 pontos
                        # 🔥 TP SEMPRE APLICADO: Modelo decide valor
                        model_tp = current_price.bid - abs(tp_value * 100)
                        new_tp = model_tp
                    
                    # Definir tipo de posição para logs
                    position_type = "LONG" if position.type == 0 else "SHORT"
                    
                    # Aplicar SL/TP apenas se calculados
                    if new_sl is not None or new_tp is not None:
                        if is_new:
                            self._log(f"🛡️ APLICANDO PROTEÇÃO AUTOMÁTICA NA NOVA POSIÇÃO:")
                        else:
                            self._log(f"🛡️ AUTO-PROTEÇÃO: {position_type} #{position.ticket} detectada sem proteção!")
                                                
                        sl_text = f"{new_sl:.2f}" if new_sl is not None else "N/A"
                        tp_text = f"{new_tp:.2f}" if new_tp is not None else "N/A"
                        self._log(f"🧠 MODELO SUGERE: SL={sl_text}, TP={tp_text}")
                        
                        # Enviar modificação
                        modify_request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": self.symbol,
                            "position": position.ticket,
                            "sl": new_sl if new_sl else position.sl,
                            "tp": new_tp if new_tp else position.tp
                        }
                        
                        result = mt5.order_send(modify_request)
                        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                            self._log(f"✅ PROTEÇÃO APLICADA: #{position.ticket} agora com SL/TP inteligente!")
                            sl_result = f"{new_sl:.2f}" if new_sl is not None else "mantido"
                            tp_result = f"{new_tp:.2f}" if new_tp is not None else "mantido"
                            self._log(f"📊 SL: {sl_result} | TP: {tp_result}")
                        else:
                            error_code = result.retcode if result else "None"
                            self._log(f"❌ ERRO ao aplicar proteção: {error_code}")
                    else:
                        # Não há SL/TP para aplicar
                        if is_new:
                            self._log(f"ℹ️ Nova posição {position_type} #{position.ticket} já tem proteção adequada")
                        else:
                            self._log(f"ℹ️ Posição {position_type} #{position.ticket} não precisa de ajustes")
            
            # Atualizar lista de posições conhecidas (remover posições fechadas)
            self.known_positions = current_tickets
                        
        except Exception as e:
            import traceback
            self._log(f"❌ ERRO na auto-proteção: {e}")
            self._log(f"📋 Detalhes do erro: {traceback.format_exc()}")

    def _manage_existing_positions(self):
        """Gerencia posições existentes (com SL/TP do agente)"""
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(symbol="GOLD")
            if positions:
                for pos in positions:
                    # Log das posições ativas com SL/TP definidos pelo agente
                    profit = pos.profit
                    sl = pos.sl
                    tp = pos.tp
                    action_type = "LONG" if pos.type == 0 else "SHORT"
                    
                    if abs(profit) > 10:  # Só logar se profit significativo
                        sl_info = f", SL: {sl:.2f}" if sl > 0 else ", SL: None"
                        tp_info = f", TP: {tp:.2f}" if tp > 0 else ", TP: None"
                        self._log(f"[POSITION] {action_type} #{pos.ticket} - P&L: ${profit:.2f}{sl_info}{tp_info}")
                        
        except Exception as e:
            self._log(f"[ERROR] Erro ao gerenciar posições: {e}")
    
    def _calculate_reward_and_info(self, action: np.ndarray, old_state: dict) -> tuple:
        """
        Método de compatibilidade com sistema de rewards modular
        Para uso em backtesting ou análise de performance
        """
        try:
            if self.reward_system:
                return self.reward_system.calculate_reward_and_info(self, action, old_state)
            else:
                # Reward básico baseado em mudança de portfolio
                current_portfolio = self.portfolio_value
                old_portfolio = old_state.get("portfolio_value", self.initial_balance)
                reward = (current_portfolio - old_portfolio) * 100.0  # Escalar para VecNormalize
                info = {
                    "reward_type": "basic",
                    "portfolio_change": current_portfolio - old_portfolio,
                    "final_reward": reward
                }
                return reward, info, False
        except Exception as e:
            self._log(f"[WARNING] Erro no cálculo de reward: {e}")
            return 0.0, {"error": str(e)}, False
    
    def _calculate_adaptive_position_size(self, action_confidence=1.0):
        """
        🚀 POSITION SIZING DINÂMICO: Adapta ao crescimento do portfolio ao vivo
        """
        try:
            # 🔥 OBTER BALANCE ATUAL DA CONTA MT5
            account_info = mt5.account_info()
            if account_info:
                current_balance = account_info.balance
                initial_balance = 1000.0  # Referência inicial
                portfolio_ratio = current_balance / initial_balance
            else:
                portfolio_ratio = 1.0
                current_balance = 1000.0
            
            # Calcular position size base como % do portfolio atual
            base_percentage = 0.10  # 10% do portfolio como base
            max_percentage = 0.16   # 16% do portfolio como máximo
            
            # Obter volatilidade atual (ATR normalizado)
            if len(self.historical_df) > 0:
                atr_5m = self.historical_df['atr_14_5m'].iloc[-1] if 'atr_14_5m' in self.historical_df.columns else 0.001
                # Usar preço atual do tick em vez de close_5m inexistente
                tick = mt5.symbol_info_tick(self.symbol)
                current_price = tick.bid if tick else 2000.0
            else:
                atr_5m = 0.001
                current_price = 2000.0
                
            volatility = atr_5m / current_price if current_price > 0 else 0.001
            
            # Normalizar volatilidade (0.001 = baixa, 0.01 = alta)
            volatility = max(min(volatility, 0.02), 0.0005)  # Limitar entre 0.05% e 2%
            
            # Calcular confiança baseada na força do sinal
            confidence_multiplier = min(action_confidence * 1.5, 1.5)  # Max 1.5x
            
            # Calcular divisor de volatilidade (maior volatilidade = menor posição)
            volatility_divisor = max(volatility * 100, 0.5)  # Min 0.5x
            
            # 🚀 PORTFOLIO SCALING: Ajustar percentual baseado no crescimento
            if portfolio_ratio > 2.0:  # Portfolio > 200% do inicial
                # Reduzir risco percentual conforme cresce (wealth preservation)
                scaling_factor = min(1.0, 2.0 / portfolio_ratio)
                base_percentage *= scaling_factor
                max_percentage *= scaling_factor
                self.log(f"[WEALTH PRESERVATION] Balance alto: ${current_balance:.2f}, reduzindo risco para {base_percentage:.1%}")
            elif portfolio_ratio < 0.8:  # Portfolio < 80% do inicial
                # Aumentar risco percentual para recuperação (controlled aggression)
                scaling_factor = min(1.2, 0.8 / portfolio_ratio)
                base_percentage *= scaling_factor
                max_percentage *= scaling_factor
                self.log(f"[RECOVERY MODE] Balance baixo: ${current_balance:.2f}, aumentando risco para {base_percentage:.1%}")
            
            # Calcular position size em % do portfolio
            position_percentage = base_percentage * confidence_multiplier / volatility_divisor
            position_percentage = max(min(position_percentage, max_percentage), 0.01)  # Entre 1% e 16%
            
            # 🔥 CONVERSÃO PARA LOTES: Baseado no preço atual do ouro
            portfolio_value_for_trade = current_balance * position_percentage
            
            # Para ouro: 1 lote = 100 onças, preço por onça
            # Valor por lote = preço_por_onça × 100
            value_per_lot = current_price * 100
            calculated_lots = portfolio_value_for_trade / value_per_lot
            
            # Aplicar limites práticos de lotes
            min_lots = 0.02  # 🔥 ATUALIZADO: Mínimo 0.02 (base lot)
            max_lots = min(0.03, current_balance / 5000)   # 🔥 ATUALIZADO: Máximo 0.03 ou baseado no portfolio
            
            final_size = max(min(calculated_lots, max_lots), min_lots)
            
            # 🔥 LOG DETALHADO PARA PRIMEIROS TRADES
            if hasattr(self.session_stats, 'total_buys') and (self.session_stats.total_buys + self.session_stats.total_sells) < 3:
                self.log(f"[DYNAMIC SIZING] Balance: ${current_balance:.2f} (ratio: {portfolio_ratio:.2f})")
                self.log(f"[DYNAMIC SIZING] Position %: {position_percentage:.1%} = ${portfolio_value_for_trade:.2f}")
                self.log(f"[DYNAMIC SIZING] Lots calculados: {calculated_lots:.3f} → Final: {final_size:.3f}")
                self.log(f"[DYNAMIC SIZING] Confidence: {action_confidence:.2f} | Volatility: {volatility:.4f}")
            
            return final_size
            
        except Exception as e:
            # Fallback inteligente baseado no balance atual
            try:
                account_info = mt5.account_info()
                if account_info:
                    fallback_size = max(0.1, min(0.16, account_info.balance / 5000))
                else:
                    fallback_size = 0.1
                self.log(f"[SIZING ERROR] Usando fallback: {fallback_size:.3f} lotes - Erro: {e}")
                return fallback_size
            except:
                return 0.1
    
    def _check_entry_filters(self, action_type):
        """
        🚀 FILTROS AFROUXADOS: Para permitir 20-30 trades/dia sem microtrading
        """
        # 🔥 FILTROS COMPLETAMENTE DESABILITADOS - COMPORTAMENTO PURO DO MODELO
        # Sempre permitir entrada - sem qualquer proteção ou filtro
        return True

    def _get_position_pnl(self, pos, current_price):
        """
        🔥 FUNÇÃO CRÍTICA: ESCALA PNL IDÊNTICA AO TREINAMENTO
        Para OURO: 1 ponto = $1 USD por 0.01 lot (escala corrigida)
        0.05 lot × 10 pontos × 100 = $50 USD (escala apropriada)
        """
        price_diff = 0
        if pos.get('type') == 'long' or (hasattr(pos, 'type') and pos.type == 0):
            price_diff = current_price - pos.get('entry_price', pos.price_open if hasattr(pos, 'price_open') else current_price)
        else:
            price_diff = pos.get('entry_price', pos.price_open if hasattr(pos, 'price_open') else current_price) - current_price
        
        lot_size = pos.get('lot_size', pos.volume if hasattr(pos, 'volume') else 0.02)
        
        # 🔥 FATOR CORRIGIDO: 100 para gerar PnL realista (compatível com treinamento)
        return price_diff * lot_size * 100

    def _get_unrealized_pnl(self):
        """
        Calcula o PnL não realizado de todas as posições abertas.
        IDÊNTICO AO TREINAMENTO
        """
        if not self.positions:
            return 0.0
        
        tick = mt5.symbol_info_tick(self.symbol)
        current_price = tick.bid if tick else 2000.0
        total_unrealized = 0.0
        
        for pos in self.positions:
            pnl = self._get_position_pnl(pos, current_price)
            total_unrealized += pnl
            
        return total_unrealized

class TradingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🚀 Legion AI Trader V1 - PPO Robot")
        self.root.geometry("1200x800")
        self.root.configure(bg='black')
        
        # 🔥 CONFIGURAÇÕES CRÍTICAS
        self.trading_active = False
        self.model = None
        self.vec_env = None
        self.env = None
        self.anti_flipflop = IntelligentAntiFlipFlop()  # 🔥 CORREÇÃO: Nome consistente
        self.session_stats = SessionStats()
        
        # 🎨 SISTEMA DE VISUALIZAÇÃO AVANÇADA
        self.visualization_system = None
        self.enable_visualization = True  # Flag para ativar/desativar visualização
        
        # Threading
        self.trading_thread = None
        self.stop_event = Event()
        
        # GUI Setup
        self.setup_interface()
        
        # 🔥 CORREÇÃO: Criar ambiente ANTES de carregar modelo
        self.env = TradingEnv()
        
        # Auto-load model
        self.auto_load_model()
        
        # 🎨 ANÁLISE PROFUNDA REATIVADA - SALVAR DADOS PARA EA
        self.enable_visualization = True  # 🔥 REATIVADO!
        self.visualization_system = None  # EA vai ler os dados
        self.model_data_file = "model_decisions.txt"  # Arquivo para EA
        self.log("🎨 [SYSTEM] Análise profunda REATIVADA - Dados salvos para EA visualizar")
        
        # 🎨 Instruções de uso
        self.log("=" * 60)
        self.log("🎨 ANÁLISE PROFUNDA DO MODELO IA - VISUALIZAÇÃO NO MT5:")
        self.log("   ▶ Clique em '🎨 Análise Profunda' para ativar/desativar")
        self.log("   🔵 Setas AZUIS = Sinais de COMPRA (confiança >60%)")
        self.log("   🔴 Setas VERMELHAS = Sinais de VENDA (confiança >60%)")
        self.log("   📊 Painel branco = Informações do modelo em tempo real")
        self.log("   🛡️ Linhas vermelhas tracejadas = Stop Loss sugerido")
        self.log("   🎯 Linhas verdes tracejadas = Take Profit sugerido")
        self.log("   🧠 Texto amarelo = Features importantes do modelo")
        self.log("   💡 TUDO aparece diretamente no gráfico do MetaTrader 5!")
        self.log("=" * 60)
    
    def setup_interface(self):
        """Interface gráfica melhorada com informações úteis"""
        self.root.title("⚔️ Legion AI Trader V1")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')
        
        # Frame principal
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Título
        title_label = tk.Label(main_frame, text="⚔️ LEGION AI TRADER V1 ⚔️", 
                              font=('Arial', 18, 'bold'), fg='#00ff88', bg='#1a1a1a')
        title_label.pack(pady=10)
        
        # Frame superior com controles e estatísticas
        top_frame = tk.Frame(main_frame, bg='#1a1a1a')
        top_frame.pack(fill=tk.X, pady=5)
        
        # Frame de controles (esquerda)
        control_frame = tk.Frame(top_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        
        tk.Label(control_frame, text="CONTROLES", font=('Arial', 12, 'bold'),
                fg='#00ff88', bg='#2d2d2d').pack(pady=5)
        
        button_frame = tk.Frame(control_frame, bg='#2d2d2d')
        button_frame.pack(pady=10)
        
        self.start_button = tk.Button(button_frame, text="▶ Iniciar Trading", 
                                     command=self.start_trading, bg='#00ff88', fg='black',
                                     font=('Arial', 10, 'bold'), width=18)
        self.start_button.pack(pady=2)
        
        self.stop_button = tk.Button(button_frame, text="⏹ Parar Trading", 
                                    command=self.stop_trading, bg='#ff4444', fg='white',
                                    font=('Arial', 10, 'bold'), width=18, state=tk.DISABLED)
        self.stop_button.pack(pady=2)
        
        # 🎨 Botão de Visualização Avançada
        self.viz_button = tk.Button(button_frame, text="🎨 Análise Profunda", 
                                   command=self.toggle_visualization, bg='#8800ff', fg='white',
                                   font=('Arial', 10, 'bold'), width=18)
        self.viz_button.pack(pady=2)
        
        # Status da visualização
        self.viz_status = tk.Label(button_frame, text="🎨 Visualização: ON" if self.enable_visualization else "🎨 Visualização: OFF", 
                                  fg='#8800ff', bg='#2d2d2d', font=('Arial', 9))
        self.viz_status.pack(pady=2)
        
        # Status do sistema
        status_frame = tk.Frame(control_frame, bg='#2d2d2d')
        status_frame.pack(pady=10, padx=10, fill=tk.X)
        
        self.status_model = tk.Label(status_frame, text="⚔️ Modelo: Carregando...", 
                                    fg='#ffaa00', bg='#2d2d2d', font=('Arial', 9))
        self.status_model.pack(anchor=tk.W)
        
        self.status_trading = tk.Label(status_frame, text="📊 Trading: Parado", 
                                      fg='#ffffff', bg='#2d2d2d', font=('Arial', 9))
        self.status_trading.pack(anchor=tk.W)
        
        self.status_connection = tk.Label(status_frame, text="🔗 MT5: Verificando...", 
                                         fg='#ffaa00', bg='#2d2d2d', font=('Arial', 9))
        self.status_connection.pack(anchor=tk.W)
        
        # Frame de estatísticas (direita)
        stats_frame = tk.Frame(top_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        stats_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(5, 0))
        
        tk.Label(stats_frame, text="ESTATÍSTICAS DA SESSÃO", font=('Arial', 12, 'bold'),
                fg='#00ff88', bg='#2d2d2d').pack(pady=5)
        
        # Grid de estatísticas
        stats_grid = tk.Frame(stats_frame, bg='#2d2d2d')
        stats_grid.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Linha 1: Balance e P&L
        row1 = tk.Frame(stats_grid, bg='#2d2d2d')
        row1.pack(fill=tk.X, pady=2)
        
        self.label_balance = tk.Label(row1, text="💰 Balance: $0.00", 
                                     fg='#ffffff', bg='#2d2d2d', font=('Arial', 10, 'bold'))
        self.label_balance.pack(side=tk.LEFT)
        
        self.label_session_pnl = tk.Label(row1, text="📈 Sessão P&L: $0.00", 
                                         fg='#00ff88', bg='#2d2d2d', font=('Arial', 10, 'bold'))
        self.label_session_pnl.pack(side=tk.RIGHT)
        
        # Linha 2: Trades
        row2 = tk.Frame(stats_grid, bg='#2d2d2d')
        row2.pack(fill=tk.X, pady=2)
        
        self.label_buys = tk.Label(row2, text="🟢 Buys: 0", 
                                  fg='#00ff88', bg='#2d2d2d', font=('Arial', 10))
        self.label_buys.pack(side=tk.LEFT)
        
        self.label_sells = tk.Label(row2, text="🔴 Sells: 0", 
                                   fg='#ff6666', bg='#2d2d2d', font=('Arial', 10))
        self.label_sells.pack(side=tk.RIGHT)
        
        # Linha 3: Win Rate e Drawdown
        row3 = tk.Frame(stats_grid, bg='#2d2d2d')
        row3.pack(fill=tk.X, pady=2)
        
        self.label_winrate = tk.Label(row3, text="🎯 Win Rate: 0%", 
                                     fg='#ffffff', bg='#2d2d2d', font=('Arial', 10))
        self.label_winrate.pack(side=tk.LEFT)
        
        self.label_drawdown = tk.Label(row3, text="📉 Drawdown: 0%", 
                                      fg='#ffaa00', bg='#2d2d2d', font=('Arial', 10))
        self.label_drawdown.pack(side=tk.RIGHT)
        
        # Linha 4: Posições e Duração
        row4 = tk.Frame(stats_grid, bg='#2d2d2d')
        row4.pack(fill=tk.X, pady=2)
        
        self.label_positions = tk.Label(row4, text="📊 Posições: 0/3", 
                                       fg='#ffffff', bg='#2d2d2d', font=('Arial', 10))
        self.label_positions.pack(side=tk.LEFT)
        
        self.label_duration = tk.Label(row4, text="⏱ Duração: 00:00:00", 
                                      fg='#ffffff', bg='#2d2d2d', font=('Arial', 10))
        self.label_duration.pack(side=tk.RIGHT)
        
        # Linha 5: Sistema Anti-Flip-Flop
        row5 = tk.Frame(stats_grid, bg='#2d2d2d')
        row5.pack(fill=tk.X, pady=2)
        
        self.label_stability = tk.Label(row5, text="🛡 Estabilidade: 100%", 
                                       fg='#00ff88', bg='#2d2d2d', font=('Arial', 10))
        self.label_stability.pack(side=tk.LEFT)
        
        self.label_cooldown = tk.Label(row5, text="⏰ Ativo há: 00:00:00", 
                                      fg='#ffffff', bg='#2d2d2d', font=('Arial', 10))
        self.label_cooldown.pack(side=tk.RIGHT)
        
        # Frame de informações de trading (inferior)
        trading_info_frame = tk.Frame(main_frame, bg='#2d2d2d', relief=tk.RAISED, bd=2)
        trading_info_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        tk.Label(trading_info_frame, text="INFORMAÇÕES DE TRADING", font=('Arial', 12, 'bold'),
                fg='#00ff88', bg='#2d2d2d').pack(pady=5)
        
        # Grid de informações detalhadas
        info_grid = tk.Frame(trading_info_frame, bg='#2d2d2d')
        info_grid.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Seção de Performance
        perf_frame = tk.LabelFrame(info_grid, text="📈 PERFORMANCE", font=('Arial', 10, 'bold'),
                                  fg='#00ff88', bg='#2d2d2d', bd=2, relief=tk.GROOVE)
        perf_frame.pack(fill=tk.X, pady=5)
        
        perf_grid = tk.Frame(perf_frame, bg='#2d2d2d')
        perf_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # Linha 1: Profit/Loss detalhado
        perf_row1 = tk.Frame(perf_grid, bg='#2d2d2d')
        perf_row1.pack(fill=tk.X, pady=2)
        
        self.label_total_profit = tk.Label(perf_row1, text="💰 Lucro Total: $0.00", 
                                          fg='#00ff88', bg='#2d2d2d', font=('Arial', 10))
        self.label_total_profit.pack(side=tk.LEFT)
        
        self.label_total_loss = tk.Label(perf_row1, text="💸 Perda Total: $0.00", 
                                        fg='#ff6666', bg='#2d2d2d', font=('Arial', 10))
        self.label_total_loss.pack(side=tk.RIGHT)
        
        # Linha 2: Trades detalhados
        perf_row2 = tk.Frame(perf_grid, bg='#2d2d2d')
        perf_row2.pack(fill=tk.X, pady=2)
        
        self.label_successful_trades = tk.Label(perf_row2, text="✅ Sucessos: 0", 
                                               fg='#00ff88', bg='#2d2d2d', font=('Arial', 10))
        self.label_successful_trades.pack(side=tk.LEFT)
        
        self.label_failed_trades = tk.Label(perf_row2, text="❌ Falhas: 0", 
                                           fg='#ff6666', bg='#2d2d2d', font=('Arial', 10))
        self.label_failed_trades.pack(side=tk.RIGHT)
        
        # Seção de Sistema
        system_frame = tk.LabelFrame(info_grid, text="⚔️ SISTEMA LEGION", font=('Arial', 10, 'bold'),
                                    fg='#00ff88', bg='#2d2d2d', bd=2, relief=tk.GROOVE)
        system_frame.pack(fill=tk.X, pady=5)
        
        system_grid = tk.Frame(system_frame, bg='#2d2d2d')
        system_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # Linha 1: Modelo e decisões
        sys_row1 = tk.Frame(system_grid, bg='#2d2d2d')
        sys_row1.pack(fill=tk.X, pady=2)
        
        self.label_model_decisions = tk.Label(sys_row1, text="🧠 Decisões: 0", 
                                             fg='#ffffff', bg='#2d2d2d', font=('Arial', 10))
        self.label_model_decisions.pack(side=tk.LEFT)
        
        self.label_avg_confidence = tk.Label(sys_row1, text="🎯 Confiança Média: 0%", 
                                            fg='#ffffff', bg='#2d2d2d', font=('Arial', 10))
        self.label_avg_confidence.pack(side=tk.RIGHT)
        
        # Linha 2: Proteções ativas
        sys_row2 = tk.Frame(system_grid, bg='#2d2d2d')
        sys_row2.pack(fill=tk.X, pady=2)
        
        self.label_protections = tk.Label(sys_row2, text="📊 Trades/h: 0.0", 
                                         fg='#ffaa00', bg='#2d2d2d', font=('Arial', 10))
        self.label_protections.pack(side=tk.LEFT)
        
        self.label_last_action = tk.Label(sys_row2, text="⚡ Última Ação: HOLD", 
                                         fg='#ffffff', bg='#2d2d2d', font=('Arial', 10))
        self.label_last_action.pack(side=tk.RIGHT)
        
        # Seção de Mercado
        market_frame = tk.LabelFrame(info_grid, text="📊 MERCADO", font=('Arial', 10, 'bold'),
                                    fg='#00ff88', bg='#2d2d2d', bd=2, relief=tk.GROOVE)
        market_frame.pack(fill=tk.X, pady=5)
        
        market_grid = tk.Frame(market_frame, bg='#2d2d2d')
        market_grid.pack(fill=tk.X, padx=10, pady=5)
        
        # Linha 1: Preço e spread
        market_row1 = tk.Frame(market_grid, bg='#2d2d2d')
        market_row1.pack(fill=tk.X, pady=2)
        
        self.label_current_price = tk.Label(market_row1, text="💎 GOLD: $0.00", 
                                           fg='#ffaa00', bg='#2d2d2d', font=('Arial', 10, 'bold'))
        self.label_current_price.pack(side=tk.LEFT)
        
        self.label_spread = tk.Label(market_row1, text="📏 Spread: 0.0", 
                                    fg='#ffffff', bg='#2d2d2d', font=('Arial', 10))
        self.label_spread.pack(side=tk.RIGHT)
        
        # Linha 2: Volatilidade e tendência
        market_row2 = tk.Frame(market_grid, bg='#2d2d2d')
        market_row2.pack(fill=tk.X, pady=2)
        
        self.label_volatility = tk.Label(market_row2, text="📈 Volatilidade: Baixa", 
                                        fg='#ffffff', bg='#2d2d2d', font=('Arial', 10))
        self.label_volatility.pack(side=tk.LEFT)
        
        self.label_trend = tk.Label(market_row2, text="🎯 Tendência: Neutra", 
                                   fg='#ffffff', bg='#2d2d2d', font=('Arial', 10))
        self.label_trend.pack(side=tk.RIGHT)
    
    def update_gui_stats(self):
        """Atualiza estatísticas na GUI em tempo real"""
        try:
            # Obter informações do MT5
            account_info = mt5.account_info()
            positions = mt5.positions_get(symbol="GOLD") or []
            
            if account_info:
                # Atualizar balance
                current_balance = account_info.balance
                self.session_stats.update_balance(current_balance)
                self.label_balance.config(text=f"💰 Balance: ${current_balance:.2f}")
                
                # Atualizar P&L da sessão
                session_pnl = self.session_stats.get_session_profit()
                pnl_color = '#00ff88' if session_pnl >= 0 else '#ff4444'
                self.label_session_pnl.config(text=f"📈 Sessão P&L: ${session_pnl:+.2f}", fg=pnl_color)
                
                # Atualizar drawdown
                drawdown_color = '#00ff88' if self.session_stats.current_drawdown < 5 else '#ffaa00' if self.session_stats.current_drawdown < 10 else '#ff4444'
                self.label_drawdown.config(text=f"📉 Drawdown: {self.session_stats.current_drawdown:.1f}%", fg=drawdown_color)
            
            # Atualizar trades
            self.label_buys.config(text=f"🟢 Buys: {self.session_stats.total_buys}")
            self.label_sells.config(text=f"🔴 Sells: {self.session_stats.total_sells}")
            
            # Atualizar win rate
            win_rate = self.session_stats.get_win_rate()
            winrate_color = '#00ff88' if win_rate >= 60 else '#ffaa00' if win_rate >= 40 else '#ff4444'
            self.label_winrate.config(text=f"🎯 Win Rate: {win_rate:.1f}%", fg=winrate_color)
            
            # Atualizar posições
            num_positions = len(positions)
            self.label_positions.config(text=f"📊 Posições: {num_positions}/3")
            
            # Atualizar duração da sessão
            duration = self.session_stats.get_session_duration()
            hours, remainder = divmod(int(duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            self.label_duration.config(text=f"⏱ Duração: {hours:02d}:{minutes:02d}:{seconds:02d}")
            
            # Atualizar sistema anti-flip-flop inteligente
            if hasattr(self, 'anti_flipflop'):
                status = self.anti_flipflop.get_system_status()
                behavior_score = status['behavior_score']
                
                stability_color = '#00ff88' if behavior_score >= 70 else '#ffaa00' if behavior_score >= 50 else '#ff4444'
                self.label_stability.config(text=f"🛡 Comportamento: {behavior_score:.0f}%", fg=stability_color)
                
                # 🔥 SUBSTITUIR COOLDOWN POR MÉTRICA ÚTIL: TEMPO DESDE ÚLTIMO TRADE
                # Calcular tempo desde último trade (mais útil que cooldown desabilitado)
                current_time = time.time()
                time_since_last_trade = current_time - self.session_stats.session_start.timestamp()
                hours, remainder = divmod(int(time_since_last_trade), 3600)
                minutes, seconds = divmod(remainder, 60)
                
                # Cor baseada na atividade recente
                if time_since_last_trade < 300:  # < 5 min
                    cooldown_color = '#00ff88'  # Verde - ativo
                elif time_since_last_trade < 1800:  # < 30 min
                    cooldown_color = '#ffaa00'  # Amarelo - moderado
                else:
                    cooldown_color = '#ff6666'  # Vermelho - inativo
                    
                self.label_cooldown.config(text=f"⏰ Ativo há: {hours:02d}:{minutes:02d}:{seconds:02d}", fg=cooldown_color)
                
                # 🔥 MÉTRICA ÚTIL: TRADES POR HORA (mais útil que bloqueios desabilitados)
                total_trades = self.session_stats.successful_trades + self.session_stats.failed_trades
                session_hours = max(1, time_since_last_trade / 3600)  # Evitar divisão por zero
                trades_per_hour = total_trades / session_hours
                
                # Cor baseada na atividade de trading
                if trades_per_hour >= 4:
                    trades_color = '#00ff88'  # Verde - muito ativo
                elif trades_per_hour >= 2:
                    trades_color = '#ffaa00'  # Amarelo - moderadamente ativo
                else:
                    trades_color = '#ff6666'  # Vermelho - pouco ativo
                    
                self.label_protections.config(text=f"📊 Trades/h: {trades_per_hour:.1f}", fg=trades_color)
                
                # Atualizar informações do modelo
                self.label_model_decisions.config(text=f"🧠 Decisões: {self.session_stats.model_decisions}")
                avg_confidence = self.session_stats.get_avg_confidence()
                self.label_avg_confidence.config(text=f"🎯 Confiança: {avg_confidence:.1f}%")
                self.label_last_action.config(text=f"⚡ Última: {self.session_stats.last_action}")
                
                # Atualizar informações de performance detalhadas
                self.label_total_profit.config(text=f"💰 Lucro: ${self.session_stats.total_profit:.2f}")
                self.label_total_loss.config(text=f"💸 Perda: ${self.session_stats.total_loss:.2f}")
                self.label_successful_trades.config(text=f"✅ Sucessos: {self.session_stats.successful_trades}")
                self.label_failed_trades.config(text=f"❌ Falhas: {self.session_stats.failed_trades}")
                
                # Atualizar informações de mercado
                tick = mt5.symbol_info_tick("GOLD")
                if tick:
                    self.label_current_price.config(text=f"💎 GOLD: ${tick.bid:.2f}")
                    spread = tick.ask - tick.bid
                    self.label_spread.config(text=f"📏 Spread: {spread:.2f}")
                    
                    # 🔥 MÉTRICAS ÚTEIS CALCULADAS DIRETAMENTE DOS DADOS MT5
                    # Calcular volatilidade real baseada no ATR
                    rates = mt5.copy_rates_from_pos("GOLD", mt5.TIMEFRAME_M5, 0, 20)
                    if rates is not None and len(rates) >= 14:
                        df_temp = pd.DataFrame(rates)
                        atr = self.env._calculate_atr_simple(df_temp) if hasattr(self, 'env') else 0.5
                        
                        # Classificar volatilidade baseada no ATR
                        if atr > 1.5:
                            volatility_level = "ALTA"
                            vol_color = '#ff6666'
                        elif atr > 0.8:
                            volatility_level = "MÉDIA"
                            vol_color = '#ffaa00'
                        else:
                            volatility_level = "BAIXA"
                            vol_color = '#00ff88'
                        
                        self.label_volatility.config(text=f"📈 ATR: {atr:.2f} ({volatility_level})", fg=vol_color)
                        
                        # Calcular tendência baseada em SMA simples
                        if len(df_temp) >= 10:
                            prices = df_temp['close']
                            sma_short = prices[-5:].mean()  # SMA 5
                            sma_long = prices[-10:].mean()  # SMA 10
                            current_price = prices.iloc[-1]
                            
                            if sma_short > sma_long and current_price > sma_short:
                                trend_direction = "BULLISH"
                                trend_color = '#00ff88'
                            elif sma_short < sma_long and current_price < sma_short:
                                trend_direction = "BEARISH" 
                                trend_color = '#ff6666'
                            else:
                                trend_direction = "LATERAL"
                                trend_color = '#ffaa00'
                                
                            self.label_trend.config(text=f"🎯 Trend: {trend_direction}", fg=trend_color)
                        else:
                            self.label_trend.config(text=f"🎯 Trend: DADOS INSUF.", fg='#ffffff')
                    else:
                        # Fallback se não conseguir dados
                        self.label_volatility.config(text=f"📈 ATR: SEM DADOS", fg='#ffffff')
                        self.label_trend.config(text=f"🎯 Trend: SEM DADOS", fg='#ffffff')
            
            # Verificar conexão MT5
            if mt5.terminal_info() is None:
                self.status_connection.config(text="🔗 MT5: Desconectado", fg='#ff4444')
            else:
                self.status_connection.config(text="🔗 MT5: Conectado", fg='#00ff88')
            
        except Exception as e:
            self.log(f"[ERRO GUI] Falha ao atualizar estatísticas: {e}")
        
        # Reagendar atualização
        if self.trading:
            self.gui_update_timer = self.root.after(2000, self.update_gui_stats)  # Atualizar a cada 2 segundos
    
    def auto_load_model(self):
        """🔥 CARREGAMENTO AUTOMÁTICO DO MODELO TREINODIFERENCIADOPPO"""
        try:
            # 🔥 USAR MODELO LEGION V1 COMPATÍVEL COM TWOHEADV3HYBRID
            # Caminho relativo à pasta do script (Modelo PPO Trader)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(script_dir, "Modelo PPO", "Legion V1.zip")
            
            if not os.path.exists(model_path):
                self.log(f"[❌ MODELO] Arquivo não encontrado: {model_path}")
                self.status_model.config(text="❌ Modelo: NÃO ENCONTRADO", fg='#ff6b6b')
                return False
            
            # Verificar se o ambiente existe
            if not hasattr(self, 'env') or self.env is None:
                self.log(f"[❌ MODELO] Ambiente não inicializado!")
                self.status_model.config(text="❌ Ambiente: ERRO", fg='#ff6b6b')
                return False
            
            # 🔥 VECNORMALIZE DO MODELO TRADER - CAMINHO CORRETO
            vec_normalize_paths = [
                "vec_normalize_final.pkl",  # Arquivo na pasta atual (Modelo PPO Trader)
                "../vec_normalize_final.pkl",  # Fallback na raiz
                "vecnormalize_final.pkl"  # Fallback final
            ]
            
            vec_normalize_path = None
            for path in vec_normalize_paths:
                if os.path.exists(path):
                    vec_normalize_path = path
                    break
            
            # 🔥 LEGION V1: USAR VECNORMALIZE DA PASTA TRADER
            usar_vecnormalize = True  # Habilitar VecNormalize com arquivo correto
            
            if usar_vecnormalize and vec_normalize_path:
                # Criar ambiente com VecNormalize (configurações do treinodiferenciadoPPO)
                temp_env = DummyVecEnv([lambda: self.env])
                self.vec_env = VecNormalize(temp_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0)
                
                # 🔥 CARREGAR VECNORMALIZE DO TREINODIFERENCIADOPPO
                try:
                    # Tentar carregar com VecNormalize.load (método oficial)
                    self.vec_env = VecNormalize.load(vec_normalize_path, temp_env)
                    self.log(f"[📊 VECNORM] VecNormalize carregado de {vec_normalize_path}")
                    
                    # 🔥 IMPORTANTE: Configurar para avaliação MAS CONTINUAR ATUALIZANDO
                    # training=False previne updates das estatísticas durante predict()
                    # MAS permite updates manuais com dados reais para melhorar normalização
                    self.vec_env.training = False  # Não atualizar durante predict()
                    self.vec_env.norm_obs = True   # Continuar normalizando observações
                    self.vec_env.norm_reward = False  # Não normalizar rewards no robô
                    
                    self.log(f"[🔧 CONFIG] VecNormalize: training=False, norm_obs=True, norm_reward=False")
                    self.log(f"[📊 ESTRATÉGIA] Preservar stats de treinamento + melhorar com dados reais")
                    
                    # 🔥 IMPORTANTE: Configurar para avaliação (não resetar estatísticas)
                    self.vec_env.training = False
                    self.log(f"[🔧 CONFIG] VecNormalize configurado para avaliação (training=False)")
                    
                    # 🔥 MANTER ESTATÍSTICAS TREINADAS - NÃO RESETAR
                    if hasattr(self.vec_env, 'obs_rms') and hasattr(self.vec_env, 'ret_rms'):
                        # Verificar estatísticas carregadas
                        obs_mean = getattr(self.vec_env.obs_rms, 'mean', None)
                        if obs_mean is not None:
                            mean_range = f"{obs_mean.min():.3f} a {obs_mean.max():.3f}"
                            self.log(f"[📊 STATS CARREGADAS] VecNorm obs_mean range: {mean_range}")
                        
                        obs_count = getattr(self.vec_env.obs_rms, 'count', 0)
                        self.log(f"[📊 EXPERIÊNCIA] VecNormalize com {obs_count} observações de treinamento")
                        self.log(f"[✅ PRESERVADO] Estatísticas de normalização do treinamento mantidas")
                    else:
                        self.log(f"[⚠️ AVISO] obs_rms ou ret_rms não encontrados no VecNormalize")
                    
                except Exception as e:
                    self.log(f"[⚠️ VECNORM] Erro ao carregar VecNormalize: {e}")
                    # Fallback: criar novo VecNormalize
                    self.vec_env = VecNormalize(temp_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0)
                    self.log(f"[📊 VECNORM] Criado novo VecNormalize com configurações padrão")
            else:
                self.vec_env = None
                if usar_vecnormalize:
                    self.log(f"[⚠️ VECNORM] Arquivo VecNormalize não encontrado - CRIANDO NOVO")
                    temp_env = DummyVecEnv([lambda: self.env])
                    self.vec_env = VecNormalize(temp_env, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=50.0)
                else:
                    self.log(f"[⚠️ VECNORM] MODO SEM NORMALIZAÇÃO para modelo genérico")
            
            # Log do modelo encontrado
            self.log(f"[🎯 MODELO] Legion V1 detectado: {model_path} - VecNormalize ATIVADO")
            
            # Carregar modelo PPO com diferentes estratégias de fallback
            model_loaded = False
            model_type = "PPO Básico"
            
            # 🔥 ESTRATÉGIA 1: Carregar com custom_objects completos
            if TWOPOLICY_AVAILABLE and TRANSFORMER_AVAILABLE:
                try:
                    self.model = RecurrentPPO.load(model_path, custom_objects={
                        'TwoHeadPolicy': TwoHeadPolicy,
                        'TwoHeadV3HybridEnhanced': TwoHeadPolicy,
                        'TransformerFeatureExtractor': TransformerFeatureExtractor,
                        'TradingTransformerFeatureExtractor': TransformerFeatureExtractor
                    }, device='cpu')
                    model_type = "PPO + Custom Classes (Full)"
                    model_loaded = True
                    self.log(f"[✅ ESTRATÉGIA 1] Modelo carregado com custom classes completas")
                except Exception as e:
                    self.log(f"[⚠️ ESTRATÉGIA 1] Falhou: {e}")
            
            # 🔥 ESTRATÉGIA 2: Carregar apenas com Policy
            if not model_loaded and TWOPOLICY_AVAILABLE:
                try:
                    self.model = RecurrentPPO.load(model_path, custom_objects={
                        'TwoHeadPolicy': TwoHeadPolicy,
                        'TwoHeadV3HybridEnhanced': TwoHeadPolicy
                    }, device='cpu')
                    model_type = "PPO + TwoHead Policy"
                    model_loaded = True
                    self.log(f"[✅ ESTRATÉGIA 2] Modelo carregado apenas com TwoHead Policy")
                except Exception as e:
                    self.log(f"[⚠️ ESTRATÉGIA 2] Falhou: {e}")
            
            # 🔥 ESTRATÉGIA 3: Carregar básico sem custom_objects
            if not model_loaded:
                try:
                    self.model = RecurrentPPO.load(model_path, device='cpu')
                    model_type = "PPO Básico (Fallback)"
                    model_loaded = True
                    self.log(f"[✅ ESTRATÉGIA 3] Modelo carregado em modo básico (sem custom classes)")
                except Exception as e:
                    self.log(f"[❌ ESTRATÉGIA 3] Falhou: {e}")
                    
            if not model_loaded:
                raise Exception("Todas as estratégias de carregamento falharam")
            
            # 🔧 GARANTIR QUE TODA A POLICY ESTÁ EM CPU
            if hasattr(self.model, 'policy'):
                self.model.policy.to('cpu')
                if hasattr(self.model.policy, 'features_extractor'):
                    self.model.policy.features_extractor.to('cpu')
                self.log("[🔧 DEVICE] Modelo forçado para CPU - device mismatch resolvido")
            
            # Status final
            if usar_vecnormalize:
                status_text = f"✅ {model_type} + VecNorm"
                status_color = '#4ecdc4'
                self.log(f"[🤖 MODO] Legion V1 com normalização ativada - MODELO DESBLOQUEADO")
            else:
                status_text = f"✅ {model_type} (SEM NORM)"
                status_color = '#ffeb3b'
                self.log(f"[🤖 MODO] Trading SEM normalização (dados RAW)")

            self.status_model.config(text=status_text, fg=status_color)
            return True
            
        except Exception as e:
            self.log(f"[❌ MODELO] Erro no carregamento: {e}")
            self.status_model.config(text="❌ Erro", fg='#ff6b6b')
            return False
    
    # 🔥 MÉTODO REMOVIDO - USANDO CARREGAMENTO AUTOMÁTICO
    
    def log(self, message):
        """Log apenas no terminal - GUI removida"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
    
    def start_trading(self):
        """Iniciar trading"""
        if not self.model:
            self.log("[ERROR] ❌ Modelo 'Legion V1' não foi carregado automaticamente!")
            self.log("[INFO] 🔄 Tentando carregar novamente...")
            if not self.auto_load_model():
                self.log("[ERROR] ❌ Falha no carregamento automático! Verifique se 'Modelo PPO Trader/Modelo PPO/Legion V1.zip' existe!")
                return
        
        # Inicializar estatísticas da sessão
        self.session_stats = SessionStats()
        
        # Obter balance inicial
        account_info = mt5.account_info()
        if account_info:
            self.session_stats.update_balance(account_info.balance)
        
        self.trading = True
        self.stop_event.clear()
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_trading.config(text="📊 Trading: Ativo", fg='#00ff88')
        
        # Iniciar atualização da GUI
        self.update_gui_stats()
        
        self.trading_thread = Thread(target=self.run_trading, daemon=True)
        self.trading_thread.start()
        
        self.log("[🚀 ⚔️ LEGION] Trading iniciado com MODELO PURO - Proteções desabilitadas!")
        self.log("[🔧 CORREÇÕES] VecNormalize desabilitado + Sistema anti-travamento ativo")
        self.log("[🔍 DIAGNÓSTICO] Verificação de dados a cada 100 steps")
        self.log("[🚨 FORÇAÇÃO] Ações forçadas após 20 HOLDs consecutivos")
        self.log("[🔧 CORREÇÕES] VecNormalize desabilitado + Sistema anti-travamento ativo")
        self.log("[🔍 DIAGNÓSTICO] Verificação de dados a cada 100 steps")
        self.log("[🚨 FORÇAÇÃO] Ações forçadas após 20 HOLDs consecutivos")
    
    def stop_trading(self):
        """Para o trading"""
        self.stop_event.set()
        self.trading_active = False
        
        # Aguardar thread terminar
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=5)
        
        # Atualizar interface
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_trading.config(text="📊 Trading: Parado", fg='#ffffff')
        
        self.log("[⏹ PARADO] Trading automatizado interrompido")
        
        # 🎨 Visualização desabilitada - nada para fechar
        if self.visualization_system:
            pass  # Visualização removida
    
    def toggle_visualization(self):
        """Ativa/desativa a visualização avançada"""
        try:
            if not self.enable_visualization:
                # Visualização desabilitada - MT5 Python API não suporta chart objects
                self.visualization_system = None
                self.enable_visualization = False  # Forçar desabilitação
                self.viz_status.config(text="🎨 Visualização: DISABLED", fg='#ffaa00')
                self.log("🎨 [SYSTEM] Visualização desabilitada - MT5 Python API não suporta chart objects")
                
            else:
                # Desativar visualização
                # Visualização já desabilitada - nada para fechar
                self.visualization_system = None
                self.enable_visualization = False
                self.viz_status.config(text="🎨 Visualização: OFF", fg='#ff4444')
                self.log("🎨 [VISUALIZATION] Análise Profunda DESATIVADA!")
                
        except Exception as e:
            self.log(f"❌ [VISUALIZATION] Erro ao alternar visualização: {e}")
            self.enable_visualization = False
            self.viz_status.config(text="🎨 Visualização: ERROR", fg='#ff4444')
    
    def run_trading(self):
        """🔥 LOOP DE TRADING COM PING A CADA 5 MINUTOS"""
        try:
            if not self.model:
                self.log("[❌ ERRO] Modelo não carregado!")
                return
                
            self.log("[🚀 TRADING] Iniciando modo automatizado...")
            step_count = 0
            self.last_ping_time = time.time()
            
            while not self.stop_event.is_set():
                try:
                    # Sistema de ping a cada 2 minutos
                    current_time = time.time()
                    if current_time - self.last_ping_time >= 120:  # 2 minutos = 120 segundos
                        account_info = mt5.account_info()
                        tick = mt5.symbol_info_tick(self.env.symbol)
                        positions = mt5.positions_get(symbol=self.env.symbol) or []
                        
                        # Verificar se dados são reais (RSI variando vs fixo em 50)
                        if len(self.env.historical_df) > 5:
                            recent_rsi = self.env.historical_df['rsi_14_5m'].tail(5).values
                            data_real = not np.allclose(recent_rsi, 50.0, atol=0.1)
                            data_status = "📈 DADOS REAIS" if data_real else "⚠️ DADOS SIMULADOS"
                        else:
                            data_status = "🔄 INICIALIZANDO"
                        
                        self.log(f"[💓 PING] Sistema ativo - Step {step_count}")
                        self.log(f"[💰 CONTA] ${account_info.balance:.2f} | Preço {self.env.symbol}: {tick.bid:.2f}")
                        self.log(f"[📊 STATUS] {len(positions)} posições | {data_status}")
                        self.last_ping_time = current_time
                    
                    # 🔍 MONITORAMENTO DE DADOS (apenas alertas críticos)
                    if step_count % 500 == 0:  # Reduzido para cada 500 steps
                        if hasattr(self.env, 'historical_df') and len(self.env.historical_df) > 10:
                            recent_data = self.env.historical_df.tail(10)
                            
                            # Verificar apenas RSI (mais confiável que preço)
                            if 'rsi_14_5m' in recent_data.columns:
                                rsi_variance = recent_data['rsi_14_5m'].var()
                                rsi_range = recent_data['rsi_14_5m'].max() - recent_data['rsi_14_5m'].min()
                                
                                # Alertar apenas se RSI realmente congelado (threshold muito baixo)
                                if rsi_variance < 0.0001 and rsi_range < 0.1:  # Valores muito mais restritivos
                                    self.log(f"🚨 DADOS CONGELADOS - RSI travado: {recent_data['rsi_14_5m'].iloc[-1]:.1f}")
                                    self.log(f"   Range RSI: {rsi_range:.3f} | Variância: {rsi_variance:.8f}")
                                elif step_count % 2000 == 0:  # Status normal a cada 2000 steps
                                    self.log(f"📊 [DADOS OK] RSI: {recent_data['rsi_14_5m'].iloc[-1]:.1f} | Var: {rsi_variance:.6f}")
                    
                    # 🔥 CORREÇÃO CRÍTICA: VecNormalize HABILITADO para melhor performance
                    USE_VECNORM = True  # 🔥 MUDANÇA: True para melhor normalização
                    
                    if USE_VECNORM and hasattr(self, 'vec_env') and self.vec_env is not None:
                        # Normalizar apenas observações, NÃO ações
                        raw_obs = self.env._get_observation()
                        raw_obs = raw_obs.reshape(1, -1)
                        normalized_obs = self.vec_env.normalize_obs(raw_obs)
                        obs = normalized_obs.flatten()
                        
                        # 🔥 VECNORMALIZE ADAPTATIVO: Dados reais são mais relevantes que históricos
                        # Atualização gradual e controlada para adaptar aos dados reais
                        
                        # 🔄 ATUALIZAÇÃO INTELIGENTE BASEADA EM MUDANÇAS
                        if step_count % 25 == 0 and step_count > 100:
                            # Detectar se dados mudaram significativamente
                            obs_mean = np.mean(np.abs(raw_obs))
                            obs_std = np.std(raw_obs)
                            
                            # Comparar com estatísticas atuais do VecNormalize
                            if hasattr(self.vec_env, 'obs_rms'):
                                current_mean = np.mean(self.vec_env.obs_rms.mean)
                                current_var = np.mean(self.vec_env.obs_rms.var)
                                
                                # Calcular diferença percentual
                                mean_diff = abs(obs_mean - current_mean) / (current_mean + 1e-8)
                                var_diff = abs(obs_std**2 - current_var) / (current_var + 1e-8)
                                
                                # Se mudança significativa (>50%), fazer update mais agressivo
                                if mean_diff > 0.5 or var_diff > 0.5:
                                    update_count = 3  # Update mais agressivo
                                    if step_count % 1000 == 0:  # Log apenas a cada 1000 steps
                                        self.log(f"🔄 [VECNORM ADAPT] Adaptação significativa - Mean: {mean_diff:.1%}, Var: {var_diff:.1%}")
                                else:
                                    update_count = 1  # Update suave
                            else:
                                update_count = 1
                            
                            # Fazer updates adaptativos
                            original_training = getattr(self.vec_env, 'training', False)
                            self.vec_env.training = True
                            
                            for _ in range(update_count):
                                _ = self.vec_env.normalize_obs(raw_obs)
                            
                            self.vec_env.training = original_training
                        
                        if step_count == 1:  # Apenas no primeiro step
                            self.log(f"✅ [VECNORM] Sistema adaptativo ativo")
                    else:
                        obs = self.env._get_observation()
                        if step_count == 1:
                            self.log(f"🔧 [INFO] VecNormalize desabilitado")
                    
                    # Fazer predição com o modelo
                    action, _states = self.model.predict(obs, deterministic=True)
                    
                    # 🎨 ANÁLISE PROFUNDA DO MODELO - SALVAR DADOS PARA EA
                    if self.enable_visualization:
                        try:
                            # Obter preço atual e portfolio
                            tick_temp = mt5.symbol_info_tick(self.env.symbol)
                            current_price = tick_temp.bid if tick_temp else 2000.0
                            account_info = mt5.account_info()
                            portfolio_value = account_info.balance if account_info else 500.0
                            
                            # 🧠 ANÁLISE PROFUNDA DO MODELO
                            model_analysis = self.analyze_model_decision_deep(obs, action, current_price, portfolio_value)
                            
                            # 💾 SALVAR DADOS PARA EA VISUALIZAR
                            self.save_model_data_for_ea(model_analysis)
                            
                        except Exception as e:
                            self.log(f"⚠️ [ANALYSIS] Erro na análise profunda: {e}")
                    
                    # 🔍 DIAGNÓSTICO INICIAL (apenas primeiros 2 steps)
                    if step_count <= 2:
                        # 🔍 DIAGNÓSTICO INICIAL CONCISO
                        policy_type = type(self.model.policy).__name__
                        self.log(f"🔍 [INIT] Step {step_count} | Policy: {policy_type} | Action: {action[:4]}")
                    
                    # Garantir que action é um array numpy
                    if not isinstance(action, np.ndarray):
                        action = np.array(action)
                    
                    # Verificar dimensão da ação
                    if len(action) < 10:
                        time.sleep(2)
                        continue
                    
                    # 🔥 PROCESSAR AÇÕES IDÊNTICAS AO TREINAMENTO
                    if len(action) >= 4:
                        entry_decision = int(action[0])
                        entry_confidence = action[1] if len(action) > 1 else 0.5
                        position_size = action[2] if len(action) > 2 else 0.5
                        mgmt_action = int(action[3]) if len(action) > 3 else 0
                        
                        # 🔥 COMPATIBILIDADE: Definir variáveis antigas para código legado
                        estrategica = entry_decision  # 0=HOLD, 1=LONG, 2=SHORT
                        taticas = action[4:10] if len(action) >= 10 else [0, 0, 0, 0, 0, 0]  # SL/TP ajustes
                        sltp_values = taticas  # Alias para compatibilidade
                        action_names = {0: "HOLD", 1: "LONG", 2: "SHORT"}  # Nomes das ações
                        
                        # 🔥 CONVERSÃO SL/TP IDÊNTICA AO TREINAMENTO
                        if len(action) >= 6:
                            sl_adjust = action[4]
                            tp_adjust = action[5]
                            
                            # Converter ajustes para pontos usando ranges do reward_system_diff
                            sl_points = abs(sl_adjust) * 100  # Max 300 pontos (liberdade total) (alinhado com diff)
                            tp_points = abs(tp_adjust) * 150  # Max 450 pontos (liberdade total) (alinhado com diff)
                            
                            # 🔥 CORREÇÃO CRÍTICA: Converter pontos para diferença de preço (multiplicar por 0.01)
                            # Para OURO: 1 ponto = 0.01 diferença de preço (ex: 2000.50 -> 2000.51 = 1 ponto)
                            sl_price_diff = sl_points * 0.01  # Converter pontos para preços (28 pontos = 0.28 diferença)
                            tp_price_diff = tp_points * 0.01  # Converter pontos para preços (41 pontos = 0.41 diferença)
                        else:
                            sl_price_diff = 0.2  # SL padrão 20 pontos
                            tp_price_diff = 0.4  # TP padrão 40 pontos
                        
                        # Contar HOLDs consecutivos
                        if entry_decision == 0:  # HOLD
                            if not hasattr(self, '_consecutive_holds'):
                                self._consecutive_holds = 0
                            self._consecutive_holds += 1
                        else:
                            self._consecutive_holds = 0
                    

                    
                    # Log a cada 10 steps para mostrar que está funcionando
                    if step_count % 10 == 0:
                        # Verificar dados reais vs simulados
                        if len(self.env.historical_df) > 0:
                            latest_data = self.env.historical_df.iloc[-1]
                            rsi_5m = latest_data.get('rsi_14_5m', 50.0)
                            # Usar preço atual do tick em vez de close_5m
                            tick_temp = mt5.symbol_info_tick(self.env.symbol)
                            price_5m = tick_temp.bid if tick_temp else 2000.0
                            
                            data_status = "📈 REAL" if abs(rsi_5m - 50.0) > 1.0 else "⚠️ SIM"
                            self.log(f"[🤖 MODELO] Step {step_count} | RSI: {rsi_5m:.1f} | Preço: {price_5m:.2f} | {data_status}")
                    
                    # 🔍 MONITORAMENTO VECNORMALIZE (apenas se problemas)
                    if step_count % 1000 == 0:  # Reduzido para cada 1000 steps
                        if hasattr(self, 'vec_env') and self.vec_env:
                            obs_norm = self.vec_env.normalize_obs(obs.reshape(1, -1)).flatten()
                            
                            # Verificar apenas anomalias críticas
                            huge_count = np.sum(np.abs(obs_norm) > 10.0)  # Valores extremos
                            if huge_count > len(obs_norm) * 0.1:  # >10% valores extremos
                                self.log(f"⚠️ [VECNORM] Step {step_count} | {huge_count} valores extremos detectados")
                            
                            # Sugerir ajustes se necessário
                            anomaly_ratio = huge_count / len(obs_norm)
                            if anomaly_ratio > 0.3:
                                self.log(f"💡 [SUGESTÃO] {anomaly_ratio:.1%} anomalias - VecNormalize se adaptando aos dados reais")
                                self.log(f"🔄 Sistema adaptativo ativo - melhorando gradualmente")
                    
                    # 🛡️ PROTEÇÃO AUTOMÁTICA: Verificar e proteger posições manuais
                    self.env._auto_protect_manual_positions(self.model, self.vec_env)
                    
                    # Obter posições atuais
                    mt5_positions = mt5.positions_get(symbol=self.env.symbol) or []
                    current_positions = len(mt5_positions)
                    
                    # Obter preço atual
                    tick = mt5.symbol_info_tick(self.env.symbol)
                    if not tick:
                        time.sleep(2)
                        continue
                    
                    # Log TODAS as ações do modelo (incluindo HOLD)
                        # Processar entrada de posição IDÊNTICO AO TREINAMENTO
                        if entry_decision > 0 and current_positions < self.env.max_positions:
                            # Calcular tamanho da posição
                            lot_size = self.env._calculate_adaptive_position_size(entry_confidence)
                            
                            # Criar posição
                            position_type = 'long' if entry_decision == 1 else 'short'
                            
                            # Calcular SL/TP
                            if position_type == 'long':
                                sl_price = tick.bid - sl_price_diff
                                tp_price = tick.bid + tp_price_diff
                            else:
                                sl_price = tick.bid + sl_price_diff
                                tp_price = tick.bid - tp_price_diff
                            
                            # Executar ordem
                            order_type = mt5.ORDER_TYPE_BUY if position_type == 'long' else mt5.ORDER_TYPE_SELL
                            response = self.env._execute_order(order_type, lot_size, sl_price, tp_price)
                            
                            action_names = {0: "HOLD", 1: "LONG", 2: "SHORT"}
                            self.log(f"[🧠 DECISÃO] {action_names[entry_decision]} | Conf: {entry_confidence:.2f} | Lot: {lot_size:.2f} | SL: {sl_price:.2f} | TP: {tp_price:.2f}")
                            self.log(f"[📋 RESULTADO] {response}")
                        
                        # Processar ações de gestão IDÊNTICO AO TREINAMENTO
                        elif mgmt_action > 0 and current_positions > 0:
                            if mgmt_action == 1:  # Fechar posição lucrativa
                                for pos in mt5_positions:
                                    pnl = self.env._get_position_pnl(pos, tick.bid)
                                    if pnl > 0:
                                        close_request = {
                                            "action": mt5.TRADE_ACTION_DEAL,
                                            "symbol": self.env.symbol,
                                            "volume": pos.volume,
                                            "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                                            "position": pos.ticket,
                                            "type_filling": self.env.filling_mode,
                                        }
                                        result = mt5.order_send(close_request)
                                        self.log(f"[💰 GESTÃO] Fechando posição lucrativa: PnL +${pnl:.2f}")
                                        break
                            elif mgmt_action == 2:  # Fechar todas as posições
                                for pos in mt5_positions:
                                    close_request = {
                                        "action": mt5.TRADE_ACTION_DEAL,
                                        "symbol": self.env.symbol,
                                        "volume": pos.volume,
                                        "type": mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY,
                                        "position": pos.ticket,
                                        "type_filling": self.env.filling_mode,
                                    }
                                    result = mt5.order_send(close_request)
                                self.log(f"[🚨 GESTÃO] Fechando todas as posições")
                        else:
                            self.log(f"[🧠 DECISÃO] HOLD | Pos: {current_positions}/{self.env.max_positions}")
                    
                    
                    
                    # 🔍 DIAGNÓSTICO REDUZIDO: Apenas quando necessário
                    if step_count % 200 == 1:  # Diagnóstico a cada 200 steps para reduzir overhead
                        self.log(f"🔍 [DIAGNÓSTICO COMPLETO] Step {step_count}")
                        
                        # 1. VERIFICAR OBSERVAÇÃO RAW vs NORMALIZADA
                        self.log(f"📊 Obs RAW[0-9]: {obs[:10]}")
                        if hasattr(self, 'vec_env') and self.vec_env is not None:
                            obs_norm = self.vec_env.normalize_obs(obs.reshape(1, -1)).flatten()
                            self.log(f"📊 Obs NORM[0-9]: {obs_norm[:10]}")
                            
                            # Verificar estatísticas de normalização
                            if hasattr(self.vec_env, 'obs_rms'):
                                obs_mean = self.vec_env.obs_rms.mean
                                obs_var = self.vec_env.obs_rms.var
                                self.log(f"📊 VecNorm Mean[0-4]: {obs_mean[:5]}")
                                self.log(f"📊 VecNorm Var[0-4]: {obs_var[:5]}")
                        
                        # 2. VERIFICAR DADOS FONTE (HISTORICAL_DF)
                        if hasattr(self.env, 'historical_df') and len(self.env.historical_df) > 0:
                            latest = self.env.historical_df.iloc[-1]
                            self.log(f"📊 DF RSI: {latest.get('rsi_14_5m', 'N/A')}")
                            self.log(f"📊 DF Returns: {latest.get('returns_5m', 'N/A')}")
                            self.log(f"📊 DF SMA20: {latest.get('sma_20_5m', 'N/A')}")
                            self.log(f"📊 DF Volatility: {latest.get('volatility_20_5m', 'N/A')}")
                        
                        # 3. VERIFICAR MAPEAMENTO OBSERVAÇÃO → FEATURES
                        if hasattr(self.env, 'feature_columns'):
                            self.log(f"📊 Feature Map: {self.env.feature_columns[:5]} ← Primeiras 5")
                        
                        # 4. VERIFICAR AÇÃO COMPLETA DO MODELO
                        self.log(f"🤖 AÇÃO COMPLETA ({len(action)}): {action}")
                        
                        # 5. DETECTAR OVER/UNDER-NORMALIZAÇÃO (MELHORADO)
                        obs_huge = np.sum(np.abs(obs) > 10)
                        obs_tiny = np.sum(np.abs(obs) < 0.001)
                        obs_zero = np.sum(np.abs(obs) < 1e-6)
                        obs_normal = np.sum((np.abs(obs) >= 0.001) & (np.abs(obs) <= 10))
                        
                        # Calcular estatísticas das observações
                        obs_mean = np.mean(np.abs(obs))
                        obs_std = np.std(obs)
                        obs_min = np.min(obs)
                        obs_max = np.max(obs)
                        
                        self.log(f"🚨 Obs Anômalas: {obs_huge} muito grandes, {obs_tiny} muito pequenas, {obs_zero} quase zero")
                        self.log(f"📊 Obs Stats: Normal={obs_normal}, Mean={obs_mean:.4f}, Std={obs_std:.4f}")
                        self.log(f"📊 Obs Range: [{obs_min:.4f}, {obs_max:.4f}]")
                        
                        # 🚨 ALERTA se muitas observações anômalas
                        total_obs = len(obs)
                        anomaly_ratio = (obs_huge + obs_tiny) / total_obs
                        if anomaly_ratio > 0.1:  # Mais de 10% anômalas (REDUZIDO para ser mais sensível)
                            self.log(f"⚠️ ALERTA: {anomaly_ratio:.1%} das observações são anômalas!")
                            self.log(f"💡 SUGESTÃO: VecNormalize pode precisar de re-calibração")
                            
                            # 🔒 PRESERVAR ESTATÍSTICAS - Anomalias são normais com dados reais
                            pass  # Não fazer nada que possa distorcer o modelo
                        # Observações normais - modelo funcionando bem
                        
                        # 6. VERIFICAR MODELO TRAVADO
                        if hasattr(self, '_last_full_action'):
                            action_diff = np.abs(action - self._last_full_action).sum()
                            self.log(f"🔄 Diferença ação anterior: {action_diff:.6f}")
                            if action_diff < 0.001:
                                self.log(f"⚠️ MODELO TRAVADO: Ações quase idênticas!")
                        self._last_full_action = action.copy()
                        
                        self.log(f"🔍 [FIM DIAGNÓSTICO] ==================")


                    
                    # Log da decisão do modelo
                    if estrategica > 0 and current_positions >= self.env.max_positions:
                        self.log(f"[⚠️ LIMITE] Modelo quer {action_names[estrategica]} mas já tem {current_positions} posições (max: {self.env.max_positions})")
                    elif estrategica > 0 and current_positions < self.env.max_positions:
                        self.log(f"[✅ EXECUTAR] Modelo quer {action_names[estrategica]} - há espaço ({current_positions}/{self.env.max_positions})")
                    
                    # 🔥 THRESHOLDS REMOVIDOS: Modelo decide tudo, sem filtros
                    sl_threshold = 0.0  # SEM threshold - modelo decide
                    tp_threshold = 0.0  # SEM threshold - modelo decide
                    
                    # 🔥 SISTEMA ANTI-FLIP-FLOP INTELIGENTE V2
                    current_time = time.time()
                    action_signature = f"{estrategica}_{'-'.join(map(str, taticas[:3]))}"
                    
                    # Adicionar contexto de mercado ao sistema anti-flip-flop
                    if hasattr(self.env, 'historical_df') and len(self.env.historical_df) > 0:
                        latest_data = self.env.historical_df.iloc[-1]
                        rsi_5m = latest_data.get('rsi_14_5m', 50.0)
                        volatility_5m = latest_data.get('volatility_20_5m', 0.5)
                        
                        # Determinar volatilidade
                        if volatility_5m > 1.5:
                            volatility = "HIGH"
                        elif volatility_5m < 0.3:
                            volatility = "LOW"
                        else:
                            volatility = "NORMAL"
                        
                        # Determinar tendência baseada em RSI
                        if rsi_5m > 60:
                            trend = "BULLISH"
                        elif rsi_5m < 40:
                            trend = "BEARISH"
                        else:
                            trend = "NEUTRAL"
                        
                        # Adicionar contexto ao sistema anti-flip-flop
                        self.anti_flipflop.add_market_context(tick.bid, volatility, trend)
                    
                    # Registrar decisão do modelo nas estatísticas
                    confidence = np.max(np.abs(action[:4])) * 100  # Confiança baseada na magnitude das ações
                    self.session_stats.add_model_decision(confidence)  # Corrigir: apenas confidence
                    
                    # 🔥 SISTEMA ANTI-FLIP-FLOP COMPLETAMENTE DESABILITADO
                    # O modelo PPO já foi treinado com controle de qualidade
                    # Não deve haver filtros adicionais no robô
                    should_block = False  # NUNCA bloquear
                    
                    # 🔥 AÇÃO ESTRATÉGICA (ABRIR NOVAS POSIÇÕES)
                    if estrategica == 1 and current_positions < self.env.max_positions:  # LONG
                        # Calcular SL/TP baseado na ação do agente
                        current_price = tick.ask
                        sl_value = sltp_values[0] if len(sltp_values) > 0 else 0.3
                        tp_value = sltp_values[1] if len(sltp_values) > 1 else 0.5
                        
                        # Converter valores [-1,1] para preços reais
                        sl_price = None
                        tp_price = None
                        
                        # 🔥 SL SEMPRE APLICADO: Conversão CORRETA OURO (1 ponto = 1.0 preço)
                        sl_points = abs(sl_value) * 100  # Max 300 pontos (liberdade total) (igual treinamento)
                        sl_price_diff = sl_points * 1.0  # OURO: 1 ponto = 1.0 diferença de preço
                        sl_price = current_price - sl_price_diff
                            
                        # 🔥 TP SEMPRE APLICADO: Conversão CORRETA OURO (1 ponto = 1.0 preço)
                        tp_points = abs(tp_value) * 150  # Max 450 pontos (liberdade total) (igual treinamento)
                        tp_price_diff = tp_points * 1.0  # OURO: 1 ponto = 1.0 diferença de preço
                        tp_price = current_price + tp_price_diff
                        
                        sl_text = f"{sl_price:.2f}" if sl_price is not None else "N/A"
                        tp_text = f"{tp_price:.2f}" if tp_price is not None else "N/A"
                        self.log(f"[🚀 EXECUTANDO] LONG @ {current_price:.2f} | SL: {sl_text} | TP: {tp_text}")
                        # Calcular tamanho dinâmico da posição
                        dynamic_lot_size = self.env._calculate_adaptive_position_size(action_confidence=1.0)
                        response = self.env._execute_order(mt5.ORDER_TYPE_BUY, dynamic_lot_size, sl_price, tp_price)
                        self._process_trade_result(response)
                        
                        # Registrar ação no sistema anti-flip-flop
                        self.anti_flipflop.update_action_executed(action_signature, current_time)
                        
                        # Atualizar estatísticas
                        if "SUCCESS" in response:
                            self.session_stats.total_buys += 1
                            self.session_stats.positions_opened += 1
                            self.session_stats.update_last_action("LONG")  # 🔥 ATUALIZAR ÚLTIMA AÇÃO
                        
                        # Se mercado fechado, aguardar mais tempo
                        if "MARKET_CLOSED" in response:
                            self.log("[⏰ AGUARDANDO] Mercado fechado - aguardando 30 minutos...")
                            time.sleep(1800)  # 30 minutos
                        
                    elif estrategica == 2 and current_positions < self.env.max_positions:  # SHORT
                        # Calcular SL/TP baseado na ação do agente
                        current_price = tick.bid
                        sl_value = sltp_values[0] if len(sltp_values) > 0 else 0.3
                        tp_value = sltp_values[1] if len(sltp_values) > 1 else 0.5
                        
                        # Converter valores [-1,1] para preços reais
                        sl_price = None
                        tp_price = None
                        
                        # 🔥 SL SEMPRE APLICADO: Conversão CORRETA OURO (1 ponto = 1.0 preço)
                        sl_points = abs(sl_value) * 100  # Max 300 pontos (liberdade total) (igual treinamento)
                        sl_price_diff = sl_points * 1.0  # OURO: 1 ponto = 1.0 diferença de preço
                        sl_price = current_price + sl_price_diff  # SHORT: SL acima do preço
                            
                        # 🔥 TP SEMPRE APLICADO: Conversão CORRETA OURO (1 ponto = 1.0 preço)
                        tp_points = abs(tp_value) * 150  # Max 450 pontos (liberdade total) (igual treinamento)
                        tp_price_diff = tp_points * 1.0  # OURO: 1 ponto = 1.0 diferença de preço
                        tp_price = current_price - tp_price_diff  # SHORT: TP abaixo do preço
                        
                        sl_text = f"{sl_price:.2f}" if sl_price is not None else "N/A"
                        tp_text = f"{tp_price:.2f}" if tp_price is not None else "N/A"
                        self.log(f"[🚀 EXECUTANDO] SHORT @ {current_price:.2f} | SL: {sl_text} | TP: {tp_text}")
                        # Calcular tamanho dinâmico da posição
                        dynamic_lot_size = self.env._calculate_adaptive_position_size(action_confidence=1.0)
                        response = self.env._execute_order(mt5.ORDER_TYPE_SELL, dynamic_lot_size, sl_price, tp_price)
                        self._process_trade_result(response)
                        
                        # Registrar ação no sistema anti-flip-flop
                        self.anti_flipflop.update_action_executed(action_signature, current_time)
                        
                        # Atualizar estatísticas
                        if "SUCCESS" in response:
                            self.session_stats.total_sells += 1
                            self.session_stats.positions_opened += 1
                            self.session_stats.update_last_action("SHORT")  # 🔥 ATUALIZAR ÚLTIMA AÇÃO
                        
                        # Se mercado fechado, aguardar mais tempo
                        if "MARKET_CLOSED" in response:
                            self.log("[⏰ AGUARDANDO] Mercado fechado - aguardando 30 minutos...")
                            time.sleep(1800)  # 30 minutos
                    
                    # Sistema antigo removido - usando AntiFlipFlopSystem avançado
                    
                    # 🔥 AÇÕES TÁTICAS (GERENCIAR POSIÇÕES EXISTENTES)
                    for i, tatica in enumerate(taticas[:current_positions]):
                        if i >= len(mt5_positions):
                            break
                            
                        position = mt5_positions[i]
                        
                        # 🔥 SISTEMA ANTI-MICRO TRADES: Verificar histórico da posição
                        position_key = f"{position.ticket}"
                        if position_key not in self.position_history:
                            self.position_history[position_key] = {
                                'open_time': current_time,
                                'close_attempts': 0,
                                'last_close_attempt': 0
                            }
                        
                        if tatica == 1:  # FECHAR POSIÇÃO
                            pos_history = self.position_history[position_key]
                            pos_history['close_attempts'] += 1
                            
                            # 🔥 MICRO TRADE CHECKS REMOVIDOS: Modelo decide quando fechar
                            position_age = current_time - pos_history['open_time']
                            pos_history['last_close_attempt'] = current_time
                            
                            self.log(f"[🎯 TÁTICA] Modelo quer FECHAR posição #{position.ticket} (tipo: {'LONG' if position.type == 0 else 'SHORT'}) - Idade: {position_age:.0f}s")
                            
                            # Fechar posição específica
                            close_request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": self.env.symbol,
                                "volume": position.volume,
                                "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                                "position": position.ticket,
                                "price": tick.bid if position.type == 0 else tick.ask,
                                "magic": 123456,
                                "comment": "Close",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": self.env.filling_mode
                            }
                            
                            result = mt5.order_send(close_request)
                            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                self.log(f"[✅ FECHOU] Posição #{position.ticket} fechada pelo agente")
                                
                                # Registrar ação no sistema anti-flip-flop
                                close_action_signature = f"close_{position.ticket}"
                                self.anti_flipflop.update_action_executed(close_action_signature, current_time)
                                
                                # Atualizar estatísticas
                                profit = position.profit
                                duration_seconds = current_time - pos_history['open_time']
                                trade_type = 'BUY' if position.type == 0 else 'SELL'
                                self.session_stats.add_trade(trade_type, profit, duration_seconds)
                                self.session_stats.positions_closed += 1
                                self.session_stats.update_last_action("CLOSE")  # 🔥 ATUALIZAR ÚLTIMA AÇÃO
                                
                                # Remover do histórico
                                if position_key in self.position_history:
                                    del self.position_history[position_key]
                            else:
                                error_code = result.retcode if result else "None"
                                self.log(f"[❌ ERRO] Falha ao fechar posição: {error_code}")
                        
                        elif tatica == 2:  # AJUSTAR SL/TP
                            # Ajustar SL/TP baseado nos valores do agente
                            sl_idx = 2 + i * 2  # Índices SL/TP para cada posição
                            tp_idx = 3 + i * 2
                            
                            if sl_idx < len(sltp_values) and tp_idx < len(sltp_values):
                                current_price = tick.bid if position.type == 0 else tick.ask
                                sl_value = sltp_values[sl_idx]
                                tp_value = sltp_values[tp_idx]
                                
                                self.log(f"[🎯 TÁTICA] Modelo quer AJUSTAR #{position.ticket}: SL={sl_value:.3f}, TP={tp_value:.3f}")
                                
                                new_sl = None
                                new_tp = None
                                
                                # 🔥 SL SEMPRE APLICADO: Conversão CORRETA OURO (1 ponto = 1.0 preço)
                                sl_points = abs(sl_value) * 100  # Max 300 pontos (liberdade total)
                                sl_price_diff = sl_points * 1.0  # OURO: 1 ponto = 1.0 diferença de preço
                                if position.type == 0:  # Long
                                    new_sl = current_price - sl_price_diff
                                else:  # Short
                                    new_sl = current_price + sl_price_diff

                                # 🔥 TP SEMPRE APLICADO: Conversão CORRETA OURO (1 ponto = 1.0 preço)
                                tp_points = abs(tp_value) * 150  # Max 450 pontos (liberdade total)
                                tp_price_diff = tp_points * 1.0  # OURO: 1 ponto = 1.0 diferença de preço
                                if position.type == 0:  # Long
                                    new_tp = current_price + tp_price_diff
                                else:  # Short
                                    new_tp = current_price - tp_price_diff
                                
                                # Modificar posição
                                # 🔥 SEMPRE APLICAR: Modelo decide todos os valores
                                self.log(f"[📝 MODIFY] Aplicando SL: {new_sl:.2f if new_sl else 'N/A'}, TP: {new_tp:.2f if new_tp else 'N/A'}")
                                
                                modify_request = {
                                    "action": mt5.TRADE_ACTION_SLTP,
                                    "symbol": self.env.symbol,
                                    "position": position.ticket,
                                    "sl": new_sl or position.sl,
                                    "tp": new_tp or position.tp
                                }
                                
                                result = mt5.order_send(modify_request)
                                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                                    self.log(f"[✅ AJUSTOU] SL/TP modificado - Posição #{position.ticket}")
                                    self.session_stats.update_last_action("ADJUST")  # 🔥 ATUALIZAR ÚLTIMA AÇÃO
                                else:
                                    error_code = result.retcode if result else "None"
                                    self.log(f"[❌ ERRO] Falha ao ajustar SL/TP: {error_code}")
                            else:
                                self.log(f"[⚠️ SKIP] Índices SL/TP fora do range para posição {i}")
                        
                        elif tatica == 0:  # MANTER
                            # Log ocasional para mostrar que modelo está monitorando
                            if step_count % 20 == 0:
                                self.log(f"[👀 MONITOR] Posição #{position.ticket} mantida pelo modelo")
                    
                    step_count += 1
                    time.sleep(1)  # 🔥 REDUZIDO: 1 segundo para maior responsividade
                    
                except Exception as e:
                    self.log(f"[❌ ERRO] Step de trading: {e}")
                    time.sleep(2)  # 🔥 REDUZIDO: 2 segundos para recovery mais rápido
                    
        except Exception as e:
            self.log(f"[❌ CRÍTICO] Erro no trading: {e}")
        finally:
            self.log("[🛑 STOP] Trading finalizado")
            self.trading = False
    
    def _process_trade_result(self, response: str):
        """Processa resultado da execução de ordem"""
        try:
            if "SUCCESS" in response:
                parts = response.split("|")
                if len(parts) >= 6:
                    ticket = parts[1]
                    price = parts[2]
                    action = parts[3]
                    sl_price = parts[4]
                    tp_price = parts[5]
                    self.log(f"[🎯 SUCESSO] {action} executado - Ticket: #{ticket}, Preço: ${price}")
                    if float(sl_price) > 0:
                        self.log(f"[🛡 PROTEÇÃO] SL definido: ${sl_price}")
                    if float(tp_price) > 0:
                        self.log(f"[🎯 ALVO] TP definido: ${tp_price}")
                    if float(sl_price) == 0 and float(tp_price) == 0:
                        self.log(f"[⚠️ RISCO] Posição sem SL/TP - risco elevado")
            elif "ERROR_MARKET_CLOSED" in response:
                self.log(f"[⏰ MERCADO] Mercado fechado - aguardando abertura")
            elif "ERROR_INVALID_ORDER" in response:
                self.log(f"[❌ ORDEM] Ordem inválida - verificar parâmetros")
            elif "ERROR_ORDER_CHECK" in response:
                self.log(f"[❌ CHECK] Ordem rejeitada pelo broker")
            elif "ERROR_NO_PRICE" in response:
                self.log(f"[❌ PREÇO] Falha ao obter cotação")
            elif "ERROR_COOLDOWN" in response:
                self.log(f"[⏰ COOLDOWN] Aguardando intervalo entre ordens")
            else:
                self.log(f"[❌ FALHA] {response}")
        except Exception as e:
            self.log(f"[❌ ERRO] Falha ao processar resultado: {e}")
    
    def _manage_existing_positions(self):
        """Gerencia posições existentes (com SL/TP do agente)"""
        try:
            import MetaTrader5 as mt5
            positions = mt5.positions_get(symbol="GOLD")
            if positions:
                for pos in positions:
                    # Log das posições ativas com SL/TP definidos pelo agente
                    profit = pos.profit
                    sl = pos.sl
                    tp = pos.tp
                    action_type = "LONG" if pos.type == 0 else "SHORT"
                    
                    if abs(profit) > 10:  # Só logar se profit significativo
                        sl_info = f", SL: {sl:.2f}" if sl > 0 else ", SL: None"
                        tp_info = f", TP: {tp:.2f}" if tp > 0 else ", TP: None"
                        self.log(f"[POSITION] {action_type} #{pos.ticket} - P&L: ${profit:.2f}{sl_info}{tp_info}")
                        
        except Exception as e:
            self.log(f"[ERROR] Erro ao gerenciar posições: {e}")
    
    def analyze_model_decision_deep(self, obs, action, current_price, portfolio_value):
        """🧠 ANÁLISE PROFUNDA DO MODELO - TODAS AS CONJECTURAS"""
        try:
            # 1. DECODIFICAR AÇÃO DO MODELO
            entry_decision = int(action[0]) if len(action) > 0 else 0
            entry_confidence = float(action[1]) if len(action) > 1 else 0.0
            position_size = float(action[2]) if len(action) > 2 else 0.0
            mgmt_action = int(action[3]) if len(action) > 3 else 0
            sl_adjust = float(action[4]) if len(action) > 4 else 0.0
            tp_adjust = float(action[5]) if len(action) > 5 else 0.0
            
            # 2. ANÁLISE DE OBSERVAÇÃO (FEATURES)
            obs_analysis = self.analyze_observation_features(obs)
            
            # 3. ANÁLISE DE CONTEXTO DE MERCADO
            market_context = self.analyze_market_context(obs, current_price)
            
            # 4. ANÁLISE DE CONFIANÇA E RISCO
            confidence_analysis = self.analyze_confidence_and_risk(action, obs)
            
            # 5. ANÁLISE DE REGIME DE MERCADO
            market_regime = self.analyze_market_regime(obs)
            
            # 6. ANÁLISE DE MOMENTUM E VOLATILIDADE
            momentum_analysis = self.analyze_momentum_volatility(obs)
            
            # 7. COMPILAR ANÁLISE COMPLETA
            analysis = {
                'timestamp': time.time(),
                'action': {
                    'entry_decision': entry_decision,
                    'entry_confidence': entry_confidence,
                    'position_size': position_size,
                    'mgmt_action': mgmt_action,
                    'sl_adjust': sl_adjust,
                    'tp_adjust': tp_adjust,
                    'action_name': {0: 'HOLD', 1: 'LONG', 2: 'SHORT'}.get(entry_decision, 'UNKNOWN')
                },
                'market': {
                    'price': current_price,
                    'portfolio': portfolio_value,
                    'context': market_context,
                    'regime': market_regime,
                    'momentum': momentum_analysis
                },
                'model': {
                    'confidence': confidence_analysis,
                    'observation': obs_analysis
                }
            }
            
            return analysis
            
        except Exception as e:
            self.log(f"❌ [ANALYSIS] Erro na análise profunda: {e}")
            return None
    
    def analyze_observation_features(self, obs):
        """📊 Analisar features da observação"""
        try:
            # Analisar primeiras 20 features mais importantes
            key_features = obs[:20] if len(obs) >= 20 else obs
            
            # Calcular estatísticas
            feature_stats = {
                'mean': float(np.mean(key_features)),
                'std': float(np.std(key_features)),
                'min': float(np.min(key_features)),
                'max': float(np.max(key_features)),
                'extreme_count': int(np.sum(np.abs(key_features) > 3.0)),
                'zero_count': int(np.sum(np.abs(key_features) < 0.001))
            }
            
            return feature_stats
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_market_context(self, obs, current_price):
        """🏛️ Analisar contexto de mercado"""
        try:
            # Extrair features de mercado se disponíveis
            context = {
                'price': current_price,
                'trend': 'NEUTRAL',
                'volatility': 'MEDIUM',
                'strength': 0.5
            }
            
            # Analisar tendência baseada nas features
            if len(obs) >= 5:
                # Assumir que as primeiras features são relacionadas a preço/retornos
                price_features = obs[:5]
                trend_signal = np.mean(price_features)
                
                if trend_signal > 0.5:
                    context['trend'] = 'BULLISH'
                    context['strength'] = min(1.0, trend_signal)
                elif trend_signal < -0.5:
                    context['trend'] = 'BEARISH'
                    context['strength'] = min(1.0, abs(trend_signal))
            
            # Analisar volatilidade
            if len(obs) >= 10:
                vol_proxy = np.std(obs[:10])
                if vol_proxy > 2.0:
                    context['volatility'] = 'HIGH'
                elif vol_proxy < 0.5:
                    context['volatility'] = 'LOW'
            
            return context
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_confidence_and_risk(self, action, obs):
        """🎯 Analisar confiança e risco"""
        try:
            entry_confidence = float(action[1]) if len(action) > 1 else 0.0
            
            # Calcular nível de risco baseado na ação e observação
            risk_level = 'LOW'
            risk_score = 0.0
            
            # Risco baseado na confiança
            if entry_confidence > 0.8:
                risk_level = 'HIGH'
                risk_score = 0.9
            elif entry_confidence > 0.5:
                risk_level = 'MEDIUM'
                risk_score = 0.6
            else:
                risk_level = 'LOW'
                risk_score = 0.3
            
            # Ajustar risco baseado na volatilidade da observação
            if len(obs) >= 10:
                obs_volatility = np.std(obs[:10])
                if obs_volatility > 2.0:
                    risk_score = min(1.0, risk_score + 0.2)
                    risk_level = 'HIGH' if risk_score > 0.7 else risk_level
            
            confidence_analysis = {
                'entry_confidence': entry_confidence,
                'risk_level': risk_level,
                'risk_score': risk_score,
                'confidence_category': 'HIGH' if entry_confidence > 0.7 else 'MEDIUM' if entry_confidence > 0.4 else 'LOW'
            }
            
            return confidence_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_market_regime(self, obs):
        """🌊 Analisar regime de mercado"""
        try:
            # Determinar regime baseado nas features
            regime = {
                'type': 'RANGING',
                'strength': 0.5,
                'stability': 'STABLE'
            }
            
            if len(obs) >= 15:
                # Analisar padrões nas features
                feature_range = np.max(obs[:15]) - np.min(obs[:15])
                feature_mean = np.mean(obs[:15])
                
                # Determinar tipo de regime
                if feature_range > 3.0:
                    regime['type'] = 'VOLATILE'
                    regime['strength'] = min(1.0, feature_range / 5.0)
                elif abs(feature_mean) > 1.0:
                    regime['type'] = 'TRENDING'
                    regime['strength'] = min(1.0, abs(feature_mean))
                
                # Determinar estabilidade
                feature_std = np.std(obs[:15])
                if feature_std > 2.0:
                    regime['stability'] = 'UNSTABLE'
                elif feature_std < 0.5:
                    regime['stability'] = 'VERY_STABLE'
            
            return regime
            
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_momentum_volatility(self, obs):
        """⚡ Analisar momentum e volatilidade"""
        try:
            momentum_analysis = {
                'momentum': 0.0,
                'momentum_strength': 'WEAK',
                'volatility': 0.0,
                'volatility_level': 'MEDIUM'
            }
            
            if len(obs) >= 20:
                # Calcular momentum (diferença entre médias de períodos diferentes)
                short_term = np.mean(obs[:5])
                long_term = np.mean(obs[5:15])
                momentum = short_term - long_term
                
                momentum_analysis['momentum'] = float(momentum)
                
                # Classificar força do momentum
                if abs(momentum) > 1.0:
                    momentum_analysis['momentum_strength'] = 'STRONG'
                elif abs(momentum) > 0.5:
                    momentum_analysis['momentum_strength'] = 'MEDIUM'
                
                # Calcular volatilidade
                volatility = np.std(obs[:20])
                momentum_analysis['volatility'] = float(volatility)
                
                # Classificar nível de volatilidade
                if volatility > 2.0:
                    momentum_analysis['volatility_level'] = 'HIGH'
                elif volatility < 0.8:
                    momentum_analysis['volatility_level'] = 'LOW'
            
            return momentum_analysis
            
        except Exception as e:
            return {'error': str(e)}
    
    def save_model_data_for_ea(self, analysis):
        """💾 Salvar dados para EA visualizar"""
        try:
            if not analysis:
                return
            
            # Preparar dados no formato que a EA espera
            action_name = analysis['action']['action_name']
            confidence = analysis['action']['entry_confidence'] * 100  # Converter para percentual
            price = analysis['market']['price']
            portfolio = analysis['market']['portfolio']
            
            # Dados estendidos para EA
            risk_level = analysis['model']['confidence']['risk_level']
            market_regime = analysis['market']['regime']['type']
            momentum = analysis['market']['momentum']['momentum']
            volatility = analysis['market']['momentum']['volatility']
            
            # Formato: ACTION|CONFIDENCE|PRICE|PORTFOLIO|RISK|REGIME|MOMENTUM|VOLATILITY
            data_line = f"{action_name}|{confidence:.1f}|{price:.2f}|{portfolio:.2f}|{risk_level}|{market_regime}|{momentum:.3f}|{volatility:.3f}"
            
            # Salvar no arquivo para EA
            try:
                with open(self.model_data_file, 'w', encoding='utf-8') as f:
                    f.write(data_line + '\n')
                    f.write(f"TIMESTAMP|{time.time()}\n")
                    f.write(f"DETAILS|{analysis['action']['entry_decision']}|{analysis['action']['position_size']:.3f}|{analysis['action']['mgmt_action']}\n")
            except Exception as file_error:
                self.log(f"⚠️ [FILE] Erro ao salvar dados: {file_error}")
            
            # Log da análise (apenas quando há decisão importante)
            if action_name != 'HOLD' or confidence > 70:
                self.log(f"🧠 [ANÁLISE PROFUNDA] {action_name} | Conf: {confidence:.1f}% | Risco: {risk_level} | Regime: {market_regime}")
                
        except Exception as e:
            self.log(f"❌ [SAVE] Erro ao salvar análise: {e}")

def main():
    """Função principal"""
    print("=" * 50)
    print("    ⚔️ LEGION AI TRADER V1")
    print("    Enhanced PPO Trading Robot")
    print("    🛡 Anti-Flip-Flop Protection")
    print("    📊 Real-time Statistics")
    print("=" * 50)
    
    root = tk.Tk()
    app = TradingApp(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[🛑] Interrompido pelo usuário")
    except Exception as e:
        print(f"[❌] Erro crítico: {e}")

if __name__ == "__main__":
    main() 
