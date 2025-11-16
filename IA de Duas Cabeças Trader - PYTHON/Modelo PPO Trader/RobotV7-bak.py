# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')

import tkinter as tk
from tkinter import scrolledtext, ttk
from threading import Thread, Event
"""
⚔️ Legion AI Trader V7 - Trading Robot PPO V7 Intuition
🧠 CONFIGURADO V7 INTUITION: Backbone Unificado + Gradient Mixing + Dynamic Trailing Stop
🎯 OBSERVATION SPACE: 2580 dimensões (129 features × 20 window)

🧠 V7 INTUITION ARCHITECTURE:
- OBSERVATION SPACE: 2580 dimensões (129 features × 20 window)  
- ACTION SPACE: 11 dimensões [entry_decision, confidence, temporal_signal, risk_appetite, market_regime_bias, sl1, sl2, sl3, tp1, tp2, tp3]
- FEATURES: Market + Technical + Positions + Intelligent = 129 features por step
- WINDOW: 20 steps × 129 features = 2580 dimensões
- UNIFIED BACKBONE: Shared processing between Actor/Critic
- GRADIENT MIXING: Cross-pollination between networks
- DYNAMIC TRAILING: Model-controlled trailing stop system

ACTION SPACE (11D) - Enhanced for Trailing Stop:
- [0] entry_decision: 0=HOLD, 1=LONG, 2=SHORT
- [1] entry_quality: [0,1] Qualidade da entrada (filtro + ajuste SL/TP)
- [2] temporal_signal: [-1,1] Sinal temporal
- [3] risk_appetite: [0,1] Apetite ao risco
- [4] market_regime_bias: [-1,1] Viés do mercado
- [5-7] sl_adjusts: SL/Trailing control pos1, pos2, pos3 ([-3,3])
- [8-10] tp_adjusts: Trailing intensity pos1, pos2, pos3 ([-3,3])

🎯 DYNAMIC TRAILING STOP:
- sl_adjust [-3,3]: Ativa/move trailing stop (abs>1.5 = ativa, abs>0.5 = move)
- tp_adjust [-3,3]: Intensidade do trailing (controla distância 15-30 pontos)
- Rewards: +1.0 execução, +0.8 ativação, +0.6 proteção, +0.4 timing

CONVERSÃO: [-3,3] → [2-8] SL, [3-15] TP pontos → SL/TP realistas (OURO: 1 ponto = $1.00)

COMPATIBILIDADE:
- 🚀 TwoHeadV7Intuition (Unified Backbone + Gradient Mixing)
- 📋 TradingTransformerFeatureExtractor
- 🔧 Enhanced Normalizer
- 🎯 Dynamic Trailing Stop System

🔧 V7 INTUITION UPDATES:
- _get_observation_v7(): Gera exatamente 2580 dimensões
- _process_v7_action(): Compatível com trailing stop dinâmico
- _verify_v7_compatibility(): Verificação específica para V7 Intuition
- auto_load_model(): Carrega modelos V7 Intuition com policy_kwargs
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
from stable_baselines3.common.vec_env import DummyVecEnv
import os
import sys

# Enhanced Normalizer - Sistema único de normalização
try:
    # Importar do arquivo local (Modelo PPO Trader)
    sys.path.insert(0, os.path.dirname(__file__))  # Adicionar pasta atual primeiro
    from enhanced_normalizer import EnhancedRunningNormalizer, create_enhanced_normalizer
except ImportError:
    # Fallback para o arquivo da raiz
    sys.path.append('..')
    from enhanced_normalizer import EnhancedVecNormalize as EnhancedRunningNormalizer, create_enhanced_normalizer

import MetaTrader5 as mt5
import sys
import warnings
import torch
from datetime import datetime, timedelta
from collections import deque, Counter
import statistics
import requests  # Para Flask server communication
import zipfile
import shutil
import tempfile
import pickle

# Configuracoes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

# 🎯 V7 MODEL PATHS - Configuração centralizada
class ModelPaths:
    """🎯 Configuração centralizada de paths para modelos V7"""
    # Caminho principal do modelo ZIP - Legion DayTrade (5.1M steps - melhor modelo)
    MODEL_ZIP_PATH = "Modelo daytrade/Legion daytrade.zip"
    
    # Diretório de extração temporária
    TEMP_EXTRACT_DIR = os.path.join(tempfile.gettempdir(), "legion_v7_models")
    
    # Arquivos esperados dentro do ZIP
    EXPECTED_FILES = {
        'model': 'pytorch_variables.pth',  # Modelo principal do SB3
        'policy': 'policy.pth',  # Policy do modelo
        'normalizer': 'enhanced_normalizer_final.pkl',  # Enhanced normalizer (fora do ZIP)
        'metadata': 'system_info.txt'  # Info do sistema
    }
    
    @classmethod
    def get_absolute_model_path(cls):
        """Retorna o caminho absoleto do modelo ZIP configurado"""
        # 🔥 USAR DIRETAMENTE O LEGION DAYTRADE.ZIP - NÃO BUSCAR AUTOMÁTICO
        if os.path.isabs(cls.MODEL_ZIP_PATH):
            return cls.MODEL_ZIP_PATH
        else:
            # Caminho relativo à pasta do RobotV7.py (Modelo PPO Trader)
            current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
            return os.path.join(current_dir, cls.MODEL_ZIP_PATH)
    
    @classmethod
    def find_latest_daytrader_model(cls):
        """Encontra o modelo DayTrader mais recente"""
        import glob
        
        # Buscar na pasta raiz do projeto
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # Patterns para buscar checkpoints DayTrader
        patterns = [
            os.path.join(project_root, "*.zip"),  # Raiz do projeto
            os.path.join(project_root, "Otimizacao/treino_principal/models/DAYTRADER/*.zip"),
            os.path.join(project_root, "checkpoints/*.zip"),
            os.path.join(project_root, "models/*.zip")
        ]
        
        all_models = []
        for pattern in patterns:
            models = glob.glob(pattern)
            for model in models:
                if "DAYTRADER" in os.path.basename(model).upper():
                    all_models.append(model)
        
        if not all_models:
            return None
            
        # Ordenar por data de modificação (mais recente primeiro)
        all_models.sort(key=os.path.getmtime, reverse=True)
        return all_models[0]
    
    @classmethod
    def ensure_extract_dir(cls):
        """Garante que o diretório de extração existe"""
        os.makedirs(cls.TEMP_EXTRACT_DIR, exist_ok=True)
        return cls.TEMP_EXTRACT_DIR

# Função auxiliar para MT5 - Correção dos erros de chart_object_delete
def safe_mt5_object_delete(obj_name):
    """Função segura para deletar objetos do MT5"""
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

# Importações para visualização
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# Configuração MT5
if not mt5.initialize():
    print(f"[ERROR] ⚠️ Falha ao inicializar MetaTrader5. Erro: {mt5.last_error()}")

# 🎯 ADICIONAR PATHS PARA IMPORTS
current_dir = os.path.dirname(os.path.abspath(__file__))  # Modelo PPO Trader
parent_dir = os.path.dirname(current_dir)  # Projeto
sys.path.insert(0, parent_dir)  # Adicionar Projeto no início

# 🧠 V7 IMPORTS - INTUITION VERSION
from trading_framework.policies.two_head_v7_intuition import TwoHeadV7Intuition, get_v7_intuition_kwargs
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
print("[INTUITION] ✅ TwoHeadV7Intuition importada - Backbone Unificado + Gradient Mixing")
print("[INTUITION] ✅ TradingTransformerFeatureExtractor importado OBRIGATÓRIO")

class BaseTradingEnv:
    """Classe base para environamento de trading com configurações básicas"""
    def __init__(self):
        self.symbol = "GOLD"
        self.initial_balance = 500.0
        self.max_positions = 3
        self.window_size = 20

class Config:
    """Configurações do sistema V7"""
    SYMBOL = "GOLD" 
    INITIAL_BALANCE = 500.0
    MAX_POSITIONS = 3
    WINDOW_SIZE = 20
    MAX_LOT_SIZE = 0.03
    BASE_LOT_SIZE = 0.02
    
    # V7 Specific
    OBSERVATION_SPACE_SIZE = 2580  # 129 features × 20 window
    FEATURES_PER_STEP = 129
    ACTION_SPACE_SIZE = 11
    
    # Features breakdown (based on obs_space.md documentation)
    MARKET_FEATURES = 16           # Market features optimized
    POSITION_FEATURES = 27         # 3 positions × 9 features each
    INTELLIGENT_V7 = 37           # V7 components optimized  
    MICROSTRUCTURE = 14           # Order flow + tick analytics
    VOLATILITY_ADVANCED = 5       # GARCH + clustering + breakout
    MARKET_CORRELATION = 4        # Inter-market correlations
    MOMENTUM_MULTI = 6            # Multi-timeframe momentum
    ENHANCED_FEATURES = 20        # Pattern recognition + regime detection

class TradingRobotV7(gym.Env):
    """🚀 Trading Robot V7 - Compatível com TwoHeadV7Simple"""
    
    def __init__(self, log_widget=None):
        super().__init__()
        self.log_widget = log_widget  # Opcional para compatibilidade
        self.symbol = Config.SYMBOL
        
        # 🔥 CONFIGURAÇÕES V7
        self.window_size = Config.WINDOW_SIZE
        self.initial_balance = Config.INITIAL_BALANCE
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        self.positions = []
        self.returns = []
        self.trades = []
        self.current_drawdown = 0.0
        self.peak_drawdown = 0.0
        self.max_lot_size = Config.MAX_LOT_SIZE
        self.max_positions = Config.MAX_POSITIONS
        self.current_positions = 0
        self.current_step = 0
        self.done = False
        self.last_order_time = 0
        self._log(f"[🔧 INIT] Sistema de cooldown inicializado - intervalo mínimo: 1s")
        
        # 🛡️ TRACKER DE POSIÇÕES: Para detectar novas posições manuais
        self.known_positions = set()  # Set com tickets de posições conhecidas
        
        # 🧠 V7 GATES - SINCRONIZAÇÃO COM DAYTRADER
        self.last_v7_gate_info = None  # Para armazenar outputs dos gates V7
        self.last_v7_outputs = None    # Para armazenar outputs V7 formatados para filtros
        self.daily_trades = []  # Para controle de boost de trades
        
        # 🔥 ACTION SPACE V7 COMPATIBLE: 11 dimensões especializadas
        self.action_space = spaces.Box(
            low=np.array([0, 0, -1, 0, -1, -3, -3, -3, -3, -3, -3]),
            high=np.array([2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3]),
            dtype=np.float32
        )
        
        # 🧠 V7 FEATURE COLUMNS - 129 features per step
        self.feature_columns = self._create_v7_feature_columns()
        
        # 🔥 OBSERVATION SPACE V7: 2580 dimensões (129 × 20)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(Config.OBSERVATION_SPACE_SIZE,),
            dtype=np.float32
        )
        
        self._log(f"[OBS SPACE V7] 🧠 {Config.OBSERVATION_SPACE_SIZE} dimensões ({Config.FEATURES_PER_STEP} features × {Config.WINDOW_SIZE} window)")
        
        # Variáveis de controle
        self.realized_balance = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.last_trade_pnl = 0.0
        self.steps_since_last_trade = 0
        self.last_action = None
        self.hold_count = 0
        self.hold_log_interval = 20  # Mostrar log de HOLD a cada 20 ocorrências
        self.base_tf = '5m'
        
        # Position sizing
        self.base_lot_size = Config.BASE_LOT_SIZE
        self.max_lot_size = Config.MAX_LOT_SIZE
        self.lot_size = self.base_lot_size  # Será calculado dinamicamente
        
        # SL/TP Ranges (aligned with daytrader.py REALISTIC_SLTP_CONFIG)
        self.sl_range_min = 2.0    # SL mínimo: 2 pontos (daytrade)
        self.sl_range_max = 8.0    # SL máximo: 8 pontos (daytrade)
        self.tp_range_min = 3.0    # TP mínimo: 3 pontos (daytrade)
        self.tp_range_max = 15.0   # TP máximo: 15 pontos (daytrade)
        self.sl_tp_step = 0.5      # Variação: 0.5 pontos
        
        # Debug counters
        self.debug_step_counter = 0
        self.debug_composite_interval = 10    # Debug composite a cada 10 steps
        self.debug_anomaly_interval = 50      # Debug anomalias a cada 50 steps
        self.last_observations = []           # Buffer para análise de anomalias
        self.obs_stats = {'mean': None, 'std': None, 'min': None, 'max': None}
        
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
                else:
                    # Verificar informações do símbolo para debug
                    symbol_info = mt5.symbol_info(self.symbol)
                    if symbol_info:
                        self._log(f"[📊 SÍMBOLO] {self.symbol} - Volume: [{symbol_info.volume_min:.3f}, {symbol_info.volume_max:.3f}] Step: {symbol_info.volume_step:.3f}")
                        self._log(f"[📊 SÍMBOLO] Spread: {symbol_info.spread} | Digits: {symbol_info.digits}")
                    else:
                        self._log(f"[WARNING] ⚠️ Não foi possível obter informações do símbolo {self.symbol}")
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
        self._initialize_historical_data_v7()
        
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
        self._log(f"[⚙️ CONFIG V7] Lot size: {self.lot_size}, Balance inicial: ${self.initial_balance}")
        self._log(f"[🎯 SL/TP RANGES] SL: {self.sl_range_min}-{self.sl_range_max} pontos, TP: {self.tp_range_min}-{self.tp_range_max} pontos (aligned with daytrader.py)")
        self._log(f"[🔒 LIMITE] Máximo de posições simultâneas: {self.max_positions}")
        self._log(f"[💰 LOT SIZE] Base: {self.base_lot_size} | Max: {self.max_lot_size} | Dynamic sizing: ATIVO")
        
        # Cache para normalização de preços
        self._price_min_max_cache = None
        
        # 🎯 V7 Model Loading - Variáveis de controle
        self.model = None
        self.normalizer = None
        self.model_loaded = False
        self.model_metadata = None
        
        # Tentar carregar modelo automaticamente
        self._log("[🤖 V7 MODEL] Iniciando carregamento automático...")
        try:
            self.auto_load_v7_model()
        except Exception as e:
            self._log(f"[WARNING] Falha no carregamento automático do modelo: {e}")
    
    def _create_v7_feature_columns(self):
        """🔧 Criar TODAS as colunas de features para V7 - 129 features total para historical_df"""
        all_columns = []
        
        # 🎯 1. MARKET FEATURES (16 features)
        base_features_5m = [
            'returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 
            'stoch_k', 'bb_position', 'trend_strength', 'atr_14'
        ]
        
        high_quality_features = [
            'volume_momentum', 'price_position', 'breakout_strength', 
            'trend_consistency', 'support_resistance', 'volatility_regime', 'market_structure'
        ]
        
        # Market features com sufixo _5m para base + high quality direto
        all_columns.extend([f"{f}_5m" for f in base_features_5m])  # 9
        all_columns.extend(high_quality_features)  # 7
        # Total market: 16
        
        # 🎯 2. MICROSTRUCTURE FEATURES (14 features)
        microstructure_features = [
            'bid_ask_spread', 'order_flow_imbalance', 'tick_direction', 'volume_imbalance',
            'price_impact', 'liquidity_score', 'market_depth', 'order_arrival_rate',
            'trade_size_avg', 'volatility_clustering', 'jump_detection', 'regime_stability',
            'tick_volume_ratio', 'price_acceleration'
        ]
        all_columns.extend(microstructure_features)  # 14
        
        # 🎯 3. VOLATILITY ADVANCED (5 features)
        volatility_features = [
            'garch_vol', 'realized_vol', 'vol_clustering', 'vol_breakout', 'vol_regime'
        ]
        all_columns.extend(volatility_features)  # 5
        
        # 🎯 4. MARKET CORRELATION (4 features)
        correlation_features = [
            'gold_dollar_corr', 'gold_bonds_corr', 'vol_correlation', 'regime_correlation'
        ]
        all_columns.extend(correlation_features)  # 4
        
        # 🎯 5. MOMENTUM MULTI (6 features)
        momentum_features = [
            'momentum_5m', 'momentum_15m', 'momentum_1h', 'momentum_confluence', 
            'trend_strength_multi', 'momentum_divergence'
        ]
        all_columns.extend(momentum_features)  # 6
        
        # 🎯 6. ENHANCED FEATURES (20 features)
        enhanced_features = [
            'pattern_bullish', 'pattern_bearish', 'pattern_reversal', 'pattern_continuation',
            'regime_trending', 'regime_ranging', 'regime_volatile', 'regime_calm',
            'support_level', 'resistance_level', 'breakout_probability', 'reversion_probability',
            'trend_maturity', 'cycle_position', 'seasonal_effect', 'news_sentiment',
            'volatility_smile', 'skew_indicator', 'tail_risk', 'liquidity_risk'
        ]
        all_columns.extend(enhanced_features)  # 20
        
        # 🔥 VERIFICAR TOTAL: 16 + 14 + 5 + 4 + 6 + 20 = 65 market-based features
        # As outras 64 features (positions=27 + intelligent=37) são calculadas dinamicamente em _get_observation_v7
        
        self._log(f"[V7 FEATURES] Historical dataframe columns: {len(all_columns)} market-based features")
        self._log(f"[V7 FEATURES] Total features per step: {len(all_columns)} + {Config.POSITION_FEATURES} + {Config.INTELLIGENT_V7} = {len(all_columns) + Config.POSITION_FEATURES + Config.INTELLIGENT_V7}")
        
        return all_columns
    
    def _initialize_historical_data_v7(self):
        """🔧 Inicializa dados históricos V7 com 129 features"""
        try:
            # Carregar dados dos últimos 1000 bars de M5
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
            
            # Criar múltiplos timeframes para V7
            df_5m = df.copy()
            df_15m = df.resample('15T').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
            }).dropna()
            df_1h = df.resample('1H').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
            }).dropna()
            
            # Calcular features para cada timeframe
            self.historical_df = pd.DataFrame(index=df_5m.index)
            
            # Processar 5m, 15m e 1h para V7
            for tf_name, tf_df in [('5m', df_5m), ('15m', df_15m), ('1h', df_1h)]:
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
                
                # Trend Strength
                returns = close_col.pct_change().fillna(0)
                self.historical_df[f'trend_strength_{tf_name}'] = returns.rolling(10).mean().fillna(0)
                
                self.historical_df[f'atr_14_{tf_name}'] = self._calculate_atr(tf_df, 14)
            
            # 🎯 CALCULAR FEATURES DE ALTA QUALIDADE V7
            close_5m = df_5m['close']
            high_5m = df_5m['high']
            low_5m = df_5m['low']
            volume_5m = df_5m['tick_volume']
            
            # Volume momentum
            volume_sma = volume_5m.rolling(20).mean().fillna(1)
            self.historical_df['volume_momentum'] = (volume_5m / volume_sma).fillna(1.0)
            
            # Price position
            high_20 = high_5m.rolling(20).max()
            low_20 = low_5m.rolling(20).min()
            self.historical_df['price_position'] = ((close_5m - low_20) / (high_20 - low_20).replace(0, 1)).fillna(0.5)
            
            # Volatility ratio
            vol_short = close_5m.rolling(5).std().fillna(0.01)
            vol_long = close_5m.rolling(20).std().fillna(0.01)
            self.historical_df['volatility_ratio'] = (vol_short / vol_long).fillna(1.0)
            
            # Intraday range
            self.historical_df['intraday_range'] = ((high_5m - low_5m) / close_5m.replace(0, 1)).fillna(0)
            
            # Market regime
            sma_20 = close_5m.rolling(20).mean()
            atr_14 = (high_5m - low_5m).rolling(14).mean()
            self.historical_df['market_regime'] = (abs(close_5m - sma_20) / atr_14.replace(0, 1)).fillna(0.5)
            
            # Spread pressure
            intraday_range = high_5m - low_5m
            volatility_avg = intraday_range.rolling(20).mean()
            spread_pressure = (intraday_range / close_5m.replace(0, 1)) / (volatility_avg / close_5m.replace(0, 1)).replace(0, 1)
            self.historical_df['spread_pressure'] = spread_pressure.clip(0, 5).fillna(1.0)
            
            # Session momentum
            self.historical_df['session_momentum'] = close_5m.pct_change(periods=48).fillna(0)
            
            # Time of day
            hours = pd.to_datetime(df_5m.index).hour
            self.historical_df['time_of_day'] = np.sin(2 * np.pi * hours / 24)
            
            # Tick momentum
            price_changes = close_5m.diff()
            tick_momentum = price_changes.rolling(5).apply(lambda x: (x > 0).sum() - (x < 0).sum()).fillna(0)
            self.historical_df['tick_momentum'] = (tick_momentum / 5.0).fillna(0)
            
            # 🔥 NORMALIZAR E LIMPAR DADOS V7
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
                        
            self._log(f"[INFO V7] ✅ Dados históricos carregados: {len(self.historical_df)} registros com {len(self.feature_columns)} features")
            
        except Exception as e:
            self._log(f"[ERROR] Erro ao inicializar dados históricos V7: {e}")
            # Fallback: criar dataframe vazio
            self.historical_df = pd.DataFrame()
            for col in self.feature_columns:
                self.historical_df[col] = [0.0] * 100
    
    def _calculate_rsi(self, price_series, period=14):
        """Calcula RSI"""
        try:
            delta = price_series.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss.replace(0, 0.01)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50.0)  # RSI padrão = 50
        except:
            return pd.Series([50.0] * len(price_series), index=price_series.index)
    
    def _calculate_atr(self, df, period=14):
        """Calcula ATR (Average True Range)"""
        try:
            high = df['high']
            low = df['low']
            close = df['close']
            
            high_low = high - low
            high_close_prev = np.abs(high - close.shift())
            low_close_prev = np.abs(low - close.shift())
            
            true_range = np.maximum(high_low, np.maximum(high_close_prev, low_close_prev))
            atr = true_range.rolling(window=period).mean()
            
            return atr.fillna(0.001)
        except:
            return pd.Series([0.001] * len(df), index=df.index)
    
    def _get_observation_v7(self):
        """🚀 Obter observação V7 - 2580 dimensões (129 features × 20 window)"""
        try:
            # Obter preço atual
            if self.mt5_connected:
                tick = mt5.symbol_info_tick(self.symbol)
                current_price = tick.bid if tick else 2000.0
            else:
                current_price = 2000.0
            
            # 🔥 1. MARKET FEATURES (36 features)
            if len(self.historical_df) > 0 and len(self.feature_columns) > 0:
                recent_data = self.historical_df[self.feature_columns].tail(self.window_size).values
                
                # SEM PADDING - FALHA SE DADOS INSUFICIENTES
                if len(recent_data) < self.window_size:
                    raise Exception(f"DADOS INSUFICIENTES V7: {len(recent_data)} < {self.window_size}")
            else:
                recent_data = np.zeros((self.window_size, len(self.feature_columns)))
            
            # 🔥 2. POSITION FEATURES (27 features - 3 positions × 9 features each)
            positions_obs = np.zeros((self.max_positions, 9), dtype=np.float32)
            
            # Obter posições atuais do MT5
            if self.mt5_connected:
                mt5_positions = mt5.positions_get(symbol=self.symbol)
                if mt5_positions is None:
                    mt5_positions = []
            else:
                mt5_positions = []
            
            # Processar posições (máximo 3)
            for i in range(self.max_positions):
                if i < len(mt5_positions):
                    pos = mt5_positions[i]
                    
                    # Feature 0: Position type (0=none, 1=long, 2=short)
                    positions_obs[i, 0] = 1.0 if pos.type == mt5.POSITION_TYPE_BUY else 2.0
                    
                    # Feature 1: Volume normalizado
                    positions_obs[i, 1] = pos.volume / 1.0  # Normalizar volume
                    
                    # Feature 2: Entry price normalizado
                    if not hasattr(self, '_price_min_max_cache') or self._price_min_max_cache is None:
                        # Calcular range de preços para normalização
                        if len(self.historical_df) > 0:
                            close_values = []
                            for col in self.feature_columns:
                                if 'close' in col.lower() or 'sma' in col.lower():
                                    close_values = self.historical_df[col].values
                                    break
                            
                            if len(close_values) == 0:
                                close_values = [current_price] * 100
                                
                            self._price_min_max_cache = {
                                'min': np.min(close_values),
                                'max': np.max(close_values), 
                                'range': np.max(close_values) - np.min(close_values) if np.max(close_values) > np.min(close_values) else 1.0
                            }
                        else:
                            self._price_min_max_cache = {
                                'min': current_price - 100,
                                'max': current_price + 100,
                                'range': 200
                            }
                    
                    positions_obs[i, 2] = (pos.price_open - self._price_min_max_cache['min']) / self._price_min_max_cache['range']
                    
                    # Feature 3: PnL atual normalizado
                    pnl = self._get_position_pnl(pos, current_price) / 1000
                    positions_obs[i, 3] = pnl
                    
                    # Features 4-5: SL e TP
                    positions_obs[i, 4] = pos.sl if pos.sl > 0 else 0
                    positions_obs[i, 5] = pos.tp if pos.tp > 0 else 0
                    
                    # Feature 6: Position age
                    try:
                        position_time = getattr(pos, 'time', None) or getattr(pos, 'time_setup', None)
                        if position_time:
                            position_age_seconds = time.time() - position_time
                            position_age_steps = position_age_seconds / 300  # 5 minutos por step
                            total_steps = len(self.historical_df) if len(self.historical_df) > 0 else 1000
                            positions_obs[i, 6] = position_age_steps / total_steps
                        else:
                            positions_obs[i, 6] = 0.1
                    except:
                        positions_obs[i, 6] = 0.1
                    
                    # Feature 7: Volume da posição normalizado
                    positions_obs[i, 7] = pos.volume / 1.0
                    
                    # Feature 8: Distância até SL/TP normalizada  
                    if pos.sl > 0:
                        sl_distance = abs(current_price - pos.sl) / current_price
                        positions_obs[i, 8] = np.clip(sl_distance, 0.0, 0.1)
                    elif pos.tp > 0:
                        tp_distance = abs(current_price - pos.tp) / current_price
                        positions_obs[i, 8] = np.clip(tp_distance, 0.0, 0.1)
                    else:
                        positions_obs[i, 8] = 0.0
                else:
                    positions_obs[i, :] = 0  # Slot vazio
            
            # 🔥 3. INTELLIGENT FEATURES (11 features) - V7 específico
            intelligent_components = self._generate_intelligent_components_v7(current_price)
            intelligent_features = self._flatten_intelligent_components_v7(intelligent_components)
            
            # Tile das posições e intelligent features para cada timestep
            tile_positions = np.tile(positions_obs.flatten(), (self.window_size, 1))  # (20, 27)
            tile_intelligent = np.tile(intelligent_features, (self.window_size, 1))    # (20, 11)
            
            # 🔥 CONCATENAR TUDO: market (36) + positions (27) + intelligent (11) = 74 features
            obs = np.concatenate([recent_data, tile_positions, tile_intelligent], axis=1)
            
            # 🔥 VERIFICAR EXATAMENTE 74 FEATURES POR STEP
            current_features_per_step = obs.shape[1]
            expected_features_per_step = Config.FEATURES_PER_STEP
            
            if current_features_per_step != expected_features_per_step:
                self._log(f"[OBS-V7-FIX] Ajustando features: {current_features_per_step} → {expected_features_per_step}")
                if current_features_per_step > expected_features_per_step:
                    obs = obs[:, :expected_features_per_step]
                else:
                    raise Exception(f"V7 FEATURES INSUFICIENTES: {current_features_per_step} < {expected_features_per_step}")
            
            # Flatten para formato final (20 × 129 = 2580)
            flat_obs = obs.flatten().astype(np.float32)
            
            # 🔥 CLIPPING E VALIDAÇÃO V7
            flat_obs = np.clip(flat_obs, -100.0, 100.0)
            flat_obs = np.nan_to_num(flat_obs, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # VERIFICAR EXATAMENTE 2580 DIMENSÕES
            if flat_obs.shape[0] != Config.OBSERVATION_SPACE_SIZE:
                raise Exception(f"V7 SHAPE INCORRETO: {flat_obs.shape[0]} != {Config.OBSERVATION_SPACE_SIZE}")
            
            # Verificações de integridade
            assert flat_obs.shape == self.observation_space.shape, f"V7 Obs shape {flat_obs.shape} != expected {self.observation_space.shape}"
            assert not np.any(np.isnan(flat_obs)), f"V7 Observação contém NaN"
            assert not np.any(np.isinf(flat_obs)), f"V7 Observação contém Inf"
            
            return flat_obs
            
        except Exception as e:
            self._log(f"[ERROR] Erro ao obter observação V7: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _generate_intelligent_components_v7(self, current_price):
        """🧠 GERAR COMPONENTES INTELIGENTES V7 - 11 features especializadas"""
        try:
            # 🎯 V7 Intelligent Components (11 total)
            components = {
                'market_regime_signal': self._get_market_regime_signal_v7(current_price),
                'volatility_context': self._get_volatility_context_v7(current_price),
                'momentum_confluence': self._get_momentum_confluence_v7(current_price),
                'risk_assessment': self._get_risk_assessment_v7(current_price),
                'pattern_recognition': self._get_pattern_recognition_v7(current_price),
                'liquidity_analysis': self._get_liquidity_analysis_v7(current_price),
                'temporal_context': self._get_temporal_context_v7(),
                'market_fatigue': self._get_market_fatigue_v7(current_price),
                'entry_quality': self._get_entry_quality_v7(current_price),
                'position_sizing_hint': self._get_position_sizing_hint_v7(),
                'exit_signal_strength': self._get_exit_signal_strength_v7(current_price)
            }
            
            return components
            
        except Exception as e:
            # Fallback com dados padrão V7
            return {
                'market_regime_signal': 0.0,
                'volatility_context': 0.5,
                'momentum_confluence': 0.0,
                'risk_assessment': 0.1,
                'pattern_recognition': 0.0,
                'liquidity_analysis': 0.5,
                'temporal_context': 0.0,
                'market_fatigue': 0.0,
                'entry_quality': 0.5,
                'position_sizing_hint': 0.5,
                'exit_signal_strength': 0.0
            }
    
    def _flatten_intelligent_components_v7(self, components):
        """🔧 Achatar componentes inteligentes V7 para array"""
        try:
            flattened = []
            component_order = [
                'market_regime_signal', 'volatility_context', 'momentum_confluence',
                'risk_assessment', 'pattern_recognition', 'liquidity_analysis', 
                'temporal_context', 'market_fatigue', 'entry_quality',
                'position_sizing_hint', 'exit_signal_strength'
            ]
            
            for key in component_order:
                value = components.get(key, 0.0)
                if isinstance(value, (int, float)):
                    flattened.append(np.float32(value))
                else:
                    flattened.append(np.float32(0.0))
            
            # Garantir exatamente 37 features (INTELLIGENT_V7)
            while len(flattened) < Config.INTELLIGENT_V7:
                flattened.append(np.float32(0.0))
            
            return np.array(flattened[:Config.INTELLIGENT_V7], dtype=np.float32)
            
        except Exception as e:
            return np.zeros(Config.INTELLIGENT_V7, dtype=np.float32)
    
    # 🧠 V7 Intelligent Component Methods
    def _get_market_regime_signal_v7(self, current_price):
        """V7: Market regime signal (-1 to 1)"""
        try:
            if self.mt5_connected and len(self.historical_df) > 20:
                # Usar dados históricos para detectar regime
                close_col = None
                for col in self.feature_columns:
                    if 'close' in col.lower() or 'sma_20_5m' in col:
                        close_col = self.historical_df[col].tail(20)
                        break
                
                if close_col is not None and len(close_col) > 0:
                    trend = np.mean(np.diff(close_col.values))
                    volatility = np.std(close_col.values)
                    return np.clip(trend / (volatility + 0.001), -1.0, 1.0)
            return 0.0
        except:
            return 0.0
    
    def _get_volatility_context_v7(self, current_price):
        """V7: Volatility context (0 to 1)"""
        try:
            if len(self.historical_df) > 20:
                vol_col = None
                for col in self.feature_columns:
                    if 'volatility' in col.lower():
                        vol_col = self.historical_df[col].tail(20)
                        break
                
                if vol_col is not None and len(vol_col) > 0:
                    current_vol = vol_col.iloc[-1]
                    avg_vol = np.mean(vol_col.values)
                    return np.clip(current_vol / (avg_vol + 0.001), 0.0, 2.0) / 2.0
            return 0.5
        except:
            return 0.5
    
    def _get_momentum_confluence_v7(self, current_price):
        """V7: Momentum confluence (-1 to 1)"""
        try:
            if len(self.historical_df) > 10:
                # Usar múltiplos indicadores de momentum
                rsi_signal = 0.0
                trend_signal = 0.0
                
                for col in self.feature_columns:
                    if 'rsi' in col.lower():
                        rsi_val = self.historical_df[col].iloc[-1]
                        rsi_signal = (rsi_val - 50) / 50  # Normalizar -1 a 1
                        break
                
                for col in self.feature_columns:
                    if 'trend_strength' in col.lower():
                        trend_signal = self.historical_df[col].iloc[-1]
                        break
                
                return np.clip((rsi_signal + trend_signal) / 2, -1.0, 1.0)
            return 0.0
        except:
            return 0.0
    
    def _get_risk_assessment_v7(self, current_price):
        """V7: Risk assessment (0 to 1)"""
        try:
            # Baseado em posições abertas e volatilidade
            num_positions = len(mt5.positions_get(symbol=self.symbol)) if self.mt5_connected else 0
            position_risk = num_positions / self.max_positions
            
            volatility_risk = 0.1
            if len(self.historical_df) > 0:
                for col in self.feature_columns:
                    if 'volatility' in col.lower():
                        volatility_risk = min(self.historical_df[col].iloc[-1], 1.0)
                        break
            
            return np.clip((position_risk + volatility_risk) / 2, 0.0, 1.0)
        except:
            return 0.1
    
    def _get_pattern_recognition_v7(self, current_price):
        """V7: Pattern recognition signal (-1 to 1)"""
        try:
            if len(self.historical_df) > 5:
                # Análise simples de padrões baseada em BB position
                for col in self.feature_columns:
                    if 'bb_position' in col.lower():
                        bb_pos = self.historical_df[col].iloc[-1]
                        # Converter de [0,1] para [-1,1] onde extremos indicam reversão
                        return (bb_pos - 0.5) * 2
            return 0.0
        except:
            return 0.0
    
    def _get_liquidity_analysis_v7(self, current_price):
        """V7: Liquidity analysis (0 to 1)"""
        try:
            # Baseado em volume momentum
            for col in self.feature_columns:
                if 'volume_momentum' in col.lower():
                    vol_momentum = self.historical_df[col].iloc[-1] if len(self.historical_df) > 0 else 1.0
                    return np.clip(vol_momentum / 2.0, 0.0, 1.0)
            return 0.5
        except:
            return 0.5
    
    def _get_temporal_context_v7(self):
        """V7: Temporal context (-1 to 1)"""
        try:
            current_hour = pd.Timestamp.now().hour
            # Sinal baseado em horários de maior volatilidade do ouro
            if 13 <= current_hour <= 17:  # Horário de maior movimento (overlap London/NY)
                return 1.0
            elif 8 <= current_hour <= 12:  # Horário europeu
                return 0.5
            elif 21 <= current_hour <= 23:  # Horário asiático
                return -0.5
            else:
                return -1.0  # Horários de menor movimento
        except:
            return 0.0
    
    def _get_market_fatigue_v7(self, current_price):
        """V7: Market fatigue detection (0 to 1)"""
        try:
            # Baseado em range intraday e volatilidade recente
            if len(self.historical_df) > 10:
                ranges = []
                for col in self.feature_columns:
                    if 'intraday_range' in col.lower():
                        ranges = self.historical_df[col].tail(10).values
                        break
                
                if len(ranges) > 0:
                    current_range = ranges[-1]
                    avg_range = np.mean(ranges)
                    # Fadiga alta quando range atual é muito menor que média
                    fatigue = 1.0 - (current_range / (avg_range + 0.001))
                    return np.clip(fatigue, 0.0, 1.0)
            return 0.0
        except:
            return 0.0
    
    def _get_entry_quality_v7(self, current_price):
        """V7: Entry quality assessment (0 to 1)"""
        try:
            # Combinação de múltiplos fatores para qualidade de entrada
            quality_score = 0.5
            
            # Fator 1: Posição do preço na banda de Bollinger
            for col in self.feature_columns:
                if 'bb_position' in col.lower() and len(self.historical_df) > 0:
                    bb_pos = self.historical_df[col].iloc[-1]
                    # Melhores entradas nos extremos das bandas
                    if bb_pos < 0.2 or bb_pos > 0.8:
                        quality_score += 0.3
                    break
            
            # Fator 2: Confluence de timeframes
            trend_confluence = 0
            for tf in ['5m', '15m', '1h']:
                for col in self.feature_columns:
                    if f'trend_strength_{tf}' in col and len(self.historical_df) > 0:
                        trend = self.historical_df[col].iloc[-1]
                        if abs(trend) > 0.1:  # Trend significativo
                            trend_confluence += 1
                        break
            
            if trend_confluence >= 2:  # Pelo menos 2 timeframes confirmando
                quality_score += 0.2
            
            return np.clip(quality_score, 0.0, 1.0)
        except:
            return 0.5
    
    def _get_position_sizing_hint_v7(self):
        """V7: Position sizing hint (0 to 1)"""
        try:
            # Baseado em posições atuais e drawdown
            current_positions = len(mt5.positions_get(symbol=self.symbol)) if self.mt5_connected else 0
            position_ratio = current_positions / self.max_positions
            
            # Menor size se já temos muitas posições
            size_hint = 1.0 - position_ratio
            
            # Ajustar baseado em performance recente
            if hasattr(self, 'peak_drawdown') and self.peak_drawdown > 0.1:
                size_hint *= 0.5  # Reduzir size se drawdown alto
            
            return np.clip(size_hint, 0.1, 1.0)
        except:
            return 0.5
    
    def _get_exit_signal_strength_v7(self, current_price):
        """V7: Exit signal strength (-1 to 1)"""
        try:
            exit_signal = 0.0
            
            # Verificar posições com risco de reversão
            if self.mt5_connected:
                positions = mt5.positions_get(symbol=self.symbol)
                if positions and len(positions) > 0:
                    for pos in positions:
                        pnl = self._get_position_pnl(pos, current_price)
                        
                        # Sinal de saída forte se posição muito no lucro ou prejuízo
                        if pnl > 50:  # Lucro alto - considerar saída
                            exit_signal += 0.5
                        elif pnl < -30:  # Prejuízo alto - considerar saída
                            exit_signal -= 0.5
            
            return np.clip(exit_signal, -1.0, 1.0)
        except:
            return 0.0
    
    def _get_position_pnl(self, position, current_price):
        """Calcula PnL de uma posição"""
        try:
            if position.type == mt5.POSITION_TYPE_BUY:
                pnl = (current_price - position.price_open) * position.volume * 100  # Para GOLD
            else:  # SELL
                pnl = (position.price_open - current_price) * position.volume * 100
            return pnl
        except:
            return 0.0
    
    def _process_v7_action(self, action):
        """🚀 Processar ação V7 - 11 dimensões especializadas"""
        try:
            if not self.mt5_connected:
                self._log("[WARNING] MT5 não conectado - ação ignorada")
                return "ERROR_NO_MT5"
            
            # 🔥 V7 ACTION SPACE (11D):
            # [0] entry_decision: 0=HOLD, 1=LONG, 2=SHORT
            # [1] entry_quality: [0,1] Qualidade da entrada (filtro + ajuste SL/TP)
            # [2] temporal_signal: [-1,1] Sinal temporal
            # [3] risk_appetite: [0,1] Apetite ao risco
            # [4] market_regime_bias: [-1,1] Viés do mercado
            # [5-7] sl_adjusts: SL para pos1, pos2, pos3 ([-3,3])
            # [8-10] tp_adjusts: TP para pos1, pos2, pos3 ([-3,3])
            
            entry_decision = int(np.clip(action[0], 0, 2))
            entry_quality = np.clip(action[1], 0.0, 1.0)
            temporal_signal = np.clip(action[2], -1.0, 1.0)
            risk_appetite = np.clip(action[3], 0.0, 1.0)
            market_regime_bias = np.clip(action[4], -1.0, 1.0)
            
            sl_adjusts = [np.clip(action[5], -3, 3), np.clip(action[6], -3, 3), np.clip(action[7], -3, 3)]
            tp_adjusts = [np.clip(action[8], -3, 3), np.clip(action[9], -3, 3), np.clip(action[10], -3, 3)]
            
            # 🔥 PROCESSAR SL/TP ADJUSTMENTS PRIMEIRO
            self._process_sl_tp_adjustments_v7(sl_adjusts, tp_adjusts)
            
            # 🔥 PROCESSAR ENTRADA BASEADA EM V7 LOGIC
            if entry_decision == 0:  # HOLD
                return "HOLD"
            
            # 🚀 V7 DECISION FILTERS REMOVIDOS - ALINHADO COM DAYTRADER: "V7 INTUITION DECIDE TUDO - SEM FILTROS LOCAIS"
            # entry_allowed, filter_reason = self._check_entry_filters_v7(entry_decision)
            # if not entry_allowed:
            #     self._log(f"[V7 FILTER] {filter_reason}")
            #     return "HOLD_V7_GATES_FAILED"
            
            # 🚀 REGIME FILTER REMOVIDO - ALINHADO COM DAYTRADER: "V7 INTUITION DECIDE TUDO"
            # if abs(market_regime_bias) > 0.5:  # Forte viés de mercado
            #     regime_direction = 1.0 if market_regime_bias > 0 else -1.0
            #     if decision_direction != regime_direction:
            #         self._log(f"[V7 FILTER] Market regime bias conflitante: {market_regime_bias:.2f} vs decision {decision_direction}")
            #         return "HOLD_REGIME_CONFLICT"
            
            # 🎯 EXECUTAR ENTRADA SE PASSOU POR TODOS OS FILTROS
            
            # 🎯 FILTRO DE QUALIDADE MÍNIMA (ALINHADO COM DAYTRADER)
            MIN_QUALITY_THRESHOLD = 0.4  # Só entrar se qualidade > 40%
            if entry_quality < MIN_QUALITY_THRESHOLD:
                return "HOLD_LOW_QUALITY"
            
            # Calcular volume baseado em V7 factors (sem usar entry_quality)
            base_volume = self._calculate_v7_volume_fixed(risk_appetite, current_positions)
            
            # Calcular SL/TP baseado em V7 logic (usando entry_quality para ajuste)
            sl_price, tp_price = self._calculate_v7_sl_tp_quality(entry_decision, entry_quality, risk_appetite)
            
            # Executar ordem
            order_type = mt5.ORDER_TYPE_BUY if entry_decision == 1 else mt5.ORDER_TYPE_SELL
            result = self._execute_order_v7(order_type, base_volume, sl_price, tp_price, 
                                          entry_quality, temporal_signal, market_regime_bias)
            
            if "SUCCESS" in result:
                action_type = "📈 LONG" if entry_decision == 1 else "📉 SHORT"
                stars = '⭐' * min(5, int(entry_quality * 5))
                self._log(f"[🎯 V7 TRADE] {action_type} | Quality: {entry_quality:.2f} {stars} | Vol: {base_volume:.3f} | Risk: {risk_appetite:.2f}")
                return result
            else:
                return result
                
        except Exception as e:
            self._log(f"[ERROR] Erro ao processar ação V7: {e}")
            return "ERROR_PROCESSING"
    
    def _process_sl_tp_adjustments_v7(self, sl_adjusts, tp_adjusts):
        """🔧 Processar ajustes de SL/TP V7 para posições existentes"""
        try:
            if not self.mt5_connected:
                return
            
            positions = mt5.positions_get(symbol=self.symbol)
            if not positions:
                return
            
            for i, pos in enumerate(positions[:3]):  # Máximo 3 posições
                if i < len(sl_adjusts) and i < len(tp_adjusts):
                    sl_adjust = sl_adjusts[i]
                    tp_adjust = tp_adjusts[i]
                    
                    # Aplicar ajustes apenas se significativos
                    if abs(sl_adjust) > 0.5 or abs(tp_adjust) > 0.5:
                        self._modify_position_v7(pos, sl_adjust, tp_adjust)
                        
        except Exception as e:
            self._log(f"[WARNING] Erro ao ajustar SL/TP V7: {e}")
    
    def _modify_position_v7(self, position, sl_adjust, tp_adjust):
        """🔧 Modificar posição individual V7"""
        try:
            # Obter preço atual
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return
            
            current_price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask
            
            # Calcular novos SL/TP baseados nos ajustes (aligned with daytrader.py)
            # Converter ajuste [-3,3] para ranges realistas:
            # SL: [-3,3] -> [2.0,8.0] pontos  
            # TP: [-3,3] -> [3.0,15.0] pontos
            
            sl_points = self.sl_range_min + (sl_adjust + 3) * (self.sl_range_max - self.sl_range_min) / 6
            tp_points = self.tp_range_min + (tp_adjust + 3) * (self.tp_range_max - self.tp_range_min) / 6
            
            # Arredondar para múltiplos de 0.5 pontos
            sl_points = round(sl_points * 2) / 2
            tp_points = round(tp_points * 2) / 2
            
            # Garantir limites
            sl_points = max(self.sl_range_min, min(sl_points, self.sl_range_max))
            tp_points = max(self.tp_range_min, min(tp_points, self.tp_range_max))
            
            new_sl = 0
            new_tp = 0
            
            if abs(sl_adjust) > 0.5:  # Aplicar SL apenas se ajuste significativo
                if position.type == mt5.POSITION_TYPE_BUY:
                    new_sl = current_price - sl_points
                else:
                    new_sl = current_price + sl_points
            else:
                new_sl = position.sl  # Manter SL atual
            
            if abs(tp_adjust) > 0.5:  # Aplicar TP apenas se ajuste significativo  
                if position.type == mt5.POSITION_TYPE_BUY:
                    new_tp = current_price + tp_points
                else:
                    new_tp = current_price - tp_points
            else:
                new_tp = position.tp  # Manter TP atual
            
            # Modificar posição se houve mudança
            if new_sl != position.sl or new_tp != position.tp:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": self.symbol,
                    "position": position.ticket,
                    "sl": new_sl if new_sl > 0 else 0,
                    "tp": new_tp if new_tp > 0 else 0
                }
                
                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self._log(f"[🔧 V7 MODIFY] Pos #{position.ticket} | SL: {new_sl:.2f} | TP: {new_tp:.2f}")
                else:
                    self._log(f"[WARNING] Falha ao modificar posição #{position.ticket}")
                    
        except Exception as e:
            self._log(f"[WARNING] Erro ao modificar posição V7: {e}")
    
    def _calculate_v7_volume_fixed(self, risk_appetite, current_positions):
        """🎯 Calcular volume V7 SEM usar entry_quality (position sizing hardcoded)"""
        try:
            # Volume base
            base_volume = self.base_lot_size
            
            # Ajuste APENAS por risk appetite (0-1 -> 0.7-1.3x)
            risk_multiplier = 0.7 + (risk_appetite * 0.6)
            
            # Redução por posições existentes
            position_multiplier = max(0.3, 1.0 - (current_positions / self.max_positions) * 0.7)
            
            # Volume final (SEM confidence multiplier)
            final_volume = base_volume * risk_multiplier * position_multiplier
            
            # Limitar ao máximo permitido
            final_volume = min(final_volume, self.max_lot_size)
            final_volume = max(final_volume, 0.01)  # Mínimo 0.01 lote
            
            return round(final_volume, 2)
            
        except:
            return self.base_lot_size
    
    def _calculate_v7_sl_tp_quality(self, entry_decision, entry_quality, risk_appetite):
        """🎯 Calcular SL/TP V7 usando Entry Quality Score"""
        try:
            # Obter preço atual
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return None, None
            
            current_price = tick.ask if entry_decision == 1 else tick.bid
            
            # Base ranges (aligned with daytrader.py)
            # SL: 2.0-8.0 pontos, TP: 3.0-15.0 pontos
            base_sl = self.sl_range_min + risk_appetite * (self.sl_range_max - self.sl_range_min)
            base_tp = self.tp_range_min + 0.6 * (self.tp_range_max - self.tp_range_min)  # Usar base fixa
            
            # 🎯 AJUSTE POR QUALIDADE DA ENTRADA
            # Quality alta = SL mais próximo (mais confiante) + TP mais longe (mais ambicioso)
            quality_factor = (entry_quality - 0.5) * 0.4  # [-0.2, +0.2]
            
            # Ajustar SL: quality alta = SL mais próximo (mais confiante)
            sl_points = base_sl * (1.0 - quality_factor * 0.5)  # -10% se quality=1.0
            
            # Ajustar TP: quality alta = TP mais longe (mais ambicioso) 
            tp_points = base_tp * (1.0 + quality_factor)      # +20% se quality=1.0
            
            # Arredondar para múltiplos de 0.5 pontos
            sl_points = round(sl_points * 2) / 2
            tp_points = round(tp_points * 2) / 2
            
            # Garantir limites
            sl_points = max(self.sl_range_min, min(sl_points, self.sl_range_max))
            tp_points = max(self.tp_range_min, min(tp_points, self.tp_range_max))
            
            # Calcular preços finais
            if entry_decision == 1:  # LONG
                sl_price = current_price - sl_points
                tp_price = current_price + tp_points
            else:  # SHORT
                sl_price = current_price + sl_points
                tp_price = current_price - tp_points
            
            return sl_price, tp_price
            
        except:
            return None, None
    
    def _execute_order_v7(self, order_type, volume, sl_price, tp_price, 
                         entry_quality, temporal_signal, market_regime_bias):
        """🚀 Executar ordem V7 com lógica especializada"""
        try:
            # 🚀 COOLDOWN REMOVIDO - ALINHADO COM DAYTRADER: "V7 INTUITION DECIDE TUDO"
            current_time = time.time()
            # time_since_last = current_time - self.last_order_time
            # if time_since_last < 1:
            #     self._log(f"⏱️ [COOLDOWN] Ordem bloqueada - {time_since_last:.2f}s desde última ordem (mín: 1s)")
            #     return "ERROR_COOLDOWN"
            
            self.last_order_time = current_time
            
            # Verificar se mercado está aberto
            market_status = self._check_market_hours_v7()
            if market_status != "OPEN":
                return f"ERROR_MARKET_{market_status}"
            
            # Obter preço atual
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return "ERROR_NO_PRICE"
            
            price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            
            # Preparar requisição V7
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "magic": 234567,  # Magic number diferente para V7
                "comment": f"V7Robot|Q:{entry_quality:.2f}|T:{temporal_signal:.2f}|R:{market_regime_bias:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.filling_mode
            }
            
            # Adicionar SL/TP se calculados
            if sl_price is not None and sl_price > 0:
                request["sl"] = sl_price
            if tp_price is not None and tp_price > 0:
                request["tp"] = tp_price
            
            # Verificar ordem
            check_result = mt5.order_check(request)
            if not check_result or (check_result.retcode != 0 and check_result.retcode != mt5.TRADE_RETCODE_DONE):
                return f"ERROR_ORDER_CHECK|{check_result.retcode if check_result else 'None'}"
            
            # Executar ordem
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                action_type = "📈 LONG" if order_type == mt5.ORDER_TYPE_BUY else "📉 SHORT"
                sl_info = f" | SL: {sl_price:.2f}" if sl_price else ""
                tp_info = f" | TP: {tp_price:.2f}" if tp_price else ""
                self._log(f"[🎯 V7 TRADE] {action_type} #{result.order} @ {price:.2f}{sl_info}{tp_info}")
                return f"SUCCESS|{result.order}|{price}|{action_type}|{sl_price or 0}|{tp_price or 0}"
            else:
                error_code = result.retcode if result else "None"
                return f"ERROR_EXECUTION|{error_code}"
                
        except Exception as e:
            self._log(f"[ERROR] Erro na execução V7: {e}")
            return "ERROR_EXCEPTION"
    
    def _check_market_hours_v7(self):
        """🕐 Verificar horários de mercado V7"""
        try:
            from datetime import datetime
            now = datetime.now()
            weekday = now.weekday()  # 0=Monday, 6=Sunday
            hour = now.hour
            
            # GOLD: Domingo 19:00 BRT até Sexta 21:00 BRT
            if weekday == 5:  # Saturday
                return "CLOSED"
            elif weekday == 6 and hour < 19:  # Sunday before 19:00 BRT
                return "CLOSED"
            elif weekday == 4 and hour >= 21:  # Friday after 21:00 BRT
                return "CLOSED"
            else:
                return "OPEN"
                
        except:
            return "UNKNOWN"
    
    def _verify_v7_compatibility(self, model_path=None):
        """🔍 Verificar compatibilidade V7"""
        try:
            self._log("[🔍 V7 VERIFY] Iniciando verificação de compatibilidade...")
            
            # 1. Verificar observation space
            expected_obs_size = Config.OBSERVATION_SPACE_SIZE
            actual_obs_size = self.observation_space.shape[0]
            
            if actual_obs_size != expected_obs_size:
                self._log(f"[❌ V7] Observation space incorreto: {actual_obs_size} != {expected_obs_size}")
                return False
            
            # 2. Verificar action space
            expected_action_size = Config.ACTION_SPACE_SIZE
            actual_action_size = self.action_space.shape[0]
            
            if actual_action_size != expected_action_size:
                self._log(f"[❌ V7] Action space incorreto: {actual_action_size} != {expected_action_size}")
                return False
            
            # 3. Verificar features por step
            expected_features = Config.FEATURES_PER_STEP
            actual_features = len(self.feature_columns) + Config.POSITION_FEATURES + Config.INTELLIGENT_V7
            
            if actual_features != expected_features:
                self._log(f"[❌ V7] Features per step incorreto: {actual_features} != {expected_features}")
                return False
            
            # 4. Testar geração de observação
            try:
                test_obs = self._get_observation_v7()
                if test_obs.shape[0] != expected_obs_size:
                    self._log(f"[❌ V7] Observação gerada com tamanho incorreto: {test_obs.shape[0]}")
                    return False
            except Exception as e:
                self._log(f"[❌ V7] Erro ao gerar observação teste: {e}")
                return False
            
            # 5. Verificar modelo se fornecido ou carregado
            model_to_test = None
            if model_path and os.path.exists(model_path):
                try:
                    # Tentar carregar modelo para verificar compatibilidade
                    model_to_test = RecurrentPPO.load(model_path)
                    self._log(f"[🔍 V7] Testando modelo externo: {model_path}")
                except Exception as e:
                    self._log(f"[❌ V7] Erro ao carregar modelo externo: {e}")
                    return False
            elif self.model_loaded and self.model is not None:
                # Usar modelo já carregado
                model_to_test = self.model
                self._log(f"[🔍 V7] Testando modelo carregado")
            
            if model_to_test is not None:
                try:
                    # Verificar se o modelo aceita nossa observation
                    test_action, _ = model_to_test.predict(test_obs.reshape(1, -1), deterministic=True)
                    
                    if test_action.shape[1] != expected_action_size:
                        self._log(f"[❌ V7] Modelo produz ação com tamanho incorreto: {test_action.shape[1]} != {expected_action_size}")
                        return False
                        
                    self._log(f"[✅ V7] Modelo compatível verificado")
                    
                except Exception as e:
                    self._log(f"[❌ V7] Erro ao verificar compatibilidade do modelo: {e}")
                    return False
            
            # 🎯 VERIFICAÇÕES V7 ESPECÍFICAS
            
            # 6. Verificar TwoHeadV7Simple compatibility
            if model_path and "v7" not in model_path.lower():
                self._log(f"[⚠️ V7] Modelo pode não ser V7: {model_path}")
            
            # 7. Verificar breakdown de features
            market_features = len(self.feature_columns)
            position_features = Config.POSITION_FEATURES
            intelligent_features = Config.INTELLIGENT_V7
            
            self._log(f"[✅ V7] Features breakdown:")
            self._log(f"  - Market: {market_features} (expected: {Config.MARKET_FEATURES})")
            self._log(f"  - Position: {position_features}")
            self._log(f"  - Intelligent: {intelligent_features}")
            self._log(f"  - Total: {market_features + position_features + intelligent_features}")
            
            # 8. Verificar MT5 connection
            if not self.mt5_connected:
                self._log(f"[⚠️ V7] MT5 não conectado - funcionalidade limitada")
            
            # 9. Status do modelo carregado
            model_status = "❌ Não carregado"
            if self.model_loaded:
                model_status = "✅ Carregado e compatível"
            elif self.model is not None:
                model_status = "⚠️ Carregado mas não verificado"
            
            self._log(f"[✅ V7] Compatibilidade verificada com sucesso!")
            self._log(f"  - Obs Space: {actual_obs_size}D")
            self._log(f"  - Action Space: {actual_action_size}D") 
            self._log(f"  - Features/Step: {expected_features}")
            self._log(f"  - Window Size: {Config.WINDOW_SIZE}")
            self._log(f"  - Modelo: {model_status}")
            self._log(f"  - Normalizer: {'✅ Disponível' if self.normalizer else '❌ Não disponível'}")
            self._log(f"  - Model Path: {ModelPaths.get_absolute_model_path()}")
            
            # Verificar Entry Head V7
            entry_head_status = self._verify_entry_head_v7()
            self._log(f"  - Entry Head V7: {entry_head_status}")
            
            return True
            
        except Exception as e:
            self._log(f"[❌ V7] Erro na verificação de compatibilidade: {e}")
            return False
    
    def _log(self, message):
        """Log helper method"""
        try:
            timestamp = pd.Timestamp.now().strftime("%H:%M:%S")
            log_message = f"[{timestamp}] {message}"
            
            if self.log_widget:
                # GUI logging
                self.log_widget.insert(tk.END, log_message + "\n")
                self.log_widget.see(tk.END)
            else:
                # Console logging
                print(log_message)
                sys.stdout.flush()  # Force immediate output
                
        except Exception as e:
            print(f"[LOG ERROR] {e}: {message}")
    
    def reset(self):
        """Reset environment V7"""
        try:
            self.current_step = 0
            self.done = False
            self.last_action = None
            self.hold_count = 0
            
            # Atualizar dados históricos
            self._initialize_historical_data_v7()
            
            return self._get_observation_v7()
            
        except Exception as e:
            self._log(f"[ERROR] Erro no reset V7: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def step(self, action):
        """Step V7 - executar ação e retornar nova observação"""
        try:
            # Processar ação V7
            action_result = self._process_v7_action(action)
            
            # Obter nova observação
            observation = self._get_observation_v7()
            
            # Calcular reward (simplificado para V7)
            reward = self._calculate_v7_reward(action_result)
            
            # Verificar se done
            done = self._check_v7_done()
            
            # Info dict
            info = {
                'action_result': action_result,
                'step': self.current_step,
                'v7_compatible': True
            }
            
            self.current_step += 1
            self.last_action = action
            
            return observation, reward, done, info
            
        except Exception as e:
            self._log(f"[ERROR] Erro no step V7: {e}")
            return self._get_observation_v7(), 0.0, True, {'error': str(e)}
    
    def _calculate_v7_reward(self, action_result):
        """🎯 Calcular reward V7 (simplificado)"""
        try:
            if "SUCCESS" in action_result:
                return 1.0  # Reward por executar trade
            elif "HOLD" in action_result:
                return 0.0  # Neutro por hold
            else:
                return -0.1  # Pequena penalidade por erro
        except:
            return 0.0
    
    def _check_v7_done(self):
        """Verificar se episódio terminou"""
        try:
            # Episódio termina se muitos steps ou erro crítico
            if self.current_step > 1000:
                return True
            
            # Verificar drawdown excessivo
            if hasattr(self, 'current_drawdown') and self.current_drawdown > 0.5:
                return True
                
            return False
            
        except:
            return False
    
    def auto_load_v7_model(self):
        """🤖 Carregar modelo V7 automaticamente do ZIP"""
        try:
            self._log("[🤖 V7 AUTO-LOAD] Iniciando carregamento automático...")
            
            # 1. Verificar se ZIP existe
            zip_path = ModelPaths.get_absolute_model_path()
            self._log(f"[🤖 V7 AUTO-LOAD] Procurando ZIP em: {zip_path}")
            
            if not os.path.exists(zip_path):
                self._log(f"[❌ V7 AUTO-LOAD] ZIP não encontrado: {zip_path}")
                return False
            
            # Verificar integridade do ZIP antes de carregar
            try:
                import zipfile
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    file_list = zip_ref.namelist()
                    self._log(f"[🔍 V7 AUTO-LOAD] Arquivos no ZIP: {file_list[:5]}...")  # Mostrar primeiros 5
                    
                    # Verificar se contém arquivos SB3 essenciais
                    has_data = any('data' in f.lower() for f in file_list)
                    has_policy = any('policy' in f.lower() or 'pytorch' in f.lower() for f in file_list)
                    self._log(f"[🔍 V7 AUTO-LOAD] ZIP contém: data={has_data}, policy={has_policy}")
                    
                    if not (has_data or has_policy):
                        self._log("[❌ V7 AUTO-LOAD] ZIP não contém arquivos SB3 válidos")
                        return False
            except Exception as zip_error:
                self._log(f"[❌ V7 AUTO-LOAD] Erro ao verificar ZIP: {zip_error}")
                return False
            
            # 2. Carregar modelo com policy_kwargs para V7 Intuition
            # 🎯 INTUITION POLICY KWARGS
            intuition_kwargs = get_v7_intuition_kwargs()
            
            # Configurar policy_kwargs para compatibilidade (sem policy_class)
            policy_kwargs = intuition_kwargs
            
            self.model = RecurrentPPO.load(zip_path, policy_kwargs=policy_kwargs)
            self._log(f"[✅ V7 INTUITION] Modelo carregado com TwoHeadV7Intuition de: {zip_path}")
            
            # 🔒 CRÍTICO: Colocar modelo em modo de inferência (congelar pesos)
            self.model.policy.eval()
            frozen_params = 0
            total_params = 0
            for param in self.model.policy.parameters():
                param.requires_grad = False
                frozen_params += param.numel()
                total_params += param.numel()
            
            self._log(f"[🔒 V7 AUTO-LOAD] Pesos do modelo CONGELADOS - modo inferência")
            self._log(f"[🔒 V7 AUTO-LOAD] Parâmetros congelados: {frozen_params:,} / {total_params:,} (100%)")
            
            self.model_loaded = True
            
            # 3. Carregar normalizer (está fora do ZIP)
            normalizer_path = os.path.join(os.path.dirname(zip_path), "enhanced_normalizer_final.pkl")
            if os.path.exists(normalizer_path):
                import pickle
                with open(normalizer_path, 'rb') as f:
                    self.normalizer = pickle.load(f)
                self._log(f"[✅ V7 AUTO-LOAD] Normalizer carregado: {normalizer_path}")
            else:
                self._log("[⚠️ V7 AUTO-LOAD] Normalizer não encontrado - usando padrão")
                self.normalizer = None
            
            # 5. Inicializar metadados padrão
            self.model_metadata = {}
            
            # 6. Verificar compatibilidade V7 (skip for direct ZIP loading)
            # Modelo já foi carregado com sucesso pelo SB3, então é compatível
            
            self._log("[🎉 V7 AUTO-LOAD] Modelo V7 carregado com sucesso!")
            return True
            
        except Exception as e:
            self._log(f"[❌ V7 AUTO-LOAD] Erro no carregamento automático: {e}")
            self.model = None
            self.normalizer = None
            self.model_loaded = False
            return False
    
    def _extract_model_zip(self, zip_path):
        """🗂️ Extrair arquivos do modelo ZIP"""
        try:
            self._log(f"[🗂️ V7 EXTRACT] Extraindo ZIP: {zip_path}")
            
            # Garantir diretório de extração
            extract_dir = ModelPaths.ensure_extract_dir()
            
            # Limpar diretório anterior se existir
            if os.path.exists(extract_dir):
                shutil.rmtree(extract_dir)
            os.makedirs(extract_dir, exist_ok=True)
            
            extracted_files = {}
            
            # Extrair ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                
                # Mapear arquivos extraídos
                for expected_key, expected_filename in ModelPaths.EXPECTED_FILES.items():
                    # Procurar arquivo no diretório extraído
                    for root, dirs, files in os.walk(extract_dir):
                        for file in files:
                            if expected_filename and (file == expected_filename or file.endswith(expected_filename)):
                                full_path = os.path.join(root, file)
                                extracted_files[expected_key] = full_path
                                self._log(f"[🗂️ V7 EXTRACT] Encontrado {expected_key}: {full_path}")
                                break
                        if expected_key in extracted_files:
                            break
            
            # Verificar se pelo menos o modelo foi encontrado
            if 'model' not in extracted_files:
                self._log("[❌ V7 EXTRACT] Modelo principal não encontrado no ZIP")
                return None
            
            self._log(f"[✅ V7 EXTRACT] Extração concluída: {len(extracted_files)} arquivos")
            return extracted_files
            
        except Exception as e:
            self._log(f"[❌ V7 EXTRACT] Erro na extração: {e}")
            return None
    
    def get_model_prediction(self, observation):
        """🧠 Obter predição do modelo V7 carregado"""
        try:
            if not self.model_loaded or self.model is None:
                self._log("[⚠️ V7 PREDICT] Modelo não carregado")
                return None
            
            # Capturar observação original para debug
            original_obs = observation.copy() if isinstance(observation, np.ndarray) else observation
            
            # Aplicar normalização se disponível
            if self.normalizer is not None:
                try:
                    observation = self.normalizer.normalize_obs(observation)
                except Exception as e:
                    self._log(f"[⚠️ V7 PREDICT] Erro na normalização: {e}")
            
            # 🔒 Obter predição em modo de inferência (sem gradientes)
            with torch.no_grad():
                action, _states = self.model.predict(observation, deterministic=False)
            
            # 🧠 CAPTURAR GATES V7 PARA SINCRONIZAÇÃO COM DAYTRADER
            try:
                gate_info = self._get_v7_entry_head_info(observation, detailed=True)
                if gate_info:
                    self.last_v7_gate_info = gate_info
                    # 🎯 CRIAR ESTRUTURA COMPATÍVEL COM FILTROS V7 INTUITION
                    self.last_v7_outputs = {
                        'gates': {
                            'entry_conf': float(gate_info.get('confidence_gate', 0.5)),
                            'mgmt_conf': float(gate_info.get('validation_gate', 0.5)),
                            'regime_id': 2,  # Default: sideways (pode ser melhorado)
                            'regime_name': 'sideways',
                            'specialization_divergence': float(1.0 - gate_info.get('composite_score', 0.5)),
                            'actor_attention_mean': float(gate_info.get('temporal_gate', 0.5)),
                            'critic_attention_mean': float(gate_info.get('risk_gate', 0.5))
                        }
                    }
            except Exception as e:
                self._log(f"[⚠️ V7 GATES] Erro ao capturar gates: {e}")
                self.last_v7_gate_info = None
                self.last_v7_outputs = None
            
            # Incrementar contador de debug
            self.debug_step_counter += 1
            
            # 🔥 DEBUG DA ENTRY HEAD CONTROLADO POR INTERVALO
            if self.debug_step_counter % self.debug_composite_interval == 0:
                try:
                    gate_info = self._get_v7_entry_head_info(observation, detailed=True)
                    if gate_info:
                        self._log_composite_debug(gate_info, self.debug_step_counter)
                    else:
                        # 🔥 FALLBACK: Mostrar pelo menos informações básicas das predições
                        try:
                            obs = self._get_observation_v7()
                            if self.model and obs is not None:
                                action, _states = self.model.predict(obs, deterministic=True)
                                self._log(f"[🎯 V7 BASIC] Step {self.debug_step_counter} | Action[0]:{action[0]:.2f} | Quality:{action[1]:.2f} | Risk:{action[3]:.2f}")
                            else:
                                self._log(f"[⚠️ V7 DEBUG] Gate info não disponível - Step {self.debug_step_counter}")
                        except:
                            self._log(f"[⚠️ V7 DEBUG] Gate info não disponível - Step {self.debug_step_counter}")
                except Exception as e:
                    self._log(f"[❌ V7 DEBUG] Erro ao capturar gate info: {e}")
            
            # Debug de observações anômalas a cada 50 steps
            if self.debug_step_counter % self.debug_anomaly_interval == 0:
                try:
                    self._debug_anomalous_observations_v7(original_obs, observation, self.debug_step_counter)
                except Exception as e:
                    self._log(f"[❌ OBS DEBUG] Erro no debug de anomalias: {e}")
            
            # Manter buffer das últimas observações para análise de anomalias
            self._update_observation_buffer(observation)
            
            return action
            
        except Exception as e:
            self._log(f"[❌ V7 PREDICT] Erro na predição: {e}")
            return None
    
    def _get_v7_entry_head_info(self, observation, detailed=False):
        """🎯 Acessar informações do Entry Head V7"""
        try:
            # Verificar se modelo tem política V7 com Entry Head
            if hasattr(self.model, 'policy') and hasattr(self.model.policy, 'entry_head'):
                policy = self.model.policy
                
                # Preparar entrada para o Entry Head
                import torch
                if isinstance(observation, np.ndarray):
                    obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                else:
                    obs_tensor = observation
                
                # 🔥 FIX: Detectar device do modelo e mover tensors para o mesmo device
                model_device = next(policy.parameters()).device
                obs_tensor = obs_tensor.to(model_device)
                
                # Acessar Entry Head diretamente (modo eval)
                policy.eval()
                with torch.no_grad():
                    # Usar dados reais da observação processados pelo features extractor
                    try:
                        # Simular processamento através do pipeline completo
                        features_dim = policy.v7_shared_lstm_hidden if hasattr(policy, 'v7_shared_lstm_hidden') else 256
                        # Entry Head concatena os 3 sinais e espera input_dim total = features_dim
                        # Então: entry_signal + management_signal + market_context = features_dim
                        # Usando: 112 + 112 + 32 = 256
                        signal_dim = (features_dim - 32) // 2  # 112 para cada signal
                        
                        # 🔥 MÉTODO SIMPLIFICADO: Predição direta para capturar comportamento
                        obs_real = self._get_observation_v7()
                        if obs_real is None:
                            return None
                            
                        action, _states = self.model.predict(obs_real, deterministic=True)
                        
                        # Criar mock gate_info baseado nas ações (funciona sempre)
                        if len(action) >= 4:
                            entry_decision = action[0]
                            entry_quality = action[1] 
                            temporal_signal = action[2] if len(action) > 2 else 0.0
                            risk_appetite = action[3] if len(action) > 3 else 0.5
                            
                            # Mock realista baseado nas ações
                            gate_info = {
                                'temporal_gate': torch.tensor(abs(float(temporal_signal))),
                                'validation_gate': torch.tensor(float(entry_quality)),
                                'risk_gate': torch.tensor(float(risk_appetite)),
                                'market_gate': torch.tensor(0.5),  # Neutro
                                'quality_gate': torch.tensor(float(entry_quality)),
                                'confidence_gate': torch.tensor(float(entry_quality)),
                                'entry_decision': float(entry_decision),
                                'entry_quality': float(entry_quality),
                                'temporal_signal': float(temporal_signal),
                                'risk_appetite': float(risk_appetite)
                            }
                        else:
                            return None
                        
                        # Calcular composite score (como na policy)
                        if gate_info and detailed:
                            temporal_gate = gate_info.get('temporal_gate', torch.tensor(0.0, device=model_device))
                            validation_gate = gate_info.get('validation_gate', torch.tensor(0.0, device=model_device))
                            risk_gate = gate_info.get('risk_gate', torch.tensor(0.0, device=model_device))
                            market_gate = gate_info.get('market_gate', torch.tensor(0.0, device=model_device))
                            quality_gate = gate_info.get('quality_gate', torch.tensor(0.0, device=model_device))
                            confidence_gate = gate_info.get('confidence_gate', torch.tensor(0.0, device=model_device))
                            
                            # Composite score calculation (same as V7 policy)
                            composite_score = (
                                temporal_gate * 0.20 +      # 20% - timing
                                validation_gate * 0.20 +    # 20% - validation
                                risk_gate * 0.25 +          # 25% - risk
                                market_gate * 0.10 +        # 10% - market conditions
                                quality_gate * 0.10 +       # 10% - technical quality
                                confidence_gate * 0.15      # 15% - confidence
                            )
                            
                            gate_info['composite_score'] = composite_score
                            gate_info['threshold'] = 0.6  # V7 threshold
                            gate_info['passes_threshold'] = composite_score > 0.6
                        
                        return gate_info
                        
                    except Exception as inner_e:
                        # Fallback to simpler method
                        signal_dim = (features_dim - 32) // 2  # 112 para cada signal
                        dummy_entry_signal = torch.randn(1, signal_dim, device=model_device)
                        dummy_mgmt_signal = torch.randn(1, signal_dim, device=model_device)
                        dummy_context = torch.randn(1, 32, device=model_device)
                        _, confidence, gate_info = policy.entry_head(dummy_entry_signal, dummy_mgmt_signal, dummy_context)
                        return gate_info
                    
            return None
            
        except Exception as e:
            # Silencioso - Entry Head não acessível diretamente
            return None
    
    def _log_composite_debug(self, gate_info, step):
        """📊 Debug TEMPO REAL do Composite Gate V7"""
        try:
            if gate_info and isinstance(gate_info, dict):
                # Extrair informações do composite
                composite_score = gate_info.get('composite_score', torch.tensor(0.0))
                threshold = gate_info.get('threshold', 0.6)
                passes = gate_info.get('passes_threshold', False)
                
                # Extrair gates individuais
                temporal = gate_info.get('temporal_gate', torch.tensor(0.0))
                validation = gate_info.get('validation_gate', torch.tensor(0.0))
                risk = gate_info.get('risk_gate', torch.tensor(0.0))
                market = gate_info.get('market_gate', torch.tensor(0.0))
                quality = gate_info.get('quality_gate', torch.tensor(0.0))
                confidence = gate_info.get('confidence_gate', torch.tensor(0.0))
                
                if hasattr(composite_score, 'item'):
                    score_val = composite_score.item()
                    passes_str = "✅ PASS" if passes else "❌ BLOCK"
                    
                    # 🔥 LOG ULTRA-COMPACTO PARA REDUZIR SPAM
                    final_gate = gate_info.get('final_gate', torch.tensor(0.0))
                    final_status = "�G" if final_gate.item() > 0.5 else "🔴"
                    
                    # 🎯 LOG FOCUSED ON ENTRY QUALITY (mais útil)
                    entry_decision = gate_info.get('entry_decision', 0)
                    entry_quality = gate_info.get('entry_quality', 0)
                    temporal_signal = gate_info.get('temporal_signal', 0)
                    risk_appetite = gate_info.get('risk_appetite', 0)
                    
                    # Decision interpretation
                    decision_str = "HOLD" if entry_decision < 0.5 else ("LONG" if entry_decision < 1.5 else "SHORT")
                    quality_stars = "⭐" * min(5, int(entry_quality * 5))
                    
                    self._log(f"[🎯 V7 QUALITY] Step {step} | {decision_str} | Quality: {entry_quality:.2f} {quality_stars}")
                    self._log(f"    🔹 Composite: {score_val:.2f} {passes_str} | Temporal: {temporal_signal:.2f} | Risk: {risk_appetite:.2f}")
                    self._log(f"    🔹 Gates: T:{temporal.item():.2f} V:{validation.item():.2f} R:{risk.item():.2f} Q:{quality.item():.2f}")
                    
                    # Análise de contribuição
                    contributions = {
                        'temporal': temporal.item() * 0.20,
                        'validation': validation.item() * 0.20,
                        'risk': risk.item() * 0.25,
                        'market': market.item() * 0.10,
                        'quality': quality.item() * 0.10,
                        'confidence': confidence.item() * 0.15
                    }
                    
                    # Encontrar maior e menor contribuidor
                    max_contrib = max(contributions.items(), key=lambda x: x[1])
                    min_contrib = min(contributions.items(), key=lambda x: x[1])
                    
                    self._log(f"    🔹 Best: {max_contrib[0]}({max_contrib[1]:.3f}) | Worst: {min_contrib[0]}({min_contrib[1]:.3f})")
                    
                    # Mostrar scores individuais detalhados
                    scores = gate_info.get('scores', {})
                    if scores:
                        self._log(f"    🔹 Detailed: MTF:{scores.get('mtf', torch.tensor(0.0)).item():.2f} Pattern:{scores.get('pattern', torch.tensor(0.0)).item():.2f} Lookahead:{scores.get('lookahead', torch.tensor(0.0)).item():.2f} Fatigue:{scores.get('fatigue', torch.tensor(0.0)).item():.2f}")
                    
                    # Mostrar final gate status
                    final_gate = gate_info.get('final_gate', torch.tensor(0.0))
                    if hasattr(final_gate, 'item'):
                        final_status = "🟢 OPEN" if final_gate.item() > 0.5 else "🔴 CLOSED"
                        self._log(f"    🔹 Final Gate: {final_gate.item():.3f} | {final_status}")
                    
        except Exception as e:
            self._log(f"[❌ DEBUG] Erro no log composite: {e}")
    
    def _update_observation_buffer(self, observation):
        """🔄 Atualizar buffer de observações para análise de anomalias"""
        try:
            if isinstance(observation, np.ndarray):
                # Manter apenas as últimas 50 observações
                self.last_observations.append(observation.copy())
                if len(self.last_observations) > 50:
                    self.last_observations.pop(0)
                
                # Atualizar estatísticas se temos observações suficientes
                if len(self.last_observations) >= 10:
                    obs_array = np.array(self.last_observations)
                    self.obs_stats = {
                        'mean': np.mean(obs_array, axis=0),
                        'std': np.std(obs_array, axis=0),
                        'min': np.min(obs_array, axis=0),
                        'max': np.max(obs_array, axis=0)
                    }
        except Exception as e:
            pass
    
    def _debug_anomalous_observations_v7(self, original_obs, normalized_obs, step):
        """🔍 Debug V7 - Qualidade das observações ANTES e DEPOIS da normalização (a cada 50 steps)"""
        try:
            self._log(f"[🔍 V7 QUALITY DEBUG - Step {step}] Análise PRÉ e PÓS normalização")
            self._log("=" * 80)
            
            # 🔥 ANÁLISE PRÉ-NORMALIZAÇÃO (Observação Original)
            if isinstance(original_obs, np.ndarray):
                orig_mean = np.mean(original_obs)
                orig_std = np.std(original_obs)
                orig_min = np.min(original_obs)
                orig_max = np.max(original_obs)
                orig_zeros = np.sum(original_obs == 0.0)
                orig_nans = np.sum(np.isnan(original_obs))
                orig_infs = np.sum(np.isinf(original_obs))
                
                # 🔥 ANÁLISE DETALHADA DE VALORES
                orig_very_small = np.sum(np.abs(original_obs) < 0.001)  # Valores muito pequenos
                orig_small = np.sum((np.abs(original_obs) >= 0.001) & (np.abs(original_obs) < 0.1))  # Pequenos
                orig_normal = np.sum((np.abs(original_obs) >= 0.1) & (np.abs(original_obs) < 10.0))  # Normais
                orig_large = np.sum((np.abs(original_obs) >= 10.0) & (np.abs(original_obs) < 100.0))  # Grandes
                orig_extreme = np.sum(np.abs(original_obs) >= 100.0)  # Extremos
                
                self._log(f"📊 [PRÉ-NORM] Stats: mean={orig_mean:.4f} std={orig_std:.4f} range=[{orig_min:.4f}, {orig_max:.4f}]")
                self._log(f"🔍 [PRÉ-NORM] Qualidade: zeros={orig_zeros} nans={orig_nans} infs={orig_infs} total={len(original_obs)}")
                self._log(f"📏 [PRÉ-NORM] Distribuição: muito_pequenos={orig_very_small} pequenos={orig_small} normais={orig_normal} grandes={orig_large} extremos={orig_extreme}")
                
                # Detectar features problemáticas PRÉ-normalização
                extreme_orig = np.where(np.abs(original_obs) > 100.0)[0]
                if len(extreme_orig) > 0:
                    self._log(f"🚨 [PRÉ-NORM] {len(extreme_orig)} features extremas (|val| > 100)")
                    for idx in extreme_orig[:3]:  # Mostrar apenas 3
                        self._log(f"    Feature[{idx}]: {original_obs[idx]:.4f}")
                
                # Análise de distribuição PRÉ-normalização
                percentiles_orig = np.percentile(original_obs, [1, 5, 25, 50, 75, 95, 99])
                self._log(f"📈 [PRÉ-NORM] Percentis: P1={percentiles_orig[0]:.3f} P5={percentiles_orig[1]:.3f} P25={percentiles_orig[2]:.3f} P50={percentiles_orig[3]:.3f} P75={percentiles_orig[4]:.3f} P95={percentiles_orig[5]:.3f} P99={percentiles_orig[6]:.3f}")
            
            # 🔥 ANÁLISE PÓS-NORMALIZAÇÃO (Observação Normalizada)
            if isinstance(normalized_obs, np.ndarray):
                norm_mean = np.mean(normalized_obs)
                norm_std = np.std(normalized_obs)
                norm_min = np.min(normalized_obs)
                norm_max = np.max(normalized_obs)
                norm_zeros = np.sum(normalized_obs == 0.0)
                norm_nans = np.sum(np.isnan(normalized_obs))
                norm_infs = np.sum(np.isinf(normalized_obs))
                
                # 🔥 ANÁLISE DETALHADA DE VALORES PÓS-NORMALIZAÇÃO
                norm_very_small = np.sum(np.abs(normalized_obs) < 0.001)  # Valores muito pequenos
                norm_small = np.sum((np.abs(normalized_obs) >= 0.001) & (np.abs(normalized_obs) < 0.1))  # Pequenos
                norm_normal = np.sum((np.abs(normalized_obs) >= 0.1) & (np.abs(normalized_obs) < 3.0))  # Normais
                norm_large = np.sum((np.abs(normalized_obs) >= 3.0) & (np.abs(normalized_obs) < 10.0))  # Grandes
                norm_extreme = np.sum(np.abs(normalized_obs) >= 10.0)  # Extremos
                
                self._log(f"📊 [PÓS-NORM] Stats: mean={norm_mean:.4f} std={norm_std:.4f} range=[{norm_min:.4f}, {norm_max:.4f}]")
                self._log(f"🔍 [PÓS-NORM] Qualidade: zeros={norm_zeros} nans={norm_nans} infs={norm_infs} total={len(normalized_obs)}")
                self._log(f"📏 [PÓS-NORM] Distribuição: muito_pequenos={norm_very_small} pequenos={norm_small} normais={norm_normal} grandes={norm_large} extremos={norm_extreme}")
                
                # Detectar features problemáticas PÓS-normalização
                extreme_norm = np.where(np.abs(normalized_obs) > 10.0)[0]
                if len(extreme_norm) > 0:
                    self._log(f"🚨 [PÓS-NORM] {len(extreme_norm)} features extremas (|val| > 10)")
                    for idx in extreme_norm[:3]:  # Mostrar apenas 3
                        self._log(f"    Feature[{idx}]: {normalized_obs[idx]:.4f}")
                
                # Análise de distribuição PÓS-normalização
                percentiles_norm = np.percentile(normalized_obs, [1, 5, 25, 50, 75, 95, 99])
                self._log(f"📈 [PÓS-NORM] Percentis: P1={percentiles_norm[0]:.3f} P5={percentiles_norm[1]:.3f} P25={percentiles_norm[2]:.3f} P50={percentiles_norm[3]:.3f} P75={percentiles_norm[4]:.3f} P95={percentiles_norm[5]:.3f} P99={percentiles_norm[6]:.3f}")
                
                # 🔥 COMPARAÇÃO E IMPACTO DA NORMALIZAÇÃO
                if isinstance(original_obs, np.ndarray) and len(original_obs) == len(normalized_obs):
                    # Calcular mudanças causadas pela normalização
                    abs_change = np.mean(np.abs(normalized_obs - original_obs))
                    relative_change = abs_change / (np.mean(np.abs(original_obs)) + 1e-8)
                    
                    # Features que mudaram drasticamente
                    big_changes = np.where(np.abs(normalized_obs - original_obs) > 5.0)[0]
                    
                    self._log(f"🔄 [IMPACTO] Mudança absoluta média: {abs_change:.4f}")
                    self._log(f"🔄 [IMPACTO] Mudança relativa média: {relative_change:.4f}")
                    
                    if len(big_changes) > 0:
                        self._log(f"⚡ [IMPACTO] {len(big_changes)} features com mudanças drásticas (>5.0)")
                        for idx in big_changes[:3]:  # Mostrar apenas 3
                            orig_val = original_obs[idx]
                            norm_val = normalized_obs[idx]
                            change = norm_val - orig_val
                            self._log(f"    Feature[{idx}]: {orig_val:.3f} → {norm_val:.3f} (Δ={change:.3f})")
                    
                    # 🔥 ANÁLISE COMPARATIVA DETALHADA
                    zero_change = orig_zeros - norm_zeros
                    extreme_change = orig_extreme - norm_extreme
                    
                    self._log(f"🔄 [COMPARATIVO] Zeros: {orig_zeros} → {norm_zeros} (Δ={zero_change:+d})")
                    self._log(f"🔄 [COMPARATIVO] Extremos: {orig_extreme} → {norm_extreme} (Δ={extreme_change:+d})")
                    
                    # Efetividade da normalização
                    if orig_std > 0:
                        normalization_effectiveness = (orig_std - norm_std) / orig_std * 100
                        self._log(f"📈 [EFETIVIDADE] Redução de std: {normalization_effectiveness:.1f}%")
                    
                    # Centralização
                    centering_improvement = abs(orig_mean) - abs(norm_mean)
                    self._log(f"🎯 [CENTRALIZAÇÃO] Melhoria na centralização: {centering_improvement:+.4f}")
                
                # 🎯 AVALIAÇÃO DE QUALIDADE GERAL
                quality_score = 100.0
                
                # Penalizar NaNs e Infs
                if norm_nans > 0:
                    quality_score -= min(50.0, norm_nans * 5.0)
                    self._log(f"❌ [QUALIDADE] Penalidade por NaNs: -{min(50.0, norm_nans * 5.0):.1f}")
                
                if norm_infs > 0:
                    quality_score -= min(30.0, norm_infs * 3.0)
                    self._log(f"❌ [QUALIDADE] Penalidade por Infs: -{min(30.0, norm_infs * 3.0):.1f}")
                
                # Penalizar valores extremos
                if len(extreme_norm) > 0:
                    extreme_penalty = min(20.0, len(extreme_norm) * 2.0)
                    quality_score -= extreme_penalty
                    self._log(f"⚠️ [QUALIDADE] Penalidade por extremos: -{extreme_penalty:.1f}")
                
                # Avaliar distribuição (ideal: mean~0, std~1 para dados normalizados)
                if abs(norm_mean) > 0.5:
                    mean_penalty = min(10.0, abs(norm_mean) * 10.0)
                    quality_score -= mean_penalty
                    self._log(f"📊 [QUALIDADE] Penalidade por mean descentrado: -{mean_penalty:.1f}")
                
                quality_score = max(0.0, quality_score)
                
                if quality_score >= 90:
                    quality_status = "🟢 EXCELENTE"
                elif quality_score >= 75:
                    quality_status = "🟡 BOA"
                elif quality_score >= 50:
                    quality_status = "🟠 REGULAR"
                else:
                    quality_status = "🔴 RUIM"
                
                self._log(f"🎯 [QUALIDADE FINAL] Score: {quality_score:.1f}/100 - Status: {quality_status}")
            
            self._log("=" * 80)
            
        except Exception as e:
            self._log(f"[❌ QUALITY DEBUG] Erro na análise V7: {e}")
            import traceback
            self._log(f"[❌ QUALITY DEBUG] Traceback: {traceback.format_exc()}")
    
    def _debug_anomalous_observations(self, current_obs, step):
        """🔍 Debug de observações anômalas (LEGACY - mantido para compatibilidade)"""
        # Redirecionar para a nova função V7
        self._debug_anomalous_observations_v7(current_obs, current_obs, step)
    
    def _log_entry_head_analysis(self, gate_info):
        """📊 Log básico da análise do Entry Head V7 (legacy)"""
        try:
            if gate_info and isinstance(gate_info, dict):
                # Extrair principais gates
                temporal = gate_info.get('temporal_gate', torch.tensor(0.0))
                validation = gate_info.get('validation_gate', torch.tensor(0.0))
                risk = gate_info.get('risk_gate', torch.tensor(0.0))
                confidence = gate_info.get('confidence_gate', torch.tensor(0.0))
                
                if hasattr(temporal, 'item'):
                    self._log(f"[🎯 V7 ENTRY HEAD] T:{temporal.item():.2f} V:{validation.item():.2f} R:{risk.item():.2f} C:{confidence.item():.2f}")
                    
        except Exception as e:
            # Silent fail - Entry Head analysis opcional
            pass
    
    def _verify_entry_head_v7(self):
        """🔍 Verificar se Entry Head V7 está disponível e funcional"""
        try:
            if not self.model or not hasattr(self.model, 'policy'):
                return "❌ Modelo não disponível"
            
            policy = self.model.policy
            
            # Verificar se tem Entry Head
            if not hasattr(policy, 'entry_head'):
                return "❌ Entry Head não encontrado"
            
            entry_head = policy.entry_head
            
            # Verificar gates especializados
            required_gates = [
                'horizon_analyzer', 'mtf_validator', 'pattern_memory_validator',
                'risk_gate_entry', 'regime_gate', 'lookahead_gate', 'fatigue_detector',
                'momentum_filter', 'volatility_filter', 'volume_filter', 'trend_strength_filter',
                'confidence_estimator'
            ]
            
            missing_gates = []
            for gate in required_gates:
                if not hasattr(entry_head, gate):
                    missing_gates.append(gate)
            
            if missing_gates:
                return f"⚠️ Missing gates: {len(missing_gates)}/{len(required_gates)}"
            
            # Verificar se é instância correta
            class_name = entry_head.__class__.__name__
            if class_name != 'SpecializedEntryHead':
                return f"⚠️ Wrong class: {class_name}"
            
            return "✅ SpecializedEntryHead com 11 gates ativos"
            
        except Exception as e:
            return f"❌ Erro na verificação: {str(e)[:30]}..."
    
    def reload_v7_model(self):
        """🔄 Recarregar modelo V7"""
        try:
            self._log("[🔄 V7 RELOAD] Recarregando modelo...")
            
            # Limpar modelo atual
            self.model = None
            self.normalizer = None
            self.model_loaded = False
            self.model_metadata = None
            
            # Recarregar
            return self.auto_load_v7_model()
            
        except Exception as e:
            self._log(f"[❌ V7 RELOAD] Erro no recarregamento: {e}")
            return False
    
    def _start_live_trading(self):
        """🚀 Iniciar trading ao vivo com dados reais do MT5"""
        try:
            self._log("[🚀 LIVE TRADING] Iniciando modo automatizado...")
            step_count = 0
            last_ping_time = time.time()
            
            while True:
                try:
                    # Sistema de ping a cada 10 minutos (reduzir spam nos logs)
                    current_time = time.time()
                    if current_time - last_ping_time >= 600:  # 10 minutos
                        if self.mt5_connected:
                            account_info = mt5.account_info()
                            tick = mt5.symbol_info_tick(self.symbol)
                            positions = mt5.positions_get(symbol=self.symbol) or []
                            
                            # Sistema ativo - logs reduzidos para evitar spam
                            if step_count % 100 == 0:  # Log apenas a cada 100 steps
                                self._log(f"[💓 PING] Sistema ativo - Step {step_count}")
                                if account_info and tick:
                                    self._log(f"[💰 CONTA] ${account_info.balance:.2f} | {len(positions)} posições")
                        last_ping_time = current_time
                    
                    # 🔥 OBTER OBSERVAÇÃO REAL DO MT5
                    obs = self._get_observation_v7()
                    
                    # 🔥 OBTER PREDIÇÃO DO MODELO (COM LOGS DA ENTRY HEAD)
                    action = self.get_model_prediction(obs)
                    
                    if action is not None:
                        # 🔥 PROCESSAR AÇÃO DO MODELO V7
                        action_analysis = self._process_v7_action(action)
                        
                        # 🔥 EXECUTAR DECISÃO NO MT5
                        if self.mt5_connected:
                            self._execute_v7_decision(action_analysis)
                        else:
                            self._log(f"[SIMULAÇÃO] {action_analysis['action_name']} - MT5 desconectado")
                    
                    step_count += 1
                    
                    # 🔥 DELAY PARA EVITAR COOLDOWN E SPAM (2 segundos)
                    time.sleep(2)
                    
                except KeyboardInterrupt:
                    self._log("👋 Trading interrompido pelo usuário")
                    break
                except Exception as e:
                    self._log(f"[❌ TRADING] Erro no loop: {e}")
                    time.sleep(60)  # Aguardar 1 minuto em caso de erro
                    
        except Exception as e:
            self._log(f"[❌ LIVE TRADING] Erro fatal: {e}")
    
    def _start_passive_monitoring(self):
        """📊 Monitoramento passivo sem trading"""
        try:
            self._log("[📊 MONITORING] Modo monitoramento passivo...")
            
            while True:
                try:
                    if self.mt5_connected:
                        account_info = mt5.account_info()
                        tick = mt5.symbol_info_tick(self.symbol)
                        positions = mt5.positions_get(symbol=self.symbol) or []
                        
                        if account_info and tick:
                            # Monitor passivo - spam removido
                            pass
                    
                    # 🔥 COMPORTAMENTO PURO - SEM DELAYS NO MONITORAMENTO
                    pass  # Monitoramento contínuo sem delays
                    
                except KeyboardInterrupt:
                    self._log("👋 Monitoramento interrompido pelo usuário")
                    break
                except Exception as e:
                    self._log(f"[❌ MONITOR] Erro: {e}")
                    time.sleep(60)
                    
        except Exception as e:
            self._log(f"[❌ MONITORING] Erro fatal: {e}")
    
    def _process_v7_action(self, action):
        """🧠 Processar ação do modelo V7 - ACTION SPACE 11D"""
        try:
            if not isinstance(action, (list, tuple, np.ndarray)):
                action = np.array([action])
            
            # Garantir 11 dimensões para compatibilidade V7
            if len(action) < 11:
                action = np.pad(action, (0, 11 - len(action)), mode='constant')
            
            # V7 ACTION SPACE: [entry_decision, confidence, temporal_signal, risk_appetite, market_regime_bias, sl1, sl2, sl3, tp1, tp2, tp3]
            entry_decision = int(np.clip(action[0], 0, 2))  # 0=HOLD, 1=LONG, 2=SHORT
            entry_quality = float(np.clip(action[1], 0, 1))  # [0,1] Qualidade
            temporal_signal = float(np.clip(action[2], -1, 1))  # [-1,1] Sinal temporal
            risk_appetite = float(np.clip(action[3], 0, 1))  # [0,1] Apetite ao risco
            market_regime_bias = float(np.clip(action[4], -1, 1))  # [-1,1] Viés do regime
            
            # SL/TP para cada posição ([-3,3] → pontos reais)
            sl_adjusts = [float(action[i]) for i in range(5, 8)]  # [5-7] SL positions
            tp_adjusts = [float(action[i]) for i in range(8, 11)]  # [8-10] TP positions
            
            # Converter [-3,3] para pontos reais usando a mesma lógica do daytrader.py
            sl_points = []
            tp_points = []
            
            for i in range(3):  # Para cada posição
                # SL: normalizar [-3,3] para [2,8] pontos (daytrader.py)
                sl_val = self.sl_range_min + (sl_adjusts[i] + 3) * (self.sl_range_max - self.sl_range_min) / 6
                sl_val = round(sl_val * 2) / 2  # Arredondar para múltiplos de 0.5
                sl_val = np.clip(sl_val, self.sl_range_min, self.sl_range_max)
                sl_points.append(sl_val)
                
                # TP: normalizar [-3,3] para [3,15] pontos (daytrader.py)
                tp_val = self.tp_range_min + (tp_adjusts[i] + 3) * (self.tp_range_max - self.tp_range_min) / 6
                tp_val = round(tp_val * 2) / 2  # Arredondar para múltiplos de 0.5
                tp_val = np.clip(tp_val, self.tp_range_min, self.tp_range_max)
                tp_points.append(tp_val)
            
            # 🔥 FILTROS V7 - Verificações antes da execução
            
            # Filter 1: Verificar limite de posições (SILENCIOSO)
            if entry_decision in [1, 2]:  # BUY ou SELL
                if self.mt5_connected:
                    current_positions = mt5.positions_get(symbol=self.symbol)
                    pos_count = len(current_positions) if current_positions else 0
                    if pos_count >= self.max_positions:
                        entry_decision = 0  # Forçar HOLD silenciosamente
            
            # 🚀 SEGUNDO FILTRO DE QUALITY REMOVIDO - ALINHADO COM DAYTRADER (apenas 0.4)
            # if entry_decision in [1, 2] and entry_quality < 0.6:
            #     self._log(f"🚫 [V7-FILTER] Entry Quality baixa: {entry_quality:.2f} < 0.6 - Forçando HOLD")
            #     entry_decision = 0  # Forçar HOLD
            
            # Mapear ação para nome
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action_name = action_names.get(entry_decision, 'UNKNOWN')
            
            # Calcular position size baseado APENAS no apetite ao risco (quality não afeta volume)
            position_size = risk_appetite
            
            return {
                'entry_decision': entry_decision,
                'entry_quality': entry_quality,
                'temporal_signal': temporal_signal,
                'risk_appetite': risk_appetite,
                'market_regime_bias': market_regime_bias,
                'position_size': position_size,
                'sl_adjusts': sl_adjusts,
                'tp_adjusts': tp_adjusts,
                'sl_points': sl_points,  # Pontos reais para MT5
                'tp_points': tp_points,  # Pontos reais para MT5
                'action_name': action_name,
                'raw_action': action.tolist()  # Manter ação original para debug
            }
            
        except Exception as e:
            self._log(f"❌ [V7-ACTION] Erro ao processar ação: {e}")
            return {
                'entry_decision': 0,
                'entry_quality': 0.0,
                'temporal_signal': 0.0,
                'risk_appetite': 0.0,
                'market_regime_bias': 0.0,
                'position_size': 0.0,
                'sl_adjusts': [0.0, 0.0, 0.0],
                'tp_adjusts': [0.0, 0.0, 0.0],
                'sl_points': [0.0, 0.0, 0.0],
                'tp_points': [0.0, 0.0, 0.0],
                'action_name': 'HOLD',
                'raw_action': [0.0] * 11
            }
    
    def _execute_v7_decision(self, action_analysis):
        """🧠 Executar decisão do modelo V7 no MT5"""
        try:
            if not self.mt5_connected:
                self._log("⚠️ [V7-EXECUÇÃO] MT5 não conectado - simulação apenas")
                return
                
            action_name = action_analysis['action_name']
            entry_quality = abs(action_analysis['entry_quality'])
            
            # V7 LOG: Mostrar ação processada (detalhado apenas para BUY/SELL)
            if action_name in ['BUY', 'SELL']:
                temporal = action_analysis.get('temporal_signal', 0.0)
                risk_app = action_analysis.get('risk_appetite', 0.0)
                regime_bias = action_analysis.get('market_regime_bias', 0.0)
                self._log(f"🧠 [V7-DECISION] {action_name} | Quality: {entry_quality:.3f} | Temporal: {temporal:.3f} | Risk: {risk_app:.3f} | Regime: {regime_bias:.3f}")
                self._log(f"🎯 [V7-ATTEMPT] Tentando executar {action_name} - verificando condições...")
            
            # Obter preço atual
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                self._log("[❌ ERRO] Não foi possível obter preço atual")
                return
            
            current_price = tick.bid
            
            # Executar ordem baseada na decisão
            if action_name == 'BUY':
                self.hold_count = 0  # Reset contador de HOLD
                self._execute_buy_order_v7(current_price, entry_quality, action_analysis)
            elif action_name == 'SELL':
                self.hold_count = 0  # Reset contador de HOLD
                self._execute_sell_order_v7(current_price, entry_quality, action_analysis)
            else:
                # HOLD - modelo decidiu não fazer nada (silencioso quando há 3 posições)
                self.hold_count += 1
                
                # Verificar se há posições abertas para decidir se loga
                if self.mt5_connected:
                    current_positions = mt5.positions_get(symbol=self.symbol)
                    pos_count = len(current_positions) if current_positions else 0
                    
                    # Só loga HOLD se não tiver 3 posições abertas
                    if pos_count < self.max_positions:
                        if self.hold_count % self.hold_log_interval == 0:
                            self._log(f"📊 [V7-EXECUÇÃO] HOLD x{self.hold_count} - Modelo aguardando oportunidade")
                        elif self.hold_count == 1:
                            self._log(f"📊 [V7-EXECUÇÃO] HOLD - Modelo decidiu não operar")
                else:
                    # Se MT5 não conectado, loga normalmente
                    if self.hold_count % self.hold_log_interval == 0:
                        self._log(f"📊 [V7-EXECUÇÃO] HOLD x{self.hold_count} - Modelo aguardando oportunidade")
                    elif self.hold_count == 1:
                        self._log(f"📊 [V7-EXECUÇÃO] HOLD - Modelo decidiu não operar")
                
        except Exception as e:
            self._log(f"❌ [V7-EXECUÇÃO] Erro ao executar decisão: {e}")
    
    def _execute_buy_order_v7(self, current_price, entry_quality, action_analysis):
        """🧠 Executar ordem de compra V7 - com SL/TP inteligentes"""
        try:
            # Verificar limite de posições (máximo 3)
            if self.mt5_connected:
                current_positions = mt5.positions_get(symbol=self.symbol)
                if current_positions and len(current_positions) >= self.max_positions:
                    self._log(f"⚠️ [LIMITE] Máximo de {self.max_positions} posições atingido - COMPRA bloqueada")
                    return
            
            # Calcular volume baseado na entry quality
            volume = self._calculate_volume_by_confidence_v7(entry_quality)
            
            # Usar SL/TP do modelo se disponível
            if action_analysis and 'sl_points' in action_analysis and 'tp_points' in action_analysis:
                # Usar primeiro SL/TP do modelo (posição 0)
                sl_points = abs(action_analysis['sl_points'][0])  # Garantir positivo
                tp_points = abs(action_analysis['tp_points'][0])  # Garantir positivo
                
                # Aplicar limites de segurança (alinhado com daytrader.py)
                sl_points = np.clip(sl_points, self.sl_range_min, self.sl_range_max)  # 2-8 pontos
                tp_points = np.clip(tp_points, self.tp_range_min, self.tp_range_max)  # 3-15 pontos
                
                sl_price = current_price - (sl_points * 1.0)  # SL abaixo do preço (OURO: 1 ponto = $1.00)
                tp_price = current_price + (tp_points * 1.0)  # TP acima do preço (OURO: 1 ponto = $1.00)
                
                self._log(f"🧠 [V7-BUY] Usando SL/TP do modelo: SL={sl_points:.1f}pts | TP={tp_points:.1f}pts")
                self._log(f"💰 [V7-BUY] Preços calculados: Atual={current_price:.2f} | SL={sl_price:.2f} | TP={tp_price:.2f}")
            else:
                # Fallback para valores padrão
                sl_price = current_price - (5 * 1.0)  # 5 pontos SL (OURO: 1 ponto = $1.00)
                tp_price = current_price + (10 * 1.0)  # 10 pontos TP (OURO: 1 ponto = $1.00)
                self._log(f"📊 [BUY] Usando SL/TP padrão: SL=5pts | TP=10pts")
            
            # Executar ordem
            result = self._execute_order_v7(mt5.ORDER_TYPE_BUY, volume, sl_price, tp_price)
            
            if "SUCCESS" in result:
                # Verificar número atual de posições após execução
                current_positions = mt5.positions_get(symbol=self.symbol) if self.mt5_connected else []
                pos_count = len(current_positions) if current_positions else 0
                self._log(f"✅ [COMPRA V7] Ordem executada! Vol: {volume} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
                self._log(f"📊 [POSIÇÕES] Total atual: {pos_count}/{self.max_positions}")
            else:
                self._log(f"❌ [COMPRA V7] Falha na execução: {result}")
                
        except Exception as e:
            self._log(f"❌ [COMPRA V7] Erro: {e}")
    
    def _execute_sell_order_v7(self, current_price, entry_quality, action_analysis):
        """🧠 Executar ordem de venda V7 - com SL/TP inteligentes"""
        try:
            # Verificar limite de posições (máximo 3)
            if self.mt5_connected:
                current_positions = mt5.positions_get(symbol=self.symbol)
                if current_positions and len(current_positions) >= self.max_positions:
                    self._log(f"⚠️ [LIMITE] Máximo de {self.max_positions} posições atingido - VENDA bloqueada")
                    return
            
            # Calcular volume baseado na entry quality
            volume = self._calculate_volume_by_confidence_v7(entry_quality)
            
            # Usar SL/TP do modelo se disponível
            if action_analysis and 'sl_points' in action_analysis and 'tp_points' in action_analysis:
                # Usar primeiro SL/TP do modelo (posição 0)
                sl_points = abs(action_analysis['sl_points'][0])  # Garantir positivo
                tp_points = abs(action_analysis['tp_points'][0])  # Garantir positivo
                
                # Aplicar limites de segurança (alinhado com daytrader.py)
                sl_points = np.clip(sl_points, self.sl_range_min, self.sl_range_max)  # 2-8 pontos
                tp_points = np.clip(tp_points, self.tp_range_min, self.tp_range_max)  # 3-15 pontos
                
                sl_price = current_price + (sl_points * 1.0)  # SL acima do preço (OURO: 1 ponto = $1.00)
                tp_price = current_price - (tp_points * 1.0)  # TP abaixo do preço (OURO: 1 ponto = $1.00)
                
                self._log(f"🧠 [V7-SELL] Usando SL/TP do modelo: SL={sl_points:.1f}pts | TP={tp_points:.1f}pts")
                self._log(f"💰 [V7-SELL] Preços calculados: Atual={current_price:.2f} | SL={sl_price:.2f} | TP={tp_price:.2f}")
            else:
                # Fallback para valores padrão
                sl_price = current_price + (5 * 1.0)  # 5 pontos SL (OURO: 1 ponto = $1.00)
                tp_price = current_price - (10 * 1.0)  # 10 pontos TP (OURO: 1 ponto = $1.00)
                self._log(f"📊 [SELL] Usando SL/TP padrão: SL=5pts | TP=10pts")
            
            # Executar ordem
            result = self._execute_order_v7(mt5.ORDER_TYPE_SELL, volume, sl_price, tp_price)
            
            if "SUCCESS" in result:
                # Verificar número atual de posições após execução
                current_positions = mt5.positions_get(symbol=self.symbol) if self.mt5_connected else []
                pos_count = len(current_positions) if current_positions else 0
                self._log(f"✅ [VENDA V7] Ordem executada! Vol: {volume} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
                self._log(f"📊 [POSIÇÕES] Total atual: {pos_count}/{self.max_positions}")
            else:
                self._log(f"❌ [VENDA V7] Falha na execução: {result}")
                
        except Exception as e:
            self._log(f"❌ [VENDA V7] Erro: {e}")
    
    def _calculate_volume_by_confidence_v7(self, entry_quality):
        """🎯 DYNAMIC SIZING V7 - Copiado do daytrader.py (hardcoded)"""
        try:
            # 🔥 LÓGICA HARDCODED DO DAYTRADER.PY - Portfolio-based scaling
            initial_portfolio_value = self.initial_balance  # 500.0
            current_portfolio_value = self._get_current_portfolio_value()
            base_lot = self.base_lot_size  # 0.02
            max_lot = self.max_lot_size    # 0.03
            growth_factor_cap = 1.6  # Cap de 60% de crescimento para controlar risco
            
            # Se o portfólio não cresceu, usa o lote base
            if current_portfolio_value <= initial_portfolio_value:
                adaptive_lot = base_lot
            else:
                # Calcular o fator de crescimento
                growth_factor = current_portfolio_value / initial_portfolio_value
                
                # Limitar o fator de crescimento para controlar o risco
                capped_growth_factor = min(growth_factor, growth_factor_cap)
                
                # Calcular o lote alvo com base no crescimento limitado
                target_lot = base_lot * capped_growth_factor
                
                # Garantir que o lote final esteja entre o mínimo (base) e o máximo absoluto
                adaptive_lot = max(base_lot, min(target_lot, max_lot))
            
            # 🎯 SEM SCALING POR QUALITY - volume já está calibrado
            final_volume = adaptive_lot
            
            # Aplicar limites finais
            final_volume = max(0.01, min(final_volume, max_lot))
            
            # 🔥 VALIDAÇÃO ADICIONAL PARA MT5
            # Garantir que o volume está em múltiplos de 0.01 (padrão para GOLD)
            final_volume = round(final_volume, 2)
            
            # Verificar se não é zero ou negativo
            if final_volume <= 0:
                final_volume = self.base_lot_size
            
            # Log detalhado do cálculo
            if hasattr(self, 'debug_step_counter') and self.debug_step_counter % 50 == 0:
                growth_factor = current_portfolio_value / initial_portfolio_value if initial_portfolio_value > 0 else 1.0
                self._log(f"[💰 DYNAMIC SIZE] Portfolio: ${current_portfolio_value:.2f} (growth: {growth_factor:.2f}x) | Adaptive: {adaptive_lot:.3f} | Quality: {entry_quality:.2f} | Final: {final_volume:.3f}")
            
            return final_volume
            
        except Exception as e:
            self._log(f"❌ [VOLUME V7] Erro no dynamic sizing: {e}")
            return self.base_lot_size
    
    def _get_current_portfolio_value(self):
        """🎯 Obter valor atual do portfolio (saldo da conta MT5)"""
        try:
            if self.mt5_connected:
                account_info = mt5.account_info()
                if account_info:
                    current_balance = account_info.balance
                    # Atualizar portfolio_value para uso em outras funções
                    self.portfolio_value = current_balance
                    return current_balance
            
            # Fallback para valor inicial se MT5 não conectado
            return self.initial_balance
            
        except Exception as e:
            self._log(f"[⚠️ PORTFOLIO] Erro ao obter saldo: {e}")
            return self.initial_balance
    
    def _execute_order_v7(self, order_type: int, volume: float, sl_price: float = None, tp_price: float = None) -> str:
        """Executar ordem V7 com SL/TP opcionais"""
        try:
            # 🚀 COOLDOWN REMOVIDO - ALINHADO COM DAYTRADER: "V7 INTUITION DECIDE TUDO"
            current_time = time.time()
            # time_since_last = current_time - self.last_order_time
            # if time_since_last < 1:
            #     self._log(f"⏱️ [COOLDOWN] Ordem bloqueada - {time_since_last:.2f}s desde última ordem (mín: 1s)")
            #     return "ERROR_COOLDOWN"
            
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
            
            # 🔥 VALIDAR VOLUME PARA O SÍMBOLO GOLD
            symbol_info = mt5.symbol_info(self.symbol)
            if symbol_info:
                volume_min = symbol_info.volume_min
                volume_max = symbol_info.volume_max
                volume_step = symbol_info.volume_step
                
                # Ajustar volume para o step correto
                volume = round(volume / volume_step) * volume_step
                
                # Aplicar limites do símbolo
                volume = max(volume_min, min(volume, volume_max))
                
                self._log(f"[📊 VOLUME] Original: {volume:.3f} | Ajustado: {volume:.3f} | Limites: [{volume_min:.3f}, {volume_max:.3f}] | Step: {volume_step:.3f}")
            else:
                self._log(f"[⚠️ VOLUME] Não foi possível obter info do símbolo {self.symbol}")
            
            # Preparar requisição com SL/TP opcionais
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "magic": 123456,
                "comment": "V7 Robot",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self.filling_mode
            }

            # Adicionar SL/TP se especificado
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
            
            if check_result.retcode != 0 and check_result.retcode != mt5.TRADE_RETCODE_DONE:
                self._log(f"[❌ ERRO] Ordem seria rejeitada: {check_result.retcode} - {check_result.comment}")
                self._log(f"[🔍 DEBUG] Volume: {volume} | Preço: {price} | Símbolo: {self.symbol}")
                
                # Log específico para erro de volume inválido
                if check_result.retcode == 10014:  # TRADE_RETCODE_INVALID_VOLUME
                    symbol_info = mt5.symbol_info(self.symbol)
                    if symbol_info:
                        self._log(f"[🔍 VOLUME DEBUG] Min: {symbol_info.volume_min} | Max: {symbol_info.volume_max} | Step: {symbol_info.volume_step}")
                
                return f"ERROR_ORDER_CHECK|{check_result.retcode}"
            
            # Executar ordem
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                action_type = "📈 LONG" if order_type == mt5.ORDER_TYPE_BUY else "📉 SHORT"
                sl_info = f" | SL: {sl_price:.2f}" if sl_price else ""
                tp_info = f" | TP: {tp_price:.2f}" if tp_price else ""
                self._log(f"[🎯 TRADE V7] {action_type} executado - #{result.order} @ {price:.2f}{sl_info}{tp_info}")
                return f"SUCCESS|{result.order}|{price}|{action_type}|{sl_price or 0}|{tp_price or 0}"
            else:
                error_code = result.retcode if result else "None"
                last_error = mt5.last_error()
                self._log(f"[❌ ERRO] Falha na ordem: {error_code} | MT5 Error: {last_error}")
                return f"ERROR_EXECUTION|{error_code}|{last_error}"
                
        except Exception as e:
            self._log(f"[❌ ORDER V7] Erro na execução: {e}")
            return f"ERROR_EXCEPTION|{e}"
    
    def get_model_info(self):
        """ℹ️ Obter informações do modelo carregado"""
        try:
            if not self.model_loaded:
                return {
                    'loaded': False,
                    'error': 'Modelo não carregado'
                }
            
            info = {
                'loaded': True,
                'model_type': type(self.model).__name__ if self.model else 'Unknown',
                'has_normalizer': self.normalizer is not None,
                'metadata': self.model_metadata or {},
                'model_path': ModelPaths.get_absolute_model_path()
            }
            
            # Adicionar informações do modelo se disponível
            if self.model:
                try:
                    info['observation_space'] = str(self.model.observation_space.shape)
                    info['action_space'] = str(self.model.action_space.shape)
                except:
                    pass
            
            return info
            
        except Exception as e:
            return {
                'loaded': False,
                'error': str(e)
            }
    
    def validate_model_zip(self, zip_path=None):
        """🔍 Validar estrutura do ZIP do modelo V7"""
        try:
            if zip_path is None:
                zip_path = ModelPaths.get_absolute_model_path()
            
            self._log(f"[🔍 V7 VALIDATE] Validando ZIP: {zip_path}")
            
            if not os.path.exists(zip_path):
                return False, f"ZIP não encontrado: {zip_path}"
            
            # Verificar se é um arquivo ZIP válido
            if not zipfile.is_zipfile(zip_path):
                return False, f"Arquivo não é um ZIP válido: {zip_path}"
            
            # Verificar conteúdo do ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                file_list = zip_ref.namelist()
                
                # Verificar se contém pelo menos um arquivo de modelo
                model_files = [f for f in file_list if f.endswith('.zip') or 'model' in f.lower()]
                if not model_files:
                    return False, "ZIP não contém arquivos de modelo válidos"
                
                # Verificar se contém normalizer (opcional mas recomendado)
                normalizer_files = [f for f in file_list if 'normalizer' in f.lower() and f.endswith('.pkl')]
                
                info = {
                    'total_files': len(file_list),
                    'model_files': len(model_files),
                    'has_normalizer': len(normalizer_files) > 0,
                    'file_list': file_list[:10]  # Primeiros 10 arquivos
                }
                
                self._log(f"[✅ V7 VALIDATE] ZIP válido com {info['total_files']} arquivos")
                return True, info
            
        except Exception as e:
            return False, f"Erro na validação: {e}"
    
    def create_model_directory(self):
        """📁 Criar diretório do modelo se não existir"""
        try:
            model_dir = os.path.dirname(ModelPaths.get_absolute_model_path())
            os.makedirs(model_dir, exist_ok=True)
            self._log(f"[📁 V7 DIR] Diretório criado/verificado: {model_dir}")
            return True
        except Exception as e:
            self._log(f"[❌ V7 DIR] Erro ao criar diretório: {e}")
            return False
    
    def run(self):
        """🚀 Executar RobotV7 - Interface simples"""
        try:
            self._log("=" * 60)
            self._log("🚀 ROBOTV7 INICIADO - Legion AI Trader V7")
            self._log("=" * 60)
            
            # Verificar compatibilidade
            if not self._verify_v7_compatibility():
                self._log("[❌ FATAL] RobotV7 não é compatível!")
                return
            
            # Status do modelo
            model_info = self.get_model_info()
            if model_info['loaded']:
                self._log(f"[✅ MODEL] {model_info['model_type']} carregado")
                self._log(f"[✅ NORM] {'Normalizer ativo' if model_info['has_normalizer'] else 'Sem normalizer'}")
            else:
                self._log(f"[❌ MODEL] Modelo não carregado: {model_info.get('error', 'Unknown')}")
            
            # Testar observação
            test_obs = self.reset()
            self._log(f"[✅ OBS] Observação V7 gerada: {test_obs.shape} ({test_obs.dtype})")
            
            # Testar predição se modelo carregado
            if model_info['loaded']:
                try:
                    action = self.get_model_prediction(test_obs)
                    if action is not None:
                        if hasattr(action, 'shape'):
                            self._log(f"[✅ PREDICT] Predição V7: {action.shape} - {action}")
                        else:
                            self._log(f"[✅ PREDICT] Predição V7: {action}")
                    else:
                        self._log("[❌ PREDICT] Falha na predição")
                except Exception as e:
                    self._log(f"[❌ PREDICT] Erro na predição: {e}")
            
            self._log("=" * 60)
            self._log("✅ ROBOTV7 PRONTO PARA OPERAÇÃO!")
            self._log("💡 Para usar: robot.step(action) ou robot.get_model_prediction(obs)")
            self._log("=" * 60)
            
            # Demo do debug system (25 predições para testar)
            if model_info['loaded']:
                self._log("🧪 EXECUTANDO DEMO DO DEBUG SYSTEM...")
                try:
                    for i in range(25):
                        test_obs = self.reset()
                        self.get_model_prediction(test_obs)
                        if i % 5 == 0:
                            self._log(f"   Demo prediction {i+1}/25...")
                    self._log("✅ Demo do debug system completo!")
                except Exception as e:
                    self._log(f"⚠️ Erro no demo: {e}")
                self._log("=" * 60)
            
            # 🚀 TRADING AO VIVO COM LOGS DA ENTRY HEAD
            if model_info['loaded']:
                self._log("🚀 INICIANDO TRADING AO VIVO COM DEBUG DA ENTRY HEAD...")
                self._log("💡 Logs da Entry Head aparecerão a cada predição!")
                self._log("=" * 60)
                
                # 🔒 Verificar se modelo está em modo de inferência
                if self.model and hasattr(self.model, 'policy'):
                    is_training = self.model.policy.training
                    self._log(f"[🔒 MODO] Modelo em modo: {'TREINAMENTO' if is_training else 'INFERÊNCIA'} ✅")
                    if is_training:
                        self._log("[⚠️ AVISO] Modelo deveria estar em modo inferência!")
                else:
                    self._log("[⚠️ AVISO] Não foi possível verificar modo do modelo")
                
                # Configurar debug com intervalo de 10 steps
                self.debug_composite_interval = 10   # Debug a cada 10 steps
                self.debug_anomaly_interval = 50     # Debug de anomalias a cada 50 steps
                
                self._start_live_trading()
            else:
                self._log("⚠️ Modelo não carregado - apenas monitoramento passivo")
                self._start_passive_monitoring()
                
        except KeyboardInterrupt:
            self._log("👋 RobotV7 finalizado pelo usuário")
                
        except Exception as e:
            self._log(f"[❌ FATAL] Erro no RobotV7: {e}")
            import traceback
            traceback.print_exc()

    # 🚀 V7 INTUITION GATES - REMOVIDOS PARA ALINHAMENTO COM DAYTRADER.PY: "V7 INTUITION DECIDE TUDO - SEM FILTROS LOCAIS"
    # def _check_entry_filters_v7(self, action_type):
    #     """
    #     🧠 FILTROS V7 INTUITION: Gates V7 REAIS - CORRIGIDO
    #     
    #     Usa os gates que a V7 Intuition REALMENTE produz:
    #     - entry_decision, entry_conf (SpecializedEntryHead)
    #     - mgmt_decision, mgmt_conf (TwoHeadDecisionMaker)
    #     - regime_id, actor_attention, critic_attention (UnifiedBackbone)
    #     """
    #     try:
    #         # 🎯 VERIFICAÇÃO CORRETA: Gates V7 reais
    #         if hasattr(self, 'last_v7_outputs') and self.last_v7_outputs:
    #             v7_passed, v7_reason = self._apply_v7_intuition_filters(action_type, self.last_v7_outputs)
    #             return v7_passed, v7_reason
    #         
    #         # Se não há outputs V7, aprovar (modelo decide)
    #         return True, "V7 Intuition Outputs não disponíveis - Aprovado"
    #         
    #     except Exception as e:
    #         # Em caso de erro, aprovar (não bloquear modelo)
    #         return True, f"V7 Entry Filters: Erro {str(e)[:50]} - Aprovado"

    # def _apply_v7_intuition_filters(self, action_type, v7_outputs):
    #     """
    #     🧠 FILTROS V7 INTUITION REAIS: Usa gates que a V7 REALMENTE produz
    #     
    #     Gates V7 Intuition REAIS:
    #     - entry_decision, entry_conf (SpecializedEntryHead)
    #     - mgmt_decision, mgmt_conf (TwoHeadDecisionMaker)  
    #     - regime_id, actor_attention, critic_attention (UnifiedBackbone)
    #     
    #     NÃO usa gates V5 (long_signal, short_signal) que NÃO EXISTEM na V7!
    #     """
    #     try:
    #         if 'gates' not in v7_outputs:
    #             return True, "Gates V7 Intuition não disponíveis - Aprovado"
    #         
    #         gates = v7_outputs['gates']
    #         
    #         # 🧠 FILTROS BASEADOS NOS GATES REAIS DA V7 INTUITION
    #         
    #         # 1. Entry Confidence Filter (SpecializedEntryHead)
    #         entry_conf = gates.get('entry_conf', 0.5)
    #         if entry_conf < 0.4:  # Baixa confiança na entrada
    #             print(f"[🚫 V7 FILTER] Entry confidence baixa: {entry_conf:.3f}")
    #             return False, f"Entry confidence baixa: {entry_conf:.3f}"
    #         
    #         # 2. Management Confidence Filter (TwoHeadDecisionMaker)
    #         mgmt_conf = gates.get('mgmt_conf', 0.5)
    #         if mgmt_conf < 0.3:  # Gestão muito insegura
    #             print(f"[🚫 V7 FILTER] Management confidence baixa: {mgmt_conf:.3f}")
    #             return False, f"Management confidence baixa: {mgmt_conf:.3f}"
    #         
    #         # 3. Regime-Based Filter (UnifiedBackbone)
    #         regime_id = gates.get('regime_id', 2)  # Default: sideways
    #         if regime_id == 3:  # Volatile market - muito arriscado
    #             print(f"[🚫 V7 FILTER] Mercado volátil (regime {regime_id}) - reduzindo exposição")
    #             return False, f"Mercado volátil (regime {regime_id})"
    #         
    #         # 4. Backbone Specialization Filter
    #         specialization_div = gates.get('specialization_divergence', 0.0)
    #         if specialization_div > 0.9:  # Alta divergência indica conflito interno
    #             print(f"[🚫 V7 FILTER] Alta divergência backbone: {specialization_div:.3f}")
    #             return False, f"Alta divergência backbone: {specialization_div:.3f}"
    #         
    #         # ✅ Trade aprovado com gates V7 reais
    #         regime_name = gates.get('regime_name', 'unknown')
    #         actor_att = gates.get('actor_attention_mean', 0.5)
    #         critic_att = gates.get('critic_attention_mean', 0.5)
    #         
    #         print(f"[✅ V7 FILTER] TRADE APROVADO - Entry: {entry_conf:.2f}, Mgmt: {mgmt_conf:.2f}, Regime: {regime_name}")
    #         return True, f"V7 Gates: Entry={entry_conf:.2f}, Mgmt={mgmt_conf:.2f}, Regime={regime_name}"
    #         
    #     except Exception as e:
    #         return True, f"V7 Gates: Erro {str(e)[:30]} - Aprovado"

class TradingAppV7:
    """Professional Trading Dashboard for Legion AI Trader V7"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Legion AI Trader V7 - Professional Trading System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(True, True)
        
        # Configure styles
        self.setup_styles()
        
        # Robot instance
        self.robot = TradingRobotV7()
        self.trading_active = False
        self.trading_thread = None
        self.stop_event = Event()
        
        # Stats tracking
        self.session_stats = {
            'buys': 0,
            'sells': 0,
            'wins': 0,
            'losses': 0,
            'profit_loss': 0.0,
            'initial_balance': 500.0
        }
        
        self.setup_gui()
        self.robot.log_widget = self.log_text  # Conectar logs do robot à GUI
        
        # Start periodic updates
        self.update_stats()
        
    def setup_styles(self):
        """Configure modern TTK styles"""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure custom styles
        self.style.configure('Title.TLabel', 
                           foreground='#00d4ff', 
                           background='#1e1e1e',
                           font=('Segoe UI', 18, 'bold'))
                           
        self.style.configure('Header.TLabel',
                           foreground='#ffffff',
                           background='#2d2d2d',
                           font=('Segoe UI', 11, 'bold'))
                           
        self.style.configure('Stat.TLabel',
                           foreground='#e1e1e1',
                           background='#2d2d2d',
                           font=('Segoe UI', 10))
                           
        self.style.configure('Value.TLabel',
                           foreground='#00ff88',
                           background='#2d2d2d',
                           font=('Segoe UI', 12, 'bold'))
                           
        self.style.configure('Status.TFrame',
                           background='#2d2d2d',
                           relief='raised',
                           borderwidth=2)
                           
        self.style.configure('Control.TButton',
                           font=('Segoe UI', 10, 'bold'))
        
    def setup_gui(self):
        """Setup professional trading interface"""
        # Create main container with padding
        main_container = tk.Frame(self.root, bg='#1e1e1e')
        main_container.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Header section
        self.create_header(main_container)
        
        # Main content area - split into left and right panels
        content_frame = tk.Frame(main_container, bg='#1e1e1e')
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        # Left panel - Trading statistics and controls
        left_panel = tk.Frame(content_frame, bg='#1e1e1e')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 10))
        
        self.create_stats_panel(left_panel)
        self.create_control_panel(left_panel)
        self.create_system_status(left_panel)
        
        # Right panel - Logs and monitoring
        right_panel = tk.Frame(content_frame, bg='#1e1e1e')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.create_log_panel(right_panel)
        
    def create_header(self, parent):
        """Create professional header with branding"""
        header_frame = tk.Frame(parent, bg='#1e1e1e', height=80)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Title and version
        title_frame = tk.Frame(header_frame, bg='#1e1e1e')
        title_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        title = ttk.Label(title_frame, text="LEGION AI TRADER", style='Title.TLabel')
        title.pack(anchor='w')
        
        subtitle = tk.Label(title_frame, text="Professional Trading System V7.0", 
                           font=('Segoe UI', 10), fg='#888888', bg='#1e1e1e')
        subtitle.pack(anchor='w')
        
        # Status indicator
        status_frame = tk.Frame(header_frame, bg='#1e1e1e')
        status_frame.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.connection_status = tk.Label(status_frame, text="● CONNECTED", 
                                         font=('Segoe UI', 10, 'bold'), 
                                         fg='#00ff88', bg='#1e1e1e')
        self.connection_status.pack(anchor='e', pady=(10, 0))
        
        # Separator line
        separator = tk.Frame(parent, height=2, bg='#333333')
        separator.pack(fill=tk.X, pady=(10, 0))
        
    def create_stats_panel(self, parent):
        """Create trading statistics panel"""
        stats_frame = ttk.Frame(parent, style='Status.TFrame')
        stats_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Panel header
        header = ttk.Label(stats_frame, text="TRADING STATISTICS", style='Header.TLabel')
        header.pack(pady=(15, 10))
        
        # Stats grid with professional layout
        grid_frame = tk.Frame(stats_frame, bg='#2d2d2d')
        grid_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Configure grid weights for 2x2 layout
        for i in range(4):
            grid_frame.columnconfigure(i, weight=1)
        for i in range(2):
            grid_frame.rowconfigure(i, weight=1)
        
        # Create stat widgets and store references directly
        self.stat_values = {}
        
        # Row 0: Orders
        self.create_stat_row(grid_frame, 0, 0, "LONG ORDERS", "0", "#00ff88")
        self.stat_values['buys'] = grid_frame.grid_slaves(row=0, column=1)[0]
        
        self.create_stat_row(grid_frame, 0, 2, "SHORT ORDERS", "0", "#ff6b6b") 
        self.stat_values['sells'] = grid_frame.grid_slaves(row=0, column=3)[0]
        
        # Row 1: Performance
        self.create_stat_row(grid_frame, 1, 0, "WIN RATE", "0.0%", "#ffd93d")
        self.stat_values['winrate'] = grid_frame.grid_slaves(row=1, column=1)[0]
        
        self.create_stat_row(grid_frame, 1, 2, "P&L SESSION", "$0.00", "#00d4ff")
        self.stat_values['pl'] = grid_frame.grid_slaves(row=1, column=3)[0]
        
    def create_stat_row(self, parent, row, col, label, value, color):
        """Create a professional stat display"""
        # Label
        lbl = tk.Label(parent, text=label, font=('Segoe UI', 9), 
                      fg='#cccccc', bg='#2d2d2d')
        lbl.grid(row=row, column=col, sticky='w', padx=(10, 5), pady=10)
        
        # Value
        val = tk.Label(parent, text=value, font=('Segoe UI', 14, 'bold'), 
                      fg=color, bg='#2d2d2d')
        val.grid(row=row, column=col+1, sticky='e', padx=(5, 10), pady=10)
        
    def create_control_panel(self, parent):
        """Create control panel with professional buttons"""
        control_frame = ttk.Frame(parent, style='Status.TFrame')
        control_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Panel header
        header = ttk.Label(control_frame, text="SYSTEM CONTROL", style='Header.TLabel')
        header.pack(pady=(15, 15))
        
        # Button container
        button_container = tk.Frame(control_frame, bg='#2d2d2d')
        button_container.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Professional buttons
        self.start_button = tk.Button(button_container, text="START TRADING", 
                                     command=self.toggle_trading,
                                     font=('Segoe UI', 11, 'bold'),
                                     bg='#00ff88', fg='#000000', 
                                     activebackground='#00cc6a',
                                     relief='flat', bd=0, padx=20, pady=8)
        self.start_button.pack(fill=tk.X, pady=(10, 5))
        
        reset_button = tk.Button(button_container, text="RESET STATISTICS",
                                command=self.reset_stats,
                                font=('Segoe UI', 10),
                                bg='#404040', fg='#ffffff',
                                activebackground='#505050',
                                relief='flat', bd=0, padx=20, pady=6)
        reset_button.pack(fill=tk.X, pady=(5, 10))
        
    def create_system_status(self, parent):
        """Create system status panel"""
        status_frame = ttk.Frame(parent, style='Status.TFrame')
        status_frame.pack(fill=tk.X)
        
        # Panel header
        header = ttk.Label(status_frame, text="MARKET STATUS", style='Header.TLabel')
        header.pack(pady=(15, 15))
        
        # Status container
        status_container = tk.Frame(status_frame, bg='#2d2d2d')
        status_container.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        # Trend indicator
        trend_frame = tk.Frame(status_container, bg='#2d2d2d')
        trend_frame.pack(fill=tk.X, pady=10)
        
        tk.Label(trend_frame, text="MARKET TREND", font=('Segoe UI', 9), 
                fg='#cccccc', bg='#2d2d2d').pack(side=tk.LEFT)
        
        self.trend_label = tk.Label(trend_frame, text="ANALYZING...", 
                                   font=('Segoe UI', 11, 'bold'), 
                                   fg='#ffd93d', bg='#2d2d2d')
        self.trend_label.pack(side=tk.RIGHT)
        
    def create_log_panel(self, parent):
        """Create professional log monitoring panel"""
        log_frame = ttk.Frame(parent, style='Status.TFrame')
        log_frame.pack(fill=tk.BOTH, expand=True)
        
        # Panel header
        header_frame = tk.Frame(log_frame, bg='#2d2d2d')
        header_frame.pack(fill=tk.X)
        
        header = ttk.Label(header_frame, text="SYSTEM MONITORING", style='Header.TLabel')
        header.pack(pady=(15, 10))
        
        # Log area with professional styling
        log_container = tk.Frame(log_frame, bg='#2d2d2d')
        log_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        # Custom scrolled text with modern look
        self.log_text = scrolledtext.ScrolledText(
            log_container,
            font=('Consolas', 9),
            bg='#1a1a1a',
            fg='#e1e1e1',
            insertbackground='#00d4ff',
            selectbackground='#404040',
            selectforeground='#ffffff',
            relief='flat',
            bd=0,
            padx=10,
            pady=10,
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)
        
        # Configure log text tags for colored output
        self.log_text.tag_configure('success', foreground='#00ff88')
        self.log_text.tag_configure('error', foreground='#ff6b6b')
        self.log_text.tag_configure('warning', foreground='#ffd93d')
        self.log_text.tag_configure('info', foreground='#00d4ff')
        
    def toggle_trading(self):
        """Alternar estado do trading"""
        if not self.trading_active:
            self.start_trading()
        else:
            self.stop_trading()
            
    def start_trading(self):
        """Iniciar trading em thread separada"""
        try:
            self.trading_active = True
            self.start_button.config(text="STOP TRADING", bg='#ff6b6b', fg='#ffffff')
            self.stop_event.clear()
            
            # Start robot in separate thread
            self.trading_thread = Thread(target=self.run_trading_loop, daemon=True)
            self.trading_thread.start()
            
            self.log("🚀 Trading V7 iniciado!")
            
        except Exception as e:
            self.log(f"❌ Erro ao iniciar trading: {e}")
            self.trading_active = False
            
    def stop_trading(self):
        """Parar trading"""
        self.trading_active = False
        self.start_button.config(text="START TRADING", bg='#00ff88', fg='#000000')
        self.stop_event.set()
        self.log("⏹️ Trading V7 parado!")
        
    def run_trading_loop(self):
        """Loop principal de trading"""
        try:
            # Initialize robot
            self.robot.run()
            
            while self.trading_active and not self.stop_event.is_set():
                # Robot is running its own loop
                # We just monitor and update stats
                time.sleep(1)  # Reduzido para 1 segundo (GUI responsiva)
                self.update_trading_stats()
                
        except Exception as e:
            self.log(f"❌ Erro no loop de trading: {e}")
            self.trading_active = False
            
    def update_trading_stats(self):
        """Atualizar estatísticas de trading"""
        try:
            if hasattr(self.robot, 'mt5_connected') and self.robot.mt5_connected:
                # Get MT5 positions and history
                positions = mt5.positions_get(symbol=self.robot.symbol) or []
                
                # Update basic stats (simplified for now)
                # In real implementation, track actual trades
                pass
                
        except Exception as e:
            self.log(f"⚠️ Erro ao atualizar stats: {e}")
            
    def calculate_trend(self):
        """Calcular trend atual"""
        try:
            if hasattr(self.robot, 'historical_data') and len(self.robot.historical_data) > 20:
                recent_prices = self.robot.historical_data['close'].tail(20)
                if recent_prices.iloc[-1] > recent_prices.iloc[0]:
                    return "🟢 BULLISH"
                elif recent_prices.iloc[-1] < recent_prices.iloc[0]:
                    return "🔴 BEARISH"
            return "🟡 NEUTRAL"
        except:
            return "🟡 NEUTRAL"
            
    def update_stats(self):
        """Update statistics in professional GUI"""
        try:
            # Update stat values using stored references
            if hasattr(self, 'stat_values'):
                self.stat_values['buys'].config(text=str(self.session_stats['buys']))
                self.stat_values['sells'].config(text=str(self.session_stats['sells']))
                
                # Calculate and update win rate
                total_trades = self.session_stats['wins'] + self.session_stats['losses']
                if total_trades > 0:
                    win_rate = (self.session_stats['wins'] / total_trades) * 100
                    self.stat_values['winrate'].config(text=f"{win_rate:.1f}%")
                else:
                    self.stat_values['winrate'].config(text="0.0%")
                
                # Update P&L with appropriate color
                pl = self.session_stats['profit_loss']
                pl_text = f"${pl:.2f}"
                pl_color = '#00ff88' if pl >= 0 else '#ff6b6b'
                self.stat_values['pl'].config(text=pl_text, fg=pl_color)
            
            # Update trend indicator
            if hasattr(self, 'trend_label'):
                trend = self.calculate_trend()
                if 'BULLISH' in trend:
                    self.trend_label.config(text="BULLISH ↗", fg='#00ff88')
                elif 'BEARISH' in trend:
                    self.trend_label.config(text="BEARISH ↘", fg='#ff6b6b')
                else:
                    self.trend_label.config(text="SIDEWAYS ↔", fg='#ffd93d')
            
            # Update connection status
            if hasattr(self, 'connection_status'):
                if hasattr(self.robot, 'mt5_connected') and self.robot.mt5_connected:
                    self.connection_status.config(text="● CONNECTED", fg='#00ff88')
                else:
                    self.connection_status.config(text="● DISCONNECTED", fg='#ff6b6b')
            
        except Exception as e:
            self.log(f"Warning: Error updating GUI - {e}")
        finally:
            # Schedule next update
            self.root.after(1500, self.update_stats)  # Update every 1.5 seconds
            
    def reset_stats(self):
        """Reset estatísticas da sessão"""
        self.session_stats = {
            'buys': 0,
            'sells': 0,
            'wins': 0,
            'losses': 0,
            'profit_loss': 0.0,
            'initial_balance': 500.0
        }
        self.log("🔄 Estatísticas resetadas!")
        
    def log(self, message):
        """Add professional formatted message to log"""
        try:
            import datetime
            timestamp = datetime.datetime.now().strftime("%H:%M:%S")
            
            # Determine message type and color
            if any(keyword in message.lower() for keyword in ['error', 'erro', 'failed', 'fail']):
                tag = 'error'
                prefix = "ERR"
            elif any(keyword in message.lower() for keyword in ['warning', 'warn', '⚠️']):
                tag = 'warning' 
                prefix = "WARN"
            elif any(keyword in message.lower() for keyword in ['success', '✅', 'connected', 'loaded']):
                tag = 'success'
                prefix = "OK"
            else:
                tag = 'info'
                prefix = "INFO"
            
            # Clean message of emojis for professional look
            clean_message = message.replace('🚀', '').replace('✅', '').replace('❌', '').replace('⚠️', '').replace('🔍', '').replace('💰', '').replace('📊', '').replace('⚙️', '').replace('🤖', '').replace('🎉', '').replace('🔄', '').replace('💡', '').strip()
            
            formatted_message = f"[{timestamp}] {prefix}: {clean_message}\n"
            
            # Insert with appropriate tag
            self.log_text.insert(tk.END, formatted_message, tag)
            self.log_text.see(tk.END)
            
            # Also print to console
            print(formatted_message.strip())
            
        except Exception as e:
            print(f"[LOG ERROR] {e}: {message}")

def main_gui():
    """Função principal para iniciar GUI"""
    root = tk.Tk()
    app = TradingAppV7(root)
    
    try:
        root.mainloop()
    except KeyboardInterrupt:
        print("\n[🛑] GUI interrompida pelo usuário")
    except Exception as e:
        print(f"[❌] Erro crítico na GUI: {e}")

# 🚀 INICIALIZAÇÃO DO ROBOTV7
if __name__ == "__main__":
    import sys
    
    # Check for console mode flag (GUI is default now)
    if len(sys.argv) > 1 and sys.argv[1] == "--console":
        print("🚀 Iniciando RobotV7 (modo console)...")
        try:
            robot = TradingRobotV7()
            robot.run()
        except Exception as e:
            print(f"❌ Erro fatal: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("🎨 Iniciando RobotV7 com GUI Profissional...")
        print("💡 Use: python RobotV7.py --console para modo terminal")
        main_gui()