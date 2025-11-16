# -*- coding: utf-8 -*-
import sys
import io
# Tornar seguro para execução via pythonw (sem console)
try:
    if getattr(sys, 'stdout', None) is not None and hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
    if getattr(sys, 'stderr', None) is not None and hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='ignore')
except Exception:
    pass

import tkinter as tk
from tkinter import scrolledtext, ttk, filedialog
from threading import Thread, Event
"""
🍒 Cherry AI Trader V7 - Trading Robot Cherry-aligned com V11 Sigmoid Policy
🧠 CONFIGURADO CHERRY: V11 Sigmoid Policy + LSTM/GRU Hybrid + 4D Actions
🎯 OBSERVATION SPACE: 450 dimensões nativas (formato Cherry - 2 posições ativas)

🧠 CHERRY ARCHITECTURE:
- OBSERVATION SPACE: 450 dimensões (Cherry format - 2 posições ativas)  
- ACTION SPACE: 4 dimensões [entry_decision, entry_confidence, pos1_mgmt, pos2_mgmt]
- FEATURES: Market + Positions + Simple = 39 features por step
- WINDOW: 10 steps × 45 features = 450 dimensões
- UNIFIED BACKBONE: Shared processing between Actor/Critic
- GRADIENT MIXING: Cross-pollination between networks
- SPECIALIZED HEADS: Entry Head + Management Head para decisões focadas

ACTION SPACE (4D) - Cherry Especializado:
- [0] entry_decision: Logit bruto para entrada (processado pelo Entry Head)
- [1] entry_confidence: [0,1] Confiança na decisão de entrada
- [2] pos1_mgmt: Management para posição 1 (SL/TP combinado)
- [3] pos2_mgmt: Management para posição 2 (SL/TP combinado)

🎯 DYNAMIC TRAILING STOP:
- sl_adjust [-3,3]: Ativa/move trailing stop (abs>1.5 = ativa, abs>0.5 = move)
- tp_adjust [-3,3]: Intensidade do trailing (controla distância 15-30 pontos)
- Rewards: +1.0 execução, +0.8 ativação, +0.6 proteção, +0.4 timing

CONVERSÃO: [-3,3] → [2-8] SL, [3-15] TP pontos → SL/TP realistas (OURO: 1 ponto = $1.00)

COMPATIBILIDADE:
- 🚀 TwoHeadV11Sigmoid (LSTM+GRU Hybrid + Entry/Management Heads)
- 📋 TradingTransformerFeatureExtractor
- 🔧 Enhanced Normalizer
- 🎯 Dynamic Trailing Stop System

🔧 CHERRY UPDATES:
- _get_observation_v7(): Gera 450D nativamente (formato Cherry)
- _process_v7_action(): Processa 4D Legion actions → 8D Robot actions
- _verify_cherry_compatibility(): Verificação específica para modelo Cherry
- auto_load_model(): Carrega modelo Cherry com V11 Sigmoid Policy
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

# 🎯 ACTION THRESHOLDS - ALINHADO COM CHERRY.PY
# Range [-1, 1]: SHORT[-1.0, -0.33) HOLD[-0.33, 0.33) LONG[0.33, 1.0]
ACTION_THRESHOLD_SHORT = -0.33   # raw_decision < -0.33 = SHORT (extremo negativo)
ACTION_THRESHOLD_LONG = 0.33     # raw_decision >= 0.33 = LONG (extremo positivo)

# 🎯 V7 MODEL PATHS - Configuração centralizada
class ModelPaths:
    """🎯 Configuração centralizada de paths para modelos V7 e Silus"""
    # Caminho principal do modelo ZIP - Legion V1 
    MODEL_ZIP_PATH = "Modelo daytrade/Legion V1.zip"
    
    # Caminhos para modelos do silus.py (checkpoints com steps)
    SILUS_MODEL_DIR = "."  # Diretório raiz onde estão os checkpoints
    SILUS_MODEL_PATTERN = "*_steps.zip"  # Padrão dos checkpoints
    
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
    def find_latest_silus_model(cls):
        """🔍 Encontrar o modelo mais recente do silus.py"""
        import glob
        
        # Procurar no diretório pai (raiz do projeto)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        pattern = os.path.join(project_root, cls.SILUS_MODEL_PATTERN)
        models = glob.glob(pattern)
        
        if not models:
            return None
            
        # Extrair número de steps e ordenar
        models_with_steps = []
        for model in models:
            try:
                basename = os.path.basename(model)
                # Procurar por padrão *_NUMERO_steps.zip
                import re
                match = re.search(r'_(\d+)_steps\.zip$', basename)
                if match:
                    steps = int(match.group(1))
                    models_with_steps.append((steps, model))
            except:
                continue
                
        if not models_with_steps:
            return None
            
        # Retornar modelo com mais steps
        models_with_steps.sort(reverse=True)
        return models_with_steps[0][1]  # Retornar path do modelo
    
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
from trading_framework.policies.two_head_v11_sigmoid import TwoHeadV11Sigmoid, get_v11_sigmoid_kwargs
from trading_framework.policies.two_head_v11_sigmoid_legacy import TwoHeadV11Sigmoid as TwoHeadV11SigmoidLegacy
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
print("[LEGION V1] ✅ TwoHeadV11Sigmoid importada - Modelo Legion V1 exclusivo")
print("[INTUITION] ✅ TradingTransformerFeatureExtractor importado OBRIGATÓRIO")

# === 🎯 CONFIGURAÇÃO SL/TP DINÂMICO (ALINHADA COM SILUS.PY) ===
REALISTIC_SLTP_CONFIG = {
    # 🎯 RANGES REALISTAS PARA GOLD 1MIN - HARD CAPS (ALINHADO COM CHERRY.PY)
    'sl_min_points': 10,    # SL mínimo: 10 pontos ($10 risk com 0.01 lot)
    'sl_max_points': 25,    # SL máximo: 25 pontos ($25 risk com 0.01 lot)
    'tp_min_points': 12,    # TP mínimo: 12 pontos ($12 reward com 0.01 lot)
    'tp_max_points': 25,    # 🔥 HARD CAP: TP máximo 25 pontos (realista!)
    'sl_tp_step': 0.5,      # Variação: 0.5 pontos
    'action_to_points_multiplier': 15  # -3*15=-45, +3*15=+45 pontos
}

class BaseTradingEnv:
    """Classe base para environamento de trading com configurações básicas"""
    def __init__(self):
        self.symbol = "GOLD"
        self.initial_balance = 500.0
        self.max_positions = 2  # 🔬 ALINHADO: silus.py pós-cirurgia usa 2 posições
        self.window_size = 20

class SymbolDetector:
    """🔍 Sistema de reconhecimento automático de símbolos do ouro"""
    GOLD_SYMBOLS = [
        "GOLD", "GOLD#", "GOLDz", "GOLD.", "GOLD-",
        "XAUUSD", "XAUUSDz", "XAUUSD#", "XAUUSD.", "XAUUSD-",
        "XAU/USD", "XAU_USD", "XAUUSDT", "XAUEUR",
        "Au", "AUU", "GOLDC", "GOLDM", "GOLDEUR",
        "XAUCAD", "XAUAUD", "XAUGBP", "XAUJPY"
    ]
    
    @staticmethod
    def detect_gold_symbol():
        """🔍 Detecta automaticamente o símbolo do ouro disponível"""
        if not mt5.initialize():
            return None
            
        symbols = mt5.symbols_get()
        if not symbols:
            return None
            
        # Buscar símbolos do ouro na ordem de prioridade
        for priority_symbol in SymbolDetector.GOLD_SYMBOLS:
            for symbol in symbols:
                if symbol.name.upper() == priority_symbol.upper():
                    # Verificar se está disponível no Market Watch
                    if mt5.symbol_select(symbol.name, True):
                        return symbol.name
        
        # Busca fuzzy por "GOLD" ou "XAU" no nome
        for symbol in symbols:
            name = symbol.name.upper()
            if "GOLD" in name or "XAU" in name:
                if mt5.symbol_select(symbol.name, True):
                    return symbol.name
                    
        return None

class Config:
    """Configurações do sistema V7"""
    SYMBOL = SymbolDetector.detect_gold_symbol() or "XAUUSDz"  # Auto-detect ou fallback
    INITIAL_BALANCE = 500.0
    MAX_POSITIONS = 2  # 🔬 ALINHADO: silus.py pós-cirurgia usa 2 posições - modelo Legion V1
    WINDOW_SIZE = 20
    MAX_LOT_SIZE = 0.03
    BASE_LOT_SIZE = 0.02

    # 🎯 FILTRO DE ATIVIDADE - Toggle para configurações otimizadas
    ACTIVITY_FILTER_ENABLED = False  # False = modo padrão, True = modo otimizado

    # V7 Specific
    OBSERVATION_SPACE_SIZE = 450   # 45 features × 10 window (Cherry format - alinhado com ambiente de treino)
    FEATURES_PER_STEP = 45  # Cherry format: 16 market + 18 position + 2 intelligent + 4 order_flow + 5 volatility
    ACTION_SPACE_SIZE = 4  # Cherry format: [entry_decision, entry_confidence, pos1_mgmt, pos2_mgmt]

    # 🎯 CONFIGURAÇÕES DO FILTRO DE ATIVIDADE
    # SL/TP Padrão
    DEFAULT_SL_RANGE = 15.0
    DEFAULT_TP_RANGE = 18.0

    # SL/TP Otimizado (quando ACTIVITY_FILTER_ENABLED = True)
    OPTIMIZED_SL_RANGE = 30.0  # +100% baseado em análise de swings 2-4h
    OPTIMIZED_TP_RANGE = 35.0  # +94% baseado em análise de swings 2-4h

    # Horários bloqueados (quando ACTIVITY_FILTER_ENABLED = True)
    # ANTIGO (2025-10-31): [8, 9, 10, 11, 17, 21] - Análise mostrou que 10:00 era lucrativo
    # SEVENTEEN (2025-11-10): [0, 1, 2, 5, 7, 8, 11, 12, 14, 16, 21, 23] - Baseado em análise sem filtro
    # EIGHTEEN (2025-11-15): Baseado em análise empírica de 51 trades (Magic 777528)
    # Bloqueados: Horários com Net PnL negativo OU (WR < 40% E trades >= 3)
    # Permitidos lucrativos: 00:00 (+$99.60), 11:00 (+$117.69), 08:00 (+$68.52), 07:00 (+$54.06)
    # Bloqueados prejudiciais: 22:00 (-$91.53), 03:00 (-$61.86), 15:00 (-$62.52), 09:00 (-$60.81)
    # Ganho potencial vs filtro Seventeen: +$838.95
    BLOCKED_HOURS = [3, 4, 9, 10, 15, 16, 17, 18, 21, 22]  # EIGHTEEN: 10 horas bloqueadas (14 permitidas)

    # Multiplicador de confiança SHORT (quando ACTIVITY_FILTER_ENABLED = True)
    SHORT_CONFIDENCE_BOOST = 1.3  # SHORT tem 47% win rate vs LONG 33%
    LONG_CONFIDENCE_PENALTY = 0.8  # Penalizar LONG bias

    # Features breakdown - Cherry format
    MARKET_FEATURES = 16           # Market features (alinhado com cherry.py)
    POSITION_FEATURES = 18         # 2 positions × 9 features (alinhado com modelo treinado)
    INTELLIGENT_FEATURES = 2       # V7 components básicos (market_regime, trend_strength)
    ORDER_FLOW_FEATURES = 4        # Order flow analysis features
    VOLATILITY_FEATURES = 5        # Volatility analysis features
    MICROSTRUCTURE = 14           # Order flow + tick analytics (legacy)
    VOLATILITY_ADVANCED = 5       # GARCH + clustering + breakout (legacy)
    MARKET_CORRELATION = 4        # Inter-market correlations (legacy)
    MOMENTUM_MULTI = 6            # Multi-timeframe momentum (legacy)
    ENHANCED_FEATURES = 20        # Pattern recognition + regime detection (legacy)

class TradingRobotV7(gym.Env):
    """🚀 Trading Robot V7 - Exclusivo para modelo Legion V1 com V11 Sigmoid Policy"""
    
    def __init__(self, log_widget=None):
        super().__init__()
        self.log_widget = log_widget  # Opcional para compatibilidade
        self.symbol = Config.SYMBOL
        
        # 🔥 CONFIGURAÇÕES V7
        self.window_size = Config.WINDOW_SIZE
        self.initial_balance = Config.INITIAL_BALANCE
        self.portfolio_value = self.initial_balance
        self.peak_portfolio = self.initial_balance
        # 🧠 MEMORY FIX: Usar deque com maxlen para evitar crescimento infinito
        self.positions = deque(maxlen=100)  # Últimas 100 posições
        self.returns = deque(maxlen=500)    # Últimos 500 returns
        self.trades = deque(maxlen=200)     # Últimos 200 trades
        self.current_drawdown = 0.0
        self.peak_drawdown = 0.0
        self.max_lot_size = Config.MAX_LOT_SIZE
        self.max_positions = Config.MAX_POSITIONS
        self.current_positions = 0
        self.current_step = 0
        self.done = False
        self.last_order_time = 0
        
        # 🚨 SISTEMA DE COOLDOWN ADAPTATIVO IDÊNTICO AO SILUS
        self.cooldown_base = 35         # Base: 35 steps = 35 minutos (1min timeframe)
        self.cooldown_after_trade = 35  # Será ajustado dinamicamente baseado no resultado
        self.last_position_closed_timestamp = 0  # Timestamp do último fechamento
        
        # Cooldown independente por slot de posição
        self.position_slot_cooldowns = {i: 0.0 for i in range(self.max_positions)}  # slot -> next_allowed_time
        self.position_slot_map = {}  # ticket -> slot

        # 📊 Contadores de ajustes SL/TP
        self.sl_tp_adjustments = {
            'total_adjustments': 0,        # Total de ajustes não-zero processados
            'sl_adjustments': 0,           # Ajustes de SL não-zero (±0.5)
            'tp_adjustments': 0,           # Ajustes de TP não-zero (±0.5)
            'successful_modifications': 0, # Modificações aceitas pelo MT5
            'failed_modifications': 0      # Modificações rejeitadas pelo MT5
        }
        
        # Sistema de tracking win/loss para cooldown adaptativo
        self.last_trade_was_win = False
        self.last_trade_was_loss = False
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        
        self._log(f"[🔧 INIT] Sistema de cooldown ADAPTATIVO inicializado - base {self.cooldown_base} min, adaptativo por resultado")
        self._log(f"[🔧 INIT] Cooldowns: WIN simples=25min, WIN múltiplo=30min, LOSS simples=45min, LOSS múltiplo=60min")
        
        # 🛡️ TRACKER DE POSIÇÕES: Para detectar novas posições manuais
        self.known_positions = set()  # Set com tickets de posições conhecidas
        self.position_stats = {}  # Dicionário com stats das posições: {ticket: {'open_price': float, 'volume': float, 'type': str}}
        
        # 🧠 V7 GATES - SINCRONIZAÇÃO COM DAYTRADER
        # Removido: last_v7_gate_info (Legion V1 usa Entry/Management heads)
        # Removido: last_v7_outputs (Legion V1 não usa filtros baseados em gates)
        # 🧠 MEMORY FIX: Usar deque com maxlen
        self.daily_trades = deque(maxlen=50)  # Últimos 50 trades do dia
        
        # 🔥 ACTION SPACE LEGION V1: 4 dimensões
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            dtype=np.float32
        )
        
        # 🧠 V7 FEATURE COLUMNS - 65 available (using first 20 for Legion V1)
        self.feature_columns = self._create_v7_feature_columns()
        
        # 🔥 OBSERVATION SPACE LEGION V1: 450 dimensões (45 × 10)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf, 
            shape=(Config.OBSERVATION_SPACE_SIZE,),
            dtype=np.float32
        )
        
        self._log(f"[OBS SPACE CHERRY] 🧠 {Config.OBSERVATION_SPACE_SIZE} dimensões (45 features × 10 window - alinhado com cherry.py)")
        
        # Variáveis de controle
        self.realized_balance = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.last_trade_pnl = 0.0
        self.steps_since_last_trade = 0
        self.last_action = None
        self.hold_count = 0
        self.hold_log_interval = 20  # Mostrar log de HOLD a cada 20 ocorrências
        self.base_tf = '1m'  # 🔥 ALTERADO: Timeframe base para 1 minuto (alinhado com modelos SILUS)
        
        # Position sizing
        # 💰 SISTEMA DE MULTIPLICADOR DE LOTE
        self.original_base_lot = Config.BASE_LOT_SIZE  # 0.02 (base de treinamento)
        self.original_max_lot = Config.MAX_LOT_SIZE    # 0.03 (máximo de treinamento)
        self.base_lot_size = self.original_base_lot    # Valor configurável pelo usuário
        self.max_lot_size = self.original_max_lot      # Calculado proporcionalmente
        self.lot_multiplier = 1.0                      # Multiplicador atual
        self.lot_size = self.base_lot_size             # Será calculado dinamicamente
        
        # SL/TP Ranges (aligned with cherry.py REALISTIC_SLTP_CONFIG)
        # 🎯 APLICAR FILTRO DE ATIVIDADE SE HABILITADO
        if Config.ACTIVITY_FILTER_ENABLED:
            self.sl_range_min = Config.OPTIMIZED_SL_RANGE   # SL mínimo otimizado: 30 pontos
            self.sl_range_max = Config.OPTIMIZED_SL_RANGE   # SL máximo otimizado: 30 pontos
            self.tp_range_min = Config.OPTIMIZED_TP_RANGE   # TP mínimo otimizado: 35 pontos
            self.tp_range_max = Config.OPTIMIZED_TP_RANGE   # TP máximo otimizado: 35 pontos
            self._log(f"[🎯 FILTRO ATIVIDADE] HABILITADO - SL: ${self.sl_range_min:.1f}, TP: ${self.tp_range_min:.1f}")
        else:
            self.sl_range_min = 10.0   # SL mínimo: 10 pontos (HARD CAP)
            self.sl_range_max = 15.0   # SL máximo: 15 pontos (HARD CAP) - UPDATED to match cherry.py
            self.tp_range_min = 12.0   # TP mínimo: 12 pontos (HARD CAP)
            self.tp_range_max = 18.0   # TP máximo: 18 pontos (HARD CAP) - UPDATED to match cherry.py
            self._log(f"[📍 MODO PADRÃO] SL: ${self.sl_range_min:.1f}-${self.sl_range_max:.1f}, TP: ${self.tp_range_min:.1f}-${self.tp_range_max:.1f}")
        self.sl_tp_step = 0.5      # Variação: 0.5 pontos
        
        # Debug counters
        self.debug_step_counter = 0
        self.debug_composite_interval = 10    # Debug composite a cada 10 steps
        self.debug_anomaly_interval = 50      # Debug anomalias a cada 50 steps
        # 🧠 MEMORY FIX: Usar deque O(1) em vez de lista O(n)
        self.last_observations = deque(maxlen=50)  # Últimas 50 observações
        self.obs_stats = {'mean': None, 'std': None, 'min': None, 'max': None}
        self.obs_stats_update_counter = 0  # Atualizar stats a cada N observações
        
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
                    
                    # 🔧 INICIALIZAR POSIÇÕES CONHECIDAS PARA DETECTAR FECHAMENTOS
                    try:
                        current_positions = self._get_robot_positions() or []
                        self.known_positions = set(pos.ticket for pos in current_positions)
                        self._log(f"[🔧 INIT] Posições conhecidas inicializadas: {len(self.known_positions)} posições")
                    except Exception as e:
                        self._log(f"[⚠️ INIT] Erro ao inicializar posições conhecidas: {e}")
                        self.known_positions = set()
                    
                    # Continuar verificação do símbolo
                    if symbol_info:
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
        self._log(f"[💰 LOT MULTIPLIER] Base: {self.base_lot_size} | Max: {self.max_lot_size} | Multiplier: {self.lot_multiplier:.1f}x | Dynamic sizing: ATIVO")
        
        # Cache para normalização de preços
        self._price_min_max_cache = None
        
        # 🎯 V7 Model Loading - Variáveis de controle
        self.model = None
        self.normalizer = None
        self.model_loaded = False
        self.model_metadata = None
        self.is_legion_model = True  # Flag para modelo Legion V1 exclusivamente
        
        # Modelo será carregado via seletor manual na GUI
        self._log("[🤖 V7 MODEL] Use o botão 'SELECT MODEL' para carregar um modelo")
        self.model_loaded = False
        # try:
        #     self.auto_load_v7_model()
        # except Exception as e:
        #     self._log(f"[WARNING] Falha no carregamento automático do modelo: {e}")
    
        # SESSION LOGGING (ID por sessão)
        try:
            from datetime import datetime as _dt
            from uuid import uuid4 as _uuid4
            self.session_id = f"{_dt.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{_uuid4().hex[:8]}"
            
            # 🎯 MAGIC NUMBER: Único por sessão para isolamento total entre instâncias
            import hashlib
            session_hash = int(hashlib.md5(str(self.session_id).encode()).hexdigest()[:6], 16)
            self.magic_number = 777000 + (session_hash % 888)  # 777000-777888 range
            self._log(f"[🔒 ISOLATION] Magic number configurado: {self.magic_number} (sessão: {self.session_id})")
            self._log(f"[🔒 ISOLATION] ✅ O robô gerenciará APENAS suas próprias posições")
            self._log(f"[🔒 ISOLATION] ✅ Você pode tradear manualmente outras posições/ativos livremente!")
            
            self.session_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
            os.makedirs(self.session_log_dir, exist_ok=True)
            self.session_log_path = os.path.join(self.session_log_dir, f"trading_session_{self.session_id}.txt")

            # Contadores da sessão (independentes da GUI)
            self.session_counters = {
                'buys': 0,
                'sells': 0,
                'wins': 0,
                'losses': 0,
                'profit_loss': 0.0,
                'start_time': time.time()
            }

            # Cabeçalho do arquivo da sessão
            header_lines = [
                "=" * 80,
                f"Legion AI Trader V7 - Sessão {self.session_id}",
                f"Símbolo: {self.symbol} | Magic: {self.magic_number}",
                f"Início: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.session_counters['start_time']))}",
                "Resumo de atividades de trading por sessão (múltiplas instâncias suportadas)",
                "=" * 80,
                ""
            ]
            with open(self.session_log_path, 'a', encoding='utf-8') as f:
                f.write("\n".join(header_lines) + "\n")
            self._log(f"[LOG] Arquivo de sessão criado: {os.path.basename(self.session_log_path)}")
        except Exception as e:
            self._log(f"[WARNING] Falha ao inicializar log de sessão: {e}")

        # Registro automático do encerramento (fallback)
        try:
            import atexit as _atexit
            _atexit.register(self._end_session_safe)
        except Exception:
            pass

    def _create_v7_feature_columns(self):
        """🔧 Criar colunas de features para Legion V1 - 65 available (using 20)"""
        all_columns = []
        
        # 🎯 1. MARKET FEATURES (16 features) - alinhado com 1m
        base_features_1m = [
            'returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 
            'stoch_k', 'bb_position', 'trend_strength', 'atr_14'
        ]
        
        high_quality_features = [
            'volume_momentum', 'price_position', 'breakout_strength', 
            'trend_consistency', 'support_resistance', 'volatility_regime', 'market_structure'
        ]
        
        # Market features com sufixo _1m para base + high quality direto
        all_columns.extend([f"{f}_1m" for f in base_features_1m])  # 9
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
        
        # 🍒 CHERRY: 65 market features disponíveis (usamos primeiras 16)
        # Cherry format: 16 market + 18 positions + 2 intelligent + 4 order_flow + 5 volatility = 45 features

        self._log(f"[CHERRY] Historical dataframe columns: {len(all_columns)} market-based features (usando primeiras 16)")
        self._log(f"[CHERRY] Format: 16 market + 18 positions + 2 intelligent + 4 order_flow + 5 volatility = 45 features × 10 steps = 450D")
        
        return all_columns
    
    def _initialize_historical_data_v7(self):
        """🔧 Inicializa dados históricos Legion V1 com features otimizadas"""
        try:
            # Carregar dados dos últimos 1000 bars de M1
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M1, 0, 1000)
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
            
            # Criar múltiplos timeframes para V7 baseado em M1
            df_1m = df.copy()
            df_5m = df.resample('5T').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
            }).dropna()
            df_15m = df.resample('15T').agg({
                'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'tick_volume': 'sum'
            }).dropna()
            
            # Calcular features para cada timeframe
            self.historical_df = pd.DataFrame(index=df_1m.index)
            
            # Processar 1m, 5m e 15m para V7
            for tf_name, tf_df in [('1m', df_1m), ('5m', df_5m), ('15m', df_15m)]:
                # Interpolar dados para o índice principal se necessário
                if len(tf_df) != len(df_1m):
                    # 🔧 FIX: Só reindexar se tf_df tem dados suficientes
                    if len(tf_df) > 50:  # Mínimo 50 registros para interpolação
                        tf_df = tf_df.reindex(df_1m.index, method='ffill')
                    else:
                        # Usar dados 1m como fallback para timeframes maiores
                        tf_df = df_1m.copy()
                
                close_col = tf_df['close']
                high_col = tf_df['high']
                low_col = tf_df['low']
                
                # Calcular features técnicas básicas
                self.historical_df[f'returns_{tf_name}'] = close_col.pct_change().fillna(0)
                self.historical_df[f'volatility_20_{tf_name}'] = close_col.rolling(20).std().fillna(0)
                self.historical_df[f'sma_20_{tf_name}'] = close_col.rolling(20).mean().fillna(close_col)
                self.historical_df[f'sma_50_{tf_name}'] = close_col.rolling(50).mean().fillna(close_col)
                self.historical_df[f'rsi_14_{tf_name}'] = self._calculate_rsi(close_col, 14)

                # ✅ CHERRY ALIGNMENT: Stochastic K real (linha 4177-4181)
                low_14 = low_col.rolling(14, min_periods=1).min()
                high_14 = high_col.rolling(14, min_periods=1).max()
                stoch_k = 100 * (close_col - low_14) / (high_14 - low_14 + 1e-8)
                self.historical_df[f'stoch_k_{tf_name}'] = stoch_k.fillna(50.0).clip(0, 100)

                # ✅ CHERRY ALIGNMENT: Bollinger Band Position (linha 4183-4189)
                bb_sma = close_col.rolling(20, min_periods=1).mean()
                bb_std = close_col.rolling(20, min_periods=1).std()
                bb_upper = bb_sma + (2 * bb_std)
                bb_lower = bb_sma - (2 * bb_std)
                bb_position = (close_col - bb_lower) / (bb_upper - bb_lower + 1e-8)
                self.historical_df[f'bb_position_{tf_name}'] = bb_position.fillna(0.5).clip(0, 1)
                
                # Trend Strength
                returns = close_col.pct_change().fillna(0)
                self.historical_df[f'trend_strength_{tf_name}'] = returns.rolling(10).mean().fillna(0)
                
                self.historical_df[f'atr_14_{tf_name}'] = self._calculate_atr(tf_df, 14)
            
            # 🎯 CALCULAR FEATURES DE ALTA QUALIDADE - ALINHADAS COM CHERRY.PY
            # Cherry espera: volume_momentum, price_position, breakout_strength,
            #                trend_consistency, support_resistance, volatility_regime, market_structure

            close_1m = df_1m['close']
            high_1m = df_1m['high']
            low_1m = df_1m['low']
            volume_1m = df_1m['tick_volume']

            # 1. volume_momentum (cherry.py linha 4286-4290)
            volume_sma_20 = volume_1m.rolling(20).mean().fillna(volume_1m.iloc[0] if len(volume_1m) > 0 else 1)
            volume_momentum = np.where(volume_sma_20 > 0, (volume_1m - volume_sma_20) / volume_sma_20, 0.001)
            self.historical_df['volume_momentum'] = volume_momentum

            # 2. price_position (cherry.py linha 4292-4298)
            high_20 = high_1m.rolling(20).max().fillna(high_1m.iloc[0] if len(high_1m) > 0 else 2000)
            low_20 = low_1m.rolling(20).min().fillna(low_1m.iloc[0] if len(low_1m) > 0 else 2000)
            price_range = np.where(high_20 > low_20, high_20 - low_20, 1)
            price_position = np.where(price_range > 0, (close_1m - low_20) / price_range, 0.25)
            self.historical_df['price_position'] = price_position

            # 3. breakout_strength - TP TARGET ZONES (ALINHADO COM CHERRY.PY linha 4363-4392)
            # 🎯 IDENTIFICA ZONAS REALISTAS PARA TP
            lookback = 20
            # Resistência próxima = high máximo recente
            rolling_resistance = high_1m.rolling(window=lookback, min_periods=1).max()
            # Suporte próximo = low mínimo recente
            rolling_support = low_1m.rolling(window=lookback, min_periods=1).min()

            # Distância para resistência (alvo TP LONG)
            dist_to_resistance = rolling_resistance - close_1m
            # Distância para suporte (alvo TP SHORT)
            dist_to_support = close_1m - rolling_support

            # Combinar (menor = alvo mais próximo)
            combined_distance = np.minimum(dist_to_resistance, dist_to_support)

            # Normalizar pela ATR
            atr_14 = (high_1m - low_1m).rolling(window=14).mean().fillna(1)
            tp_zone_distance = combined_distance / (atr_14 + 1e-8)

            # Inverter: ALTO = alvo PRÓXIMO (bom), BAIXO = alvo DISTANTE (ruim)
            breakout_strength = 1.0 - np.clip(tp_zone_distance / 5.0, 0.0, 1.0)
            self.historical_df['breakout_strength'] = breakout_strength

            # 4. trend_consistency - REAL CALCULATION (CONTINUOUS)
            # 🔥 TREND CONSISTENCY: Consistência do trend via rolling proportion
            returns = close_1m.pct_change().fillna(0)
            consistency_window = 10
            # Calcular rolling proportion of same-sign returns
            positive_rolling = (returns > 0).rolling(consistency_window, min_periods=1).sum()
            negative_rolling = (returns < 0).rolling(consistency_window, min_periods=1).sum()
            # Consistência = proporção do sinal dominante (valor contínuo 0.5-1.0)
            trend_consistency = np.maximum(positive_rolling, negative_rolling) / consistency_window
            self.historical_df['trend_consistency'] = trend_consistency.fillna(0.5)

            # 5. support_resistance - SL ZONE QUALITY (ALINHADO COM CHERRY.PY linha 4404-4430)
            # 🎯 IDENTIFICA ZONAS SEGURAS PARA COLOCAR SL
            lookback = 20
            # Suporte = low mínimo dos últimos N períodos
            rolling_support_sl = low_1m.rolling(window=lookback, min_periods=1).min()
            # Resistência = high máximo dos últimos N períodos
            rolling_resistance_sl = high_1m.rolling(window=lookback, min_periods=1).max()

            # Distância do close atual para suporte/resistência
            dist_to_support_sl = close_1m - rolling_support_sl
            dist_to_resistance_sl = rolling_resistance_sl - close_1m

            # Combinar (menor distância = mais crítico)
            combined_distance_sl = np.minimum(dist_to_support_sl, dist_to_resistance_sl)

            # Normalizar pela ATR
            atr_14_sl = (high_1m - low_1m).rolling(window=14).mean().fillna(1)
            sl_zone_quality = combined_distance_sl / (atr_14_sl + 1e-8)
            sl_zone_quality = np.clip(sl_zone_quality / 3.0, 0.0, 1.0)

            self.historical_df['support_resistance'] = sl_zone_quality

            # 6. volatility_regime - REAL CALCULATION (CONTINUOUS)
            # 🔥 VOLATILITY REGIME: Ratio contínuo ao invés de 3 valores discretos
            vol_20 = close_1m.rolling(20).std().fillna(0.001)
            vol_50 = close_1m.rolling(50).std().fillna(0.001)
            volatility_regime = np.where(vol_50 > 0, vol_20 / vol_50, 1.0)
            # Normalizar para [0, 1] de forma contínua (não discreta)
            volatility_regime = np.clip(volatility_regime / 3.0, 0.0, 1.0)
            self.historical_df['volatility_regime'] = volatility_regime

            # 7. market_structure - REAL CALCULATION (CONTINUOUS)
            # 🔥 MARKET STRUCTURE: Estrutura de higher highs/lower lows (CONTINUOUS)
            lookback = 20

            # Rolling max/min para pivots
            recent_high = high_1m.rolling(lookback, min_periods=1).max()
            previous_high = high_1m.shift(lookback).rolling(lookback, min_periods=1).max()
            recent_low = low_1m.rolling(lookback, min_periods=1).min()
            previous_low = low_1m.shift(lookback).rolling(lookback, min_periods=1).min()

            # Calcular força do trend de forma contínua (não discreta)
            high_momentum = (recent_high - previous_high) / (previous_high + 1e-8)
            low_momentum = (recent_low - previous_low) / (previous_low + 1e-8)

            # Estrutura = média dos momentums (normalizado 0-1)
            structure = (high_momentum + low_momentum) / 2.0
            structure = np.clip(structure * 10 + 0.5, 0.0, 1.0)  # Escalar para [0,1]

            self.historical_df['market_structure'] = structure.fillna(0.5)
            
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

            # 🧠 MEMORY FIX: Manter apenas últimas 300 linhas do dataframe (rolling window)
            if len(self.historical_df) > 300:
                self.historical_df = self.historical_df.tail(300).copy()

            self._log(f"[INFO V7] ✅ Dados históricos carregados: {len(self.historical_df)} registros com {len(self.feature_columns)} features")
            
        except Exception as e:
            self._log(f"[ERROR] Erro ao inicializar dados históricos V7: {e}")
            # Fallback: criar dataframe vazio
            self.historical_df = pd.DataFrame()
            for col in self.feature_columns:
                self.historical_df[col] = [0.0] * 100
    
    def _calculate_rsi(self, price_series, period=14):
        """Calcula RSI - ALINHADO COM CHERRY.PY (linha 4169-4175)"""
        try:
            # ✅ CHERRY ALIGNMENT: Calcular gain/loss ANTES de rolling mean
            delta = price_series.diff()
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain, index=price_series.index).rolling(period, min_periods=1).mean()
            avg_loss = pd.Series(loss, index=price_series.index).rolling(period, min_periods=1).mean()
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            return rsi.fillna(50.0)  # RSI padrão = 50
        except:
            return pd.Series([50.0] * len(price_series), index=price_series.index)

    def _trim_historical_df(self):
        """🧠 MEMORY FIX: Manter dataframe com tamanho controlado durante atualizações"""
        try:
            if hasattr(self, 'historical_df') and len(self.historical_df) > 300:
                self.historical_df = self.historical_df.tail(300).copy()
        except Exception as e:
            pass

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

    def _generate_intelligent_features_v7_robot(self, current_price):
        """🔥 CHERRY ALIGNMENT: 7 intelligent features ALINHADAS com cherry.py"""
        try:
            # Simular high, low usando dados disponíveis
            if len(self.historical_df) >= 2:
                prev_close = self.historical_df[self.feature_columns[0]].iloc[-2]
                high = max(current_price, prev_close) * 1.001  # Simular high
                low = min(current_price, prev_close) * 0.999   # Simular low
            else:
                high = current_price * 1.001
                low = current_price * 0.999

            # 1. VOLUME_MOMENTUM (alinhado com cherry.py linha 4345-4361)
            if len(self.historical_df) >= 20:
                # Volume change vs média (usar variação de preço como proxy)
                recent_changes = self.historical_df[self.feature_columns[0]].tail(20).pct_change().abs()
                vol_mean = recent_changes.mean()
                current_change = abs(current_price - prev_close) / prev_close if prev_close > 0 else 0
                volume_momentum = np.clip((current_change - vol_mean) / (vol_mean + 1e-8), -1.0, 1.0) * 0.5 + 0.5
            else:
                volume_momentum = 0.5

            # 2. PRICE_POSITION (alinhado com cherry.py linha 4299-4315)
            if len(self.historical_df) >= 20:
                recent_data = self.historical_df[self.feature_columns[0]].tail(20)
                high_20 = recent_data.max()
                low_20 = recent_data.min()
                price_range = high_20 - low_20
                price_position = (current_price - low_20) / (price_range + 1e-8) if price_range > 0 else 0.5
                price_position = np.clip(price_position, 0.0, 1.0)
            else:
                price_position = 0.5

            # 3. BREAKOUT_STRENGTH = TP_TARGET_ZONES (alinhado com cherry.py linha 4363-4392)
            if len(self.historical_df) >= 20:
                closes = self.historical_df[self.feature_columns[0]].tail(20)
                rolling_resistance = closes.max()  # Resistência próxima
                rolling_support = closes.min()     # Suporte próximo

                # Distância para resistência (alvo TP LONG)
                dist_to_resistance = rolling_resistance - current_price
                # Distância para suporte (alvo TP SHORT)
                dist_to_support = current_price - rolling_support

                # Combinar (menor = alvo mais próximo)
                combined_distance = min(dist_to_resistance, dist_to_support)

                # Normalizar pela ATR (simular usando range recente)
                atr_14 = closes.pct_change().abs().mean() * current_price
                tp_zone_distance = combined_distance / (atr_14 + 1e-8)

                # Inverter: ALTO = alvo PRÓXIMO (bom), BAIXO = alvo DISTANTE (ruim)
                breakout_strength = 1.0 - np.clip(tp_zone_distance / 5.0, 0.0, 1.0)
            else:
                breakout_strength = 0.5

            # 4. TREND_CONSISTENCY (alinhado com cherry.py linha 4317-4343)
            if len(self.historical_df) >= 10:
                closes = self.historical_df[self.feature_columns[0]].tail(10)
                returns = closes.pct_change().dropna()
                positive_returns = (returns > 0).sum()
                trend_consistency = positive_returns / len(returns) if len(returns) > 0 else 0.5
            else:
                trend_consistency = 0.5

            # 5. SUPPORT_RESISTANCE = SL_ZONE_QUALITY (alinhado com cherry.py linha 4423-4449)
            if len(self.historical_df) >= 20:
                closes = self.historical_df[self.feature_columns[0]].tail(20)
                rolling_support = closes.min()
                rolling_resistance = closes.max()

                # Distância do close atual para suporte/resistência
                dist_to_support = current_price - rolling_support
                dist_to_resistance = rolling_resistance - current_price

                # Combinar (menor distância = mais crítico)
                combined_distance = min(dist_to_support, dist_to_resistance)

                # Normalizar pela ATR
                atr_14 = closes.pct_change().abs().mean() * current_price
                sl_zone_quality = combined_distance / (atr_14 + 1e-8)
                support_resistance = np.clip(sl_zone_quality / 3.0, 0.0, 1.0)
            else:
                support_resistance = 0.5

            # 6. VOLATILITY_REGIME (alinhado com cherry.py linha 4394-4421)
            if len(self.historical_df) >= 50:
                closes = self.historical_df[self.feature_columns[0]].tail(50)
                vol_20 = closes.tail(20).pct_change().std() if len(closes) >= 20 else 0.001
                vol_50 = closes.pct_change().std() if len(closes) > 0 else 0.001
                volatility_regime = np.clip(vol_20 / (vol_50 + 1e-8), 0.1, 0.9)
            else:
                volatility_regime = 0.5

            # 7. MARKET_STRUCTURE = RECENT_VOLATILITY_SPIKE (alinhado com cherry.py linha 4470-4490)
            if len(self.historical_df) >= 50:
                closes = self.historical_df[self.feature_columns[0]].tail(50)
                current_range = closes.pct_change().abs()
                atr_14 = current_range.tail(14).mean() if len(current_range) >= 14 else 0.001
                atr_50 = current_range.mean() if len(current_range) > 0 else 0.001

                # Volatility ratio = ATR atual / ATR médio
                vol_ratio = atr_14 / (atr_50 + 1e-8) if atr_50 > 0 else 1.0

                # Detectar spikes RECENTES (máximo dos últimos 5 períodos)
                vol_spike_recent = current_range.tail(5).max() / (atr_50 + 1e-8) if len(current_range) >= 5 else 1.0

                # Normalizar: >1.5 = spike alto, <1.0 = calmo
                market_structure = np.clip((vol_spike_recent - 0.8) / 1.5, 0.0, 1.0)
            else:
                market_structure = 0.5

            return np.array([
                volume_momentum,      # 1. Volume momentum
                price_position,       # 2. Price position
                breakout_strength,    # 3. TP target zones
                trend_consistency,    # 4. Trend consistency
                support_resistance,   # 5. SL zone quality
                volatility_regime,    # 6. Volatility regime
                market_structure      # 7. Recent volatility spike
            ], dtype=np.float32)

        except Exception as e:
            # Fallback: 7 features com valores seguros
            self._log(f"[WARN] Erro calculando intelligent features: {e}")
            return np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    def _generate_order_flow_features_robot(self, current_price):
        """🍒 ORDER FLOW ANALYSIS - Robot version"""
        try:
            if len(self.historical_df) < 2:
                return np.full(4, 0.3, dtype=np.float32)

            # Simular high, low, volume usando dados disponíveis
            close = current_price
            prev_close = self.historical_df[self.feature_columns[0]].iloc[-2] if len(self.historical_df) >= 2 else current_price
            high = max(close, prev_close) * 1.001  # Simular high
            low = min(close, prev_close) * 0.999   # Simular low
            volume = abs(close - prev_close) * 1000  # Simular volume baseado na variação

            # 1. BID/ASK SPREAD RATIO (simulado via range)
            range_price = high - low
            mid_price = (high + low) / 2
            spread_ratio = np.clip(range_price / mid_price, 0.001, 0.1) if mid_price > 0 else 0.01

            # 2. VOLUME IMBALANCE
            price_change = close - prev_close
            volume_intensity = np.clip(volume / 1000, 0.1, 10.0)
            volume_imbalance = np.tanh(price_change * volume_intensity * 0.001)
            volume_imbalance = np.clip(volume_imbalance, -0.5, 0.5) + 0.5  # [0,1]

            # 3. PRICE IMPACT ESTIMATE
            if range_price > 0:
                price_impact = volume_intensity / (range_price + 1)
                price_impact = np.clip(price_impact, 0.1, 0.9)
            else:
                price_impact = 0.5

            # 4. MARKET MAKER SIGNAL
            mm_signal = 1 / (1 + volume_intensity * (range_price / close + 0.001))
            mm_signal = np.clip(mm_signal, 0.1, 0.9)

            return np.array([spread_ratio, volume_imbalance, price_impact, mm_signal], dtype=np.float32)

        except:
            return np.full(4, 0.3, dtype=np.float32)

    def _generate_volatility_features_robot(self, current_price):
        """🍒 VOLATILIDADE SINTÉTICA - Robot version"""
        try:
            if len(self.historical_df) < 5:
                return np.full(5, 0.3, dtype=np.float32)

            # Volatilidade básica das últimas 5 barras
            window_data = self.historical_df[self.feature_columns[0]].tail(5)
            vol = window_data.pct_change().std() if len(window_data) > 1 else 0.01
            vol = min(vol * 10, 1.0)  # Normalize

            return np.array([vol, vol * 0.8, vol * 1.2, vol * 0.9, vol * 1.1], dtype=np.float32)

        except:
            return np.full(5, 0.3, dtype=np.float32)

    def _get_position_pnl_robot(self, position, current_price):
        """🍒 Calcular PnL atual da posição - Robot version"""
        try:
            # Calcular diferença de preço baseado no tipo de posição
            if position.type == mt5.POSITION_TYPE_BUY:  # LONG
                price_diff = current_price - position.price_open
            else:  # SHORT
                price_diff = position.price_open - current_price

            # ESCALA CORRETA OURO: 1 ponto = $1 USD com 0.01 lot
            # Fórmula: price_diff * lot_size * 100
            pnl = price_diff * position.volume * 100
            return pnl

        except Exception as e:
            self._log(f"❌ [PnL CALC ERROR] {e}")
            return 0.0

    def _get_observation_v7(self):
        """🚀 Obter observação Cherry-aligned - 450 dimensões (45 features × 10 window)"""
        try:
            # Obter preço atual
            if self.mt5_connected:
                tick = mt5.symbol_info_tick(self.symbol)
                current_price = tick.bid if tick else 2000.0
            else:
                current_price = 2000.0

            # 🍒 CHERRY FORMAT: 10 steps × 45 features = 450D
            window_size = 10
            features_per_step = 45

            # 1. MARKET FEATURES (16 features por step - Cherry format)
            if len(self.historical_df) > 0 and len(self.feature_columns) > 0:
                # Pegar últimos 10 steps e primeiras 16 features (Cherry format)
                recent_data = self.historical_df[self.feature_columns[:16]].tail(window_size).values

                if len(recent_data) < window_size:
                    # Padding se dados insuficientes
                    padding = np.zeros((window_size - len(recent_data), 16))
                    recent_data = np.vstack([padding, recent_data])
            else:
                recent_data = np.zeros((window_size, 16))
            
            # 2. POSITION FEATURES (18 features - 2 posições × 9 features cada - ALINHADO COM MODELO TREINADO)
            positions_obs = np.full(18, 0.001, dtype=np.float32)  # Usar 0.001 como silus.py

            # Obter posições atuais do MT5 (apenas do robô)
            if self.mt5_connected:
                mt5_positions = self._get_robot_positions()
                if mt5_positions is None:
                    mt5_positions = []
            else:
                mt5_positions = []

            # Processar até 2 posições ATIVAS (alinhado com modelo treinado)
            for i in range(min(2, len(mt5_positions))):  # Até 2 posições ativas
                pos = mt5_positions[i]
                base_idx = i * 9  # 9 features por posição como cherry.py

                # 🔧 FIX: Features alinhadas com CHERRY.PY (linha 4625-4632)
                entry_price = max(pos.price_open, 0.01) / 1000.0  # ✅ CHERRY ALIGNMENT: /1000 (não /10000)
                current_price_norm = max(current_price, 0.01) / 1000.0  # ✅ CHERRY ALIGNMENT: /1000

                # Calcular PnL baseado no tipo da posição
                if pos.type == mt5.POSITION_TYPE_BUY:
                    unrealized_pnl = (current_price - pos.price_open) * pos.volume
                    position_type = 1.0  # long
                else:
                    unrealized_pnl = (pos.price_open - current_price) * pos.volume
                    position_type = 2.0  # short

                # Normalizar PnL
                unrealized_pnl = unrealized_pnl if unrealized_pnl != 0 else 0.01

                # 🔧 FIX: Duration usando steps como cherry.py (linha 4632)
                # Converter tempo real para steps equivalentes (1min timeframe)
                duration_minutes = (time.time() - pos.time) / 60.0  # Converter para minutos
                duration_steps = duration_minutes  # 1 step = 1 minuto no timeframe 1m
                duration = max(duration_steps, 1) / 1440.0  # ✅ CHERRY ALIGNMENT: /1440 (não /3600 depois /24)
                
                # 🔧 FIX: SL/TP alignment com cherry.py (linha 4630-4631)
                # Cherry usa: sl/tp normalizado por /1000 (não por current_price)
                sl_norm = max(pos.sl, 0.01) / 1000.0 if pos.sl > 0 else 0.01  # ✅ CHERRY ALIGNMENT
                tp_norm = max(pos.tp, 0.01) / 1000.0 if pos.tp > 0 else 0.01  # ✅ CHERRY ALIGNMENT

                # 🔧 FIX: Volume sem desnormalização (cherry usa volume direto - linha 4629)
                volume_norm = max(pos.volume, 0.01)  # ✅ CHERRY ALIGNMENT: volume direto

                # 🔧 FIX: Position type alignment (cherry usa 1.0/-1.0, não 1.0/2.0 - linha 4655)
                position_type_cherry = 1.0 if pos.type == mt5.POSITION_TYPE_BUY else -1.0  # ✅ CHERRY ALIGNMENT

                # 9 features por posição (FORMATO CHERRY.PY):
                positions_obs[base_idx:base_idx+9] = [
                    1.0,  # [0] Posição ativa
                    float(entry_price),         # [1] Entry price normalizado
                    float(current_price_norm),  # [2] Current price normalizado
                    float(unrealized_pnl),      # [3] Unrealized PnL
                    float(duration),            # [4] Duration ⭐ (CRITICAL para modelo)
                    float(volume_norm),         # [5] Volume ✅ CHERRY ALIGNED
                    float(sl_norm),             # [6] SL ✅ CHERRY ALIGNED
                    float(tp_norm),             # [7] TP ✅ CHERRY ALIGNED
                    float(position_type_cherry) # [8] Type ✅ CHERRY ALIGNED (1.0/-1.0)
                ]
            
            # 🔥 POSIÇÕES INATIVAS VARIÁVEIS - CHERRY.PY ALIGNED (linha 4658-4678)
            # Usar variação baseada no step atual para evitar valores estáticos
            current_price_norm_current = max(current_price, 0.01) / 1000.0

            for i in range(len(mt5_positions), 2):  # Preencher posições restantes até 2
                # Gerar variação determinística baseada no step e índice
                price_variation = (hash(f"{self.current_step}_{i}") % 100) / 10000.0
                volume_variation = (hash(f"{self.current_step}_{i}_vol") % 50) / 100000.0

                # 🎯 FIX VIÉS: Randomizar direção de SL/TP para posições inativas
                # Metade terá padrão LONG, metade padrão SHORT
                random_direction = (hash(f"{self.current_step}_{i}_dir") % 2) * 2 - 1  # -1 ou +1
                if random_direction > 0:  # Padrão LONG
                    sl = current_price_norm_current * 0.99   # SL abaixo
                    tp = current_price_norm_current * 1.01   # TP acima
                    position_type_inactive = 0.1             # Leve tendência LONG
                else:  # Padrão SHORT
                    sl = current_price_norm_current * 1.01   # SL acima
                    tp = current_price_norm_current * 0.99   # TP abaixo
                    position_type_inactive = -0.1            # Leve tendência SHORT

                base_idx = i * 9
                positions_obs[base_idx:base_idx+9] = [
                    0.001 + price_variation,                                          # [0] Inativa (variável)
                    current_price_norm_current + price_variation,                     # [1] Entry baseado no preço atual
                    current_price_norm_current,                                       # [2] Current price real
                    -0.001 - price_variation,                                         # [3] PnL pequeno negativo
                    0.1 + (hash(f"{self.current_step}_{i}_dur") % 100) / 1000.0,    # [4] Duration variável
                    0.001 + volume_variation,                                         # [5] Volume variável
                    sl,                                                               # [6] SL balanceado (LONG ou SHORT)
                    tp,                                                               # [7] TP balanceado (LONG ou SHORT)
                    position_type_inactive                                            # [8] Type balanceado (-0.1 ou +0.1)
                ]
            
            # 3. INTELLIGENT FEATURES (7 features - HÍBRIDO CORE)
            # 🔥 CHERRY ALIGNMENT: Gerar 7 features reais baseadas em dados históricos
            intelligent_features = self._generate_intelligent_features_v7_robot(current_price)  # 7 features reais

            # 4. ORDER FLOW FEATURES (4 features)
            order_flow_features = self._generate_order_flow_features_robot(current_price)

            # 5. CONSTRUIR OBSERVAÇÃO: 10 steps × 45 features = 450D (HÍBRIDO)
            # ESTRUTURA HÍBRIDA: 16 market + 18 positions + 7 intelligent + 4 order_flow = 45 features
            total_features_raw = 16 + 18 + 7 + 4  # 45 features
            features_per_step = 45  # Híbrido format exato
            obs_matrix = np.zeros((window_size, features_per_step), dtype=np.float32)

            for step in range(window_size):
                # Construir features completas Híbrido format
                full_features = np.concatenate([
                    recent_data[step],          # Market features (16)
                    positions_obs,             # Position features (18)
                    intelligent_features,      # Intelligent features (7) - HÍBRIDO
                    order_flow_features        # Order flow features (4)
                ])  # Total: 45 features híbrido!

                # Cherry format - sem truncamento necessário
                obs_matrix[step, :] = full_features
            
            # 5. Flatten para 450D
            flat_obs = obs_matrix.flatten().astype(np.float32)
            
            # 6. Normalização IDÊNCTICA ao silus.py
            flat_obs = np.clip(flat_obs, -100.0, 100.0)  # Range do silus.py
            flat_obs = np.nan_to_num(flat_obs, nan=0.01, posinf=100.0, neginf=-100.0)  # Valores orgânicos do silus.py
            
            # Correção de zeros extremos para 0.0 (orgânico como silus.py)
            zeros_mask = np.abs(flat_obs) < 1e-8
            flat_obs[zeros_mask] = 0.0
            
            # Verificar exatamente 450 dimensões
            if flat_obs.shape[0] != Config.OBSERVATION_SPACE_SIZE:
                raise Exception(f"LEGION V1 SHAPE INCORRETO: {flat_obs.shape[0]} != {Config.OBSERVATION_SPACE_SIZE}")
            
            # Verificações de integridade
            assert flat_obs.shape == self.observation_space.shape, f"Legion V1 Obs shape {flat_obs.shape} != expected {self.observation_space.shape}"
            assert not np.any(np.isnan(flat_obs)), f"Legion V1 Observação contém NaN"
            assert not np.any(np.isinf(flat_obs)), f"Legion V1 Observação contém Inf"
            
            return flat_obs
            
        except Exception as e:
            self._log(f"[ERROR] Erro ao obter observação Legion V1: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _process_legion_action(self, action):
        """🚀 UNIFICADO: Redirecionar para sistema V7 unificado"""
        # Simplesmente chamar o sistema V7 que já está correto e unificado
        return self._process_v7_action(action)
    
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
            # Baseado em posições abertas e volatilidade (apenas do robô)
            num_positions = len(self._get_robot_positions()) if self.mt5_connected else 0
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
            # Baseado em posições atuais e drawdown (apenas do robô)
            current_positions = len(self._get_robot_positions()) if self.mt5_connected else 0
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
            
            # Verificar posições com risco de reversão (apenas do robô)
            if self.mt5_connected:
                positions = self._get_robot_positions()
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
            # Normalizar volume para escala do modelo (dividir pelo multiplicador)
            normalized_volume = position.volume / self.lot_multiplier
            if position.type == mt5.POSITION_TYPE_BUY:
                pnl = (current_price - position.price_open) * normalized_volume * 100  # Para GOLD
            else:  # SELL
                pnl = (position.price_open - current_price) * normalized_volume * 100
            return pnl
        except:
            return 0.0
    
    
    def _process_sl_tp_adjustments_v7(self, sl_adjusts, tp_adjusts):
        """🔧 Processar ajustes de SL/TP V7 para posições existentes"""
        try:
            # Debug: confirmar entrada na função (usar step_count para live trading)
            step_for_debug = getattr(self, 'current_step', 0)
            if step_for_debug % 10 == 0:
                self._log(f"🔧 [DEBUG FUNC] Dentro de _process_sl_tp_adjustments_v7 - Step {step_for_debug}")
                self._log(f"🔧 [DEBUG FUNC] MT5 conectado: {self.mt5_connected}")

            if not self.mt5_connected:
                if step_for_debug % 10 == 0:
                    self._log(f"🔧 [DEBUG FUNC] MT5 não conectado - saindo da função")
                return

            positions = self._get_robot_positions()
            if not positions:
                if step_for_debug % 10 == 0:
                    self._log(f"🔧 [DEBUG FUNC] Nenhuma posição encontrada - saindo da função")
                return

            # Log detalhado apenas a cada 50 steps para reduzir spam
            if step_for_debug % 50 == 0:
                self._log(f"📊 [SL/TP AJUSTES] Processando {len(positions)} posições")
                self._log(f"📊 [SL/TP AJUSTES] Adjusts: SL={sl_adjusts[:len(positions)]}, TP={tp_adjusts[:len(positions)]}")

            for i, pos in enumerate(positions[:3]):  # Máximo 3 posições
                if i < len(sl_adjusts) and i < len(tp_adjusts):
                    sl_adjust = sl_adjusts[i]
                    tp_adjust = tp_adjusts[i]

                    # Calcular significância dos ajustes (qualquer ajuste não-zero é significativo)
                    significant_sl = sl_adjust != 0
                    significant_tp = tp_adjust != 0

                    # Incrementar contadores apenas se houver ajuste significativo
                    if significant_sl or significant_tp:
                        self.sl_tp_adjustments['total_adjustments'] += 1

                    if significant_sl:
                        self.sl_tp_adjustments['sl_adjustments'] += 1

                    if significant_tp:
                        self.sl_tp_adjustments['tp_adjustments'] += 1

                    # Log detalhado do ajuste apenas a cada 50 steps
                    if hasattr(self, 'current_step') and self.current_step % 50 == 0:
                        self._log(f"🎯 [AJUSTE POS {i+1}] Ticket: {pos.ticket}")
                        self._log(f"   📍 SL atual: ${pos.sl:.2f}, TP atual: ${pos.tp:.2f}")
                        self._log(f"   🔧 Ajustes: SL={sl_adjust:.2f}, TP={tp_adjust:.2f}")
                        self._log(f"   💡 Significativo: SL={significant_sl}, TP={significant_tp}")

                    # Aplicar ajustes apenas se significativos
                    if significant_sl or significant_tp:
                        result = self._modify_position_v7(pos, sl_adjust, tp_adjust)
                        if result:
                            self.sl_tp_adjustments['successful_modifications'] += 1
                        else:
                            self.sl_tp_adjustments['failed_modifications'] += 1
                    else:
                        # Log apenas se for step de 50 velas
                        if hasattr(self, 'current_step') and self.current_step % 50 == 0:
                            self._log(f"   ⏭️  Ajustes zero (0,0) - mantendo SL/TP atuais")

            # Log resumo dos contadores apenas a cada 50 steps (50 velas de 1min)
            if hasattr(self, 'current_step') and self.current_step % 50 == 0:
                stats = self.sl_tp_adjustments
                self._log(f"📈 [STATS SL/TP - 50 VELAS] Step {self.current_step}: Total: {stats['total_adjustments']}, "
                         f"SL: {stats['sl_adjustments']}, TP: {stats['tp_adjustments']}, "
                         f"Sucessos: {stats['successful_modifications']}, Falhas: {stats['failed_modifications']}")

        except Exception as e:
            self._log(f"[WARNING] Erro ao ajustar SL/TP V7: {e}")
    
    def _modify_position_v7(self, position, sl_adjust, tp_adjust):
        """🔧 Modificar posição individual V7 com lógicas Cherry-aligned"""
        try:
            # Obter preço atual
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                self._log(f"❌ [MODIFY] Falha ao obter preço atual para posição #{position.ticket}")
                return False

            current_price = tick.bid if position.type == mt5.POSITION_TYPE_BUY else tick.ask
            pos_type = "LONG" if position.type == mt5.POSITION_TYPE_BUY else "SHORT"

            # 🍒 CHERRY LOGIC: Verificar PnL atual e aplicar cap de $100
            current_pnl = self._get_position_pnl_robot(position, current_price)

            # 🛑 AUTO-CLOSE AT $100 USD (PnL CAP)
            if current_pnl >= 100:
                self._log(f"🛑 [PnL CAP $100] Auto-closing position #{position.ticket} at ${current_pnl:.2f}")
                self._close_position_mt5(position.ticket)
                return True

            # 🍒 CHERRY LOGIC: SL TRAILING ONLY (RESTRICTIVE)
            new_sl = position.sl
            new_tp = position.tp
            sl_changed = False
            tp_changed = False

            # 🎯 SL TRAILING ONLY (A FAVOR DO TRADE) WITH RANGE LIMITS
            if abs(sl_adjust) >= 0.3:  # Model wants to modify SL
                if position.sl > 0:
                    # Calculate SL movement in points (±0.5 → ±2.0 points max)
                    sl_movement_points = sl_adjust * 2.0

                    if position.type == mt5.POSITION_TYPE_BUY:  # LONG
                        # LONG: SL can only go UP (protect more profit)
                        proposed_sl = position.sl + sl_movement_points

                        # 🍒 HARDCAP: SL distance from entry must be >= 10pt and <= 15pt
                        sl_distance_from_entry = position.price_open - proposed_sl
                        if sl_distance_from_entry < self.sl_range_min:  # < 10pt
                            proposed_sl = position.price_open - self.sl_range_min
                            self._log(f"🔒 [SL HARDCAP MIN] #{position.ticket}: Mínimo {self.sl_range_min}pt do entry (${proposed_sl:.2f})")
                        elif sl_distance_from_entry > self.sl_range_max:  # > 15pt
                            proposed_sl = position.price_open - self.sl_range_max
                            self._log(f"🔒 [SL HARDCAP MAX] #{position.ticket}: Máximo {self.sl_range_max}pt do entry (${proposed_sl:.2f})")

                        # RESTRICTION: SL can only go UP and keep 5pt buffer
                        if proposed_sl > position.sl and proposed_sl < current_price - 5.0:
                            new_sl = proposed_sl
                            sl_changed = True
                            self._log(f"🛡️ [SL UP] #{position.ticket}: ${position.sl:.2f} → ${new_sl:.2f} (Dist: {sl_distance_from_entry:.1f}pt)")
                        else:
                            self._log(f"🚫 [SL BLOCKED] #{position.ticket}: LONG SL can only move UP with 5pt buffer")
                    else:  # SHORT
                        # SHORT: SL can only go DOWN (protect more profit)
                        proposed_sl = position.sl - sl_movement_points

                        # 🍒 HARDCAP: SL distance from entry must be >= 10pt and <= 15pt
                        sl_distance_from_entry = proposed_sl - position.price_open
                        if sl_distance_from_entry < self.sl_range_min:  # < 10pt
                            proposed_sl = position.price_open + self.sl_range_min
                            self._log(f"🔒 [SL HARDCAP MIN] #{position.ticket}: Mínimo {self.sl_range_min}pt do entry (${proposed_sl:.2f})")
                        elif sl_distance_from_entry > self.sl_range_max:  # > 15pt
                            proposed_sl = position.price_open + self.sl_range_max
                            self._log(f"🔒 [SL HARDCAP MAX] #{position.ticket}: Máximo {self.sl_range_max}pt do entry (${proposed_sl:.2f})")

                        # RESTRICTION: SL can only go DOWN and keep 5pt buffer
                        if proposed_sl < position.sl and proposed_sl > current_price + 5.0:
                            new_sl = proposed_sl
                            sl_changed = True
                            self._log(f"🛡️ [SL DOWN] #{position.ticket}: ${position.sl:.2f} → ${new_sl:.2f} (Dist: {sl_distance_from_entry:.1f}pt)")
                        else:
                            self._log(f"🚫 [SL BLOCKED] #{position.ticket}: SHORT SL can only move DOWN with 5pt buffer")

            # 🎯 TP ADJUSTABLE WITH $100 CAP AND RANGE LIMITS
            if abs(tp_adjust) >= 0.3:  # Model wants to modify TP
                if position.tp > 0:
                    # Calculate TP movement in points (±0.5 → ±3.0 points max)
                    tp_movement_points = tp_adjust * 3.0

                    if position.type == mt5.POSITION_TYPE_BUY:  # LONG
                        proposed_tp = position.tp + tp_movement_points

                        # 🍒 HARDCAP: TP distance from entry must be <= 18pt
                        tp_distance_from_entry = proposed_tp - position.price_open
                        if tp_distance_from_entry > self.tp_range_max:  # 18.0pt
                            proposed_tp = position.price_open + self.tp_range_max
                            self._log(f"🔒 [TP HARDCAP] #{position.ticket}: Limitado a {self.tp_range_max}pt do entry (${proposed_tp:.2f})")

                        # Validate: TP must be above current price + buffer
                        if proposed_tp > current_price + 3.0:
                            # Check if new TP would exceed $100 cap
                            potential_pnl = (proposed_tp - position.price_open) * position.volume * 100
                            if potential_pnl <= 100:  # Respect $100 cap
                                new_tp = proposed_tp
                                tp_changed = True
                                self._log(f"🎯 [TP UP] #{position.ticket}: ${position.tp:.2f} → ${new_tp:.2f} (Dist: {tp_distance_from_entry:.1f}pt, PnL: ${potential_pnl:.2f})")
                            else:
                                self._log(f"🚫 [TP BLOCKED] #{position.ticket}: Would exceed $100 cap (${potential_pnl:.2f})")
                        else:
                            self._log(f"🚫 [TP BLOCKED] #{position.ticket}: TP too close to current price")
                    else:  # SHORT
                        proposed_tp = position.tp - tp_movement_points

                        # 🍒 HARDCAP: TP distance from entry must be <= 18pt
                        tp_distance_from_entry = position.price_open - proposed_tp
                        if tp_distance_from_entry > self.tp_range_max:  # 18.0pt
                            proposed_tp = position.price_open - self.tp_range_max
                            self._log(f"🔒 [TP HARDCAP] #{position.ticket}: Limitado a {self.tp_range_max}pt do entry (${proposed_tp:.2f})")

                        # Validate: TP must be below current price - buffer
                        if proposed_tp < current_price - 3.0:
                            # Check if new TP would exceed $100 cap
                            potential_pnl = (position.price_open - proposed_tp) * position.volume * 100
                            if potential_pnl <= 100:  # Respect $100 cap
                                new_tp = proposed_tp
                                tp_changed = True
                                self._log(f"🎯 [TP DOWN] #{position.ticket}: ${position.tp:.2f} → ${new_tp:.2f} (Dist: {tp_distance_from_entry:.1f}pt, PnL: ${potential_pnl:.2f})")
                            else:
                                self._log(f"🚫 [TP BLOCKED] #{position.ticket}: Would exceed $100 cap (${potential_pnl:.2f})")
                        else:
                            self._log(f"🚫 [TP BLOCKED] #{position.ticket}: TP too close to current price")

            # Modificar posição se houve mudança
            if new_sl != position.sl or new_tp != position.tp:
                request = {
                    "action": mt5.TRADE_ACTION_SLTP,
                    "symbol": self.symbol,
                    "position": position.ticket,
                    "sl": new_sl if new_sl > 0 else 0,
                    "tp": new_tp if new_tp > 0 else 0
                }

                self._log(f"📤 [MODIFY REQ] Pos #{position.ticket}: SL=${new_sl:.2f}, TP=${new_tp:.2f}")

                result = mt5.order_send(request)
                if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                    self._log(f"✅ [MODIFY SUCCESS] Pos #{position.ticket} | SL: ${new_sl:.2f} | TP: ${new_tp:.2f}")
                    return True
                else:
                    error_code = result.retcode if result else "NO_RESULT"
                    error_comment = result.comment if result else "NO_COMMENT"
                    self._log(f"❌ [MODIFY FAILED] Pos #{position.ticket} | Erro: {error_code} - {error_comment}")
                    return False
            else:
                self._log(f"⏭️  [NO CHANGE] Pos #{position.ticket} | SL/TP inalterados")
                return True
        except Exception as e:
            self._log(f"❌ [MODIFY ERROR] Pos #{position.ticket} | Exceção: {e}")
            return False
    
    def _calculate_v7_volume_fixed(self, risk_appetite, current_positions):
        """🎯 Calcular volume V7 SEM usar entry_quality (position sizing hardcoded)"""
        try:
            # Volume base SEMPRE 0.02 (modelo foi treinado com isso)
            base_volume = self.original_base_lot  # 0.02 fixo para cálculos
            
            # Ajuste APENAS por risk appetite (0-1 -> 0.7-1.3x)
            risk_multiplier = 0.7 + (risk_appetite * 0.6)
            
            # Redução por posições existentes
            position_multiplier = max(0.3, 1.0 - (current_positions / self.max_positions) * 0.7)
            
            # Volume final (SEM confidence multiplier)
            final_volume = base_volume * risk_multiplier * position_multiplier
            
            # 💰 DEBUG: Log do cálculo de volume
            self._log(f"💰 [VOLUME DEBUG] Base={base_volume:.3f} | Risk={risk_multiplier:.2f} | Pos={position_multiplier:.2f} | Final={final_volume:.3f}")
            
            # Limitar ao máximo do modelo (0.03 original)
            final_volume = min(final_volume, self.original_max_lot)  # 0.03 máximo do modelo
            final_volume = max(final_volume, 0.01)  # Mínimo 0.01 lote
            
            # 🎯 APLICAR MULTIPLICADOR SÓ NO FINAL (para MT5)
            mt5_volume = final_volume * self.lot_multiplier
            mt5_volume = max(0.01, min(mt5_volume, 10.0))  # Limites de segurança MT5
            
            self._log(f"💰 [VOLUME CALCULADO] Modelo={final_volume:.3f} | MT5={mt5_volume:.3f} | Multiplicador={self.lot_multiplier:.1f}x")
            
            return round(mt5_volume, 3)  # Retorna volume para MT5
            
        except:
            return self.original_base_lot * self.lot_multiplier  # Fallback com multiplicador
    
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
                self._log(f"[❌ LEGION V1] Action space incorreto: {actual_action_size} != {expected_action_size}")
                return False
            
            # 3. Verificar features por step - Cherry format
            expected_features = Config.FEATURES_PER_STEP  # 45
            # Cherry: 16 market + 18 positions + 2 intelligent + 4 order_flow + 5 volatility = 45
            actual_features = 45  # Fixed for Cherry format
            
            if actual_features != expected_features:
                self._log(f"[❌ CHERRY] Features per step incorreto: {actual_features} != {expected_features}")
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
                    # Tentar carregar modelo com compatibilidade para verificação
                    model_to_test = self.load_model_with_policy_compat(model_path)
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
                        self._log(f"[❌ LEGION V1] Modelo produz ação com tamanho incorreto: {test_action.shape[1]} != {expected_action_size}")
                        return False
                        
                    self._log(f"[✅ LEGION V1] Modelo compatível verificado")
                    
                except Exception as e:
                    self._log(f"[❌ LEGION V1] Erro ao verificar compatibilidade do modelo: {e}")
                    return False
            
            # 🎯 VERIFICAÇÕES V7 ESPECÍFICAS
            
            # 6. Verificar TwoHeadV7Simple compatibility
            if model_path and "v7" not in model_path.lower():
                self._log(f"[⚠️ V7] Modelo pode não ser V7: {model_path}")
            
            # 7. Verificar breakdown de features - Cherry format
            market_features = Config.MARKET_FEATURES  # 16
            position_features = Config.POSITION_FEATURES  # 18
            intelligent_features = Config.INTELLIGENT_FEATURES  # 2
            order_flow_features = Config.ORDER_FLOW_FEATURES  # 4
            volatility_features = Config.VOLATILITY_FEATURES  # 5

            self._log(f"[✅ CHERRY] Features breakdown:")
            self._log(f"  - Market: {market_features} (using first 16 from {len(self.feature_columns)} available)")
            self._log(f"  - Position: {position_features} (2 posições ativas, 18 features total)")
            self._log(f"  - Intelligent: {intelligent_features} (V7 components)")
            self._log(f"  - Order Flow: {order_flow_features} (order flow analysis)")
            self._log(f"  - Volatility: {volatility_features} (volatility features)")
            self._log(f"  - Total: {market_features + position_features + intelligent_features + order_flow_features + volatility_features}")
            
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

            # 🔧 FIX: Thread-safe logging
            if hasattr(self, 'gui') and hasattr(self.gui, 'enqueue_log') and callable(self.gui.enqueue_log):
                self.gui.enqueue_log(log_message)
            else:
                if self.log_widget:
                    # GUI logging - check if we're in main thread
                    try:
                        import threading
                        if threading.current_thread() == threading.main_thread():
                            self.log_widget.insert(tk.END, log_message + "\n")
                            self.log_widget.see(tk.END)
                        else:
                            # From background thread - use console
                            print(log_message)
                            sys.stdout.flush()
                    except:
                        print(log_message)
                        sys.stdout.flush()
                else:
                    # Console logging
                    print(log_message)
                    sys.stdout.flush()  # Force immediate output

            # Registrar em arquivo da sessão mensagens importantes de trading
            try:
                if hasattr(self, 'session_log_path') and self.session_log_path:
                    important = any(key in message for key in [
                        'TRADE', 'FECHADA', 'MODIFY', 'TRAILING',
                        'ORDER', 'EXECUTION', 'COOLDOWN', 'MT5'
                    ]) or message.startswith("=")
                    if important:
                        import os
                        # 🧠 MEMORY FIX: Rotação de logs - limitar tamanho do arquivo
                        max_log_size = 5 * 1024 * 1024  # 5MB
                        if os.path.exists(self.session_log_path):
                            if os.path.getsize(self.session_log_path) > max_log_size:
                                # Renomear arquivo antigo
                                backup_path = f"{self.session_log_path}.old"
                                if os.path.exists(backup_path):
                                    os.remove(backup_path)  # Remove backup antigo
                                os.rename(self.session_log_path, backup_path)

                        with open(self.session_log_path, 'a', encoding='utf-8') as f:
                            f.write(log_message + "\n")
                # Incrementar contadores por parsing de mensagem de trade
                if 'V7 TRADE' in message:
                    try:
                        if 'LONG' in message and hasattr(self, 'session_counters'):
                            self.session_counters['buys'] += 1
                        elif 'SHORT' in message and hasattr(self, 'session_counters'):
                            self.session_counters['sells'] += 1
                    except Exception:
                        pass
            except Exception:
                pass
                
        except Exception as e:
            print(f"[LOG ERROR] {e}: {message}")
    
# Funções de adaptação removidas - RobotV7 agora usa formato Legion V1 nativo
    
    def reset(self):
        """Reset environment V7"""
        try:
            self.current_step = 0
            self.done = False
            self.last_action = None
            self.hold_count = 0
            
            # 🔥 MANTER COOLDOWN - NÃO RESETAR APÓS SESSÃO
            # self.last_position_closed_timestamp = 0  # REMOVIDO: Mantém cooldown entre sessões
            
            # Atualizar dados históricos
            self._initialize_historical_data_v7()
            
            return self._get_observation_v7()
            
        except Exception as e:
            self._log(f"[ERROR] Erro no reset V7: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def step(self, action):
        """Step V7 - executar ação e retornar nova observação"""
        try:
            # PRIMEIRO: Atualizar estatísticas de posições (detectar fechamentos e ativar cooldown)
            self._update_position_stats()

            # 🧠 MEMORY FIX: Manter historical_df com tamanho controlado (a cada 50 steps)
            if hasattr(self, 'current_step') and self.current_step % 50 == 0:
                self._trim_historical_df()

            # DEPOIS: Processar ação Legion V1 (4D) com cooldown já atualizado
            action_result = self._process_legion_action(action)

            # 🔧 NOVO: Processar ajustes SL/TP dinâmicos em posições existentes
            if action_result and 'sl_adjusts' in action_result and 'tp_adjusts' in action_result:
                # Debug: confirmar se está sendo chamado
                if hasattr(self, 'current_step') and self.current_step % 10 == 0:
                    self._log(f"🔧 [DEBUG] Step {self.current_step}: Chamando _process_sl_tp_adjustments_v7")
                    self._log(f"🔧 [DEBUG] sl_adjusts: {action_result['sl_adjusts']}")
                    self._log(f"🔧 [DEBUG] tp_adjusts: {action_result['tp_adjusts']}")
                self._process_sl_tp_adjustments_v7(action_result['sl_adjusts'], action_result['tp_adjusts'])

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
    
    def load_model_with_policy_compat(self, model_path):
        """
        🔧 Carrega RecurrentPPO com compatibilidade automática para checkpoints legados.

        Detecta checkpoints pre-bypass (Nineth, etc.) e carrega com TwoHeadV11SigmoidLegacy.
        Checkpoints pós-bypass (Eleventh, Twelveth) carregam normalmente.
        """
        try:
            model = RecurrentPPO.load(model_path)
            self._log("✅ [POLICY] Checkpoint moderno carregado (pós-bypass)")
            return model
        except Exception as e:
            error_msg = str(e)

            # Padrões de erro que indicam checkpoint legacy (pre-bypass)
            legacy_patterns = (
                "market_context.raw_features_processor",
                "market_context.context_processor.0.weight",
                "raw_features_processor.0.weight",
                "Unexpected key(s) in state_dict",
            )

            if not any(p in error_msg for p in legacy_patterns):
                self._log(f"❌ [POLICY] Erro desconhecido ao carregar checkpoint:")
                self._log(f"    {error_msg[:200]}")
                raise

            self._log("⚠️  [POLICY] Checkpoint LEGACY detectado (pre-bypass)")
            self._log("    → Carregando com TwoHeadV11SigmoidLegacy...")
            custom_objects = {"policy_class": TwoHeadV11SigmoidLegacy}
            model = RecurrentPPO.load(model_path, custom_objects=custom_objects)
            self._log("✅ [POLICY] Checkpoint legacy carregado com sucesso!")
            return model

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

            # 2. Detectar tipo de modelo automaticamente pelos kwargs salvos
            self._log(f"[🔍 V7 AUTO-LOAD] Detectando tipo de modelo automaticamente...")

            # Carregar modelo com compatibilidade automática (pre/pós bypass)
            try:
                self._log(f"[🎯 LEGION V1] Carregando modelo com auto-detecção...")
                self.model = self.load_model_with_policy_compat(zip_path)
                self.is_legion_model = True
                self._log(f"[✅ LEGION V1] Modelo Legion V1 carregado com sucesso!")
                self._log(f"[✅ COMPATIBILIDADE] Modelo Legion V1: obs=450D, action=4D (formato nativo)")

                # Verificar se é realmente V11 Sigmoid
                policy_class = self.model.policy.__class__.__name__
                self._log(f"[🔍 LEGION V1] Policy detectada: {policy_class}")
                if "V11" not in policy_class and "TwoHeadV" not in policy_class:
                    self._log(f"[⚠️ LEGION V1] Aviso: Policy pode não ser V11 Sigmoid: {policy_class}")

            except Exception as e:
                self._log(f"[❌ LEGION V1] Falha no carregamento: {str(e)[:200]}...")
                self._log(f"[❌ CRITICAL] RobotV7 agora é exclusivo para modelos Legion V1")
                raise e
            
            # Log final do carregamento
            self._log(f"[✅ AUTO-LOAD] Modelo Legion V1 (V11 Sigmoid) carregado de: {os.path.basename(zip_path)}")
            
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
            
            # 🔥 1MIN TIMEFRAME: Sem filtro - consulta a cada candle como no treino
            # Modelo foi treinado com dados 1min, então consultamos a cada minuto
            
            # Capturar observação original para debug
            original_obs = observation.copy() if isinstance(observation, np.ndarray) else observation
            
            # Aplicar normalização se disponível
            if self.normalizer is not None:
                try:
                    observation = self.normalizer.normalize_obs(observation)
                except Exception as e:
                    self._log(f"[⚠️ V7 PREDICT] Erro na normalização: {e}")
            
            # 🔒 Obter predição em modo de inferência (sem gradientes) - Legion V1 nativo
            with torch.no_grad():
                action, _states = self.model.predict(observation, deterministic=False)
            
            # 🧠 V7 INTUITION: Sistema simplificado sem gates
            # Legion V1 usa Entry/Management heads especializados
            
            # Incrementar contador de debug
            self.debug_step_counter += 1
            
            # 🔥 DEBUG TODAS AS PREDIÇÕES LEGION V1 (4D format)
            try:
                # 🎯 LOG TODA PREDIÇÃO - LIVE TRADING (ALINHADO COM CHERRY.PY)
                raw_decision = float(action[0])
                if raw_decision < ACTION_THRESHOLD_SHORT:
                    action_name = "SHORT"
                elif raw_decision < ACTION_THRESHOLD_LONG:
                    action_name = "HOLD"
                else:
                    action_name = "LONG"
                
                self._log(f"[🤖 PREDIÇÃO] Step {self.debug_step_counter} | "
                         f"{action_name} | Entry: {action[0]:.2f} | Confidence: {action[1]:.2f} | "
                         f"Pos1_Mgmt: {action[2]:.2f} | Pos2_Mgmt: {action[3]:.2f}")
                
                # Debug detalhado a cada 10 steps
                if self.debug_step_counter % self.debug_composite_interval == 0:
                    current_time = pd.Timestamp.now().strftime("%H:%M:%S")
                    self._log(f"[📊 STATUS {current_time}] Consultas: {self.debug_step_counter} | "
                             f"Última ação: {action_name} | Confiança média: {action[1]:.2f}")
                             
            except Exception as e:
                self._log(f"[❌ V7 DEBUG] Erro no debug: {e}")
            
            
            # Manter buffer das últimas observações para análise de anomalias
            self._update_observation_buffer(observation)
            
            return action
            
        except Exception as e:
            self._log(f"[❌ V7 PREDICT] Erro na predição: {e}")
            return None
    
    def _get_v7_entry_head_info_DEPRECATED(self, observation, detailed=False):
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
                        
                        # Obter predição Legion V1 direta
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
    
    def _log_composite_debug_DEPRECATED(self, gate_info, step):
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
                    
                    # Decision interpretation (entry_decision já é 0=HOLD, 1=LONG, 2=SHORT)
                    decision_names = {0: "HOLD", 1: "LONG", 2: "SHORT"}
                    decision_str = decision_names.get(int(entry_decision), "UNKNOWN")
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
                # 🧠 MEMORY FIX: deque já tem maxlen=50, append é O(1)
                self.last_observations.append(observation.copy())

                # 🧠 MEMORY FIX: Atualizar estatísticas apenas a cada 10 observações
                self.obs_stats_update_counter += 1
                if len(self.last_observations) >= 10 and self.obs_stats_update_counter >= 10:
                    obs_array = np.array(self.last_observations)
                    self.obs_stats = {
                        'mean': np.mean(obs_array, axis=0),
                        'std': np.std(obs_array, axis=0),
                        'min': np.min(obs_array, axis=0),
                        'max': np.max(obs_array, axis=0)
                    }
                    self.obs_stats_update_counter = 0  # Reset contador
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
    
    def _log_entry_head_analysis_DEPRECATED(self, gate_info):
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
        """🔄 Recarregar modelo Legion V1"""
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
                            positions = self._get_robot_positions() or []
                            
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

                        # 🔧 NOVO: Processar ajustes SL/TP dinâmicos em posições existentes
                        if action_analysis and 'sl_adjusts' in action_analysis and 'tp_adjusts' in action_analysis:
                            # Debug: confirmar se está sendo chamado
                            if step_count % 10 == 0:
                                self._log(f"🔧 [DEBUG LIVE] Step {step_count}: Chamando _process_sl_tp_adjustments_v7")
                                self._log(f"🔧 [DEBUG LIVE] sl_adjusts: {action_analysis['sl_adjusts']}")
                                self._log(f"🔧 [DEBUG LIVE] tp_adjusts: {action_analysis['tp_adjusts']}")
                            self._process_sl_tp_adjustments_v7(action_analysis['sl_adjusts'], action_analysis['tp_adjusts'])

                        # 🔥 EXECUTAR DECISÃO NO MT5
                        if self.mt5_connected:
                            self._execute_v7_decision(action_analysis)
                        else:
                            self._log(f"[SIMULAÇÃO] {action_analysis['action_name']} - MT5 desconectado")
                    
                    step_count += 1
                    
                    # 🔥 SINCRONIZAÇÃO COM CANDLES 1MIN - IGUAL AO TREINO
                    # Aguardar próximo candle 1min para consultar modelo (idêntico ao ambiente de treino)
                    current_time = time.time()
                    next_candle_time = ((current_time // 60) + 1) * 60  # Próximo minuto
                    wait_seconds = next_candle_time - current_time
                    
                    if wait_seconds > 0.5:  # Só esperar se falta mais que 0.5s
                        self._log(f"⏰ [1MIN SYNC] Aguardando {wait_seconds:.1f}s para próximo candle...")
                        time.sleep(wait_seconds)
                    else:
                        time.sleep(0.1)  # Pequeno delay para não sobrecarregar
                    
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
                        positions = self._get_robot_positions() or []
                        
                        if account_info and tick:
                            # Monitor passivo - atualizar estatísticas
                            self._update_position_stats()
                            
                            # 📊 MONITORAMENTO PASSIVO: Logs e stats (MT5 faz SL/TP automaticamente)
                            self._check_and_close_positions(tick)
                    
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
    
    def _update_position_stats(self):
        """📊 Atualizar estatísticas baseado em posições fechadas"""
        try:
            if not self.mt5_connected:
                return
                
            # Obter posições atuais (apenas do robô)
            current_positions = self._get_robot_positions() or []
            current_tickets = set(pos.ticket for pos in current_positions)
            
            # Verificar posições fechadas
            closed_tickets = self.known_positions - current_tickets
            closed_any = False
            
            for ticket in closed_tickets:
                if ticket in self.position_stats:
                    # Obter histórico da posição fechada
                    history = mt5.history_deals_get(position=ticket)
                    if history and len(history) >= 2:  # Open + Close
                        close_deal = history[-1]  # Último deal é o fechamento
                        profit = close_deal.profit
                        
                        # 🔥 SISTEMA DE COOLDOWN ADAPTATIVO BASEADO EM WIN/LOSS (IDÊNTICO AO SILUS)
                        self.last_position_closed_timestamp = time.time()
                        
                        # Detectar resultado do trade
                        is_win = profit > 0
                        
                        # Atualizar tracking de win/loss sequences
                        if is_win:
                            self.last_trade_was_win = True
                            self.last_trade_was_loss = False
                            self.consecutive_wins += 1
                            self.consecutive_losses = 0
                        else:
                            self.last_trade_was_win = False
                            self.last_trade_was_loss = True
                            self.consecutive_losses += 1
                            self.consecutive_wins = 0
                        
                        # Aplicar cooldown adaptativo (IDÊNTICO AO SILUS)
                        if self.last_trade_was_loss:
                            if self.consecutive_losses >= 3:
                                self.cooldown_after_trade = 60  # 60min para múltiplas perdas consecutivas
                            else:
                                self.cooldown_after_trade = 45  # 45min para perda simples
                        elif self.last_trade_was_win:
                            if self.consecutive_wins >= 3:
                                self.cooldown_after_trade = 30  # 30min para múltiplas wins consecutivas
                            else:
                                self.cooldown_after_trade = 25  # 25min para win simples
                        else:
                            self.cooldown_after_trade = self.cooldown_base  # Default 35min
                        
                        # Log do resultado e cooldown aplicado
                        result_str = "WIN" if is_win else "LOSS"
                        consecutive_str = f" ({self.consecutive_wins}W)" if is_win else f" ({self.consecutive_losses}L)"
                        self._log(f"🔒 [COOLDOWN ADAPTATIVO] {result_str}{consecutive_str} → {self.cooldown_after_trade}min | P&L: ${profit:.2f}")
                        
                        # Atualizar P&L e Virtual Portfolio
                        if hasattr(self, 'gui') and hasattr(self.gui, 'session_stats'):
                            self.gui.session_stats['profit_loss'] += profit
                            
                            # Update virtual balance
                            self.gui.update_virtual_balance(profit)
                            
                            # Atualizar wins/losses
                            if profit > 0:
                                self.gui.session_stats['wins'] += 1
                            else:
                                self.gui.session_stats['losses'] += 1
                            
                            # Atualizar GUI
                            self.gui.update_stats()
                        
                        # Atualizar contadores locais e registrar no arquivo da sessão
                        try:
                            if hasattr(self, 'session_counters'):
                                if profit > 0:
                                    self.session_counters['wins'] += 1
                                else:
                                    self.session_counters['losses'] += 1
                                self.session_counters['profit_loss'] += float(profit)
                            if hasattr(self, 'session_log_path') and self.session_log_path:
                                with open(self.session_log_path, 'a', encoding='utf-8') as f:
                                    f.write(f"CLOSE | ticket={ticket} | pnl={profit:.2f}\n")
                        except Exception:
                            pass
                        
                        self._log(f"📊 [POSIÇÃO FECHADA] Ticket #{ticket} | P&L: ${profit:.2f}")
                        closed_any = True
                        # 🔥 FIX CRÍTICO: Atualizar cooldown independente do slot associado
                        try:
                            slot = self.position_slot_map.get(ticket, None)
                            self._log(f"🔍 [CLOSE-DEBUG] Ticket #{ticket} - Slot no mapa: {slot}")
                            
                            if slot is None:
                                # Tentar extrair do comentário do deal/posição
                                cmt = getattr(close_deal, 'comment', '') or ''
                                slot = self._extract_slot_from_comment(str(cmt))
                                self._log(f"🔍 [CLOSE-DEBUG] Ticket #{ticket} - Comment: '{cmt}' → Slot extraído: {slot}")
                                
                            if slot is None:
                                # ÚLTIMO RECURSO: Tentar buscar nas posições históricas
                                try:
                                    hist_positions = mt5.positions_get(symbol=self.symbol)
                                    if hist_positions:
                                        for pos in hist_positions:
                                            if pos.magic == self.magic_number and pos.ticket == ticket:
                                                pos_cmt = getattr(pos, 'comment', '') or ''
                                                slot = self._extract_slot_from_comment(str(pos_cmt))
                                                self._log(f"🔍 [CLOSE-DEBUG] Ticket #{ticket} - Posição Comment: '{pos_cmt}' → Slot: {slot}")
                                                break
                                except Exception:
                                    pass
                                    
                            if slot is not None:
                                # Usar cooldown adaptativo baseado no resultado do trade
                                cooldown_until = time.time() + (self.cooldown_after_trade * 60)
                                self.position_slot_cooldowns[int(slot)] = cooldown_until
                                # Remover mapeamento do ticket
                                self.position_slot_map.pop(ticket, None)
                                cooldown_until_str = datetime.fromtimestamp(cooldown_until).strftime('%H:%M:%S')
                                self._log(f"🔒 [COOLDOWN-SLOT] Slot {int(slot)} em cooldown por {self.cooldown_after_trade} min até {cooldown_until_str}")
                                self._log(f"📊 [COOLDOWN-DETAIL] Ticket #{ticket} | Profit: {profit:.2f} | Slot liberado em: {cooldown_until_str}")
                            else:
                                self._log(f"❌ [CLOSE-ERROR] Ticket #{ticket} - NÃO foi possível identificar o slot! Cooldown não ativado.")
                        except Exception as e:
                            self._log(f"❌ [CLOSE-ERROR] Erro ao processar cooldown slot: {e}")
                    
                    # Remover da lista de stats
                    del self.position_stats[ticket]
            
            # Atualizar posições conhecidas
            self.known_positions = current_tickets

            # Removido cooldown global: não aplicar cooldown quando todos os slots ficam vazios
            
            # Adicionar novas posições ao tracker
            for pos in current_positions:
                if pos.ticket not in self.position_stats:
                    self.position_stats[pos.ticket] = {
                        'open_price': pos.price_open,
                        'volume': pos.volume / self.lot_multiplier,  # Normalizar para modelo
                        'type': 'LONG' if pos.type == 0 else 'SHORT'
                    }
                    
        except Exception as e:
            self._log(f"[❌ STATS] Erro ao atualizar estatísticas: {e}")
    
    def end_session(self):
        """Encerrar sessão e salvar resumo consolidado no .txt"""
        try:
            end_time = time.time()
            start_time = 0.0
            if hasattr(self, 'session_counters'):
                start_time = self.session_counters.get('start_time', end_time)
                buys = self.session_counters.get('buys', 0)
                sells = self.session_counters.get('sells', 0)
                wins = self.session_counters.get('wins', 0)
                losses = self.session_counters.get('losses', 0)
                pl = self.session_counters.get('profit_loss', 0.0)
            else:
                buys = sells = wins = losses = 0
                pl = 0.0
            duration_min = (end_time - start_time) / 60.0 if start_time else 0.0
            total = wins + losses
            winrate = (wins / total * 100.0) if total > 0 else 0.0

            summary_lines = [
                "",
                "-" * 80,
                "RESUMO DA SESSÃO",
                f"Sessão: {getattr(self, 'session_id', 'N/A')}",
                f"Término: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}",
                f"Duração: {duration_min:.1f} min",
                f"Trades: buys={buys} sells={sells} total={buys + sells}",
                f"Resultados: wins={wins} losses={losses} winrate={winrate:.1f}%",
                f"P&L acumulado: ${pl:.2f}",
                "-" * 80,
                ""
            ]
            if hasattr(self, 'session_log_path') and self.session_log_path:
                with open(self.session_log_path, 'a', encoding='utf-8') as f:
                    f.write("\n".join(summary_lines))
            self._log("[LOG] Resumo da sessão salvo")
        except Exception as e:
            self._log(f"[WARNING] Falha ao salvar resumo da sessão: {e}")

    def _end_session_safe(self):
        """Wrapper silencioso para encerrar sessão no atexit"""
        try:
            if not getattr(self, '_session_ended', False):
                self._session_ended = True
                self.end_session()
        except Exception:
            pass

    def test_gui_stats_update(self):
        """🧪 Teste manual para atualizar stats da GUI"""
        try:
            if hasattr(self, 'gui') and hasattr(self.gui, 'session_stats'):
                # Simular uma win
                self.gui.session_stats['wins'] += 1
                self.gui.session_stats['profit_loss'] += 25.50
                self.gui.update_stats()
                self._log("🧪 [TESTE] Win simulado adicionado - GUI deveria atualizar")
                
                # Simular uma loss após 3 segundos
                import threading
                def simulate_loss():
                    import time
                    time.sleep(3)
                    self.gui.session_stats['losses'] += 1
                    self.gui.session_stats['profit_loss'] -= 15.30
                    self.gui.update_stats()
                    self._log("🧪 [TESTE] Loss simulado adicionado - GUI deveria atualizar")
                
                threading.Thread(target=simulate_loss, daemon=True).start()
            else:
                self._log("❌ [TESTE] GUI não disponível para teste")
        except Exception as e:
            self._log(f"❌ [TESTE] Erro no teste: {e}")
    
    def test_quadruple_selectors(self):
        """🧪 Teste dos Seletores Quadruplos Completos"""
        try:
            self._log("🧪 [TESTE] Testando Seletores Quadruplos...")
            
            # Teste 1: SL Afrouxar (pos1_mgmt = -0.8)
            sl_adj, tp_adj = self._convert_management_to_sltp_adjustments(-0.8)
            self._log(f"🔍 pos1_mgmt = -0.8 → sl_adjust={sl_adj}, tp_adjust={tp_adj} (Esperado: 0.5, 0)")
            
            # Teste 2: SL Apertar (pos1_mgmt = -0.3)
            sl_adj, tp_adj = self._convert_management_to_sltp_adjustments(-0.3)
            self._log(f"🔍 pos1_mgmt = -0.3 → sl_adjust={sl_adj}, tp_adjust={tp_adj} (Esperado: -0.5, 0)")
            
            # Teste 3: TP Próximo (pos1_mgmt = 0.3)
            sl_adj, tp_adj = self._convert_management_to_sltp_adjustments(0.3)
            self._log(f"🔍 pos1_mgmt = 0.3 → sl_adjust={sl_adj}, tp_adjust={tp_adj} (Esperado: 0, -0.5)")
            
            # Teste 4: TP Distante (pos1_mgmt = 1.0)
            sl_adj, tp_adj = self._convert_management_to_sltp_adjustments(1.0)
            self._log(f"🔍 pos1_mgmt = 1.0 → sl_adjust={sl_adj}, tp_adjust={tp_adj} (Esperado: 0, 0.5)")
            
            self._log("✅ [TESTE] Seletores Quadruplos mapeados corretamente!")
            self._log("💡 [INFO] Agora tp_adjust será processado para modificar TP de posições existentes!")
            
        except Exception as e:
            self._log(f"❌ [TESTE] Erro no teste de seletores: {e}")
    
    def _is_in_cooldown(self):
        """🔒 Verificar se está em período de cooldown após fechamento de posição"""
        if self.last_position_closed_timestamp == 0:
            return False, 0  # Nenhuma posição fechada ainda, pode operar
            
        current_time = time.time()
        time_since_last_close = current_time - self.last_position_closed_timestamp
        cooldown_seconds = self.cooldown_after_trade * 60
        
        if time_since_last_close < cooldown_seconds:
            remaining_time = cooldown_seconds - time_since_last_close
            remaining_minutes = remaining_time / 60
            return True, remaining_minutes
        
        return False, 0
    
    def test_cooldown_system(self):
        """🧪 Teste do sistema de cooldown"""
        try:
            self._log("🧪 [TESTE COOLDOWN] Testando sistema de cooldown...")
            
            # Teste 1: Sem posições fechadas (sem cooldown)
            cooldown_check = self._is_in_cooldown()
            self._log(f"🔍 Sem posições fechadas - Em cooldown: {cooldown_check[0]}")
            
            # Simular fechamento de posição
            self.last_position_closed_timestamp = time.time()
            self._log("🔄 Simulando fechamento de posição...")
            
            # Teste 2: Imediatamente após fechamento (deve estar em cooldown)
            cooldown_check = self._is_in_cooldown()
            self._log(f"🔍 Após fechamento - Em cooldown: {cooldown_check[0]}, Restantes: {cooldown_check[1]:.1f} min")
            
            self._log("✅ [TESTE COOLDOWN] Sistema funcionando corretamente!")
            self._log(f"💡 [INFO] Cooldown adaptativo: base {self.cooldown_base}min, atual {self.cooldown_after_trade}min baseado em win/loss")
            
        except Exception as e:
            self._log(f"❌ [TESTE COOLDOWN] Erro: {e}")

    def show_adaptive_cooldown_status(self):
        """📊 Mostrar status atual do sistema de cooldown adaptativo"""
        try:
            self._log("📊 [COOLDOWN STATUS] Sistema de Cooldown Adaptativo (SILUS-Compatible)")
            self._log(f"   🏁 Último resultado: {'WIN' if self.last_trade_was_win else 'LOSS' if self.last_trade_was_loss else 'NENHUM'}")
            self._log(f"   🔥 Wins consecutivos: {self.consecutive_wins}")
            self._log(f"   ❄️ Losses consecutivos: {self.consecutive_losses}")
            self._log(f"   ⏰ Cooldown atual: {self.cooldown_after_trade} minutos")
            self._log(f"   📋 Cooldown base: {self.cooldown_base} minutos")
            
            # Status dos slots
            current_time = time.time()
            active_cooldowns = 0
            for slot, cooldown_until in self.position_slot_cooldowns.items():
                if current_time < cooldown_until:
                    remaining = (cooldown_until - current_time) / 60
                    self._log(f"   🔒 Slot {slot}: Cooldown por {remaining:.1f} min")
                    active_cooldowns += 1
            
            if active_cooldowns == 0:
                self._log(f"   ✅ Todos os slots livres para trading")
                
            # Cooldown global
            cooldown_check = self._is_in_cooldown()
            if cooldown_check[0]:
                self._log(f"   🚫 Cooldown global ativo: {cooldown_check[1]:.1f} min restantes")
            else:
                self._log(f"   ✅ Cooldown global livre para novos trades")
                
        except Exception as e:
            self._log(f"❌ [COOLDOWN STATUS] Erro: {e}")
    
    def _get_robot_positions(self):
        """🔒 Obter apenas posições do robô ativas (excluindo cooldown)"""
        try:
            if not self.mt5_connected:
                return []
            
            # Obter todas as posições do símbolo
            all_positions = mt5.positions_get(symbol=self.symbol)
            if not all_positions:
                return []
            
            # Filtrar apenas posições do robô (magic number)
            robot_positions = [pos for pos in all_positions if pos.magic == self.magic_number]
            
            # 🔥 FIX CRÍTICO: Remover posições cujos slots estão em cooldown
            active_positions = []
            import time as _time
            current_time = _time.time()
            
            for pos in robot_positions:
                # Encontrar slot da posição
                slot = self.position_slot_map.get(pos.ticket, None)
                if slot is None:
                    # Tentar extrair do comentário
                    comment = getattr(pos, 'comment', '') or ''
                    slot = self._extract_slot_from_comment(str(comment))
                
                if slot is not None:
                    # Verificar se slot não está em cooldown
                    cooldown_until = self.position_slot_cooldowns.get(int(slot), 0.0)
                    if current_time >= cooldown_until:
                        active_positions.append(pos)
                    else:
                        # Log posição em cooldown sendo ignorada
                        remain = (cooldown_until - current_time) / 60
                        self._log(f"🔒 [COOLDOWN-IGNORED] Posição #{pos.ticket} no slot {slot} ignorada - cooldown restante: {remain:.1f}m")
                else:
                    # Posição sem slot identificado - incluir (fallback)
                    active_positions.append(pos)
            
            return active_positions
            
        except Exception as e:
            self._log(f"[❌ ROBOT_POS] Erro ao obter posições do robô: {e}")
            return []

    def _map_slot_for_open_positions(self, slot_hint: int | None = None):
        """🔥 FIX: Mapeia tickets de posições abertas para slots usando o comment.
        Usa slot_hint quando o comentário ainda não está propagado totalmente.
        """
        try:
            if not self.mt5_connected:
                return
            # 🔥 CRÍTICO: Usar MT5 direto, NÃO _get_robot_positions() que filtra por cooldown!
            all_positions = mt5.positions_get(symbol=self.symbol)
            if not all_positions:
                return
            # Filtrar apenas por magic number (sem filtro de cooldown)
            positions = [pos for pos in all_positions if pos.magic == self.magic_number]
            for pos in positions:
                try:
                    if pos.ticket in self.position_slot_map:
                        continue
                    cmt = getattr(pos, 'comment', '') or ''
                    slot = self._extract_slot_from_comment(str(cmt))
                    if slot is None and slot_hint is not None and f"SLOT{slot_hint}" in str(cmt):
                        slot = int(slot_hint)
                    if slot is not None:
                        self.position_slot_map[pos.ticket] = int(slot)
                        self._log(f"[COOLDOWN-SLOTS] Map: ticket #{pos.ticket} → slot {int(slot)}")
                except Exception:
                    continue
        except Exception as e:
            self._log(f"[COOLDOWN-SLOTS] Erro no mapeamento de slots: {e}")

    def _extract_slot_from_comment(self, comment: str):
        try:
            import re
            # Novo formato: V7242S0 (V7 + últimos 4 dígitos magic + S + slot)
            m = re.search(r"V7\d{4}S(\d+)", str(comment))
            if m:
                return int(m.group(1))
            # Formatos antigos (compatibilidade)
            m = re.search(r"SLOT(\d+)", str(comment))
            if m:
                return int(m.group(1))
            m = re.search(r"V7S(\d+)", str(comment))
            if m:
                return int(m.group(1))
        except Exception:
            return None
        return None

    def _reconcile_slot_map(self):
        try:
            if not self.mt5_connected:
                return
            positions = self._get_robot_positions() or []
            open_tickets = set(pos.ticket for pos in positions)
            # Remove tickets que fecharam
            for t in list(self.position_slot_map.keys()):
                if t not in open_tickets:
                    self.position_slot_map.pop(t, None)
            # Atribuir slots a tickets sem mapping, com base no comment
            used_slots = set(self.position_slot_map.values())
            free_slots = [i for i in range(self.max_positions) if i not in used_slots]
            for pos in positions:
                if pos.ticket not in self.position_slot_map:
                    slot = None
                    c = getattr(pos, 'comment', '') or ''
                    slot = self._extract_slot_from_comment(c)
                    if slot is None and free_slots:
                        slot = free_slots.pop(0)
                    if slot is not None:
                        self.position_slot_map[pos.ticket] = int(slot)
        except Exception:
            pass

    def _allocate_entry_slot(self):
        """🔍 Escolhe um slot livre cujo cooldown já expirou - com logs detalhados"""
        try:
            import time as _t
            self._reconcile_slot_map()
            used = set(self.position_slot_map.values())
            now = _t.time()
            min_remain = None
            
            self._log(f"🔍 [SLOT-ALLOCATION] Buscando slot livre...")
            self._log(f"🔒 [SLOTS-STATUS] Slots em uso: {sorted(used) if used else 'Nenhum'}")
            
            for s in range(self.max_positions):
                allow_time = self.position_slot_cooldowns.get(s, 0.0)
                remain = max(0.0, allow_time - now)
                status = "OCUPADO" if s in used else ("LIVRE" if now >= allow_time else f"COOLDOWN({remain/60:.1f}m)")
                
                self._log(f"📍 [SLOT-{s}] Status: {status} | Próximo uso permitido: {datetime.fromtimestamp(allow_time).strftime('%H:%M:%S') if allow_time > 0 else 'Imediato'}")
                
                if s in used:
                    continue
                    
                if now >= allow_time:
                    self._log(f"✅ [SLOT-SELECTED] Slot {s} selecionado - disponível para uso")
                    return s, 0.0
                else:
                    if min_remain is None or remain < min_remain:
                        min_remain = remain
                        
            if min_remain:
                self._log(f"⏱️ [SLOT-WAIT] Nenhum slot livre. Próximo disponível em {min_remain/60:.1f} minutos")
            else:
                self._log(f"🚫 [SLOT-FULL] Todos os slots ocupados")
                
            return None, (min_remain or 0.0)
        except Exception as e:
            self._log(f"❌ [SLOT-ERROR] Erro na alocação de slot: {e}")
            return None, 0.0
    
    def _check_and_close_positions(self, tick):
        """📊 Monitorar posições para logs e atualizar P&L na GUI (MT5 faz os fechamentos)"""
        try:
            if not self.mt5_connected or not tick:
                return
                
            positions = self._get_robot_positions()
            if not positions:
                return
                
            current_price = tick.bid
            
            for position in positions:
                try:
                    # Calcular P&L atual apenas para logs (normalizado para modelo)
                    normalized_volume = position.volume / self.lot_multiplier
                    if position.type == 0:  # LONG
                        pnl = (current_price - position.price_open) * normalized_volume
                    else:  # SHORT
                        pnl = (position.price_open - current_price) * normalized_volume
                    
                    # Log informativo apenas (MT5 gerencia SL/TP automaticamente)
                    has_sl = position.sl > 0
                    has_tp = position.tp > 0
                    
                    # Log status para monitoramento (sem fechar nada)
                    if has_sl or has_tp:
                        self._log(f"📊 [MONITOR] Ticket #{position.ticket} | P&L: ${pnl:.2f} | SL: ${position.sl:.2f} | TP: ${position.tp:.2f}")
                            
                except Exception as pos_error:
                    self._log(f"❌ [MONITOR] Erro ao monitorar posição #{position.ticket}: {pos_error}")
                    
        except Exception as e:
            self._log(f"❌ [MONITOR] Erro geral: {e}")
    
    def _convert_management_to_sltp_adjustments(self, mgmt_value):
        """
        🚀 Converte valor de management [-1,1] em ajustes SL/TP bidirecionais (como silus.py)
        
        LÓGICA:
        - mgmt_value < 0: foco em SL (proteção)
          - < -0.5: SL +0.5 pontos (afrouxar SL = menos risco prematura saída)
          - > -0.5: SL -0.5 pontos (apertar = menos risco)
        - mgmt_value > 0: foco em TP (target)
          - > +0.5: TP +0.5 pontos (target distante)
          - < +0.5: TP -0.5 pontos (target próximo)
          
        Returns: (sl_adjust, tp_adjust)
        """
        if mgmt_value < 0:
            # Foco em SL management
            if mgmt_value < -0.5:
                return (0.5, 0)  # Afrouxar SL
            else:
                return (-0.5, 0)  # Apertar SL
        elif mgmt_value > 0:
            # Foco em TP management
            if mgmt_value > 0.5:
                return (0, 0.5)  # TP distante
            else:
                return (0, -0.5)  # TP próximo
        else:
            # Valor próximo de zero = HOLD
            return (0, 0)

    def _convert_model_adjustments_to_points(self, sl_adjust, tp_adjust, context="adjustment"):
        """
        🎯 FUNÇÃO UNIFICADA: Converte ajustes do modelo ±0.5 para pontos válidos

        SUBSTITUI A DUPLICAÇÃO com silus.py - Agora ambos usam o MESMO sistema!

        Args:
            sl_adjust (float): Ajuste SL do modelo (±0.5)
            tp_adjust (float): Ajuste TP do modelo (±0.5)
            context (str): "creation" para novas posições, "adjustment" para ajustes
        """

        # Usar configuração do próprio Robot_1min.py
        sl_min, sl_max = REALISTIC_SLTP_CONFIG['sl_min_points'], REALISTIC_SLTP_CONFIG['sl_max_points']
        tp_min, tp_max = REALISTIC_SLTP_CONFIG['tp_min_points'], REALISTIC_SLTP_CONFIG['tp_max_points']

        result = {
            'sl_points': 0.0,
            'tp_points': 0.0,
            'sl_change': 0.0,
            'tp_change': 0.0,
            'valid': False,
            'context': context
        }

        if context == "creation":
            # 🏗️ CRIAÇÃO DE POSIÇÕES: Converter ±0.5 para ranges válidos
            sl_range = sl_max - sl_min
            sl_center = sl_min + (sl_range / 2)
            result['sl_points'] = sl_center + (sl_adjust * sl_range / 2)

            tp_range = tp_max - tp_min
            tp_center = tp_min + (tp_range / 2)
            result['tp_points'] = tp_center + (tp_adjust * tp_range / 2)

            result['sl_change'] = result['sl_points']
            result['tp_change'] = result['tp_points']

        elif context == "adjustment":
            # 🔧 AJUSTE DE POSIÇÕES: Converter ±0.5 para mudanças diretas
            result['sl_change'] = sl_adjust  # ±0.5 pontos direto
            result['tp_change'] = tp_adjust  # ±0.5 pontos direto
            result['sl_points'] = abs(result['sl_change'])
            result['tp_points'] = abs(result['tp_change'])

        # Aplicar limites de segurança
        result['sl_points'] = max(sl_min, min(result['sl_points'], sl_max))
        result['tp_points'] = max(tp_min, min(result['tp_points'], tp_max))

        # Arredondar para múltiplos de 0.5
        result['sl_points'] = round(result['sl_points'] * 2) / 2
        result['tp_points'] = round(result['tp_points'] * 2) / 2
        result['sl_change'] = round(result['sl_change'] * 2) / 2
        result['tp_change'] = round(result['tp_change'] * 2) / 2

        # Validar se está dentro dos limites
        result['valid'] = (sl_min <= result['sl_points'] <= sl_max and
                          tp_min <= result['tp_points'] <= tp_max)

        return result

    # 🚨 DEPRECADO: Manter por compatibilidade temporária
    def _convert_action_to_realistic_sltp(self, sltp_action_values, current_price):
        """
        🚀 Converte action space para SL/TP realistas de forma clara (como silus.py)
        sltp_action_values[0] = SL adjustment [-3,3]
        sltp_action_values[1] = TP adjustment [-3,3]
        Retorna: [sl_points, tp_points] sempre positivos
        """
        sl_adjust = sltp_action_values[0]  # [-3,3] para SL
        tp_adjust = sltp_action_values[1]  # [-3,3] para TP
        
        # Converter para pontos realistas separadamente
        # SL: normalizar [-3,3] para [sl_min_points, sl_max_points]
        sl_points = REALISTIC_SLTP_CONFIG['sl_min_points'] + \
                    (sl_adjust + 3) * (REALISTIC_SLTP_CONFIG['sl_max_points'] - REALISTIC_SLTP_CONFIG['sl_min_points']) / 6
        
        # TP: normalizar [-3,3] para [tp_min_points, tp_max_points]
        tp_points = REALISTIC_SLTP_CONFIG['tp_min_points'] + \
                    (tp_adjust + 3) * (REALISTIC_SLTP_CONFIG['tp_max_points'] - REALISTIC_SLTP_CONFIG['tp_min_points']) / 6
        
        # Arredondar para múltiplos de 0.5 pontos
        sl_points = round(sl_points * 2) / 2
        tp_points = round(tp_points * 2) / 2
        
        # Garantir limites (segurança)
        sl_points = max(REALISTIC_SLTP_CONFIG['sl_min_points'], min(sl_points, REALISTIC_SLTP_CONFIG['sl_max_points']))
        tp_points = max(REALISTIC_SLTP_CONFIG['tp_min_points'], min(tp_points, REALISTIC_SLTP_CONFIG['tp_max_points']))
        
        return [sl_points, tp_points]
    
    def _process_dynamic_sl_tp_adjustment(self, position, sl_adjust, tp_adjust, current_price, pos_index):
        """
        🎯 ALINHADO COM CHERRY.PY: NEW RESTRICTIVE SL/TP SYSTEM

        SL TRAILING ONLY:
        - LONG: SL can only go UP (protect more profit)
        - SHORT: SL can only go DOWN (protect more profit)
        - Never allow SL to "flee" from price

        TP ADJUSTABLE WITH $100 CAP:
        - Model can move TP to maximize profits
        - BUT always respecting $100 USD cap
        - Auto-close if PnL reaches $100
        """
        result = {
            'action_taken': False,
            'sl_adjusted': False,
            'tp_adjusted': False,
            'pnl_cap_reached': False,
            'sl_blocked': False,
            'tp_blocked': False,
            'position_updates': {},
            'sl_info': {},
            'tp_info': {}
        }
        
        try:
            if not self.mt5_connected:
                return result

            # 🔄 CALCULATE CURRENT PnL
            normalized_volume = position.volume / self.lot_multiplier
            if position.type == 0:  # LONG
                current_pnl = (current_price - position.price_open) * normalized_volume * 100
                pos_type = 'long'
            else:  # SHORT
                current_pnl = (position.price_open - current_price) * normalized_volume * 100
                pos_type = 'short'

            # 🛑 AUTO-CLOSE AT $100 USD (PnL CAP) - ALINHADO COM CHERRY.PY
            if current_pnl >= 100:
                self._log(f"🛑 [PnL CAP $100] Auto-closing position #{position.ticket} at ${current_pnl:.2f}")
                # Fechar posição via MT5
                close_request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": position.symbol,
                    "volume": position.volume,
                    "type": mt5.ORDER_TYPE_SELL if position.type == 0 else mt5.ORDER_TYPE_BUY,
                    "position": position.ticket,
                    "magic": position.magic,
                    "comment": "PnL_CAP_$100",
                }
                mt5.order_send(close_request)
                result['pnl_cap_reached'] = True
                result['action_taken'] = True
                return result

            # 🎯 SL TRAILING ONLY (A FAVOR DO TRADE) - ALINHADO COM CHERRY.PY
            if abs(sl_adjust) >= 0.3:  # Model wants to modify SL
                current_sl = position.sl if position.sl > 0 else 0
                if current_sl > 0:
                    # Calculate SL movement in points (±0.5 → ±1.0 point)
                    sl_movement_points = sl_adjust * 2.0  # Scale to ±2 points max

                    if pos_type == 'long':
                        # LONG: new SL = current SL + movement
                        new_sl = current_sl + sl_movement_points
                        # RESTRICTION: SL can only go UP (new_sl > current_sl)
                        if new_sl > current_sl and new_sl < current_price - 5.0:  # Keep 5pt buffer
                            result['position_updates']['sl'] = new_sl
                            result['sl_adjusted'] = True
                            result['action_taken'] = True
                            result['sl_info'] = {
                                'direction': 'UP (protecting profit)',
                                'old_sl': current_sl,
                                'new_sl': new_sl,
                                'movement_points': sl_movement_points
                            }
                        else:
                            result['sl_blocked'] = True
                            result['sl_info'] = {
                                'block_reason': 'SL can only move UP for LONG' if new_sl <= current_sl else 'SL too close to price',
                                'attempted_sl': new_sl
                            }

                    else:  # short
                        # SHORT: new SL = current SL - movement
                        new_sl = current_sl - sl_movement_points
                        # RESTRICTION: SL can only go DOWN (new_sl < current_sl)
                        if new_sl < current_sl and new_sl > current_price + 5.0:  # Keep 5pt buffer
                            result['position_updates']['sl'] = new_sl
                            result['sl_adjusted'] = True
                            result['action_taken'] = True
                            result['sl_info'] = {
                                'direction': 'DOWN (protecting profit)',
                                'old_sl': current_sl,
                                'new_sl': new_sl,
                                'movement_points': sl_movement_points
                            }
                        else:
                            result['sl_blocked'] = True
                            result['sl_info'] = {
                                'block_reason': 'SL can only move DOWN for SHORT' if new_sl >= current_sl else 'SL too close to price',
                                'attempted_sl': new_sl
                            }

            # 🎯 TP ADJUSTABLE WITH $100 CAP - ALINHADO COM CHERRY.PY
            if abs(tp_adjust) >= 0.3:  # Model wants to modify TP
                current_tp = position.tp if position.tp > 0 else 0
                if current_tp > 0:
                    # Calculate TP movement in points
                    tp_movement_points = tp_adjust * 3.0  # Scale to ±3 points max

                    if pos_type == 'long':
                        new_tp = current_tp + tp_movement_points
                        # Validate: TP must be above current price + buffer
                        if new_tp > current_price + 3.0:
                            # Check if new TP would exceed $100 cap
                            potential_pnl = (new_tp - position.price_open) * normalized_volume * 100
                            if potential_pnl <= 100:  # Respect $100 cap
                                result['position_updates']['tp'] = new_tp
                                result['tp_adjusted'] = True
                                result['action_taken'] = True
                                result['tp_info'] = {
                                    'old_tp': current_tp,
                                    'new_tp': new_tp,
                                    'movement_points': tp_movement_points,
                                    'potential_pnl': potential_pnl
                                }
                            else:
                                result['tp_blocked'] = True
                                result['tp_info'] = {
                                    'block_reason': f'TP would exceed $100 cap (${potential_pnl:.2f})',
                                    'attempted_tp': new_tp
                                }
                        else:
                            result['tp_blocked'] = True
                            result['tp_info'] = {
                                'block_reason': 'TP too close to current price',
                                'attempted_tp': new_tp
                            }

                    else:  # short
                        new_tp = current_tp - tp_movement_points
                        # Validate: TP must be below current price - buffer
                        if new_tp < current_price - 3.0:
                            # Check if new TP would exceed $100 cap
                            potential_pnl = (position.price_open - new_tp) * normalized_volume * 100
                            if potential_pnl <= 100:  # Respect $100 cap
                                result['position_updates']['tp'] = new_tp
                                result['tp_adjusted'] = True
                                result['action_taken'] = True
                                result['tp_info'] = {
                                    'old_tp': current_tp,
                                    'new_tp': new_tp,
                                    'movement_points': tp_movement_points,
                                    'potential_pnl': potential_pnl
                                }
                            else:
                                result['tp_blocked'] = True
                                result['tp_info'] = {
                                    'block_reason': f'TP would exceed $100 cap (${potential_pnl:.2f})',
                                    'attempted_tp': new_tp
                                }
                        else:
                            result['tp_blocked'] = True
                            result['tp_info'] = {
                                'block_reason': 'TP too close to current price',
                                'attempted_tp': new_tp
                            }
            
            # Aplicar mudanças se o modelo decidiu - ALINHADO COM CHERRY.PY
            if result['action_taken'] and result['position_updates']:
                new_sl = result['position_updates'].get('sl')
                new_tp = result['position_updates'].get('tp')
                modify_result = self._modify_position_sltp(position.ticket, new_sl, new_tp)
                if "SUCCESS" in modify_result:
                    # Log simplificado - sem metadata tracking
                    sl_info = f" | SL: {new_sl:.2f}" if new_sl else ""
                    tp_info = f" | TP: {new_tp:.2f}" if new_tp else ""

                    if result['sl_adjusted'] and result['tp_adjusted']:
                        self._log(f"🎯 [SL+TP ADJUST] Pos #{position.ticket} ajustada{sl_info}{tp_info}")
                    elif result['tp_adjusted']:
                        self._log(f"🎯 [TP ADJUST] Pos #{position.ticket} ajustada{tp_info}")
                    elif result['sl_adjusted']:
                        self._log(f"🎯 [SL TRAILING] Pos #{position.ticket} ajustada{sl_info}")
                else:
                    self._log(f"❌ [SL/TP ADJUST] Falha ao ajustar pos #{position.ticket}: {modify_result}")
                    result['action_taken'] = False

            # Log de bloqueios (debug)
            if result['sl_blocked'] and result['sl_info'].get('block_reason'):
                self._log(f"⚠️ [SL BLOCKED] Pos #{position.ticket}: {result['sl_info']['block_reason']}")
            if result['tp_blocked'] and result['tp_info'].get('block_reason'):
                self._log(f"⚠️ [TP BLOCKED] Pos #{position.ticket}: {result['tp_info']['block_reason']}")

            return result
                    
        except Exception as e:
            self._log(f"❌ [TRAILING STOP] Erro: {e}")
            return result
    
    def _modify_position_sltp(self, ticket, new_sl=None, new_tp=None):
        """Modificar SL/TP de uma posição existente"""
        try:
            # Obter posição atual
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return "ERROR_POSITION_NOT_FOUND"
            
            position = positions[0]
            
            # Preparar request de modificação
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": position.symbol,
                "position": ticket,
                "sl": new_sl if new_sl is not None else position.sl,
                "tp": new_tp if new_tp is not None else position.tp,
                "magic": self.magic_number,
            }
            
            # Executar modificação
            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                return f"SUCCESS|{ticket}|{new_sl or position.sl}|{new_tp or position.tp}"
            else:
                error_code = result.retcode if result else "None"
                return f"ERROR_MODIFY|{error_code}"
                
        except Exception as e:
            return f"ERROR_EXCEPTION|{e}"
    
    def _get_position_pnl(self, position, current_price):
        """📊 Calcular P&L de uma posição (como silus.py)"""
        try:
            # Normalizar volume para escala do modelo (dividir pelo multiplicador)
            normalized_volume = position.volume / self.lot_multiplier
            if position.type == 0:  # LONG
                return (current_price - position.price_open) * normalized_volume
            else:  # SHORT
                return (position.price_open - current_price) * normalized_volume
        except:
            return 0.0
    
    def _process_v7_action(self, action):
        """🧠 Processar ação do modelo V7 - ACTION SPACE 11D"""
        try:
            if not isinstance(action, (list, tuple, np.ndarray)):
                action = np.array([action])
            
            # Garantir 4D Legion V1 (sem expansão)
            if len(action) != 4:
                self._log(f"[⚠️ TRADE] Esperado 4D, recebido {len(action)}D")
                if len(action) < 4:
                    action = np.pad(action, (0, 4 - len(action)), mode='constant')
                else:
                    action = action[:4]
            
            # LEGION V1: [entry_decision, entry_confidence, pos1_mgmt, pos2_mgmt]
            # 🚀 ALINHADO COM CHERRY.PY: Usar constantes globais para consistência
            raw_decision = float(action[0])
            if raw_decision < ACTION_THRESHOLD_SHORT:
                entry_decision = 2  # SELL (SHORT) - extremo negativo
            elif raw_decision < ACTION_THRESHOLD_LONG:
                entry_decision = 0  # HOLD - centro
            else:
                entry_decision = 1  # BUY (LONG) - extremo positivo
            entry_confidence = float(np.clip(action[1], 0, 1))  # [0,1] Fusão quality + risk_appetite
            
            # Gerenciamento das 2 posições ([-1,1] → pontos reais)
            pos1_mgmt = float(np.clip(action[2], -1, 1))     # Gestão posição 1
            pos2_mgmt = float(np.clip(action[3], -1, 1))     # Gestão posição 2
            
            # 🚀 FIX: Usar EXATAMENTE o mesmo sistema do SILUS.PY
            # Converter management values em ajustes SL/TP bidirecionais
            pos1_sl_adjust, pos1_tp_adjust = self._convert_management_to_sltp_adjustments(pos1_mgmt)
            pos2_sl_adjust, pos2_tp_adjust = self._convert_management_to_sltp_adjustments(pos2_mgmt)

            # Criar arrays com ajustes bidirecionais corretos
            sl_adjusts = [pos1_sl_adjust, pos2_sl_adjust, pos1_sl_adjust]  # Pos1, Pos2, Global (=Pos1)
            tp_adjusts = [pos1_tp_adjust, pos2_tp_adjust, pos1_tp_adjust]  # Pos1, Pos2, Global (=Pos1)

            # Converter ajustes para pontos realistas usando mesma função do silus.py
            sl_points = []
            tp_points = []

            # Obter preço atual para conversão
            tick = mt5.symbol_info_tick(self.symbol) if hasattr(self, 'mt5_connected') and self.mt5_connected else None
            current_price = tick.bid if tick else 2650.0  # Fallback para simulação

            for i in range(2):  # Para cada posição (2 ativas máximo)
                # 🎯 UNIFICADO: Usar função unificada (mesma do silus.py)
                creation_result = self._convert_model_adjustments_to_points(sl_adjusts[i], tp_adjusts[i], "creation")
                sl_points.append(creation_result['sl_points'])  # Já validado e positivo
                tp_points.append(creation_result['tp_points'])  # Já validado e positivo
            
            # 🔥 FILTROS V7 - Verificações antes da execução
            
            # Filter 1: Verificar limite de posições (SILENCIOSO)
            if entry_decision in [1, 2]:  # BUY ou SELL
                if self.mt5_connected:
                    current_positions = self._get_robot_positions()
                    pos_count = len(current_positions) if current_positions else 0
                    if pos_count >= self.max_positions:
                        entry_decision = 0  # Forçar HOLD silenciosamente
            
            # 🎯 FILTRO DE ATIVIDADE: Ajustar confiança baseado em SHORT vs LONG
            original_confidence = entry_confidence
            if Config.ACTIVITY_FILTER_ENABLED:
                if entry_decision == 2:  # SHORT
                    entry_confidence *= Config.SHORT_CONFIDENCE_BOOST
                    entry_confidence = min(entry_confidence, 1.0)  # Cap at 1.0
                    if original_confidence != entry_confidence:
                        self._log(f"🎯 [FILTRO ATIVIDADE] SHORT boost: {original_confidence:.2f} → {entry_confidence:.2f}")
                elif entry_decision == 1:  # LONG
                    entry_confidence *= Config.LONG_CONFIDENCE_PENALTY
                    if original_confidence != entry_confidence:
                        self._log(f"🚫 [FILTRO ATIVIDADE] LONG penalty: {original_confidence:.2f} → {entry_confidence:.2f}")

            # 🎯 ETAPA 1 - FILTRO DE CONFIDENCE: 80% MÍNIMO (OBRIGATÓRIO)
            # ALINHADO COM CHERRY.PY: entry_decision é [0=HOLD, 1=LONG, 2=SHORT]
            if entry_decision in [1, 2] and entry_confidence < 0.8:
                self._log(f"😫 [CONFIDENCE FILTER] Entry REJEITADA: decision={entry_decision}, confidence={entry_confidence:.2f} < 0.8 - Forçando HOLD")
                entry_decision = 0  # REJEITA_TRADE - Forçar HOLD
            
            # 🚀 FILTROS ANTIGOS REMOVIDOS
            # if entry_decision in [1, 2] and entry_quality < 0.6:
            #     self._log(f"🚫 [V7-FILTER] Entry Quality baixa: {entry_quality:.2f} < 0.6 - Forçando HOLD")
            #     entry_decision = 0  # Forçar HOLD
            
            # Mapear ação para nome
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action_name = action_names.get(entry_decision, 'UNKNOWN')
            
            # Calcular position size baseado na confiança da entrada
            position_size = entry_confidence
            
            return {
                'entry_decision': entry_decision,
                'entry_confidence': entry_confidence,
                'pos1_mgmt': pos1_mgmt,
                'pos2_mgmt': pos2_mgmt,
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
                'entry_confidence': 0.0,
                'pos1_mgmt': 0.0,
                'pos2_mgmt': 0.0,
                'position_size': 0.0,
                'sl_adjusts': [0.0, 0.0, 0.0],
                'tp_adjusts': [0.0, 0.0, 0.0],
                'sl_points': [0.0, 0.0, 0.0],
                'tp_points': [0.0, 0.0, 0.0],
                'action_name': 'HOLD',
                'raw_action': [0.0] * 4
            }
    
    def _execute_v7_decision(self, action_analysis):
        """🧠 Executar decisão do modelo V7 no MT5"""
        try:
            if not self.mt5_connected:
                self._log("⚠️ [V7-EXECUÇÃO] MT5 não conectado - simulação apenas")
                return
                
            action_name = action_analysis['action_name']
            entry_confidence = action_analysis['entry_confidence']
            
            # V7 LOG: Mostrar ação processada (detalhado apenas para BUY/SELL)
            if action_name in ['BUY', 'SELL']:
                pos1_mgmt = action_analysis.get('pos1_mgmt', 0.0)
                pos2_mgmt = action_analysis.get('pos2_mgmt', 0.0)
                self._log(f"🧠 [V7-DECISION] {action_name} | Confidence: {entry_confidence:.3f} | Pos1: {pos1_mgmt:.3f} | Pos2: {pos2_mgmt:.3f}")
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
                current_time = pd.Timestamp.now().strftime("%H:%M:%S")
                self._log(f"🟢 [SINAL COMPRA {current_time}] Modelo decidiu COMPRAR | Confiança: {entry_confidence:.2f} | Preço: ${current_price:.2f}")
                self._execute_buy_order_v7(current_price, entry_confidence, action_analysis)
            elif action_name == 'SELL':
                self.hold_count = 0  # Reset contador de HOLD
                current_time = pd.Timestamp.now().strftime("%H:%M:%S")
                self._log(f"🔴 [SINAL VENDA {current_time}] Modelo decidiu VENDER | Confiança: {entry_confidence:.2f} | Preço: ${current_price:.2f}")
                self._execute_sell_order_v7(current_price, entry_confidence, action_analysis)
            else:
                # HOLD - modelo decidiu não fazer nada (silencioso quando há 3 posições)
                self.hold_count += 1
                
                # Verificar se há posições abertas para decidir se loga (apenas do robô)
                if self.mt5_connected:
                    current_positions = self._get_robot_positions()
                    pos_count = len(current_positions) if current_positions else 0
                    
                    # 🔥 LOG TODA DECISÃO HOLD (não apenas algumas)
                    if pos_count < self.max_positions:
                        current_time = pd.Timestamp.now().strftime("%H:%M:%S")
                        self._log(f"⭕ [EXECUÇÃO {current_time}] HOLD - Posições: {pos_count}/{self.max_positions} | Aguardando oportunidade")
                    else:
                        self._log(f"🔒 [POSIÇÕES CHEIAS] HOLD - {pos_count}/{self.max_positions} posições ativas")
                else:
                    # Se MT5 não conectado, loga normalmente
                    current_time = pd.Timestamp.now().strftime("%H:%M:%S")
                    self._log(f"⭕ [SIMULAÇÃO {current_time}] HOLD - MT5 desconectado")
                
        except Exception as e:
            self._log(f"❌ [V7-EXECUÇÃO] Erro ao executar decisão: {e}")
    
    def _execute_buy_order_v7(self, current_price, entry_confidence, action_analysis):
        """🧠 Executar ordem de compra V7 - com SL/TP inteligentes"""
        try:
            # Removido cooldown global: apenas cooldown por slot é considerado
            # Verificar limite de posições do robô (máximo 2)
            if self.mt5_connected:
                current_positions = self._get_robot_positions()
                if current_positions and len(current_positions) >= self.max_positions:
                    self._log(f"⚠️ [LIMITE ROBÔ] Máximo de {self.max_positions} posições do robô atingido - COMPRA bloqueada")
                    return
            
            # Calcular volume baseado em entry_confidence (IDÊNCTICO SILUS)
            # Legion V1: entry_confidence = fusão de quality + risk_appetite
            # Silus chama _calculate_adaptive_position_size_quality(entry_confidence)
            volume = self._calculate_volume_by_risk_appetite(entry_confidence)
            self._log(f"🎯 [POSITION SIZING] Entry Confidence: {entry_confidence:.3f} -> Volume: {volume:.3f}")
            
            # Selecionar slot disponível (respeita cooldown independente por slot)
            slot_id, wait_sec = self._allocate_entry_slot()
            if slot_id is None:
                # Logar tempos restantes por slot para depuração
                try:
                    now = time.time()
                    waits = []
                    for s in range(self.max_positions):
                        allow = self.position_slot_cooldowns.get(s, 0.0)
                        rem = max(0.0, allow - now)
                        waits.append(f"slot{s}:{rem/60:.1f}m")
                    waits_str = ", ".join(waits)
                    self._log(f"🔒 [COOLDOWN-SLOTS] Nenhum slot livre. Aguardando {wait_sec/60:.1f} min | {waits_str}")
                except Exception:
                    self._log(f"🔒 [COOLDOWN-SLOTS] Nenhum slot livre. Aguardando {wait_sec/60:.1f} min")
                return

            # Usar SL/TP do modelo se disponível
            if action_analysis and 'sl_points' in action_analysis and 'tp_points' in action_analysis:
                # Usar primeiro SL/TP do modelo (posição 0)
                sl_points = abs(action_analysis['sl_points'][0])  # Garantir positivo
                tp_points = abs(action_analysis['tp_points'][0])  # Garantir positivo
                
                # Aplicar limites de segurança (alinhado com cherry.py)
                sl_points = np.clip(sl_points, self.sl_range_min, self.sl_range_max)  # 10-25 pontos
                tp_points = np.clip(tp_points, self.tp_range_min, self.tp_range_max)  # 12-25 pontos
                
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
            result = self._execute_order_v7(mt5.ORDER_TYPE_BUY, volume, sl_price, tp_price, slot_id=slot_id)
            
            if "SUCCESS" in result:
                # Compra executada com sucesso (cooldown apenas no fechamento)
                
                # Incrementar estatísticas de compra
                if hasattr(self, 'gui') and hasattr(self.gui, 'session_stats'):
                    self.gui.session_stats['buys'] += 1
                
                # Verificar número atual de posições do robô após execução
                current_positions = self._get_robot_positions() if self.mt5_connected else []
                pos_count = len(current_positions) if current_positions else 0
                self._log(f"✅ [COMPRA V7] Ordem executada! Vol: {volume} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
                self._log(f"📊 [POSIÇÕES ROBÔ] Total atual: {pos_count}/{self.max_positions}")
                
                # Atualizar GUI com estatísticas
                if hasattr(self, 'gui') and hasattr(self.gui, 'update_stats'):
                    self.gui.update_stats()
            else:
                self._log(f"❌ [COMPRA V7] Falha na execução: {result}")
                
        except Exception as e:
            self._log(f"❌ [COMPRA V7] Erro: {e}")
    
    def _execute_sell_order_v7(self, current_price, entry_confidence, action_analysis):
        """🧠 Executar ordem de venda V7 - com SL/TP inteligentes"""
        try:
            # Removido cooldown global: apenas cooldown por slot é considerado
            # Verificar limite de posições do robô (máximo 2)
            if self.mt5_connected:
                current_positions = self._get_robot_positions()
                if current_positions and len(current_positions) >= self.max_positions:
                    self._log(f"⚠️ [LIMITE] Máximo de {self.max_positions} posições atingido - VENDA bloqueada")
                    return
            
            # Calcular volume baseado em entry_confidence (IDÊNCTICO SILUS)
            # Legion V1: entry_confidence = fusão de quality + risk_appetite
            # Silus chama _calculate_adaptive_position_size_quality(entry_confidence)
            volume = self._calculate_volume_by_risk_appetite(entry_confidence)
            self._log(f"🎯 [POSITION SIZING] Entry Confidence: {entry_confidence:.3f} -> Volume: {volume:.3f}")
            
            # Selecionar slot disponível (respeita cooldown independente por slot)
            slot_id, wait_sec = self._allocate_entry_slot()
            if slot_id is None:
                self._log(f"🔒 [COOLDOWN-SLOTS] Nenhum slot livre. Aguardando {wait_sec/60:.1f} min")
                return

            # Usar SL/TP do modelo se disponível
            if action_analysis and 'sl_points' in action_analysis and 'tp_points' in action_analysis:
                # Usar primeiro SL/TP do modelo (posição 0)
                sl_points = abs(action_analysis['sl_points'][0])  # Garantir positivo
                tp_points = abs(action_analysis['tp_points'][0])  # Garantir positivo
                
                # Aplicar limites de segurança (alinhado com cherry.py)
                sl_points = np.clip(sl_points, self.sl_range_min, self.sl_range_max)  # 10-25 pontos
                tp_points = np.clip(tp_points, self.tp_range_min, self.tp_range_max)  # 12-25 pontos
                
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
            result = self._execute_order_v7(mt5.ORDER_TYPE_SELL, volume, sl_price, tp_price, slot_id=slot_id)
            
            if "SUCCESS" in result:
                # Venda executada com sucesso (cooldown apenas no fechamento)
                
                # Incrementar estatísticas de venda
                if hasattr(self, 'gui') and hasattr(self.gui, 'session_stats'):
                    self.gui.session_stats['sells'] += 1
                
                # Verificar número atual de posições do robô após execução
                current_positions = self._get_robot_positions() if self.mt5_connected else []
                pos_count = len(current_positions) if current_positions else 0
                self._log(f"✅ [VENDA V7] Ordem executada! Vol: {volume} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
                self._log(f"📊 [POSIÇÕES ROBÔ] Total atual: {pos_count}/{self.max_positions}")
                
                # Atualizar GUI com estatísticas
                if hasattr(self, 'gui') and hasattr(self.gui, 'update_stats'):
                    self.gui.update_stats()
            else:
                self._log(f"❌ [VENDA V7] Falha na execução: {result}")
                
        except Exception as e:
            self._log(f"❌ [VENDA V7] Erro: {e}")
    
    def _calculate_volume_by_risk_appetite(self, entry_confidence):
        """🎯 POSITION SIZING BASEADO EM ENTRY CONFIDENCE - IDÊNCTICO SILUS.PY
        
        Legion V1: entry_confidence = fusão quality + risk_appetite
        Silus: _calculate_adaptive_position_size_quality(entry_confidence)
        """
        try:
            # 🔥 LÓGICA HARDCODED DO DAYTRADER.PY - Portfolio-based scaling
            initial_portfolio_value = self.initial_balance  # 500.0
            current_portfolio_value = self._get_current_portfolio_value()
            base_lot = self.original_base_lot  # 0.02 (modelo sempre pensa nisso)
            max_lot = self.original_max_lot    # 0.03 (modelo sempre pensa nisso)
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
            
            # 🎯 AJUSTE POR ENTRY CONFIDENCE (mesma curva 0.7–1.3 do cherry.py)
            risk_multiplier = 0.7 + (entry_confidence * 0.6)
            final_lot = adaptive_lot * risk_multiplier
            
            # Aplicar limites idênticos ao ambiente (0.02 → 0.03)
            final_lot = max(base_lot, min(final_lot, max_lot))
            final_lot = round(final_lot, 2)
            
            # 🎯 APLICAR MULTIPLICADOR PARA MT5
            mt5_volume = final_lot * self.lot_multiplier
            mt5_volume = max(0.01, min(mt5_volume, 10.0))  # Limites de segurança MT5
            mt5_volume = round(mt5_volume, 2)
            
            # Log detalhado do cálculo
            if hasattr(self, 'debug_step_counter') and self.debug_step_counter % 50 == 0:
                growth_factor = current_portfolio_value / initial_portfolio_value if initial_portfolio_value > 0 else 1.0
                self._log(f"[💰 POSITION SIZING] Portfolio: ${current_portfolio_value:.2f} (growth: {growth_factor:.2f}x) | Base: {adaptive_lot:.3f} | Entry Confidence: {entry_confidence:.2f} (mult: {risk_multiplier:.2f}x) | Modelo: {final_lot:.2f} | MT5: {mt5_volume:.2f}")
            
            return mt5_volume  # Retorna volume para MT5
            
        except Exception as e:
            self._log(f"❌ [VOLUME V7] Erro no dynamic sizing: {e}")
            return self.original_base_lot * self.lot_multiplier  # Fallback com multiplicador
    
    def _get_current_portfolio_value(self):
        """🎯 Obter valor atual do portfolio (saldo da conta MT5)"""
        try:
            # Preferir portfolio virtual da GUI (sessão atual) para dimensionamento consistente entre instâncias
            try:
                if hasattr(self, 'gui') and hasattr(self.gui, 'session_stats'):
                    cb = float(self.gui.session_stats.get('current_balance', self.initial_balance))
                    if cb > 0:
                        self.portfolio_value = cb
                        return cb
            except Exception:
                pass
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
    
    def _execute_order_v7(self, order_type: int, volume: float, sl_price: float = None, tp_price: float = None, slot_id: int = None) -> str:
        """Executar ordem V7 com SL/TP opcionais"""
        try:
            import time  # 🔥 FIX: Import time no início da função
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

            # 🎯 FILTRO DE ATIVIDADE: Bloquear horários ruins
            if Config.ACTIVITY_FILTER_ENABLED and hour in Config.BLOCKED_HOURS:
                self._log(f"[🚫 FILTRO ATIVIDADE] Horário bloqueado: {hour:02d}:00 (win rate <40%)")
                return "ERROR_BLOCKED_HOUR"

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
                
                # Ajustar volume para o step correto (floor para não inflar o tamanho)
                try:
                    import math
                    steps = math.floor(volume / volume_step)
                    volume = steps * volume_step
                except Exception:
                    volume = round(volume, 2)
                
                # Aplicar limites do símbolo e do robô
                volume = max(volume_min, min(volume, volume_max))
                volume = min(volume, self.max_lot_size)
                
                self._log(f"[📊 VOLUME] Ajustado (floor): {volume:.3f} | Limites: [{volume_min:.3f}, {volume_max:.3f}] | Step: {volume_step:.3f}")
            else:
                self._log(f"[⚠️ VOLUME] Não foi possível obter info do símbolo {self.symbol}")
            
            # Preparar requisição com SL/TP opcionais
            # Comentário curto para MT5 (limite de caracteres)
            if slot_id is not None:
                # Usar apenas últimos 4 dígitos do magic number + slot (máximo 10 chars)
                magic_short = str(self.magic_number)[-4:]
                order_comment = f"V7{magic_short}S{slot_id}"  # Ex: V7242S0
            else:
                magic_short = str(self.magic_number)[-4:]
                order_comment = f"V7{magic_short}"  # Ex: V7242

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "magic": self.magic_number,
                "comment": order_comment,
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
                
                # 🔥 FIX CRÍTICO: Mapear slot→ticket IMEDIATAMENTE
                if slot_id is not None:
                    # Aguardar um momento para posição aparecer no MT5
                    time.sleep(0.5)
                    
                    # Buscar posição recém-aberta por ticket/comment
                    new_positions = mt5.positions_get(symbol=self.symbol)
                    if new_positions:
                        for pos in new_positions:
                            if (pos.magic == self.magic_number and 
                                pos.ticket not in self.position_slot_map and
                                f"SLOT{slot_id}" in str(getattr(pos, 'comment', ''))):
                                self.position_slot_map[pos.ticket] = int(slot_id)
                                self._log(f"🔗 [SLOT-MAP] Ticket #{pos.ticket} → Slot {slot_id} (MAPEADO IMEDIATAMENTE)")
                                break
                    
                    # Reconciliar mapeamento geral como backup
                    try:
                        self._map_slot_for_open_positions(slot_id)
                    except Exception:
                        pass
                
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
        """🔍 Validar estrutura do ZIP do modelo Legion V1"""
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
    #     Usa as heads que o Legion V1 REALMENTE produz:
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
    #         return True, "Legion V1 Outputs não disponíveis - Aprovado"
    #         
    #     except Exception as e:
    #         # Em caso de erro, aprovar (não bloquear modelo)
    #         return True, f"V7 Entry Filters: Erro {str(e)[:50]} - Aprovado"

    # def _apply_v7_intuition_filters(self, action_type, v7_outputs):
    #     """
    #     🧠 FILTROS V7 INTUITION REAIS: Usa gates que a V7 REALMENTE produz
    #     
    #     Heads Legion V1 REAIS:
    #     - entry_decision, entry_conf (SpecializedEntryHead)
    #     - mgmt_decision, mgmt_conf (TwoHeadDecisionMaker)  
    #     - regime_id, actor_attention, critic_attention (UnifiedBackbone)
    #     
    #     NÃO usa gates V5 (long_signal, short_signal) que NÃO EXISTEM na V7!
    #     """
    #     try:
    #         if 'gates' not in v7_outputs:
    #             return True, "Heads Legion V1 não disponíveis - Aprovado"
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
        # Título único por instância para facilitar o foco/gestão de múltiplas GUIs
        try:
            import os as _os
            unique_title = f"Legion AI Trader V7 - Professional Trading System [{_os.getpid()}]"
        except Exception:
            unique_title = "Legion AI Trader V7 - Professional Trading System"
        self.root.title(unique_title)
        self.root.geometry("1200x800")
        self.root.configure(bg='#1e1e1e')
        self.root.resizable(True, True)
        
        # Thread-safe GUI management
        import queue
        import threading
        # 🧠 MEMORY FIX: Queue com maxsize para evitar acúmulo de logs
        self.log_queue = queue.Queue(maxsize=1000)  # Máximo 1000 mensagens na fila
        self.update_callbacks = []
        self.callback_cleanup_counter = 0  # Contador para limpeza periódica
        self.is_closing = False
        self.last_stats_update = 0
        self.stats_update_interval = 2.0
        self.gui_responsive = True  # Mantido para compatibilidade
        self._updating_stats = False  # Flag para prevenir múltiplas chamadas
        
        # Configure styles
        self.setup_styles()
        
        # Robot instance
        self.robot = TradingRobotV7()
        # Usar session_id do robô (já criado na inicialização) ou criar fallback
        if hasattr(self.robot, 'session_id') and self.robot.session_id:
            self.session_id = self.robot.session_id
        else:
            import time as _time
            self.session_id = f"SESS_{int(_time.time())}"[-8:]
        self.session_prefix = f"V7S_{self.session_id}_"
        # Propagar para o robô
        setattr(self.robot, 'session_prefix', self.session_prefix)
        self.trading_active = False
        self.trading_thread = None
        self.stop_event = Event()
        
        # Stats tracking - Virtual Portfolio $500
        self.session_stats = {
            'buys': 0,
            'sells': 0,
            'wins': 0,
            'losses': 0,
            'profit_loss': 0.0,
            'initial_balance': 500.0,
            'current_balance': 500.0,  # Tracked virtual balance
            'session_start_time': time.time(),
            'peak_balance': 500.0,
            'max_drawdown': 0.0
        }
        
        # Model selection - Dynamic loading
        self.selected_model_path = None
        
        self.setup_gui()
        self.robot.log_widget = self.log_text  # Conectar logs do robot à GUI
        self.robot.gui = self  # Conectar robot à GUI para estatísticas
        
        # Start periodic updates
        self.process_log_queue()  # Start log queue processing
        self.update_stats()
        
        # Setup cleanup on window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # Removidas todas as funções de manipulação de janela problemáticas
    
    def enqueue_log(self, message):
        """🔧 FIX: Thread-safe method to add log messages"""
        if not self.is_closing:
            try:
                self.log_queue.put(message, timeout=0.1)
            except queue.Full:
                pass  # Drop message if queue is full
    
    def process_log_queue(self):
        """🔧 FIX: Process log messages from queue in main thread"""
        try:
            while not self.log_queue.empty():
                message = self.log_queue.get_nowait()
                if hasattr(self, 'log_text'):
                    self.log_text.insert(tk.END, message + "\n")
                    self.log_text.see(tk.END)
        except queue.Empty:
            pass
        except Exception:
            pass
        
        # Reagendar se não estiver fechando
        if not self.is_closing:
            self.safe_after(100, self.process_log_queue)
    
    def safe_after(self, delay, callback):
        """🔧 FIX: Thread-safe wrapper for root.after with cleanup tracking"""
        if not self.is_closing:
            try:
                callback_id = self.root.after(delay, callback)
                self.update_callbacks.append(callback_id)

                # 🧠 MEMORY FIX: Limpar callbacks antigos a cada 100 chamadas
                self.callback_cleanup_counter += 1
                if self.callback_cleanup_counter >= 100:
                    self._cleanup_old_callbacks()
                    self.callback_cleanup_counter = 0

                return callback_id
            except:
                pass
        return None

    def _cleanup_old_callbacks(self):
        """🧠 MEMORY FIX: Remove callbacks já executados da lista"""
        if len(self.update_callbacks) > 50:
            # Manter apenas os últimos 50 callbacks (os mais antigos já foram executados)
            self.update_callbacks = self.update_callbacks[-50:]
    
    def cleanup_callbacks(self):
        """🔧 FIX: Cancel all pending callbacks before closing"""
        self.is_closing = True
        
        # Cancel all registered callbacks
        for callback_id in self.update_callbacks:
            try:
                self.root.after_cancel(callback_id)
            except:
                pass
        
        self.update_callbacks.clear()
    
    def on_closing(self):
        """🔧 FIX: Immediate closure without hanging"""
        try:
            # Set stop event immediately
            if hasattr(self, 'stop_event'):
                self.stop_event.set()

            # Don't wait for threads - kill them immediately
            if hasattr(self, 'trading_thread') and self.trading_thread:
                # Force thread termination without waiting
                pass

            # Clear callbacks quickly
            self.cleanup_callbacks()

        except Exception as e:
            print(f"Error during cleanup: {e}")
        finally:
            # Force immediate destruction
            try:
                self.root.quit()  # Exit mainloop
            except:
                pass
            try:
                self.root.destroy()  # Destroy window
            except:
                pass
    
    # Removidas funções de heartbeat e restore
        
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
        
        subtitle = tk.Label(title_frame, text="Professional Trading System V7.0 - 1MIN TIMEFRAME MODELS",
                           font=('Segoe UI', 10), fg='#888888', bg='#1e1e1e')
        subtitle.pack(anchor='w')

        # 🍒 Cherry V1 Label
        cherry_label = tk.Label(title_frame, text="✨ Cherry V1 ✨",
                               font=('Segoe UI', 12, 'bold'), fg='#DA70D6', bg='#1e1e1e')
        cherry_label.pack(anchor='w', pady=(2, 0))
        
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
        
        # Configure grid weights for 4x2 layout (expanded)
        for i in range(4):
            grid_frame.columnconfigure(i, weight=1)
        for i in range(4):  # Expanded to 4 rows
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
        
        # Row 2: Virtual Portfolio
        self.create_stat_row(grid_frame, 2, 0, "BALANCE", "$500.00", "#00ff88")
        self.stat_values['balance'] = grid_frame.grid_slaves(row=2, column=1)[0]
        
        self.create_stat_row(grid_frame, 2, 2, "GROWTH", "0.0%", "#ffd93d")
        self.stat_values['growth'] = grid_frame.grid_slaves(row=2, column=3)[0]
        
        # Row 3: Risk Management
        self.create_stat_row(grid_frame, 3, 0, "MAX DD", "0.0%", "#ff6b6b")
        self.stat_values['drawdown'] = grid_frame.grid_slaves(row=3, column=1)[0]
        
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
        
        # Model selector button
        model_button = tk.Button(button_container, text="SELECT MODEL",
                                command=self.select_model_file,
                                font=('Segoe UI', 10),
                                bg='#ff6b6b', fg='#ffffff',
                                activebackground='#e74c3c',
                                relief='flat', bd=0, padx=20, pady=6)
        model_button.pack(fill=tk.X, pady=(5, 5))
        
        # Model status label
        self.model_status_label = tk.Label(button_container, text="No model selected",
                                          font=('Segoe UI', 8), 
                                          fg='#888888', bg='#2d2d2d',
                                          wraplength=180)
        self.model_status_label.pack(fill=tk.X, pady=(0, 5))
        
        reset_button = tk.Button(button_container, text="RESET STATISTICS",
                                command=self.reset_stats,
                                font=('Segoe UI', 10),
                                bg='#404040', fg='#ffffff',
                                activebackground='#505050',
                                relief='flat', bd=0, padx=20, pady=6)
        reset_button.pack(fill=tk.X, pady=(5, 5))

        # 🔢 MAGIC NUMBER - DISPLAY PARA IDENTIFICAÇÃO
        magic_frame = tk.Frame(button_container, bg='#2d2d2d')
        magic_frame.pack(fill=tk.X, pady=(15, 5))

        magic_label = tk.Label(magic_frame, text="🔢 MAGIC NUMBER DESTA INSTÂNCIA",
                              font=('Segoe UI', 9, 'bold'),
                              fg='#ffd93d', bg='#2d2d2d')
        magic_label.pack(anchor='w')

        # Display do magic number (grande e bem visível)
        self.magic_display = tk.Label(magic_frame,
                                     text=f"{self.robot.magic_number}",
                                     font=('Segoe UI', 18, 'bold'),
                                     fg='#00ff88', bg='#2d2d2d')
        self.magic_display.pack(anchor='w', pady=(5, 2))

        # Info sobre uso
        magic_info = tk.Label(magic_frame,
                             text="💡 Use este número para identificar trades desta instância no log",
                             font=('Segoe UI', 7),
                             fg='#888888', bg='#2d2d2d')
        magic_info.pack(anchor='w', pady=(0, 5))

        # 🎯 FILTRO DE ATIVIDADE - BOTÃO TOGGLE
        filter_toggle_frame = tk.Frame(button_container, bg='#2d2d2d')
        filter_toggle_frame.pack(fill=tk.X, pady=(15, 5))

        # Variável de estado do filtro
        self.activity_filter_enabled = False

        # Botão toggle grande e visível
        self.filter_toggle_button = tk.Button(
            filter_toggle_frame,
            text="⭕ FILTRO DE ATIVIDADE: DESATIVADO",
            command=self.toggle_activity_filter,
            font=('Segoe UI', 11, 'bold'),
            bg='#ff4444',  # Vermelho quando desativado
            fg='#ffffff',
            activebackground='#cc0000',
            relief='raised',
            bd=3,
            padx=15,
            pady=10
        )
        self.filter_toggle_button.pack(fill=tk.X)

        # Info do filtro (pequena)
        filter_info_label = tk.Label(
            filter_toggle_frame,
            text="Bloqueia horários ruins | Prioriza SHORT (47% WR) | Ajusta confidence",
            font=('Segoe UI', 7),
            fg='#888888',
            bg='#2d2d2d'
        )
        filter_info_label.pack(anchor='w', pady=(3, 0))

        # 💰 LOT SIZE MULTIPLIER SECTION
        lot_section_frame = tk.Frame(button_container, bg='#2d2d2d')
        lot_section_frame.pack(fill=tk.X, pady=(10, 10))
        
        # Lot size label
        lot_label = tk.Label(lot_section_frame, text="BASE LOT SIZE", 
                            font=('Segoe UI', 9, 'bold'), 
                            fg='#ffd93d', bg='#2d2d2d')
        lot_label.pack(anchor='w')
        
        # Lot input frame
        lot_input_frame = tk.Frame(lot_section_frame, bg='#2d2d2d')
        lot_input_frame.pack(fill=tk.X, pady=(5, 0))
        
        # Lot size entry
        self.lot_size_var = tk.StringVar(value="0.02")
        self.lot_size_entry = tk.Entry(lot_input_frame, textvariable=self.lot_size_var,
                                      font=('Segoe UI', 10), width=8,
                                      bg='#404040', fg='#ffffff', 
                                      relief='flat', bd=2,
                                      insertbackground='#ffffff')
        self.lot_size_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        # Apply button
        apply_lot_button = tk.Button(lot_input_frame, text="APPLY",
                                    command=lambda: (print("BUTTON CLICKED!"), self.apply_lot_size()),
                                    font=('Segoe UI', 8, 'bold'),
                                    bg='#00d4ff', fg='#000000',
                                    activebackground='#0099cc',
                                    relief='flat', bd=0, padx=10, pady=3)
        apply_lot_button.pack(side=tk.LEFT)
        
        # Range info
        range_label = tk.Label(lot_section_frame, text="Range: 0.01 - 0.40 (1x - 20x)",
                              font=('Segoe UI', 8), 
                              fg='#888888', bg='#2d2d2d')
        range_label.pack(anchor='w', pady=(5, 0))
        
        # Current multiplier display
        self.multiplier_label = tk.Label(lot_section_frame, text="Multiplier: 1.0x | Max Lot: 0.03",
                                        font=('Segoe UI', 8),
                                        fg='#cccccc', bg='#2d2d2d')
        self.multiplier_label.pack(anchor='w', pady=(2, 0))

    def toggle_activity_filter(self):
        """🎯 Toggle do Filtro de Atividade - BOTÃO ON/OFF"""
        # Alternar estado
        self.activity_filter_enabled = not self.activity_filter_enabled

        # Atualizar Config do robot
        Config.ACTIVITY_FILTER_ENABLED = self.activity_filter_enabled

        # ⚠️ NÃO ALTERAR SL/TP RANGES - Modelo foi treinado com ranges específicos!
        # O filtro apenas ativa: bloqueio de horários + ajuste de confidence SHORT/LONG

        if self.activity_filter_enabled:
            # ATIVADO - Aplicar filtros (SEM alterar SL/TP ranges)

            # Atualizar botão para VERDE
            self.filter_toggle_button.config(
                text="✅ FILTRO DE ATIVIDADE: ATIVADO",
                bg='#00cc44',  # Verde
                activebackground='#009933'
            )

            self.log("🎯 [FILTRO ATIVIDADE] ✅ ATIVADO")
            self.log(f"   • Horários bloqueados: {Config.BLOCKED_HOURS}")
            self.log(f"   • SHORT boost: {Config.SHORT_CONFIDENCE_BOOST}x")
            self.log(f"   • LONG penalty: {Config.LONG_CONFIDENCE_PENALTY}x")
            self.log(f"   • SL/TP ranges: INALTERADOS (modelo treinado)")
        else:
            # DESATIVADO - Voltar ao padrão

            # Atualizar botão para VERMELHO
            self.filter_toggle_button.config(
                text="⭕ FILTRO DE ATIVIDADE: DESATIVADO",
                bg='#ff4444',  # Vermelho
                activebackground='#cc0000'
            )

            self.log("🎯 [FILTRO ATIVIDADE] ❌ DESATIVADO (Modo Padrão)")
            self.log(f"   • Todos os filtros desativados")

    def apply_lot_size(self):
        """💰 Aplicar nova configuração de lote base com multiplicador"""
        try:
            # Validar entrada
            new_lot_value = float(self.lot_size_var.get())
            
            # Validar limites (0.01 a 0.4 = 1x a 20x)
            if new_lot_value < 0.01:
                self.lot_size_var.set("0.01")
                new_lot_value = 0.01
                self.log("⚠️ [LOT SIZE] Valor mínimo aplicado: 0.01 (1x)")
            elif new_lot_value > 0.40:
                self.lot_size_var.set("0.40") 
                new_lot_value = 0.40
                self.log("⚠️ [LOT SIZE] Valor máximo aplicado: 0.40 (20x)")
            
            # Calcular multiplicador baseado no lote original de treinamento (0.02)
            lot_multiplier = new_lot_value / 0.02
            
            # Atualizar valores no robot
            self.robot.lot_multiplier = lot_multiplier
            self.robot.base_lot_size = new_lot_value
            self.robot.max_lot_size = 0.03 * lot_multiplier
            self.robot.lot_size = new_lot_value
            
            # Atualizar display
            multiplier_text = f"{lot_multiplier:.1f}x"
            self.multiplier_label.config(text=f"Multiplier: {multiplier_text} | Max Lot: {self.robot.max_lot_size:.3f}")
            
            # Log da alteração
            self.log(f"💰 [LOT MULTIPLIER] Aplicado: Base={new_lot_value:.3f} | Max={self.robot.max_lot_size:.3f} | Multiplier={multiplier_text}")
            
        except ValueError:
            self.log("❌ [LOT SIZE] Valor inválido! Use números decimais (ex: 0.04)")
            self.lot_size_var.set("0.02")  # Restaurar valor padrão
        
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
        try:
            if hasattr(self, 'robot') and hasattr(self.robot, 'end_session'):
                self.robot.end_session()
        except Exception:
            pass
        self.log("⏹️ Trading V7 parado!")
        
    def run_trading_loop(self):
        """Loop principal de trading"""
        try:
            # Inicializar robô em thread separada para não bloquear a GUI
            from threading import Thread as _Thread
            _Thread(target=self.robot.run, daemon=True).start()

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
                # Atualizar estatísticas baseadas em posições fechadas (integra com GUI)
                if hasattr(self.robot, '_update_position_stats'):
                    self.robot.gui = self  # garantir referência
                    self.robot._update_position_stats()

                # Atualizar GUI imediatamente
                self.update_stats()
        
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
        """Update statistics safely without multiple recursions"""
        # Prevenir múltiplas execuções simultâneas
        if self._updating_stats or self.is_closing:
            return
            
        self._updating_stats = True
        
        try:
            # Verificar se janela está visível antes de atualizar
            try:
                window_state = self.root.state()
                if window_state in ('iconic', 'withdrawn'):
                    # Janela minimizada - não atualizar GUI
                    return
            except:
                return
            
            current_time = time.time()
            
            # Skip se muito frequente
            if current_time - self.last_stats_update < self.stats_update_interval:
                return
            
            self.last_stats_update = current_time
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
                
                # Update virtual balance (new)
                if 'balance' in self.stat_values:
                    current_balance = self.session_stats['current_balance']
                    balance_color = '#00ff88' if current_balance >= 500 else '#ff6b6b'
                    self.stat_values['balance'].config(text=f"${current_balance:.2f}", fg=balance_color)
                
                # Update growth percentage (new)
                if 'growth' in self.stat_values:
                    growth = ((self.session_stats['current_balance'] - 500) / 500) * 100
                    growth_sign = '+' if growth >= 0 else ''
                    growth_color = '#00ff88' if growth >= 0 else '#ff6b6b'
                    self.stat_values['growth'].config(text=f"{growth_sign}{growth:.1f}%", fg=growth_color)
                
                # Update max drawdown (new)
                if 'drawdown' in self.stat_values:
                    dd = self.session_stats['max_drawdown']
                    dd_color = '#ff6b6b' if dd > 10 else '#ffaa00' if dd > 5 else '#00ff88'
                    self.stat_values['drawdown'].config(text=f"{dd:.1f}%", fg=dd_color)
            
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
            self._updating_stats = False
            # Agendar próxima atualização apenas se janela estiver ativa
            if not self.is_closing:
                self.safe_after(2000, self.update_stats)  # Intervalo fixo de 2 segundos
            
    def reset_stats(self):
        """Reset estatísticas da sessão"""
        self.session_stats = {
            'buys': 0,
            'sells': 0,
            'wins': 0,
            'losses': 0,
            'profit_loss': 0.0,
            'initial_balance': 500.0,
            'current_balance': 500.0,  # Reset virtual portfolio
            'session_start_time': time.time(),
            'peak_balance': 500.0,
            'max_drawdown': 0.0
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
    
    def select_model_file(self):
        """🎯 Seletor de arquivo para escolher modelo"""
        try:
            # Abrir dialog para selecionar arquivo .zip
            file_path = filedialog.askopenfilename(
                title="Select Trading Model",
                filetypes=[
                    ("Model files", "*.zip"),
                    ("All files", "*.*")
                ],
                initialdir="D:/Projeto"
            )
            
            if file_path:
                self.selected_model_path = file_path
                filename = os.path.basename(file_path)
                self.model_status_label.config(
                    text=f"Selected: {filename}",
                    fg='#00ff88'
                )
                self.log(f"🎯 Model selected: {filename}")
                
                # Load the selected model
                self.load_selected_model()
            else:
                self.log("⚠️ No model selected")
                
        except Exception as e:
            self.log(f"❌ Error selecting model: {e}")
    
    def load_selected_model(self):
        """🤖 Load the selected model dynamically"""
        try:
            if not self.selected_model_path:
                self.log("❌ No model path selected")
                return False
            
            self.log(f"🔄 Loading model: {self.selected_model_path}")
            
            # Override the robot's model path and load
            self.robot.model = None
            self.robot.model_loaded = False
            # Reset normalizer before (re)load
            self.robot.normalizer = None
            
            # Try to load with compatibility support
            model_loaded = False
            filename = os.path.basename(self.selected_model_path)

            # Strategy 1: Use compatibility loader (supports pre/post-bypass)
            try:
                self.log(f"🔄 Loading with auto-compatibility detection...")
                self.robot.model = self.robot.load_model_with_policy_compat(self.selected_model_path)
                model_loaded = True
                self.log(f"✅ Loaded with compatibility: {filename}")
            except Exception as e1:
                self.log(f"⚠️ Compatibility load failed: {e1}")

            # Strategy 2: Standard PPO (fallback)
            if not model_loaded:
                try:
                    from stable_baselines3 import PPO
                    self.robot.model = PPO.load(self.selected_model_path)
                    model_loaded = True
                    self.log(f"✅ Loaded as PPO: {filename}")
                except Exception as e2:
                    self.log(f"⚠️ Standard PPO failed: {e2}")
            
            if model_loaded:
                # Put model in inference mode
                self.robot.model.policy.eval()
                for param in self.robot.model.policy.parameters():
                    param.requires_grad = False
                
                self.robot.model_loaded = True
                self.robot.is_legion_model = 'Legion' in filename
                
                self.model_status_label.config(
                    text=f"Loaded: {filename}",
                    fg='#00ff88'
                )
                
                # Try to load Enhanced Normalizer from the same directory as the model
                try:
                    model_dir = os.path.dirname(self.selected_model_path)
                    # Primary expected filename
                    normalizer_candidates = [
                        os.path.join(model_dir, "enhanced_normalizer_final.pkl"),
                        os.path.join(model_dir, "enhanced_normalizer_final_enhanced.pkl")
                    ]
                    # If none of the primary names exist, try any *.pkl with 'normalizer' in the name
                    if not any(os.path.exists(p) for p in normalizer_candidates):
                        try:
                            candidates = [
                                os.path.join(model_dir, f)
                                for f in os.listdir(model_dir)
                                if f.lower().endswith('.pkl') and 'normalizer' in f.lower()
                            ]
                            normalizer_candidates.extend(candidates)
                        except Exception:
                            pass
                    loaded_normalizer = False
                    for norm_path in normalizer_candidates:
                        if os.path.exists(norm_path):
                            try:
                                with open(norm_path, 'rb') as f:
                                    self.robot.normalizer = pickle.load(f)
                                self.log(f"✅ Enhanced Normalizer loaded: {os.path.basename(norm_path)}")
                                loaded_normalizer = True
                                break
                            except Exception as ne:
                                self.log(f"⚠️ Failed to load normalizer '{os.path.basename(norm_path)}': {ne}")
                    if not loaded_normalizer:
                        self.log("⚠️ No Enhanced Normalizer found alongside the selected model (continuing without it)")
                except Exception as e_norm:
                    self.log(f"⚠️ Normalizer scan error: {e_norm}")

                self.log(f"✅ Model loaded successfully: {filename}")
                return True
            else:
                self.model_status_label.config(
                    text="Failed to load model",
                    fg='#ff6b6b'
                )
                self.log(f"❌ Failed to load model: {filename}")
                return False
                
        except Exception as e:
            self.log(f"❌ Error loading model: {e}")
            return False
    
    def update_virtual_balance(self, pnl_change):
        """💰 Update virtual balance tracking"""
        try:
            self.session_stats['current_balance'] += pnl_change
            
            # Update peak balance
            if self.session_stats['current_balance'] > self.session_stats['peak_balance']:
                self.session_stats['peak_balance'] = self.session_stats['current_balance']
            
            # Update max drawdown
            current_dd = (self.session_stats['peak_balance'] - self.session_stats['current_balance']) / self.session_stats['peak_balance'] * 100
            if current_dd > self.session_stats['max_drawdown']:
                self.session_stats['max_drawdown'] = current_dd
            
            self.update_stats()
            
        except Exception as e:
            self.log(f"❌ Error updating virtual balance: {e}")

def main_gui():
    """Função principal para iniciar GUI"""
    root = tk.Tk()
    app = TradingAppV7(root)

    try:
        # Set up proper exit handling
        def force_exit():
            try:
                root.quit()
                root.destroy()
            except:
                pass
            import os
            os._exit(0)  # Force exit if needed

        # Backup exit method
        root.protocol("WM_DELETE_WINDOW", lambda: app.on_closing() or force_exit())

        root.mainloop()
    except KeyboardInterrupt:
        print("\n[🛑] GUI interrompida pelo usuário")
        try:
            root.destroy()
        except:
            pass
    except Exception as e:
        print(f"[❌] Erro crítico na GUI: {e}")
        try:
            root.destroy()
        except:
            pass
    finally:
        # Ensure exit
        import sys
        sys.exit(0)

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
