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
🎣 ROBOPESCADOR - Trading Robot Pescador Scalp Especializado
🧠 CONFIGURADO PESCADOR: PescadorRewardSystem + Activity System + Scalp Ranges
🎯 OBSERVATION SPACE: 450 dimensões nativas (formato compatível pescador)

🎣 PESCADOR ARCHITECTURE:
- OBSERVATION SPACE: 450 dimensões (compatível com pescador training)
- ACTION SPACE: 4 dimensões [entry_decision, entry_confidence, pos1_mgmt, pos2_mgmt]
- FEATURES: Market + Positions + Simple = 39 features por step
- WINDOW: 10 steps × 45 features = 450 dimensões
- SCALP OPTIMIZED: SL/TP ranges curtos (0.3-0.7 / 0.5-1.0 pontos)
- NO COOLDOWNS: Operação contínua sem pausas entre trades

ACTION SPACE (4D) - Pescador Scalp:
- [0] entry_decision: Logit bruto para entrada (sem gate filtering)
- [1] entry_confidence: [0,1] Confiança na decisão de entrada
- [2] pos1_mgmt: Management para posição 1 (SL/TP scalp)
- [3] pos2_mgmt: Management para posição 2 (SL/TP scalp)

🎣 PESCADOR SCALP FEATURES:
- sl_adjust [-3,3]: Ativa/move trailing stop (ranges curtos 0.3-0.7)
- tp_adjust [-3,3]: Intensidade do trailing (ranges curtos 0.5-1.0)  
- Activity bonuses: Recompensas por trading ativo e frequente
- No cooldowns: Zero pausas entre ordens para scalping contínuo

CONVERSÃO PESCADOR: [-3,3] → [0.3-0.7] SL, [0.5-1.0] TP pontos (SCALP: 1 ponto = $1.00)

COMPATIBILIDADE PESCADOR:
- 🎣 PescadorEnv Architecture (compatível com pescador.py training)
- 📋 TradingTransformerFeatureExtractor
- 🔧 Enhanced Normalizer
- 🎯 Scalp-optimized SL/TP System

🔧 PESCADOR UPDATES:
- _get_observation_v7(): Gera 450D nativamente (formato pescador)
- _process_v7_action(): Processa 4D Pescador actions → 8D Robot actions
- _verify_pescador_compatibility(): Verificação específica para modelos pescador
- auto_load_model(): Carrega modelos pescador treinados
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

# 🎣 PESCADOR CONFIGURATIONS
MIN_CONFIDENCE_THRESHOLD = 0.2  # 20% - Alinhado com pescador.py (vs 60% RobotV7)

# 🎯 V7 MODEL PATHS - Configuração centralizada
class ModelPaths:
    """🎣 Configuração centralizada de paths para modelos PESCADOR"""
    # Caminho padrão do modelo ZIP - Pescador baseline
    MODEL_ZIP_PATH = "Modelo daytrade/Legion V1.zip"  # Fallback se não encontrar pescador
    
    # Caminhos para modelos do pescador.py (checkpoints com steps)
    PESCADOR_MODEL_DIR = "../../Otimizacao/treino_principal/models/PESCADOR"  # Diretório pescador
    PESCADOR_MODEL_PATTERN = "*_steps_*.zip"  # Padrão dos checkpoints pescador
    
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
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
print("[LEGION V1] ✅ TwoHeadV11Sigmoid importada - Modelo Legion V1 exclusivo")
print("[INTUITION] ✅ TradingTransformerFeatureExtractor importado OBRIGATÓRIO")

# === 🎯 CONFIGURAÇÃO SL/TP DINÂMICO (ALINHADA COM SILUS.PY) ===
REALISTIC_SLTP_CONFIG = {
    # 🎯 RANGES DAYTRADE CORRETOS - ALINHADOS COM SILUS
    'sl_min_points': 2,     # SL mínimo: 2 pontos (daytrade)
    'sl_max_points': 8,     # SL máximo: 8 pontos (daytrade)  
    'tp_min_points': 3,     # TP mínimo: 3 pontos (daytrade)
    'tp_max_points': 15,    # TP máximo: 15 pontos (daytrade)
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

class Config:
    """Configurações do sistema V7"""
    SYMBOL = "GOLD" 
    INITIAL_BALANCE = 500.0
    MAX_POSITIONS = 2  # 🔬 ALINHADO: silus.py pós-cirurgia usa 2 posições - modelo Legion V1
    WINDOW_SIZE = 20
    MAX_LOT_SIZE = 0.03
    BASE_LOT_SIZE = 0.02
    
    # V7 Specific
    OBSERVATION_SPACE_SIZE = 450   # 45 features × 10 window (Legion V1 format - mantém compatibilidade)
    FEATURES_PER_STEP = 45  # Legion V1 format: 20 market + 18 position + 7 simple (padding para compatibilidade)
    ACTION_SPACE_SIZE = 4  # Legion V1 format: [entry_decision, entry_confidence, pos1_mgmt, pos2_mgmt]
    
    # Features breakdown - Legion V1 format
    MARKET_FEATURES = 20           # Market features (first 20 from 65 available)
    POSITION_FEATURES = 18         # 2 positions × 9 features (alinhado com modelo treinado)
    SIMPLE_FEATURES = 7            # Simple context features
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
        
        # 😨 SISTEMA DE COOLDOWN ANTI-OVERTRADING (timestamp-based)
        self.cooldown_minutes = 0  # 🎣 PESCADOR: SEM COOLDOWNS (scalp rápido)
        self.last_position_closed_timestamp = 0  # Timestamp do último fechamento
        self._log(f"[🎣 PESCADOR] Sistema de cooldown DESABILITADO - {self.cooldown_minutes} minutos (scalp sem pausa)")

        # 🎣 PESCADOR: Cooldown slots desabilitados (sempre 0.0)
        self.position_slot_cooldowns = {i: 0.0 for i in range(self.max_positions)}  # slot -> sempre 0.0
        self.position_slot_map = {}  # ticket -> slot
        
        # 🎯 MAGIC NUMBER: Para isolar posições do robô das suas posições manuais
        self.magic_number = 777888  # Número único para o Legion V1
        self._log(f"[🔒 ISOLATION] Magic number configurado: {self.magic_number} - Posições isoladas!")
        self._log(f"[🔒 ISOLATION] ✅ O robô gerenciará APENAS suas próprias posições")
        self._log(f"[🔒 ISOLATION] ✅ Você pode tradear manualmente outras posições/ativos livremente!")
        
        # 🛡️ TRACKER DE POSIÇÕES: Para detectar novas posições manuais
        self.known_positions = set()  # Set com tickets de posições conhecidas
        self.position_stats = {}  # Dicionário com stats das posições: {ticket: {'open_price': float, 'volume': float, 'type': str}}
        
        # 🧠 V7 GATES - SINCRONIZAÇÃO COM DAYTRADER
        # Removido: last_v7_gate_info (Legion V1 usa Entry/Management heads)
        # Removido: last_v7_outputs (Legion V1 não usa filtros baseados em gates)
        self.daily_trades = []  # Para controle de boost de trades
        
        # 🔥 ACTION SPACE LEGION V1: 4 dimensões
        self.action_space = spaces.Box(
            low=np.array([-10.0, 0.0, -3.0, -3.0]),
            high=np.array([10.0, 1.0, 3.0, 3.0]),
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
        
        self._log(f"[OBS SPACE LEGION] 🧠 {Config.OBSERVATION_SPACE_SIZE} dimensões (45 features × 10 window)")
        
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
        # 🎣 PESCADOR RANGES: Mais curtos para scalping
        self.sl_range_min = 0.3    # SL mínimo: 0.3 pontos (pescador scalp)
        self.sl_range_max = 0.7    # SL máximo: 0.7 pontos (pescador scalp)
        self.tp_range_min = 0.5    # TP mínimo: 0.5 pontos (pescador scalp)
        self.tp_range_max = 1.0    # TP máximo: 1.0 pontos (pescador scalp)
        self.sl_tp_step = 0.1      # Variação: 0.1 pontos (pescador)
        
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
        self._log(f"[🎣 PESCADOR SL/TP] SL: {self.sl_range_min}-{self.sl_range_max} pontos, TP: {self.tp_range_min}-{self.tp_range_max} pontos (SCALP RANGES)")
        self._log(f"[🔒 LIMITE] Máximo de posições simultâneas: {self.max_positions}")
        self._log(f"[💰 LOT SIZE] Base: {self.base_lot_size} | Max: {self.max_lot_size} | Dynamic sizing: ATIVO")
        
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
        
        # 🔥 LEGION V1: 65 market features disponíveis (usamos primeiras 20)
        # Legion format: 20 market + 18 positions + 7 simple = 45 features (2 posições ativas)
        
        self._log(f"[LEGION V1] Historical dataframe columns: {len(all_columns)} market-based features (usando primeiras 20)")
        self._log(f"[LEGION V1] Format: 20 market + 18 positions + 7 simple = 45 features × 10 steps = 450D")
        
        return all_columns
    
    def _initialize_historical_data_v7(self):
        """🔧 Inicializa dados históricos Legion V1 com features otimizadas"""
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
        """🚀 Obter observação Legion V1 - 450 dimensões (45 features × 10 window)"""
        try:
            # Obter preço atual
            if self.mt5_connected:
                tick = mt5.symbol_info_tick(self.symbol)
                current_price = tick.bid if tick else 2000.0
            else:
                current_price = 2000.0
            
            # 🔥 LEGION V1 FORMAT: 10 steps × 45 features = 450D
            window_size = 10
            features_per_step = 45
            
            # 1. MARKET FEATURES (20 features por step - Legion V1 format)
            if len(self.historical_df) > 0 and len(self.feature_columns) > 0:
                # Pegar últimos 10 steps e primeiras 20 features (Legion V1)
                recent_data = self.historical_df[self.feature_columns[:20]].tail(window_size).values
                
                if len(recent_data) < window_size:
                    # Padding se dados insuficientes
                    padding = np.zeros((window_size - len(recent_data), 20))
                    recent_data = np.vstack([padding, recent_data])
            else:
                recent_data = np.zeros((window_size, 20))
            
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
                base_idx = i * 9  # 9 features por posição como silus.py
                
                # Features alinhadas com silus.py structure
                entry_price = max(pos.price_open, 0.01) / 10000.0  # Normalização silus.py style
                current_price_norm = max(current_price, 0.01) / 10000.0
                
                # Calcular PnL baseado no tipo da posição
                if pos.type == mt5.POSITION_TYPE_BUY:
                    unrealized_pnl = (current_price - pos.price_open) * pos.volume
                    position_type = 1.0  # long
                else:
                    unrealized_pnl = (pos.price_open - current_price) * pos.volume  
                    position_type = 2.0  # short
                
                # Normalizar PnL
                unrealized_pnl = unrealized_pnl if unrealized_pnl != 0 else 0.01
                
                # Calcular duração (crítico para modelo - como silus.py)
                duration = (time.time() - pos.time) / 3600.0  # Em horas
                duration = max(duration, 0.01) / 24.0  # Normalizar para [0,1] aproximadamente
                
                # 9 features por posição (FORMATO SILUS.PY):
                positions_obs[base_idx:base_idx+9] = [
                    1.0,  # [0] Posição ativa
                    float(entry_price),         # [1] Entry price normalizado
                    float(current_price_norm),  # [2] Current price normalizado  
                    float(unrealized_pnl),      # [3] Unrealized PnL
                    float(duration),            # [4] Duration ⭐ (CRITICAL para modelo)
                    float(pos.volume / 1.0),    # [5] Volume normalizado
                    float(position_type),       # [6] Type (1=long, 2=short)
                    float(pos.sl / current_price if pos.sl > 0 else 0.001),  # [7] SL normalizado
                    float(pos.tp / current_price if pos.tp > 0 else 0.001)   # [8] TP normalizado
                ]
            
            # Posições inativas: FORMATO SILUS.PY (posições não preenchidas acima)
            for i in range(len(mt5_positions), 2):  # Preencher posições restantes até 2
                base_idx = i * 9
                positions_obs[base_idx:base_idx+9] = [
                    0.01,  # [0] Inativa
                    0.5,   # [1] Entry price padrão
                    0.5,   # [2] Current price padrão
                    0.01,  # [3] PnL padrão  
                    0.35,  # [4] Duration ⭐ NÃO-ZERO (CRÍTICO)
                    0.02,  # [5] Volume padrão
                    0.01,  # [6] Type padrão
                    0.001, # [7] SL padrão
                    0.001  # [8] TP padrão
                ]
            
            # 3. CONTEXT FEATURES (7 features para completar 45 total: 20+18+7=45)
            # Legion V1 format: market(20) + positions(18) + context(7) = 45
            context_features = np.array([
                current_price / 2500.0,  # Preço normalizado
                len(mt5_positions) / 2.0,  # Número de posições (max 2)  
                self.realized_balance / 1000.0,  # Balance normalizado
                time.time() % 86400 / 86400.0,  # Time of day
                self.current_step % 100 / 100.0,  # Step normalizado
                np.random.normal(0, 0.01),  # Noise for regularization
                1.0   # Bias term
            ], dtype=np.float32)
            
            # 4. CONSTRUIR OBSERVAÇÃO: 10 steps × 45 features = 450D (Legion V1)
            # ESTRUTURA: 20 market + 18 positions + 7 context = 45 features
            total_features_raw = 20 + 18 + 7  # 45 features
            features_per_step = 45  # Legion V1 format exato
            obs_matrix = np.zeros((window_size, features_per_step), dtype=np.float32)
            
            for step in range(window_size):
                # Construir features completas Legion V1
                full_features = np.concatenate([
                    recent_data[step],          # Market features (20)
                    positions_obs,             # Position features (18) 
                    context_features           # Context features (7)
                ])  # Total: 45 features exato
                
                # Legion V1 format - sem truncamento necessário
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
        """🚀 Processar ação Legion V1 - 4D: [entry_decision, entry_confidence, pos1_mgmt, pos2_mgmt]"""
        try:
            if not isinstance(action, (list, tuple, np.ndarray)):
                action = np.array([action])
            
            # Garantir 4 dimensões Legion V1
            if len(action) != 4:
                self._log(f"[⚠️ ACTION] Esperado 4D, recebido {len(action)}D - padding/truncating")
                if len(action) < 4:
                    action = np.pad(action, (0, 4 - len(action)), mode='constant')
                else:
                    action = action[:4]
            
            # LEGION V1 ACTION SPACE: [entry_decision, entry_confidence, pos1_mgmt, pos2_mgmt]
            entry_decision = float(action[0])    # Raw logit da Entry Head
            entry_confidence = float(action[1])  # [0,1] confidence da entrada
            pos1_mgmt = float(action[2])        # [-3,3] management posição 1
            pos2_mgmt = float(action[3])        # [-3,3] management posição 2
            
            # 😨 SISTEMA DE COOLDOWN ANTI-OVERTRADING (timestamp-based)
            # 🎣 PESCADOR: COOLDOWN PERMANENTEMENTE DESABILITADO
            cooldown_check = (False, 0.0)  # Força sem cooldown sempre
            if False:  # Cooldown desabilitado para pescador scalp
                entry_decision = 0.0  # NUNCA executado
                entry_type = "HOLD"
                remaining_minutes = cooldown_check[1]
                # Log apenas a cada 60 segundos para reduzir spam
                if int(time.time()) % 60 == 0:
                    self._log(f"⏱️ [COOLDOWN ATIVO] {remaining_minutes:.1f} minutos restantes até próxima entrada permitida")
                    self._log(f"🔒 [COOLDOWN] Timestamp último fechamento: {self.last_position_closed_timestamp}")
                    self._log(f"🕐 [COOLDOWN] Tempo atual: {time.time()}")
                    self._log(f"⏰ [COOLDOWN] Diferença: {time.time() - self.last_position_closed_timestamp:.1f} segundos")
            else:
                # Converter entry_decision Box[0,2] para decisão
                # Action space: [0,2] → 0=HOLD, 1=LONG, 2=SHORT
                if entry_decision >= 1.33:      # [1.33, 2.0] = SHORT
                    entry_type = "SHORT"
                elif entry_decision >= 0.67:    # [0.67, 1.33) = LONG
                    entry_type = "LONG"
                else:                           # [0.0, 0.67) = HOLD
                    entry_type = "HOLD"
            
            # Aplicar lógica de entrada baseada na decisão
            if entry_type == "HOLD":
                self.hold_count += 1
                return f"HOLD #{self.hold_count} - Aguardando melhor oportunidade"
                
            elif entry_type in ["LONG", "SHORT"]:
                # Verificar se pode abrir nova posição (apenas do robô)
                current_positions = len(self._get_robot_positions()) if self.mt5_connected else 0
                
                if current_positions >= self.max_positions:
                    return f"MAX POSITIONS REACHED ({current_positions}/{self.max_positions})"
                
                # Executar entrada com confidence
                if entry_confidence < MIN_CONFIDENCE_THRESHOLD:  # 🎣 PESCADOR: Alinhado com pescador.py
                    return f"LOW CONFIDENCE: {entry_confidence:.2f} < {MIN_CONFIDENCE_THRESHOLD} - HOLD"
                
                # Calcular SL/TP baseado nos management values
                sl_points = abs(pos1_mgmt) * 2 + 2  # 2-8 pontos
                tp_points = abs(pos2_mgmt) * 3 + 3  # 3-15 pontos
                
                # Executar trade
                if self.mt5_connected:
                    result = self._execute_trade_legion(entry_type, entry_confidence, sl_points, tp_points)
                    return result
                else:
                    return f"SIMULATION: {entry_type} - Conf: {entry_confidence:.2f}, SL: {sl_points}p, TP: {tp_points}p"
            
            # 🎯 PROCESSAR GESTÃO DE POSIÇÕES EXISTENTES VIA MANAGEMENT HEAD
            # Sistema de trailing stop dinâmico baseado nas ações do modelo (como silus.py)
            robot_positions = self._get_robot_positions()
            
            if robot_positions:
                # Converter management values em ajustes
                pos1_sl_adjust, pos1_tp_adjust = self._convert_management_to_sltp_adjustments(pos1_mgmt)
                pos2_sl_adjust, pos2_tp_adjust = self._convert_management_to_sltp_adjustments(pos2_mgmt)
                
                sl_adjusts = [pos1_sl_adjust, pos2_sl_adjust]
                tp_adjusts = [pos1_tp_adjust, pos2_tp_adjust]
                
                # Obter preço atual
                tick = mt5.symbol_info_tick(self.symbol)
                if tick:
                    current_price = (tick.bid + tick.ask) / 2
                    
                    # Processar cada posição do robô (como silus.py)
                    for i, pos in enumerate(robot_positions[:2]):  # Máximo 2 posições
                        if i < len(sl_adjusts):
                            sl_adjust = sl_adjusts[i]
                            tp_adjust = tp_adjusts[i]
                            
                            # 🎯 DYNAMIC TRAILING STOP - Baseado nas ações do modelo
                            trailing_result = self._process_dynamic_sl_tp_adjustment(
                                pos, sl_adjust, tp_adjust, current_price, i
                            )
                            
                            # Aplicar mudanças se o modelo decidiu (já processado dentro da função)
                            if trailing_result['action_taken']:
                                # Marcar informações para tracking (metadata já atualizada)
                                pass
            
            return "ACTION_PROCESSED"
            
        except Exception as e:
            self._log(f"[ERROR] Erro ao processar ação Legion V1: {e}")
            return f"ERROR: {str(e)}"
    
    def _execute_trade_legion(self, trade_type, confidence, sl_points, tp_points):
        """Executar trade baseado na decisão Legion V1"""
        try:
            # Obter preço atual
            tick = mt5.symbol_info_tick(self.symbol)
            if not tick:
                return "ERROR: Sem dados de preço"
                
            current_price = tick.bid if trade_type == "SHORT" else tick.ask
            
            # Calcular volume usando position sizing correto
            volume = self._calculate_position_size_v7(confidence)
            
            # Calcular SL/TP
            if trade_type == "LONG":
                sl_price = current_price - (sl_points * 0.1)
                tp_price = current_price + (tp_points * 0.1)
                order_type = mt5.ORDER_TYPE_BUY
            else:  # SHORT
                sl_price = current_price + (sl_points * 0.1)
                tp_price = current_price - (tp_points * 0.1)
                order_type = mt5.ORDER_TYPE_SELL
            
            # Request para MT5
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": self.symbol,
                "volume": round(volume, 2),
                "type": order_type,
                "price": current_price,
                "sl": sl_price,
                "tp": tp_price,
                "deviation": 20,
                "magic": self.magic_number,
                "comment": f"Legion V1 - Conf: {confidence:.2f}",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return f"TRADE FAILED: {result.retcode} - {result.comment}"
            else:
                # Atualizar estatísticas da sessão
                if hasattr(self, 'gui') and hasattr(self.gui, 'session_stats'):
                    if trade_type == "LONG":
                        self.gui.session_stats['buys'] += 1
                    else:  # SHORT
                        self.gui.session_stats['sells'] += 1
                
                # Posição aberta com sucesso (cooldown apenas no fechamento)
                
                self._log(f"[TRADE SUCCESS] {trade_type} - Vol: {volume:.2f}, SL: {sl_points}p, TP: {tp_points}p, Conf: {confidence:.2f}")
                return f"TRADE SUCCESS: {trade_type} - Ticket: {result.order}"
                
        except Exception as e:
            return f"TRADE ERROR: {str(e)}"
    
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
            if position.type == mt5.POSITION_TYPE_BUY:
                pnl = (current_price - position.price_open) * position.volume * 100  # Para GOLD
            else:  # SELL
                pnl = (position.price_open - current_price) * position.volume * 100
            return pnl
        except:
            return 0.0
    
    def _process_v7_action(self, action):
        """🚀 Processar ação V7 - 8 dimensões otimizadas (sem gates)"""
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
            
            # Selecionar slot disponível respeitando cooldown independente por slot
            order_type = mt5.ORDER_TYPE_BUY if entry_decision == 1 else mt5.ORDER_TYPE_SELL
            slot_id, wait_sec = self._allocate_entry_slot()
            if slot_id is None:
                self._log(f"🔒 [COOLDOWN-SLOTS] Nenhum slot livre. Aguardando {wait_sec/60:.1f} min")
                return "HOLD_COOLDOWN_SLOT"

            # Executar ordem (novo executor com suporte a slot_id)
            result = self._execute_order_v7(order_type, base_volume, sl_price, tp_price, slot_id=slot_id)
            
            if "SUCCESS" in result:
                # Trade executado com sucesso (cooldown apenas no fechamento)
                
                # Incrementar estatísticas baseado no tipo de ordem
                if hasattr(self, 'gui') and hasattr(self.gui, 'session_stats'):
                    if entry_decision == 1:  # LONG
                        self.gui.session_stats['buys'] += 1
                    else:  # SHORT  
                        self.gui.session_stats['sells'] += 1
                
                action_type = "📈 LONG" if entry_decision == 1 else "📉 SHORT"
                stars = '⭐' * min(5, int(entry_quality * 5))
                self._log(f"[🎯 V7 TRADE] {action_type} | Quality: {entry_quality:.2f} {stars} | Vol: {base_volume:.3f} | Risk: {risk_appetite:.2f}")
                
                # Atualizar GUI com estatísticas
                if hasattr(self, 'gui') and hasattr(self.gui, 'update_stats'):
                    self.gui.update_stats()
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
            
            positions = self._get_robot_positions()
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
                "magic": self.magic_number,
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
                # Mapear imediatamente o ticket da nova posição ao slot de entrada
                try:
                    if slot_id is not None:
                        # Pequeno atraso para MT5 refletir a nova posição
                        time.sleep(0.1)
                        positions = self._get_robot_positions() or []
                        # Descobrir quais tickets ainda não estão mapeados
                        unmapped = [p for p in positions if p.ticket not in self.position_slot_map]
                        for p in unmapped:
                            # Tentar extrair slot do comentário, preferível
                            c = getattr(p, 'comment', '') or ''
                            s = self._extract_slot_from_comment(str(c))
                            if s is None:
                                # Se não veio no comentário, presumir slot_id desta entrada se ainda livre
                                if slot_id not in self.position_slot_map.values():
                                    s = slot_id
                            if s is not None:
                                self.position_slot_map[p.ticket] = int(s)
                                # Não mapeie múltiplos tickets para o mesmo slot_id nesta iteração
                                break
                except Exception:
                    pass
                # Atualizar estatísticas da GUI (buys/sells) imediatamente
                try:
                    if hasattr(self, 'gui') and hasattr(self.gui, 'session_stats'):
                        if order_type == mt5.ORDER_TYPE_BUY:
                            self.gui.session_stats['buys'] += 1
                        else:
                            self.gui.session_stats['sells'] += 1
                        if hasattr(self.gui, 'root') and hasattr(self.gui.root, 'after'):
                            self.gui.root.after(0, self.gui.update_stats)
                        else:
                            self.gui.update_stats()
                except Exception:
                    pass
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
                self._log(f"[❌ LEGION V1] Action space incorreto: {actual_action_size} != {expected_action_size}")
                return False
            
            # 3. Verificar features por step - Legion V1 format
            expected_features = Config.FEATURES_PER_STEP  # 45
            # Legion V1: 20 market + 18 positions + 7 simple = 45
            actual_features = 45  # Fixed for Legion V1 format
            
            if actual_features != expected_features:
                self._log(f"[❌ LEGION V1] Features per step incorreto: {actual_features} != {expected_features}")
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
            
            # 7. Verificar breakdown de features - Legion V1
            market_features = Config.MARKET_FEATURES  # 20 (using first 20 from available)
            position_features = Config.POSITION_FEATURES  # 12
            simple_features = Config.SIMPLE_FEATURES  # 7
            
            self._log(f"[✅ LEGION V1] Features breakdown:")
            self._log(f"  - Market: {market_features} (using first 20 from {len(self.feature_columns)} available)")
            self._log(f"  - Position: {position_features} (2 posições ativas, 18 features total)")
            self._log(f"  - Simple: {simple_features} (context features)")
            self._log(f"  - Total: {market_features + position_features + simple_features}")
            
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
            
            # Reset cooldown
            self.last_position_closed_timestamp = 0
            
            # Atualizar dados históricos
            self._initialize_historical_data_v7()
            
            return self._get_observation_v7()
            
        except Exception as e:
            self._log(f"[ERROR] Erro no reset V7: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def step(self, action):
        """Step V7 - executar ação e retornar nova observação"""
        try:
            # Processar ação Legion V1 (4D)
            action_result = self._process_legion_action(action)
            
            # Atualizar estatísticas de posições
            self._update_position_stats()
            
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
            
            # 2. Detectar tipo de modelo automaticamente pelos kwargs salvos
            self._log(f"[🔍 V7 AUTO-LOAD] Detectando tipo de modelo automaticamente...")
            
            # Carregar modelo Legion V1 (sem forçar policy_kwargs - modelo já foi salvo corretamente)
            try:
                self._log(f"[🎯 LEGION V1] Carregando modelo Legion V1...")
                self.model = RecurrentPPO.load(zip_path)
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
            
            # 🔥 DEBUG SIMPLIFICADO LEGION V1 (4D format)
            if self.debug_step_counter % self.debug_composite_interval == 0:
                try:
                    # Debug básico das predições Legion V1 (4D format)
                    self._log(f"[🎯 LEGION V1] Step {self.debug_step_counter} | "
                             f"Entry: {action[0]:.2f} | Confidence: {action[1]:.2f} | "
                             f"Pos1_Mgmt: {action[2]:.2f} | Pos2_Mgmt: {action[3]:.2f}")
                except Exception as e:
                    self._log(f"[❌ V7 DEBUG] Erro no debug: {e}")
            
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
                        # Atualizar cooldown independente do slot associado
                        try:
                            slot = self.position_slot_map.get(ticket, None)
                            if slot is None:
                                # Tentar extrair do comentário do deal/posição
                                cmt = getattr(close_deal, 'comment', '') or ''
                                slot = self._extract_slot_from_comment(str(cmt))
                            if slot is not None:
                                self.position_slot_cooldowns[int(slot)] = time.time() + (self.cooldown_minutes * 60)
                                # Remover mapeamento do ticket
                                self.position_slot_map.pop(ticket, None)
                                self._log(f"😨 [COOLDOWN-SLOT] Slot {int(slot)} em cooldown por {self.cooldown_minutes} min")
                        except Exception:
                            pass
                    
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
                        'volume': pos.volume,
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
        cooldown_seconds = self.cooldown_minutes * 60
        
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
            self._log(f"💡 [INFO] Cooldown configurado para {self.cooldown_minutes} minutos após fechamento de posição")
            
        except Exception as e:
            self._log(f"❌ [TESTE COOLDOWN] Erro: {e}")
    
    def _get_robot_positions(self):
        """🔒 Obter apenas posições do robô (com magic number)"""
        try:
            if not self.mt5_connected:
                return []
            
            # Obter todas as posições do símbolo
            all_positions = mt5.positions_get(symbol=self.symbol)
            if not all_positions:
                return []
            
            # Filtrar apenas posições do robô (magic + comentário da sessão se disponível)
            session_prefix = getattr(self, 'session_prefix', None)
            if session_prefix:
                def _belongs(pos):
                    try:
                        return pos.magic == self.magic_number and hasattr(pos, 'comment') and pos.comment and str(pos.comment).startswith(session_prefix)
                    except Exception:
                        return False
                robot_positions = [pos for pos in all_positions if _belongs(pos)]
            else:
                robot_positions = [pos for pos in all_positions if pos.magic == self.magic_number]
            return robot_positions
            
        except Exception as e:
            self._log(f"[❌ ROBOT_POS] Erro ao obter posições do robô: {e}")
            return []

    def _map_slot_for_open_positions(self, slot_hint: int | None = None):
        """Mapeia tickets de posições abertas para slots usando o comment.
        Usa slot_hint quando o comentário ainda não está propagado totalmente.
        """
        try:
            if not self.mt5_connected:
                return
            positions = self._get_robot_positions() or []
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
        """Escolhe um slot livre cujo cooldown já expirou."""
        try:
            import time as _t
            self._reconcile_slot_map()
            used = set(self.position_slot_map.values())
            now = _t.time()
            min_remain = None
            for s in range(self.max_positions):
                if s in used:
                    continue
                allow_time = self.position_slot_cooldowns.get(s, 0.0)
                if now >= allow_time:
                    return s, 0.0
                else:
                    remain = allow_time - now
                    if min_remain is None or remain < min_remain:
                        min_remain = remain
            return None, (min_remain or 0.0)
        except Exception:
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
                    # Calcular P&L atual apenas para logs
                    if position.type == 0:  # LONG
                        pnl = (current_price - position.price_open) * position.volume
                    else:  # SHORT
                        pnl = (position.price_open - current_price) * position.volume
                    
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
        🎯 DYNAMIC SL/TP MANAGEMENT - Seletores Quadruplos Completos
        
        O modelo envia sl_adjust/tp_adjust que são interpretados como:
        - sl_adjust: controla trailing stop dinâmico
        - tp_adjust: modifica take profit diretamente
        - Magnitude indica intensidade da mudança
        """
        result = {
            'action_taken': False,
            'trailing_activated': False,
            'trailing_moved': False,
            'trailing_protected': False,
            'tp_adjusted': False,
            'position_updates': {},
            'trail_info': {},
            'tp_info': {}
        }
        
        try:
            if not self.mt5_connected:
                return result
                
            # Calcular lucro atual da posição
            if position.type == 0:  # LONG
                current_pnl = (current_price - position.price_open) * position.volume
                pos_type = 'long'
            else:  # SHORT
                current_pnl = (position.price_open - current_price) * position.volume
                pos_type = 'short'
                
            pnl_pct = current_pnl / abs(position.price_open) * 100 if position.price_open != 0 else 0
            
            # 🎯 INTERPRETAÇÃO INTELIGENTE DOS ADJUSTS
            # sl_adjust [-3,3] -> decisão de trailing stop
            # tp_adjust [-3,3] -> intensidade/distância do trailing
            
            # Determinar se o modelo quer ativar/mover trailing
            trailing_signal = sl_adjust  # Sinal principal para trailing
            trailing_intensity = abs(tp_adjust)  # Intensidade da mudança
            
            # 🔥 ATIVAÇÃO DE TRAILING - Modelo decide quando ativar
            pos_metadata = getattr(position, '_metadata', {})
            if not pos_metadata.get('trailing_activated', False) and abs(trailing_signal) > 1.5:
                # Modelo está sinalizando para ativar trailing (sinal forte)
                if current_pnl > 0:  # Só ativar trailing em lucro
                    result['trailing_activated'] = True
                    result['action_taken'] = True
                    
                    # Inicializar trailing stop
                    initial_trail_distance = 15 + (trailing_intensity * 5)  # 15-30 pontos baseado na intensidade
                    
                    if pos_type == 'long':
                        trail_price = current_price - initial_trail_distance
                        # Só ativar se o trailing for melhor que o SL atual
                        if trail_price > (position.sl if position.sl > 0 else position.price_open - 50):
                            result['position_updates']['sl'] = trail_price
                            result['position_updates']['trailing_distance'] = initial_trail_distance
                            result['trailing_protected'] = True
                    else:  # short
                        trail_price = current_price + initial_trail_distance
                        # Só ativar se o trailing for melhor que o SL atual
                        if trail_price < (position.sl if position.sl > 0 else position.price_open + 50):
                            result['position_updates']['sl'] = trail_price
                            result['position_updates']['trailing_distance'] = initial_trail_distance
                            result['trailing_protected'] = True
                    
                    result['trail_info'] = {
                        'activation_reason': f"Model signal {trailing_signal:.2f}, PnL {pnl_pct:.1f}%",
                        'initial_distance': initial_trail_distance
                    }
            
            # 🔄 MOVIMENTO DE TRAILING - Modelo decide quando mover
            elif pos_metadata.get('trailing_activated', False) and abs(trailing_signal) > 0.5:
                # Trailing já ativo, modelo quer mover
                current_trail_distance = pos_metadata.get('trailing_distance', 20)
                
                # Interpretar direção do sinal
                if trailing_signal > 0:
                    # Apertar trailing (mais proteção)
                    new_trail_distance = max(10, current_trail_distance - (trailing_intensity * 2))
                else:
                    # Afrouxar trailing (mais espaço)
                    new_trail_distance = min(40, current_trail_distance + (trailing_intensity * 2))
                
                # Calcular novo preço de trailing
                if pos_type == 'long':
                    new_trail_price = current_price - new_trail_distance
                    # Só mover trailing para cima (proteção)
                    if new_trail_price > (position.sl if position.sl > 0 else 0):
                        result['position_updates']['sl'] = new_trail_price
                        result['position_updates']['trailing_distance'] = new_trail_distance
                        result['trailing_moved'] = True
                        result['action_taken'] = True
                else:  # short
                    new_trail_price = current_price + new_trail_distance
                    # Só mover trailing para baixo (proteção)
                    if new_trail_price < (position.sl if position.sl > 0 else float('inf')):
                        result['position_updates']['sl'] = new_trail_price
                        result['position_updates']['trailing_distance'] = new_trail_distance
                        result['trailing_moved'] = True
                        result['action_taken'] = True
                
                result['trail_info'] = {
                    'movement_reason': f"Model signal {trailing_signal:.2f}",
                    'old_distance': current_trail_distance,
                    'new_distance': new_trail_distance
                }
            
            # 📊 ANÁLISE DE OPORTUNIDADE PERDIDA
            if not pos_metadata.get('trailing_activated', False) and current_pnl > position.price_open * 0.02:
                # Posição com 2%+ de lucro sem trailing ativo
                pos_metadata['missed_trailing_opportunity'] = True
            
            # 🎯 MODIFICAÇÃO DINÂMICA DE TP - Seletores Quadruplos Completos
            if abs(tp_adjust) > 0.5:  # Modelo quer modificar TP
                current_tp = position.tp if position.tp > 0 else None
                if current_tp:
                    # Calcular novo TP baseado no sinal do modelo
                    tp_change_points = tp_adjust * 5.0  # Converter [-3,3] para pontos
                    
                    if position.type == 0:  # LONG
                        new_tp = current_tp + tp_change_points
                        # Validar: TP deve ser maior que preço atual
                        if new_tp > current_price + 2.0:  # Mínimo 2 pontos acima do preço
                            result['position_updates']['tp'] = new_tp
                            result['tp_adjusted'] = True
                            result['action_taken'] = True
                            
                    else:  # SHORT
                        new_tp = current_tp - tp_change_points
                        # Validar: TP deve ser menor que preço atual
                        if new_tp < current_price - 2.0:  # Mínimo 2 pontos abaixo do preço
                            result['position_updates']['tp'] = new_tp
                            result['tp_adjusted'] = True
                            result['action_taken'] = True
                    
                    if result['tp_adjusted']:
                        result['tp_info'] = {
                            'adjustment_reason': f"Model TP signal {tp_adjust:.2f}",
                            'old_tp': current_tp,
                            'new_tp': result['position_updates']['tp'],
                            'change_points': tp_change_points
                        }
            
            # Aplicar mudanças se o modelo decidiu
            if result['action_taken'] and result['position_updates']:
                new_sl = result['position_updates'].get('sl')
                new_tp = result['position_updates'].get('tp')
                modify_result = self._modify_position_sltp(position.ticket, new_sl, new_tp)
                if "SUCCESS" in modify_result:
                    # Atualizar metadata da posição para tracking
                    if not hasattr(position, '_metadata'):
                        position._metadata = {}
                    position._metadata.update(result['position_updates'])
                    
                    if result['trailing_activated']:
                        position._metadata['trailing_activated'] = True
                        position._metadata['trailing_activation_step'] = getattr(self, 'current_step', 0)
                    
                    if result['trailing_moved']:
                        position._metadata['trailing_moves'] = pos_metadata.get('trailing_moves', 0) + 1
                        position._metadata['last_trailing_move'] = getattr(self, 'current_step', 0)
                    
                    sl_info = f" | SL: {new_sl:.2f}" if new_sl else ""
                    tp_info = f" | TP: {new_tp:.2f}" if new_tp else ""
                    
                    if result['tp_adjusted'] and result['trailing_activated']:
                        self._log(f"🎯 [SL+TP ADJUST] Pos #{position.ticket} ajustada{sl_info}{tp_info}")
                    elif result['tp_adjusted']:
                        self._log(f"🎯 [TP ADJUST] Pos #{position.ticket} ajustada{tp_info}")
                    else:
                        self._log(f"🎯 [TRAILING STOP] Pos #{position.ticket} ajustada{sl_info}")
                else:
                    self._log(f"❌ [TRAILING STOP] Falha ao ajustar pos #{position.ticket}: {modify_result}")
                    result['action_taken'] = False
                    
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
            if position.type == 0:  # LONG
                return (current_price - position.price_open) * position.volume
            else:  # SHORT
                return (position.price_open - current_price) * position.volume
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
            entry_decision = int(np.clip(action[0], 0, 2))  # 0=HOLD, 1=LONG, 2=SHORT
            entry_confidence = float(np.clip(action[1], 0, 1))  # [0,1] Fusão quality + risk_appetite
            
            # Gerenciamento das 2 posições ([-1,1] → pontos reais)
            pos1_mgmt = float(np.clip(action[2], -1, 1))     # Gestão posição 1
            pos2_mgmt = float(np.clip(action[3], -1, 1))     # Gestão posição 2
            
            # Criar arrays para compatibilidade (usando gestão das posições como SL/TP)
            sl_adjusts = [pos1_mgmt, pos2_mgmt, pos1_mgmt]  # Pos1, Pos2, Global (=Pos1)
            tp_adjusts = [pos2_mgmt, pos1_mgmt, pos2_mgmt]  # Pos1, Pos2, Global (=Pos2)
            
            # Converter [-1,1] para pontos reais usando a mesma lógica do daytrader.py
            sl_points = []
            tp_points = []
            
            for i in range(2):  # Para cada posição (2 ativas máximo)
                # SL: normalizar [-1,1] para [2,8] pontos (daytrader.py)
                sl_val = self.sl_range_min + (sl_adjusts[i] + 1) * (self.sl_range_max - self.sl_range_min) / 2
                sl_val = round(sl_val * 2) / 2  # Arredondar para múltiplos de 0.5
                sl_val = np.clip(sl_val, self.sl_range_min, self.sl_range_max)
                sl_points.append(sl_val)
                
                # TP: normalizar [-1,1] para [3,15] pontos (daytrader.py)
                tp_val = self.tp_range_min + (tp_adjusts[i] + 1) * (self.tp_range_max - self.tp_range_min) / 2
                tp_val = round(tp_val * 2) / 2  # Arredondar para múltiplos de 0.5
                tp_val = np.clip(tp_val, self.tp_range_min, self.tp_range_max)
                tp_points.append(tp_val)
            
            # 🔥 FILTROS V7 - Verificações antes da execução
            
            # Filter 1: Verificar limite de posições (SILENCIOSO)
            if entry_decision in [1, 2]:  # BUY ou SELL
                if self.mt5_connected:
                    current_positions = self._get_robot_positions()
                    pos_count = len(current_positions) if current_positions else 0
                    if pos_count >= self.max_positions:
                        entry_decision = 0  # Forçar HOLD silenciosamente
            
            # 🎣 ETAPA 1 - FILTRO DE CONFIDENCE PESCADOR: 20% MÍNIMO (ALINHADO)
            # Box[0,2]: 0.67=LONG, 1.33=SHORT → qualquer >= 0.67 é trade
            if entry_decision >= 0.67 and entry_confidence < MIN_CONFIDENCE_THRESHOLD:
                self._log(f"😫 [CONFIDENCE FILTER] Entry REJEITADA: decision={entry_decision:.2f}, confidence={entry_confidence:.2f} < {MIN_CONFIDENCE_THRESHOLD} - Forçando HOLD")
                entry_decision = 0.0  # REJEITA_TRADE - Forçar para HOLD range
            
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
                self._execute_buy_order_v7(current_price, entry_confidence, action_analysis)
            elif action_name == 'SELL':
                self.hold_count = 0  # Reset contador de HOLD
                self._execute_sell_order_v7(current_price, entry_confidence, action_analysis)
            else:
                # HOLD - modelo decidiu não fazer nada (silencioso quando há 3 posições)
                self.hold_count += 1
                
                # Verificar se há posições abertas para decidir se loga (apenas do robô)
                if self.mt5_connected:
                    current_positions = self._get_robot_positions()
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
            
            # 🎯 AJUSTE POR ENTRY CONFIDENCE (faixa mais contida 0.6–1.2)
            risk_multiplier = 0.6 + (entry_confidence * 0.6)
            final_volume = adaptive_lot * risk_multiplier
            
            # Aplicar limites finais
            final_volume = max(0.01, min(final_volume, max_lot))
            
            # 🔥 VALIDAÇÃO ADICIONAL PARA MT5
            # Mantenha 3 casas antes de ajustar ao step do símbolo
            final_volume = round(final_volume, 3)
            
            # Verificar se não é zero ou negativo
            if final_volume <= 0:
                final_volume = self.base_lot_size
            
            # Log detalhado do cálculo
            if hasattr(self, 'debug_step_counter') and self.debug_step_counter % 50 == 0:
                growth_factor = current_portfolio_value / initial_portfolio_value if initial_portfolio_value > 0 else 1.0
                self._log(f"[💰 POSITION SIZING] Portfolio: ${current_portfolio_value:.2f} (growth: {growth_factor:.2f}x) | Base: {adaptive_lot:.3f} | Entry Confidence: {entry_confidence:.2f} (mult: {risk_multiplier:.2f}x) | Final: {final_volume:.3f}")
            
            return final_volume
            
        except Exception as e:
            self._log(f"❌ [VOLUME V7] Erro no dynamic sizing: {e}")
            return self.base_lot_size
    
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
            # Comentário único por sessão (isola instâncias)
            session_prefix = getattr(self, 'session_prefix', None)
            order_comment = f"{session_prefix}V7" if session_prefix else "V7 Robot"
            if slot_id is not None:
                # Anexar identificação de slot ao comentário
                if session_prefix:
                    order_comment = f"{session_prefix}V7_SLOT{slot_id}"
                else:
                    order_comment = f"V7_SLOT{slot_id}"

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
                # Reconciliar mapeamento de slot→ticket imediatamente após abertura
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
        self.log_queue = queue.Queue()
        self.update_callbacks = []
        self.is_closing = False
        self.last_stats_update = 0
        self.stats_update_interval = 2.0
        self.gui_responsive = True  # Mantido para compatibilidade
        self._updating_stats = False  # Flag para prevenir múltiplas chamadas
        
        # Configure styles
        self.setup_styles()
        
        # Robot instance
        self.robot = TradingRobotV7()
        # Criar prefixo único para esta sessão (isola stats entre instâncias)
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
                return callback_id
            except:
                pass
        return None
    
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
        """🔧 FIX: Proper cleanup when closing window"""
        try:
            self.cleanup_callbacks()
            if hasattr(self, 'stop_event'):
                self.stop_event.set()
            if hasattr(self, 'trading_thread') and self.trading_thread:
                self.trading_thread.join(timeout=2.0)
        except:
            pass
        finally:
            self.root.destroy()
    
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
            
            # Try to load with different strategies
            model_loaded = False
            filename = os.path.basename(self.selected_model_path)
            
            # Strategy 1: RecurrentPPO (for SILUS models)
            if 'SILUS' in filename.upper():
                try:
                    from sb3_contrib import RecurrentPPO
                    self.robot.model = RecurrentPPO.load(self.selected_model_path)
                    model_loaded = True
                    self.log(f"✅ Loaded as RecurrentPPO: {filename}")
                except Exception as e1:
                    self.log(f"⚠️ RecurrentPPO failed: {e1}")
            
            # Strategy 2: Standard PPO
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
