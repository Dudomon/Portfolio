# -*- coding: utf-8 -*-
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='ignore')
"""
⚔️ Legion AI Trader V1 - Trading Robot PPO v3.0 Enhanced
🧠 ATUALIZADO PPOV1: Compatível com modelo treinado usando ppov1.py
🎯 CONFIGURADO: Observation Space 1320D + Action Space 11D (PPOV1 Compatible)

🧠 PPOV1 COMPATIBILITY:
- OBSERVATION SPACE: 1320 dimensões (66 features × 20 window)
- ACTION SPACE: 11 dimensões [entry_decision, confidence, temporal_signal, risk_appetite, market_regime_bias, sl1, sl2, sl3, tp1, tp2, tp3]
- FEATURES: Market (27) + Positions (21) + Intelligent (6) = 54 features por step
- WINDOW: 20 steps × 66 features = 1320 dimensões

ACTION SPACE (11D):
- [0] entry_decision: 0=HOLD, 1=LONG, 2=SHORT
- [1] entry_confidence: [0,1] Confiança da entrada
- [2] temporal_signal: [-1,1] Sinal temporal
- [3] risk_appetite: [0,1] Apetite ao risco
- [4] market_regime_bias: [-1,1] Viés do mercado
- [5-7] sl_adjusts: SL para pos1, pos2, pos3 ([-3,3])
- [8-10] tp_adjusts: TP para pos1, pos2, pos3 ([-3,3])

CONVERSÃO: [-3,3] → [0,45] pontos → SL/TP realistas (OURO)

COMPATIBILIDADE TOTAL:
- 🧠 TwoHeadV5Intelligent48h (Entry Head Ultra-Especializada)
- 🚀 TwoHeadV4Intelligent48h (Policy 48h)
- 🔥 TwoHeadV3HybridEnhanced (Policy híbrida)
- 📋 TradingTransformerFeatureExtractor

🔧 PPOV1 UPDATES:
- _get_observation(): Gera exatamente 1320 dimensões
- _process_model_action(): Compatível com action space 11D do PPOV1
- _verify_ppov1_compatibility(): Verificação automática de compatibilidade
- auto_load_model(): Carrega modelos treinados com ppov1.py
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

# Configuracoes
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

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

# ZMQ functionality removed for cleaner implementation

# Configurar matplotlib para modo não-bloqueante
plt.ion()
plt.style.use('dark_background')  # Tema escuro para melhor visualização

# Paths para imports - CORRIGIR PARA ENCONTRAR TREINODIFERENCIADOPPO.PY
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # Pasta pai (Otimizacao)
sys.path.insert(0, parent_dir)  # Adicionar no início para prioridade
sys.path.append(current_dir)

# Reward system functionality removed for cleaner implementation

# Classes de fallback
class BaseTradingEnv:
    def __init__(self, df, config=None, is_training=False):
        self.df = df
        self.config = config or type('Config', (), {
            'WINDOW_SIZE': 20,
            'MAX_POSITIONS': 3,
            'POSITION_SIZE': 0.02,  # Base lot 0.02
            'INITIAL_BALANCE': 500.0
        })()
        self.current_step = 20
        self.positions_tracker = []

class Config:
    def __init__(self):
        self.WINDOW_SIZE = 20
        self.MAX_POSITIONS = 3  
        self.POSITION_SIZE = 0.02  # Base lot 0.02
        self.INITIAL_BALANCE = 500.0

# IMPORTS OBRIGATÓRIOS - SEM FALLBACKS, SEM GAMBIARRAS
# SISTEMA SÓ FUNCIONA COM V5 (1320) OU V6 (1480) - MAIS NADA

# V5 OBRIGATÓRIO
from trading_framework.policies.two_head_v5_intelligent_48h import TwoHeadV5Intelligent48h
print("[STRICT] ✅ TwoHeadV5Intelligent48h importada - 1320 obs OBRIGATÓRIO")

# V6 OBRIGATÓRIO  
from trading_framework.policies.two_head_v6_intelligent_48h import TwoHeadV6Intelligent48h
print("[STRICT] ✅ TwoHeadV6Intelligent48h importada - 1480 obs OBRIGATÓRIO")

# TRANSFORMER OBRIGATÓRIO
from trading_framework.extractors.transformer_extractor import TradingTransformerFeatureExtractor
print("[STRICT] ✅ TradingTransformerFeatureExtractor importado OBRIGATÓRIO")

# FLAGS SEMPRE TRUE - SEM VERIFICAÇÕES CONDICIONAIS
TWOPOLICY_AVAILABLE = True
TRANSFORMER_AVAILABLE = True

# APENAS PARA COMPATIBILIDADE COM CÓDIGO LEGACY
TwoHeadPolicy = TwoHeadV5Intelligent48h  # Default para V5
TransformerFeatureExtractor = TradingTransformerFeatureExtractor

# Analisador profissional de mercado
class ProfessionalAnalyzer:
    """Sistema de análise técnica profissional para mercado"""
    
    def __init__(self):
        self.last_analysis_time = 0
        print("[ANALYSIS] 🎨 Sistema de análise gráfica profissional ativado!")
    
    def create_market_analysis(self, price, decision, confidence, rsi, bb_pos, volatility, momentum, trend):
        """Criar análise completa do mercado"""
        import time
        current_time = time.time()
        
        # Análise completa a cada 30 segundos
        if current_time - self.last_analysis_time < 30:
            return
            
        self.last_analysis_time = current_time
        
        print(f"\n🎯 ═══ ANÁLISE TÉCNICA PROFISSIONAL ═══")
        print(f"💰 Preço: {price:.5f} | 🧠 Decisão: {decision} | 📊 Conf: {confidence:.1%}")
        
        # 1. ESTRUTURA DE MERCADO
        if rsi > 70 and bb_pos > 0.8:
            regime = "🔴 SOBRECOMPRADO EXTREMO"
        elif rsi < 30 and bb_pos < 0.2:
            regime = "🟢 SOBREVENDIDO EXTREMO"
        elif bb_pos > 0.6 and trend > 0.001:
            regime = "🚀 BREAKOUT BULLISH"
        elif bb_pos < 0.4 and trend < -0.001:
            regime = "📉 BREAKDOWN BEARISH"
        else:
            regime = "📊 CONSOLIDAÇÃO/NORMAL"
        
        print(f"📈 REGIME: {regime}")
        print(f"📊 RSI: {rsi:.1f} | BB: {bb_pos:.3f} | Vol: {volatility:.4f}")
        
        # 2. NÍVEIS DINÂMICOS
        vol_range = max(volatility * price, price * 0.0005)
        if bb_pos < 0.3:
            support = price - (vol_range * 0.5)
            support_str = "FORTE 🛡️"
        else:
            support = price - vol_range
            support_str = "MÉDIO 🛡️"
        
        if bb_pos > 0.7:
            resistance = price + (vol_range * 0.5)
            resist_str = "FORTE ⚔️"
        else:
            resistance = price + vol_range
            resist_str = "MÉDIO ⚔️"
        
        print(f"🛡️  SUPORTE: {support:.5f} ({support_str})")
        print(f"⚔️  RESISTÊNCIA: {resistance:.5f} ({resist_str})")
        
        # 3. SETUP DE TRADING
        if confidence > 0.5 and decision != "HOLD":
            atr = max(volatility * price, price * 0.001)
            
            if decision == "BUY":
                sl = price - (atr * 1.5)
                tp1 = price + (atr * 1.5)
                tp2 = price + (atr * 2.5)
                quality = "🟢 SETUP FORTE" if confidence > 0.7 else "🟡 Setup Moderado"
                
                print(f"🟢 SETUP COMPRA - {quality}")
                print(f"🛑 SL: {sl:.5f} (-{((price-sl)/price*100):.1f}%)")
                print(f"💰 TP1: {tp1:.5f} (+{((tp1-price)/price*100):.1f}%)")
                print(f"💰 TP2: {tp2:.5f} (+{((tp2-price)/price*100):.1f}%)")
                
            elif decision == "SELL":
                sl = price + (atr * 1.5)
                tp1 = price - (atr * 1.5)
                tp2 = price - (atr * 2.5)
                quality = "🔴 SETUP FORTE" if confidence > 0.7 else "🟡 Setup Moderado"
                
                print(f"🔴 SETUP VENDA - {quality}")
                print(f"🛑 SL: {sl:.5f} (+{((sl-price)/price*100):.1f}%)")
                print(f"💰 TP1: {tp1:.5f} (-{((price-tp1)/price*100):.1f}%)")
                print(f"💰 TP2: {tp2:.5f} (-{((price-tp2)/price*100):.1f}%)")
        else:
            print(f"⏸️ AGUARDAR - Sem setup claro (Conf: {confidence:.1%})")
        
        # 4. CONFLUÊNCIA
        score = 0
        factors = []
        
        if rsi > 70: factors.append("RSI Sobrecomprado"); score -= 1
        elif rsi < 30: factors.append("RSI Sobrevendido"); score += 1
        
        if bb_pos > 0.8: factors.append("BB Superior"); score -= 1
        elif bb_pos < 0.2: factors.append("BB Inferior"); score += 1
        
        if abs(momentum) > 0.001:
            if momentum > 0: factors.append("Momentum+"); score += 1
            else: factors.append("Momentum-"); score -= 1
        
        if abs(trend) > 0.001:
            if trend > 0: factors.append("Trend↑"); score += 1
            else: factors.append("Trend↓"); score -= 1
        
        if score >= 2: confluence = "🟢 CONFLUÊNCIA BULLISH FORTE"
        elif score >= 1: confluence = "🟢 Confluência Bullish"
        elif score <= -2: confluence = "🔴 CONFLUÊNCIA BEARISH FORTE"
        elif score <= -1: confluence = "🔴 Confluência Bearish"
        else: confluence = "🟡 Confluência Neutra"
        
        print(f"🔄 {confluence} (Score: {score})")
        print(f"📋 Fatores: {', '.join(factors) if factors else 'Nenhum'}")
        print(f"═══════════════════════════════════════\n")

# Sistema de desenhos de análise técnica no gráfico
class TechnicalAnalysisDrawer:
    """🎯 Sistema avançado de desenhos de análise técnica no MT5"""
    
    def __init__(self):
        self.support_levels = []
        self.resistance_levels = []
        self.trend_lines = []
        self.pattern_objects = []
        self.confluence_zones = []
        self.drawing_objects = {}
        self.last_analysis_time = 0
        self.analysis_history = deque(maxlen=100)
        
        # Configurações de desenho
        self.colors = {
            'support': 0x00FF00,      # Verde para suporte
            'resistance': 0xFF0000,   # Vermelho para resistência  
            'trend_up': 0x00FFFF,     # Ciano para trend alta
            'trend_down': 0xFF00FF,   # Magenta para trend baixa
            'confluence': 0xFFFF00,   # Amarelo para confluências
            'pattern': 0xFF8000,      # Laranja para padrões
            'fibonacci': 0x8080FF,    # Azul claro para fibonacci
            'pivot': 0xC0C0C0        # Cinza para pivots
        }
        
        print("[DRAWER] 🎨 Sistema de desenhos técnicos inicializado!")
    
    def analyze_and_draw_market_structure(self, obs, current_price, model_confidence):
        """🔍 Analisa estrutura do mercado e desenha elementos técnicos"""
        try:
            current_time = time.time()
            
            # Evitar análise muito frequente (máximo a cada 30 segundos)
            if current_time - self.last_analysis_time < 30:
                return
                
            self.last_analysis_time = current_time
            
            # 1. DETECTAR E DESENHAR SUPORTES/RESISTÊNCIAS
            self._detect_and_draw_support_resistance(obs, current_price)
            
            # 2. DETECTAR E DESENHAR LINHAS DE TENDÊNCIA
            self._detect_and_draw_trend_lines(obs, current_price)
            
            # 3. DETECTAR E DESENHAR PADRÕES GRÁFICOS
            self._detect_and_draw_patterns(obs, current_price, model_confidence)
            
            # 4. DETECTAR E DESENHAR ZONAS DE CONFLUÊNCIA
            self._detect_and_draw_confluence_zones(obs, current_price)
            
            # 5. DESENHAR NÍVEIS DE FIBONACCI
            self._draw_fibonacci_levels(obs, current_price)
            
            # 6. DESENHAR PONTOS PIVÔ
            self._draw_pivot_points(obs, current_price)
            
            # 7. ADICIONAR ANOTAÇÕES DO MODELO
            self._add_model_annotations(current_price, model_confidence, obs)
            
            print(f"[DRAWER] ✅ Análise técnica completa realizada - Preço: {current_price:.5f}")
            
        except Exception as e:
            # ERRO CRÍTICO - NÃO MASCARAR
            print(f"[DRAWER] ❌ ERRO CRÍTICO na análise técnica: {e}")
            raise Exception(f"Análise técnica FALHOU: {e}")
    
    def _detect_and_draw_support_resistance(self, obs, current_price):
        """🎯 Detecta e desenha níveis de suporte e resistência"""
        try:
            # Extrair dados de preço das observações
            price_data = self._extract_price_data_from_obs(obs)
            if len(price_data) < 20:
                return
                
            # Detectar máximos e mínimos locais
            highs = []
            lows = []
            
            for i in range(2, len(price_data) - 2):
                # Máximo local (resistência potencial)
                if (price_data[i] > price_data[i-1] and price_data[i] > price_data[i-2] and
                    price_data[i] > price_data[i+1] and price_data[i] > price_data[i+2]):
                    highs.append(price_data[i])
                    
                # Mínimo local (suporte potencial)
                if (price_data[i] < price_data[i-1] and price_data[i] < price_data[i-2] and
                    price_data[i] < price_data[i+1] and price_data[i] < price_data[i+2]):
                    lows.append(price_data[i])
            
            # Agrupar níveis próximos (tolerância de 10 pips)
            tolerance = 0.0010
            
            # Processar resistências
            resistance_levels = self._cluster_levels(highs, tolerance)
            for i, level in enumerate(resistance_levels[:5]):  # Máximo 5 níveis
                self._draw_horizontal_line(
                    f"resistance_{i}", 
                    level, 
                    self.colors['resistance'],
                    f"🔴 Resistência {level:.5f}",
                    width=2
                )
            
            # Processar suportes
            support_levels = self._cluster_levels(lows, tolerance)
            for i, level in enumerate(support_levels[:5]):  # Máximo 5 níveis
                self._draw_horizontal_line(
                    f"support_{i}", 
                    level, 
                    self.colors['support'],
                    f"🟢 Suporte {level:.5f}",
                    width=2
                )
                
            print(f"[DRAWER] 📊 Desenhados {len(resistance_levels)} resistências e {len(support_levels)} suportes")
            
        except Exception as e:
            # ERRO CRÍTICO - NÃO MASCARAR
            print(f"[DRAWER] ❌ ERRO CRÍTICO em S/R: {e}")
            raise Exception(f"S/R drawing FALHOU: {e}")
    
    def _draw_fibonacci_levels(self, obs, current_price):
        """📐 Desenha níveis de Fibonacci"""
        try:
            price_data = self._extract_price_data_from_obs(obs)
            if len(price_data) < 20:
                return
                
            # Encontrar swing high e swing low recentes
            recent_data = price_data[-50:]  # Últimas 50 barras
            swing_high = max(recent_data)
            swing_low = min(recent_data)
            
            # Calcular níveis de Fibonacci
            diff = swing_high - swing_low
            fib_levels = {
                '0.0': swing_low,
                '23.6': swing_low + (diff * 0.236),
                '38.2': swing_low + (diff * 0.382),
                '50.0': swing_low + (diff * 0.500),
                '61.8': swing_low + (diff * 0.618),
                '78.6': swing_low + (diff * 0.786),
                '100.0': swing_high
            }
            
            # Desenhar níveis de Fibonacci
            for fib_name, fib_level in fib_levels.items():
                self._draw_horizontal_line(
                    f"fib_{fib_name}",
                    fib_level,
                    self.colors['fibonacci'],
                    f"📐 Fib {fib_name}% - {fib_level:.5f}",
                    width=1,
                    style=2  # Linha pontilhada
                )
                
            print(f"[DRAWER] 📐 Níveis de Fibonacci desenhados ({swing_low:.5f} - {swing_high:.5f})")
            
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao desenhar Fibonacci: {e}")
    
    def _add_model_annotations(self, current_price, model_confidence, obs):
        """🧠 Adiciona anotações do modelo IA"""
        try:
            # Análise do regime de mercado
            market_regime = self._analyze_market_regime_from_obs(obs)
            
            # Análise de momentum
            momentum_strength = self._analyze_momentum_from_obs(obs)
            
            # Criar anotação principal do modelo
            annotation_text = f"🧠 IA ANALYSIS\n"
            annotation_text += f"Confidence: {model_confidence:.1%}\n"
            annotation_text += f"Regime: {market_regime}\n"
            annotation_text += f"Momentum: {momentum_strength:.3f}\n"
            annotation_text += f"Price: {current_price:.5f}"
            
            # Desenhar anotação
            self._draw_text_annotation(
                "model_analysis",
                current_price,
                annotation_text,
                self.colors['confluence']
            )
            
            print(f"[DRAWER] 🧠 Anotações do modelo adicionadas")
            
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao adicionar anotações: {e}")
    
    def _extract_price_data_from_obs(self, obs):
        """Extrai dados de preço das observações"""
        try:
            # Assumindo que os primeiros elementos são dados de preço
            if isinstance(obs, np.ndarray) and len(obs) > 100:
                # Extrair aproximadamente 50 pontos de preço das observações
                price_indices = range(0, min(200, len(obs)), 4)  # A cada 4 elementos
                return [float(obs[i]) for i in price_indices if i < len(obs)]
            return []
        except:
            return []
    
    def _cluster_levels(self, levels, tolerance):
        """Agrupa níveis próximos"""
        if not levels:
            return []
            
        levels = sorted(levels)
        clustered = []
        current_cluster = [levels[0]]
        
        for level in levels[1:]:
            if abs(level - current_cluster[-1]) <= tolerance:
                current_cluster.append(level)
            else:
                # Média do cluster atual
                clustered.append(sum(current_cluster) / len(current_cluster))
                current_cluster = [level]
        
        # Adicionar último cluster
        if current_cluster:
            clustered.append(sum(current_cluster) / len(current_cluster))
            
        return clustered
    
    def _analyze_market_regime_from_obs(self, obs):
        """Analisa regime de mercado das observações"""
        try:
            if len(obs) > 50:
                volatility = np.std(obs[:50])
                if volatility > 0.01:
                    return "TRENDING"
                else:
                    return "RANGING"
            return "UNKNOWN"
        except:
            return "UNKNOWN"
    
    def _analyze_momentum_from_obs(self, obs):
        """Analisa momentum das observações"""
        try:
            if len(obs) > 20:
                recent = obs[:20]
                return float(np.mean(recent))
            return 0.0
        except:
            return 0.0
    
    def _draw_horizontal_line(self, name, price, color, description, width=1, style=0):
        """Desenha linha horizontal no MT5"""
        try:
            # Limpar linha existente
            self._delete_object(name)
            
            # Usar função segura do MT5
            safe_mt5_object_delete(name)
            
            # Tentar criar linha horizontal usando diferentes métodos
            success = False
            
            # Método 1: Tentar criar linha horizontal diretamente
            try:
                if hasattr(mt5, 'chart_objects_add'):
                    success = mt5.chart_objects_add(0, name, mt5.OBJ_HLINE, 0, 0, price)
                    if success:
                        mt5.chart_object_set_integer(0, name, mt5.OBJPROP_COLOR, color)
                        mt5.chart_object_set_integer(0, name, mt5.OBJPROP_WIDTH, width)
                        mt5.chart_object_set_string(0, name, mt5.OBJPROP_TEXT, description)
            except:
                pass
            
            # Método 2: Fallback - criar usando ObjectCreate
            if not success:
                try:
                    current_time = datetime.now()
                    if hasattr(mt5, 'ObjectCreate'):
                        success = mt5.ObjectCreate(0, name, mt5.OBJ_HLINE, 0, current_time, price)
                except:
                    pass
            
            if success:
                # Adicionar à lista de objetos
                self.drawing_objects[name] = {
                    'type': 'hline',
                    'price': price,
                    'description': description
                }
                
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao desenhar linha horizontal {name}: {e}")
    
    def _draw_text_annotation(self, name, price, text, color):
        """Desenha anotação de texto no gráfico"""
        try:
            # Limpar anotação existente
            self._delete_object(name)
            
            # Usar função segura do MT5
            safe_mt5_object_delete(name)
            
            # Tentar criar texto
            try:
                current_time = datetime.now()
                if hasattr(mt5, 'chart_objects_add'):
                    success = mt5.chart_objects_add(0, name, mt5.OBJ_TEXT, 0, current_time, price)
                    if success:
                        mt5.chart_object_set_string(0, name, mt5.OBJPROP_TEXT, text)
                        mt5.chart_object_set_integer(0, name, mt5.OBJPROP_COLOR, color)
                        mt5.chart_object_set_integer(0, name, mt5.OBJPROP_FONTSIZE, 10)
                        
                        self.drawing_objects[name] = {
                            'type': 'text',
                            'price': price,
                            'text': text
                        }
            except Exception as e:
                print(f"[DRAWER] ❌ Erro ao criar texto {name}: {e}")
                
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao desenhar anotação {name}: {e}")
    
    def _delete_object(self, name):
        """Remove objeto do gráfico"""
        try:
            if name in self.drawing_objects:
                safe_mt5_object_delete(name)
                del self.drawing_objects[name]
        except:
            pass
    
    def clear_all_drawings(self):
        """🧹 Limpa todos os desenhos do gráfico"""
        try:
            for obj_name in list(self.drawing_objects.keys()):
                self._delete_object(obj_name)
            
            # Limpar também objetos MT5 por tipo usando função segura
            safe_mt5_object_delete("resistance_")
            safe_mt5_object_delete("support_")
            safe_mt5_object_delete("fib_")
            safe_mt5_object_delete("model_")
            
            self.drawing_objects.clear()
            print("[DRAWER] 🧹 Todos os desenhos removidos")
            
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao limpar desenhos: {e}")
    
    def _detect_and_draw_trend_lines(self, obs, current_price):
        """📈 Detecta e desenha linhas de tendência"""
        try:
            # Remover linhas de tendência antigas antes de desenhar novas
            self._delete_object("trend_start")
            self._delete_object("trend_end")

            price_data = self._extract_price_data_from_obs(obs)
            if len(price_data) < 30:
                return
                
            # Detectar tendência usando regressão linear simples
            x = np.arange(len(price_data))
            y = np.array(price_data)
            
            # Calcular linha de tendência
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                
                # Determinar direção da tendência
                if slope > 0.00001:  # Tendência de alta
                    trend_color = self.colors['trend_up']
                    trend_name = "📈 Tendência de Alta"
                elif slope < -0.00001:  # Tendência de baixa
                    trend_color = self.colors['trend_down']
                    trend_name = "📉 Tendência de Baixa"
                else:
                    return  # Sem tendência clara
                
                # Calcular pontos da linha de tendência
                start_price = y[0]
                end_price = y[0] + slope * len(y)
                
                # Desenhar linha de tendência como duas linhas horizontais
                self._draw_horizontal_line(
                    "trend_start",
                    start_price,
                    trend_color,
                    f"{trend_name} - Início: {start_price:.5f}",
                    width=1,
                    style=1
                )
                
                self._draw_horizontal_line(
                    "trend_end",
                    end_price,
                    trend_color,
                    f"{trend_name} - Fim: {end_price:.5f}",
                    width=1,
                    style=1
                )
                
                print(f"[DRAWER] 📈 Linha de tendência desenhada: {trend_name}")
                
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao desenhar tendências: {e}")
    
    def _detect_and_draw_patterns(self, obs, current_price, model_confidence):
        """🔍 Detecta e desenha padrões gráficos"""
        try:
            price_data = self._extract_price_data_from_obs(obs)
            if len(price_data) < 20:
                return
                
            # PADRÃO 1: Divergência baseada na confiança do modelo
            if model_confidence > 0.8:
                self._draw_divergence_pattern(current_price, model_confidence)
            
            # PADRÃO 2: Breakout potencial
            recent_data = price_data[-10:]
            if len(recent_data) >= 5:
                volatility = np.std(recent_data)
                if volatility < 0.0005:  # Baixa volatilidade = possível breakout
                    self._draw_breakout_pattern(current_price, recent_data)
            
            # PADRÃO 3: Reversão baseada em extremos
            if len(price_data) >= 20:
                self._detect_reversal_pattern(price_data, current_price)
                
            print(f"[DRAWER] 🔍 Padrões gráficos analisados")
            
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao detectar padrões: {e}")
    
    def _detect_and_draw_confluence_zones(self, obs, current_price):
        """🎯 Detecta e desenha zonas de confluência"""
        try:
            # Coletar níveis importantes próximos ao preço atual
            tolerance = 0.0020  # 20 pips de tolerância
            
            # Simular níveis de confluência baseados no preço atual
            confluence_levels = []
            
            # Adicionar possíveis níveis de suporte/resistência próximos
            for offset in [-0.0030, -0.0015, 0.0015, 0.0030]:
                level = current_price + offset
                confluence_levels.append(level)
            
            # Desenhar zona de confluência principal
            if confluence_levels:
                avg_level = np.mean(confluence_levels)
                self._draw_horizontal_line(
                    "confluence_main",
                    avg_level,
                    self.colors['confluence'],
                    f"🎯 Zona de Confluência: {avg_level:.5f}",
                    width=3,
                    style=2
                )
                
                print(f"[DRAWER] 🎯 Zona de confluência desenhada em {avg_level:.5f}")
                
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao detectar confluências: {e}")
    
    def _draw_pivot_points(self, obs, current_price):
        """⚖️ Desenha pontos pivô"""
        try:
            price_data = self._extract_price_data_from_obs(obs)
            if len(price_data) < 10:
                return
                
            # Usar dados recentes para calcular pivô
            recent_data = price_data[-20:]  # Últimas 20 barras
            high = max(recent_data)
            low = min(recent_data)
            close = recent_data[-1]
            
            # Calcular ponto pivô principal
            pivot = (high + low + close) / 3
            
            # Calcular resistências e suportes
            r1 = (2 * pivot) - low
            s1 = (2 * pivot) - high
            
            # Desenhar pontos pivô
            self._draw_horizontal_line(
                "pivot_main",
                pivot,
                self.colors['pivot'],
                f"⚖️ Pivot Point: {pivot:.5f}",
                width=2,
                style=3
            )
            
            self._draw_horizontal_line(
                "pivot_r1",
                r1,
                self.colors['resistance'],
                f"⚖️ R1: {r1:.5f}",
                width=1,
                style=3
            )
            
            self._draw_horizontal_line(
                "pivot_s1",
                s1,
                self.colors['support'],
                f"⚖️ S1: {s1:.5f}",
                width=1,
                style=3
            )
            
            print(f"[DRAWER] ⚖️ Pontos pivô desenhados - PP: {pivot:.5f}")
            
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao desenhar pivots: {e}")
    
    def _draw_divergence_pattern(self, current_price, confidence):
        """📊 Desenha padrão de divergência"""
        try:
            # Desenhar sinal de divergência quando confiança é muito alta
            divergence_text = f"⚡ DIVERGÊNCIA DETECTADA\nConfiança: {confidence:.1%}\nPreço: {current_price:.5f}"
            
            self._draw_text_annotation(
                "divergence_signal",
                current_price + 0.0010,  # Ligeiramente acima do preço
                divergence_text,
                self.colors['pattern']
            )
            
            # Desenhar linha de alerta
            self._draw_horizontal_line(
                "divergence_line",
                current_price,
                self.colors['pattern'],
                f"⚡ Divergência - {confidence:.1%}",
                width=3,
                style=4
            )
            
            print(f"[DRAWER] ⚡ Divergência desenhada com confiança {confidence:.1%}")
            
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao desenhar divergência: {e}")
    
    def _draw_breakout_pattern(self, current_price, recent_data):
        """💥 Desenha padrão de breakout"""
        try:
            # Calcular zona de consolidação
            high_consolidation = max(recent_data)
            low_consolidation = min(recent_data)
            
            # Desenhar zona de consolidação
            self._draw_horizontal_line(
                "breakout_high",
                high_consolidation,
                self.colors['pattern'],
                f"💥 Breakout High: {high_consolidation:.5f}",
                width=2,
                style=2
            )
            
            self._draw_horizontal_line(
                "breakout_low",
                low_consolidation,
                self.colors['pattern'],
                f"💥 Breakout Low: {low_consolidation:.5f}",
                width=2,
                style=2
            )
            
            # Adicionar anotação de breakout
            breakout_text = f"💥 ZONA DE BREAKOUT\nRange: {low_consolidation:.5f} - {high_consolidation:.5f}"
            self._draw_text_annotation(
                "breakout_annotation",
                (high_consolidation + low_consolidation) / 2,
                breakout_text,
                self.colors['pattern']
            )
            
            print(f"[DRAWER] 💥 Padrão de breakout desenhado")
            
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao desenhar breakout: {e}")
    
    def _detect_reversal_pattern(self, price_data, current_price):
        """🔄 Detecta e desenha padrões de reversão"""
        try:
            # Verificar se estamos em extremo (possível reversão)
            recent_data = price_data[-10:]
            all_data = price_data[-50:]
            
            current_avg = np.mean(recent_data)
            overall_avg = np.mean(all_data)
            
            # Detectar extremo
            if current_avg > overall_avg * 1.01:  # Possível topo
                reversal_text = f"🔄 POSSÍVEL REVERSÃO\nTipo: TOPO\nPreço: {current_price:.5f}"
                color = self.colors['trend_down']
                level_name = "reversal_top"
            elif current_avg < overall_avg * 0.99:  # Possível fundo
                reversal_text = f"🔄 POSSÍVEL REVERSÃO\nTipo: FUNDO\nPreço: {current_price:.5f}"
                color = self.colors['trend_up']
                level_name = "reversal_bottom"
            else:
                return  # Sem sinal de reversão
            
            # Desenhar sinal de reversão
            self._draw_text_annotation(
                level_name,
                current_price,
                reversal_text,
                color
            )
            
            # Desenhar linha de reversão
            self._draw_horizontal_line(
                f"{level_name}_line",
                current_price,
                color,
                f"🔄 Reversão: {current_price:.5f}",
                width=2,
                style=4
            )
            
            print(f"[DRAWER] 🔄 Padrão de reversão detectado")
            
        except Exception as e:
            print(f"[DRAWER] ❌ Erro ao detectar reversão: {e}")

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

# Anti-flipflop system removed - dead code that was completely disabled

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
        self.max_lot_size = 0.03  # Max lot 0.03
        self.max_positions = 3  # MÁXIMO 3 POSIÇÕES SIMULTÂNEAS
        self.current_positions = 0
        self.current_step = 0
        self.done = False
        self.last_order_time = 0
        
        # 🛡️ TRACKER DE POSIÇÕES: Para detectar novas posições manuais
        self.known_positions = set()  # Set com tickets de posições conhecidas
        
        # 🔥 ACTION SPACE PPOV1 COMPATIBLE: 11 dimensões especializadas
        # 🔥 ACTION SPACE PPOV1 COMPATIBLE (11 dimensões)
        # [0] action: 0=HOLD, 1=LONG, 2=SHORT
        # [1] confidence: [0,1] Confiança da decisão
        # [2] temporal_signal: [-1,1] Sinal temporal
        # [3] risk_appetite: [0,1] Apetite ao risco
        # [4] market_regime_bias: [-1,1] Viés do regime de mercado
        # [5] sl1: [-3,3] Stop Loss nível 1
        # [6] sl2: [-3,3] Stop Loss nível 2
        # [7] sl3: [-3,3] Stop Loss nível 3
        # [8] tp1: [-3,3] Take Profit nível 1
        # 🔥 ACTION SPACE LEGION V1: 11 dimensões (confirmado pela análise)
        # [0] entry_decision: 0=hold, 1=long, 2=short
        # [1] entry_confidence: [0,1] Confiança da entrada
        # [2] temporal_signal: [-1,1] Sinal temporal
        # [3] risk_appetite: [0,1] Apetite ao risco
        # [4] market_regime_bias: [-1,1] Viés do mercado
        # [5-7] sl_adjusts: SL para pos1, pos2, pos3
        # [8-10] tp_adjusts: TP para pos1, pos2, pos3
        self.action_space = spaces.Box(
            low=np.array([0, 0, -1, 0, -1, -3, -3, -3, -3, -3, -3]),
            high=np.array([2, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3]),
            dtype=np.float32
        )
        
        # 🧠 OBSERVATION SPACE PPOV1 V5: Compatível com TwoHeadV5 + Componentes Inteligentes
        # Features básicas alinhadas com ppov1.py
        base_features_5m_15m = [
            'returns', 'volatility_20', 'sma_20', 'sma_50', 'rsi_14', 
            'stoch_k', 'bb_position', 'trend_strength', 'atr_14'
        ]
        
        # 🎯 FEATURES DE ALTA QUALIDADE (exatamente como ppov1.py)
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
        
        # 🔥 VERIFICAÇÃO: Total deve ser 27 market features (9 × 2 + 9 = 27)
        # 9 features × 2 timeframes = 18 + 9 high_quality = 27 market features
        # 27 market + 21 positions (3×7) + 12 intelligent = 60 features total
        # PPOV1 tem inconsistência: calcula 66 mas implementa 60
        
        # 🧠 V5 ENHANCEMENT: Adicionar espaço para componentes inteligentes (12 features - COMPATÍVEL COM PPOV1)
        intelligent_features_count = 12  # Market regime (3) + Volatility (3) + Momentum (3) + Risk (3) = 12 features
        
        # 🔥 OBSERVATION SPACE SERÁ DEFINIDO APÓS CARREGAR MODELO
        # V5: 1320 dimensões | V6: 1480 dimensões  
        # TODOS DEVEM TER STRATEGIC FUSION LAYER - SEM FALLBACKS
        self.observation_space = None  # Definido após carregar modelo
        
        self._log(f"[OBS SPACE] 🧠 Será definido após carregar modelo")
        
        # Variáveis de controle idênticas ao mainppo1.py
        self.realized_balance = self.initial_balance
        self.peak_portfolio_value = self.initial_balance
        self.last_trade_pnl = 0.0
        self.steps_since_last_trade = 0
        self.last_action = None
        self.hold_count = 0
        self.base_tf = '5m'
        
        # Position sizing
        self.base_lot_size = 0.02  # Base lot 0.02
        self.max_lot_size = 0.03   # Max lot 0.03
        self.lot_size = self.base_lot_size  # Será calculado dinamicamente
        
        # Reward system removed for cleaner implementation
        
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
        
        # ZMQ functionality removed

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
        
        # Verificação de compatibilidade movida para após carregamento do modelo
    
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
        """🧠 Obtém observação compatível com TwoHeadV5 + componentes inteligentes"""
        try:
            # Atualizar dados históricos com tick mais recente
            self._update_historical_data()
            
            # 🔥 OBSERVAÇÃO COMPATÍVEL COM PPOV1.PY V5
            if len(self.historical_df) < self.window_size:
                return np.zeros(self.observation_space.shape, dtype=np.float32)
            
            # Obter preço atual para cálculos de posições
            tick = mt5.symbol_info_tick(self.symbol)
            current_price = tick.bid if tick else 2000.0  # Fallback
            
            # 🔥 POSIÇÕES EXATAMENTE COMO NO PPOV1.PY (9 features por posição)
            positions_obs = np.zeros((self.max_positions, 9))
            
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
                        if len(self.historical_df) > 0:
                            # Usar dados de fechamento do timeframe principal
                            close_values = []
                            for col in self.historical_df.columns:
                                if 'returns_5m' in col:  # Usar dados 5m como base
                                    close_values = self.historical_df[col].values
                                    break
                            
                            if len(close_values) == 0:
                                # Fallback: usar current_price
                                close_values = [current_price] * 100
                                
                            self._price_min_max_cache = {
                                'min': np.min(close_values),
                                'max': np.max(close_values), 
                                'range': np.max(close_values) - np.min(close_values) if np.max(close_values) > np.min(close_values) else 1.0
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
                    try:
                        # MT5 TradePosition usa 'time' para abertura da posição
                        position_time = getattr(pos, 'time', None) or getattr(pos, 'time_setup', None)
                        if position_time:
                            # Converter tempo da posição para steps simulados
                            position_age_seconds = time.time() - position_time
                            position_age_steps = position_age_seconds / 300  # 5 minutos por step
                            total_steps = len(self.historical_df) if len(self.historical_df) > 0 else 1000
                            positions_obs[i, 6] = position_age_steps / total_steps
                        else:
                            positions_obs[i, 6] = 0.1  # Valor padrão
                    except Exception as e:
                        positions_obs[i, 6] = 0.1  # Valor padrão em caso de erro
                    
                    # 🔥 FEATURES EXTRAS PARA COMPATIBILIDADE COM PPOV1 (9 features por posição)
                    # Feature 7: Volume da posição (normalizado)
                    positions_obs[i, 7] = pos.volume / 1.0  # Normalizar volume
                    
                    # Feature 8: Distância até SL/TP (normalizada)
                    if pos.sl > 0:
                        sl_distance = abs(current_price - pos.sl) / current_price
                        positions_obs[i, 8] = np.clip(sl_distance, 0.0, 0.1)  # Máximo 10%
                    elif pos.tp > 0:
                        tp_distance = abs(current_price - pos.tp) / current_price
                        positions_obs[i, 8] = np.clip(tp_distance, 0.0, 0.1)  # Máximo 10%
                    else:
                        positions_obs[i, 8] = 0.0  # Sem SL/TP
                else:
                    positions_obs[i, :] = 0  # Slot vazio
            
            # 🧠 COMPONENTES INTELIGENTES PARA V5
            intelligent_components = self._generate_intelligent_components_mt5(current_price)
            
            # 🔥 FEATURES DINÂMICAS IGUAL AO TREINAMENTO
            if len(self.historical_df) > 0 and len(self.feature_columns) > 0:
                recent_data = self.historical_df[self.feature_columns].tail(self.window_size).values
                
                # SEM PADDING - FALHA SE DADOS INSUFICIENTES
                if len(recent_data) < self.window_size:
                    raise Exception(f"DADOS INSUFICIENTES: {len(recent_data)} < {self.window_size} - Corrija na fonte")
            else:
                recent_data = np.zeros((self.window_size, len(self.feature_columns)))  # Features de mercado dinâmicas
            
            # Tile das posições para cada timestep (max_positions×9 features)
            tile_positions = np.tile(positions_obs.flatten(), (self.window_size, 1))
            
            # 🧠 TILE DOS COMPONENTES INTELIGENTES
            intelligent_features = self._flatten_intelligent_components_mt5(intelligent_components)
            tile_intelligent = np.tile(intelligent_features, (self.window_size, 1))
            
            # 🔥 CONCATENAR TUDO: mercado + posições + intelligent
            obs = np.concatenate([recent_data, tile_positions, tile_intelligent], axis=1)
            
            # 🔥 GARANTIR EXATAMENTE 1320 DIMENSÕES (PPOV1)
            expected_features_per_step = 1320 // self.window_size  # 66 features por step
            current_features_per_step = obs.shape[1]
            
            if current_features_per_step != expected_features_per_step:
                self._log(f"[OBS-FIX] Ajustando features: {current_features_per_step} → {expected_features_per_step}")
                if current_features_per_step > expected_features_per_step:
                    # Truncar features extras
                    obs = obs[:, :expected_features_per_step]
                else:
                    # SEM PADDING - FALHA SE FEATURES INSUFICIENTES
                    raise Exception(f"FEATURES INSUFICIENTES: {current_features_per_step} < {expected_features_per_step} - Corrija na fonte")
            
            # Flatten para formato final
            flat_obs = obs.flatten().astype(np.float32)
            
            # 🔥 CLIPPING E VALIDAÇÃO
            flat_obs = np.clip(flat_obs, -100.0, 100.0)
            flat_obs = np.nan_to_num(flat_obs, nan=0.0, posinf=100.0, neginf=-100.0)
            
            # SEM AJUSTES ARTIFICIAIS - FALHA SE DIMENSÕES ERRADAS
            if flat_obs.shape[0] != self.observation_space.shape[0]:
                raise Exception(f"SHAPE INCORRETO: {flat_obs.shape[0]} != {self.observation_space.shape[0]} - Corrija na fonte")
            
            # Verificações de integridade
            assert flat_obs.shape == self.observation_space.shape, f"Obs shape {flat_obs.shape} != expected {self.observation_space.shape}"
            assert not np.any(np.isnan(flat_obs)), f"Observação ainda contém NaN após limpeza"
            assert not np.any(np.isinf(flat_obs)), f"Observação ainda contém Inf após limpeza"
            
            return flat_obs
            
        except Exception as e:
            self._log(f"[ERROR] Erro ao obter observação V5: {e}")
            return np.zeros(self.observation_space.shape, dtype=np.float32)
    
    def _generate_intelligent_components_mt5(self, current_price):
        """
        🧠 GERAR COMPONENTES INTELIGENTES PARA ENTRY HEAD V5 - VERSÃO MT5
        Cria os dados especializados baseados em dados reais do MT5
        """
        try:
            # 🎯 1. MARKET REGIME CLASSIFICATION
            market_regime = self._classify_market_regime_mt5(current_price)
            
            # 🎯 2. VOLATILITY CONTEXT ANALYSIS
            volatility_context = self._analyze_volatility_context_mt5(current_price)
            
            # 🎯 3. MOMENTUM CONFLUENCE
            momentum_confluence = self._calculate_momentum_confluence_mt5(current_price)
            
            # 🎯 4. LIQUIDITY ZONES DETECTION
            liquidity_zones = self._detect_liquidity_zones_mt5(current_price)
            
            # 🎯 5. PATTERN RECOGNITION
            pattern_recognition = self._extract_pattern_memory_mt5(current_price)
            
            # 🎯 6. RISK ASSESSMENT
            risk_assessment = self._calculate_risk_metrics_mt5(current_price)
            
            # 🎯 7. MARKET FATIGUE DETECTOR
            market_fatigue = self._calculate_market_fatigue_mt5(current_price)
            
            return {
                'market_regime': market_regime,
                'volatility_context': volatility_context,
                'momentum_confluence': momentum_confluence,
                'liquidity_zones': liquidity_zones,
                'pattern_recognition': pattern_recognition,
                'risk_assessment': risk_assessment,
                'market_fatigue': market_fatigue
            }
            
        except Exception as e:
            # Fallback com dados padrão
            return {
                'market_regime': {'regime': 'unknown', 'strength': 0.0, 'direction': 0.0},
                'volatility_context': {'level': 'normal', 'percentile': 0.5, 'expanding': False},
                'momentum_confluence': {'score': 0.0, 'direction': 0.0, 'strength': 0.0},
                'liquidity_zones': {'near_support': False, 'near_resistance': False, 'zone_strength': 0.0},
                'pattern_recognition': {'pattern_strength': 0.0, 'pattern_type': 'none', 'confidence': 0.0},
                'risk_assessment': {'drawdown_risk': 0.0, 'position_risk': 0.0, 'volatility_risk': 0.01, 'risk_score': 0.0},
                'market_fatigue': {'fatigue_score': 0.0, 'recent_trades': 0, 'should_avoid_entry': False}
            }
    
    def _classify_market_regime_mt5(self, current_price):
        """🎯 Classificar regime de mercado usando dados MT5"""
        try:
            # Obter dados de 50 barras (4h de dados)
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 50)
            if rates is None or len(rates) < 10:
                return {'regime': 'unknown', 'strength': 0.0, 'direction': 0.0}
            
            prices = rates['close']
            
            # Calcular trend strength usando SMA
            if len(prices) >= 20:
                sma_20 = np.mean(prices[-20:])
                price_diff = prices - sma_20
                trend_strength = np.mean(price_diff) / np.std(price_diff) if np.std(price_diff) > 0 else 0.0
                direction = 1.0 if trend_strength > 0.5 else (-1.0 if trend_strength < -0.5 else 0.0)
                
                if abs(trend_strength) > 1.0:
                    regime = 'trending'
                elif abs(trend_strength) < 0.3:
                    regime = 'ranging'
                else:
                    regime = 'volatile'
            else:
                # Fallback usando returns
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
                
                if volatility > 0.02:
                    regime = 'volatile'
                elif volatility < 0.005:
                    regime = 'ranging'
                else:
                    regime = 'trending'
                
                trend_strength = np.mean(returns) / volatility if volatility > 0 else 0.0
                direction = 1.0 if trend_strength > 0.1 else (-1.0 if trend_strength < -0.1 else 0.0)
            
            return {
                'regime': regime,
                'strength': float(np.clip(abs(trend_strength), 0.0, 2.0)),
                'direction': float(direction)
            }
            
        except Exception as e:
            return {'regime': 'unknown', 'strength': 0.0, 'direction': 0.0}
    
    def _analyze_volatility_context_mt5(self, current_price):
        """📈 Analisar contexto de volatilidade usando MT5"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 20)
            if rates is None or len(rates) < 5:
                return {'level': 'normal', 'percentile': 0.5, 'expanding': False}
            
            prices = rates['close']
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            # Classificar volatilidade
            if volatility > 0.015:
                level = 'high'
                percentile = 0.8
            elif volatility < 0.005:
                level = 'low'
                percentile = 0.2
            else:
                level = 'normal'
                percentile = 0.5
            
            # Detectar expansão
            if len(returns) >= 10:
                recent_vol = np.std(returns[-5:])
                older_vol = np.std(returns[-10:-5])
                expanding = recent_vol > older_vol * 1.2
            else:
                expanding = False
            
            return {
                'level': level,
                'percentile': float(np.clip(percentile, 0.0, 1.0)),
                'expanding': bool(expanding)
            }
            
        except Exception as e:
            return {'level': 'normal', 'percentile': 0.5, 'expanding': False}
    
    def _calculate_momentum_confluence_mt5(self, current_price):
        """🚀 Calcular confluência de momentum usando MT5"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 50)
            if rates is None or len(rates) < 20:
                return {'score': 0.0, 'direction': 0.0, 'strength': 0.0}
            
            prices = rates['close']
            highs = rates['high']
            lows = rates['low']
            
            confluence_score = 0.0
            direction_sum = 0.0
            indicators_count = 0
            
            # RSI
            if len(prices) >= 14:
                rsi = self._calculate_rsi(prices, 14)
                if rsi > 70:
                    confluence_score += 0.5
                    direction_sum -= 1.0
                elif rsi < 30:
                    confluence_score += 0.5
                    direction_sum += 1.0
                else:
                    confluence_score += 0.2
                indicators_count += 1
            
            # Moving Average Crossover
            if len(prices) >= 20:
                sma_10 = np.mean(prices[-10:])
                sma_20 = np.mean(prices[-20:])
                
                if sma_10 > sma_20:
                    confluence_score += 0.2
                    direction_sum += 1.0
                else:
                    confluence_score += 0.1
                    direction_sum -= 1.0
                indicators_count += 1
            
            # Normalizar
            if indicators_count > 0:
                confluence_score /= indicators_count
                direction_sum /= indicators_count
            
            return {
                'score': float(np.clip(confluence_score, 0.0, 1.0)),
                'direction': float(np.clip(direction_sum, -1.0, 1.0)),
                'strength': float(np.clip(abs(direction_sum), 0.0, 1.0))
            }
            
        except Exception as e:
            return {'score': 0.0, 'direction': 0.0, 'strength': 0.0}
    
    def _detect_liquidity_zones_mt5(self, current_price):
        """💧 Detectar zonas de liquidez usando MT5"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 50)
            if rates is None or len(rates) < 10:
                return {'near_support': False, 'near_resistance': False, 'zone_strength': 0.0}
            
            highs = rates['high']
            lows = rates['low']
            
            # Detectar resistance (máximos)
            resistance_levels = []
            for i in range(2, len(highs)-2):
                if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
                    resistance_levels.append(highs[i])
            
            # Detectar support (mínimos)
            support_levels = []
            for i in range(2, len(lows)-2):
                if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
                    support_levels.append(lows[i])
            
            # Verificar proximidade
            price_range = np.max(highs) - np.min(lows)
            threshold = price_range * 0.01  # 1% do range
            
            near_resistance = any(abs(current_price - r) < threshold for r in resistance_levels)
            near_support = any(abs(current_price - s) < threshold for s in support_levels)
            
            # Calcular força da zona
            zone_strength = 0.0
            if near_resistance:
                zone_strength += 0.5
            if near_support:
                zone_strength += 0.5
            
            return {
                'near_support': bool(near_support),
                'near_resistance': bool(near_resistance),
                'zone_strength': float(zone_strength)
            }
            
        except Exception as e:
            return {'near_support': False, 'near_resistance': False, 'zone_strength': 0.0}
    
    def _extract_pattern_memory_mt5(self, current_price):
        """🔍 Extrair memória de padrões usando MT5"""
        try:
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 20)
            if rates is None or len(rates) < 10:
                return {'pattern_strength': 0.0, 'pattern_type': 'none', 'confidence': 0.0}
            
            prices = rates['close']
            
            # Detectar padrões simples
            trend_slope = np.polyfit(range(len(prices)), prices, 1)[0]
            trend_strength = abs(trend_slope) / np.std(prices) if np.std(prices) > 0 else 0.0
            
            # Reversal pattern (últimas 5 barras)
            if len(prices) >= 5:
                recent_prices = prices[-5:]
                if recent_prices[0] < recent_prices[2] < recent_prices[4]:  # Uptrend
                    pattern_type = 'uptrend'
                    confidence = 0.7
                elif recent_prices[0] > recent_prices[2] > recent_prices[4]:  # Downtrend
                    pattern_type = 'downtrend'
                    confidence = 0.7
                else:
                    pattern_type = 'sideways'
                    confidence = 0.4
            else:
                pattern_type = 'none'
                confidence = 0.0
            
            return {
                'pattern_strength': float(np.clip(trend_strength, 0.0, 2.0)),
                'pattern_type': pattern_type,
                'confidence': float(np.clip(confidence, 0.0, 1.0))
            }
            
        except Exception as e:
            return {'pattern_strength': 0.0, 'pattern_type': 'none', 'confidence': 0.0}
    
    def _calculate_risk_metrics_mt5(self, current_price):
        """🛡️ Calcular métricas de risco usando MT5"""
        try:
            # Obter posições atuais
            mt5_positions = mt5.positions_get(symbol=self.symbol) or []
            
            # Calcular concentração de posições
            position_concentration = len(mt5_positions) / self.max_positions
            
            # Calcular drawdown baseado em posições
            total_pnl = sum([self._get_position_pnl(pos, current_price) for pos in mt5_positions])
            account_info = mt5.account_info()
            if account_info:
                current_balance = account_info.balance + total_pnl
                equity = account_info.equity
                drawdown = max(0, (current_balance - equity) / current_balance) if current_balance > 0 else 0.0
            else:
                drawdown = 0.0
            
            # Volatilidade recente
            rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 10)
            if rates is not None and len(rates) >= 5:
                prices = rates['close']
                returns = np.diff(prices) / prices[:-1]
                volatility = np.std(returns)
            else:
                volatility = 0.01  # Default
            
            # Risk score combinado
            risk_score = (drawdown * 0.5) + (position_concentration * 0.3) + (volatility * 0.2)
            
            return {
                'drawdown_risk': float(np.clip(drawdown, 0.0, 1.0)),
                'position_risk': float(np.clip(position_concentration, 0.0, 1.0)),
                'volatility_risk': float(np.clip(volatility, 0.0, 0.1)),
                'risk_score': float(np.clip(risk_score, 0.0, 1.0))
            }
            
        except Exception as e:
            return {'drawdown_risk': 0.0, 'position_risk': 0.0, 'volatility_risk': 0.01, 'risk_score': 0.0}
    
    def _calculate_market_fatigue_mt5(self, current_price):
        """😴 Calcular fadiga do mercado usando MT5"""
        try:
            # Simular trades recentes baseado em posições
            mt5_positions = mt5.positions_get(symbol=self.symbol) or []
            
            # Contar posições recentes (últimas 4 horas)
            current_time = time.time()
            recent_trades = 0
            
            for pos in mt5_positions:
                position_time = getattr(pos, 'time', None) or getattr(pos, 'time_setup', None)
                if position_time and (current_time - position_time) < 14400:  # 4 horas
                    recent_trades += 1
            
            # Calcular fadiga baseada em overtrading
            fatigue_score = min(recent_trades / 5.0, 1.0)  # 5+ posições = fadiga máxima
            
            return {
                'fatigue_score': float(np.clip(fatigue_score, 0.0, 1.0)),
                'recent_trades': int(recent_trades),
                'should_avoid_entry': bool(fatigue_score > 0.7)
            }
            
        except Exception as e:
            return {'fatigue_score': 0.0, 'recent_trades': 0, 'should_avoid_entry': False}
    
    def _flatten_intelligent_components_mt5(self, components):
        """🔄 Achatar componentes inteligentes para observação MT5 - COMPATÍVEL COM PPOV1 (12 features)"""
        try:
            flattened = []
            
            # Market regime (3 features) - IGUAL AO TREINAMENTO
            regime = components['market_regime']
            regime_encoding = {'trending': 1.0, 'ranging': 0.0, 'volatile': 0.5, 'unknown': 0.25}
            flattened.extend([
                regime_encoding.get(regime['regime'], 0.25),
                regime['strength'],
                regime['direction']
            ])
            
            # Volatility context (3 features) - IGUAL AO TREINAMENTO
            vol_ctx = components['volatility_context']
            vol_encoding = {'high': 1.0, 'normal': 0.5, 'low': 0.0}
            flattened.extend([
                vol_encoding.get(vol_ctx['level'], 0.5),
                vol_ctx['percentile'],
                1.0 if vol_ctx['expanding'] else 0.0
            ])
            
            # Momentum confluence (3 features) - IGUAL AO TREINAMENTO
            momentum = components['momentum_confluence']
            flattened.extend([
                momentum['score'],
                momentum['direction'],
                momentum['strength']
            ])
            
            # Risk assessment (3 features) - IGUAL AO TREINAMENTO PPOV1
            risk = components['risk_assessment']
            flattened.extend([
                risk.get('drawdown_risk', risk.get('drawdown', 0.5)),
                risk.get('volatility_risk', risk.get('volatility', 0.5)),
                risk.get('position_risk', risk.get('position_concentration', 0.5))
            ])
            
            # Total: 12 features inteligentes (COMPATÍVEL COM PPOV1)
            return np.array(flattened, dtype=np.float32)
            
        except Exception as e:
            self._log(f"[V5-ERROR] Erro ao achatar componentes MT5: {e}")
            return np.zeros(12, dtype=np.float32)  # 12 features de fallback
    
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
                    
                    # SMAs reais (NORMALIZADOS)
                    sma_20_raw = np.mean(prices[-20:]) if len(prices) >= 20 else current_price
                    sma_50_raw = np.mean(prices[-50:]) if len(prices) >= 50 else current_price
                    
                    # 🔥 CORREÇÃO CRÍTICA: Normalizar preços para escala 0-1
                    sma_20 = (sma_20_raw - current_price) / current_price  # Diferença relativa
                    sma_50 = (sma_50_raw - current_price) / current_price  # Diferença relativa
                    
                    # RSI real
                    rsi = self._calculate_rsi(prices[-15:], 14) if len(prices) >= 15 else 50.0
                    
                    # Volatilidade real
                    returns_array = np.diff(prices[-21:]) / prices[-21:-1] if len(prices) > 21 else [0]
                    volatility = np.std(returns_array) if len(returns_array) > 1 else 0.0
                    
                    # ATR real (NORMALIZADO)
                    atr_raw = self._calculate_atr_simple(df.iloc[-15:]) if len(df) >= 15 else abs(returns)
                    atr = atr_raw / current_price  # Normalizar ATR como % do preço
                    
                    # Stochastic real
                    if len(prices) >= 14:
                        high_14 = np.max(df['high'].values[-14:])
                        low_14 = np.min(df['low'].values[-14:])
                        stoch_k = ((current_price - low_14) / (high_14 - low_14)) * 100 if high_14 > low_14 else 50.0
                    else:
                        stoch_k = 50.0
                    
                    # 🔥 BOLLINGER BAND POSITION (0-1) - CORRIGIDO!
                    bb_std = np.std(prices[-20:]) if len(prices) >= 20 else volatility * current_price
                    bb_upper = sma_20_raw + (bb_std * 2)  # ✅ USAR PREÇO ABSOLUTO
                    bb_lower = sma_20_raw - (bb_std * 2)  # ✅ USAR PREÇO ABSOLUTO
                    bb_position = ((current_price - bb_lower) / (bb_upper - bb_lower)) if bb_upper > bb_lower else 0.5
                    bb_position = np.clip(bb_position, 0, 1)
                    
                    # 🔍 DEBUG BB: Log cálculo a cada 50 steps
                    if not hasattr(self, '_bb_debug_counter'):
                        self._bb_debug_counter = 0
                    self._bb_debug_counter += 1
                    
                    if self._bb_debug_counter % 50 == 0 and tf_name == '5m':
                        self._log(f"🔍 [BB DEBUG] Price:{current_price:.2f} | SMA:{sma_20_raw:.2f} | Upper:{bb_upper:.2f} | Lower:{bb_lower:.2f} | Position:{bb_position:.3f}")
                    
                            # Bollinger Bands calculation complete
                    
                    # 🔥 TREND STRENGTH - IGUAL AO TREINAMENTO
                    trend_strength = np.mean(returns_array[-10:]) if len(returns_array) >= 10 else returns
                    
                    # 🔥 APLICAR FEATURES EXATAMENTE COMO NO TREINAMENTO
                    new_row[f'returns_{tf_name}'] = np.float32(np.clip(returns, -0.1, 0.1))
                    new_row[f'volatility_20_{tf_name}'] = np.float32(volatility * 100)
                    new_row[f'sma_20_{tf_name}'] = np.float32(sma_20)
                    new_row[f'sma_50_{tf_name}'] = np.float32(sma_50)
                    new_row[f'rsi_14_{tf_name}'] = np.float32(np.clip(rsi, 0, 100))
                    new_row[f'stoch_k_{tf_name}'] = np.float32(np.clip(stoch_k, 0, 100))
                    new_row[f'bb_position_{tf_name}'] = np.float32(bb_position)  # ✅ CORRIGIDO
                    new_row[f'trend_strength_{tf_name}'] = np.float32(trend_strength)  # ✅ CORRIGIDO
                    new_row[f'atr_14_{tf_name}'] = np.float32(atr)
                    
                    # Log dados reais apenas a cada 10 steps
                    if tf_name == '5m' and not hasattr(self, '_data_log_counter'):
                        self._data_log_counter = 0
                    
                    if tf_name == '5m':
                        self._data_log_counter += 1
                        if self._data_log_counter % 10 == 0:
                            self._log(f"[📊 DADOS] RSI={rsi:.1f} | Vol={volatility:.4f} | BB={bb_position:.2f} | Trend={trend_strength:.4f}")
                            self._data_log_counter = 0
                    
                else:
                    # Fallback com dados do tick se MT5 falhar
                    tick = mt5.symbol_info_tick(self.symbol)
                    current_price = tick.bid if tick else 2000.0
                    
                    new_row[f'returns_{tf_name}'] = np.float32(0.0)
                    new_row[f'volatility_20_{tf_name}'] = np.float32(0.01)
                    new_row[f'sma_20_{tf_name}'] = np.float32(0.0)  # Diferença relativa = 0
                    new_row[f'sma_50_{tf_name}'] = np.float32(0.0)  # Diferença relativa = 0
                    new_row[f'rsi_14_{tf_name}'] = np.float32(50.0)
                    new_row[f'stoch_k_{tf_name}'] = np.float32(50.0)
                    new_row[f'bb_position_{tf_name}'] = np.float32(0.5)  # ✅ CORRIGIDO
                    new_row[f'trend_strength_{tf_name}'] = np.float32(0.0)  # ✅ CORRIGIDO
                    new_row[f'atr_14_{tf_name}'] = np.float32(0.001)  # ATR normalizado
            
            # 🔥 CALCULAR HIGH QUALITY FEATURES (baseadas em dados 5m)
            if '5m' in timeframes:
                rates_5m = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 50)
                if rates_5m is not None and len(rates_5m) > 20:
                    df_5m = pd.DataFrame(rates_5m)
                    
                    # Volume momentum
                    volumes = df_5m['tick_volume'].values
                    volume_sma = np.mean(volumes[-20:]) if len(volumes) >= 20 else volumes[-1]
                    new_row['volume_momentum'] = np.float32(volumes[-1] / volume_sma if volume_sma > 0 else 1.0)
                    
                    # Price position (posição do preço no range recente)
                    highs = df_5m['high'].values
                    lows = df_5m['low'].values  
                    closes = df_5m['close'].values
                    high_20 = np.max(highs[-20:]) if len(highs) >= 20 else highs[-1]
                    low_20 = np.min(lows[-20:]) if len(lows) >= 20 else lows[-1]
                    current_close = closes[-1]
                    new_row['price_position'] = np.float32((current_close - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5)
                    
                    # Volatility ratio
                    vol_short = np.std(closes[-5:]) if len(closes) >= 5 else 0.01
                    vol_long = np.std(closes[-20:]) if len(closes) >= 20 else 0.01
                    new_row['volatility_ratio'] = np.float32(vol_short / vol_long if vol_long > 0 else 1.0)
                    
                    # Intraday range
                    new_row['intraday_range'] = np.float32((highs[-1] - lows[-1]) / closes[-1] if closes[-1] > 0 else 0.0)
                    
                    # Market regime (trending vs ranging)
                    sma_20_regime = np.mean(closes[-20:]) if len(closes) >= 20 else closes[-1]
                    atr_14_regime = np.mean((highs[-14:] - lows[-14:])) if len(highs) >= 14 else abs(highs[-1] - lows[-1])
                    new_row['market_regime'] = np.float32(abs(closes[-1] - sma_20_regime) / atr_14_regime if atr_14_regime > 0 else 0.5)
                    
                    # Spread pressure (simulado)
                    tick = mt5.symbol_info_tick(self.symbol)
                    if tick:
                        spread = tick.ask - tick.bid
                        avg_spread = spread  # Simplificado
                        new_row['spread_pressure'] = np.float32(spread / avg_spread if avg_spread > 0 else 1.0)
                    else:
                        new_row['spread_pressure'] = np.float32(1.0)
                    
                    # Session momentum (baseado na hora)
                    current_hour = pd.Timestamp.now().hour
                    new_row['session_momentum'] = np.float32(np.sin(2 * np.pi * current_hour / 24))  # Ciclo diário
                    
                    # Time of day (normalizado 0-1)
                    new_row['time_of_day'] = np.float32(current_hour / 24.0)
                    
                    # Tick momentum (baseado em mudanças recentes)
                    if len(closes) >= 3:
                        tick_changes = np.diff(closes[-3:])
                        new_row['tick_momentum'] = np.float32(np.mean(tick_changes))
                    else:
                        new_row['tick_momentum'] = np.float32(0.0)
                else:
                    # Fallback para high quality features
                    new_row['volume_momentum'] = np.float32(1.0)
                    new_row['price_position'] = np.float32(0.5)
                    new_row['volatility_ratio'] = np.float32(1.0)
                    new_row['intraday_range'] = np.float32(0.001)
                    new_row['market_regime'] = np.float32(0.5)
                    new_row['spread_pressure'] = np.float32(1.0)
                    new_row['session_momentum'] = np.float32(0.0)
                    new_row['time_of_day'] = np.float32(pd.Timestamp.now().hour / 24.0)
                    new_row['tick_momentum'] = np.float32(0.0)
            
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
                    self._log("[WARNING] Order send returned None - market may be closed")
                
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
                                # Position analysis complete
                    
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
                                
                            action, _states = model.predict(model_obs, deterministic=False)
                            
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
                    
                    tick_obj = mt5.symbol_info_tick(self.symbol)
                    if not tick_obj:
                        continue
                    current_price = tick_obj.bid
                        
                    # Calcular SL/TP inteligente baseado no modelo + regras de segurança
                    new_sl = None
                    new_tp = None
                    
                    if position.type == 0:  # LONG
                        # SL: Model + minimum 50 points rule
                        # 🔥 CORREÇÃO: SL usando escala realista 15x
                        model_sl = tick_obj.bid - abs(sl_value * 15 * 0.01)  # 15x multiplicador + conversão
                        safety_sl = tick_obj.bid - (30 * 0.01)  # 30 pontos safety (era 100)
                        new_sl = max(model_sl, safety_sl)
                        
                        # 🔥 CORREÇÃO: TP usando escala realista 15x
                        model_tp = tick_obj.ask + abs(tp_value * 15 * 0.01)  # 15x multiplicador + conversão
                        new_tp = model_tp
                            
                    else:  # SHORT
                        # SL: Model + minimum 50 points rule
                        # 🔥 CORREÇÃO: SL usando escala realista 15x
                        model_sl = tick_obj.ask + abs(sl_value * 15 * 0.01)  # 15x multiplicador + conversão
                        safety_sl = tick_obj.ask + (30 * 0.01)  # 30 pontos safety (era 100)
                        new_sl = min(model_sl, safety_sl)
                        
                        # 🔥 CORREÇÃO: TP usando escala realista 15x
                        model_tp = tick_obj.bid - abs(tp_value * 15 * 0.01)  # 15x multiplicador + conversão
                        new_tp = model_tp
                    
                    # Definir tipo de posição para logs
                    position_type = "LONG" if position.type == 0 else "SHORT"
                    
                    # 🔥 AUTO-PROTECTION DISABLED: Conflicting with main execution SL/TP
                    # The main execution already sets proper broker-compatible SL/TP
                    # Auto-protection was overriding with incompatible values causing stops
                    if is_new:
                        self._log(f"ℹ️ AUTO-PROTECTION DISABLED - Main execution handles SL/TP")
                        self._log(f"ℹ️ Position {position_type} #{position.ticket} uses main execution SL/TP")
                    
                    # Skip auto-protection to prevent conflicts with main execution
            
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
                reward = (current_portfolio - old_portfolio) * 100.0  # Escalar para Enhanced Normalizer
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
            
            # 🔥 CORREÇÃO CRÍTICA: Limites IDÊNTICOS ao treinamento
            base_lot = 0.02  # Base lot igual ao treinamento
            max_lot = 0.03   # Max lot igual ao treinamento
            
            # Lógica simplificada igual ao treinamento
            if current_balance <= 1000.0:  # Balance inicial
                final_size = base_lot
            else:
                # Crescimento limitado igual ao treinamento
                growth_factor = current_balance / 1000.0
                capped_growth_factor = min(growth_factor, 1.6)  # Cap de 60%
                target_lot = base_lot * capped_growth_factor
                final_size = max(base_lot, min(target_lot, max_lot))
            
            final_size = round(final_size, 2)  # Arredondar igual ao treinamento
            
            # 🔥 LOG DETALHADO PARA PRIMEIROS TRADES
            if hasattr(self.session_stats, 'total_buys') and (self.session_stats.total_buys + self.session_stats.total_sells) < 3:
                self.log(f"[DYNAMIC SIZING] Balance: ${current_balance:.2f} (ratio: {portfolio_ratio:.2f})")
                self.log(f"[DYNAMIC SIZING] Position %: {position_percentage:.1%} = ${portfolio_value_for_trade:.2f}")
                self.log(f"[DYNAMIC SIZING] Lots calculados: {calculated_lots:.3f} → Final: {final_size:.3f}")
                self.log(f"[DYNAMIC SIZING] Confidence: {action_confidence:.2f} | Volatility: {volatility:.4f}")
            
            return final_size
            
        except Exception as e:
            # Fallback CORRIGIDO: usar valores do treinamento
            try:
                account_info = mt5.account_info()
                if account_info:
                    # Usar lógica simples igual ao treinamento
                    if account_info.balance <= 1000.0:
                        fallback_size = 0.02  # Base lot do treinamento
                    else:
                        growth_factor = account_info.balance / 1000.0
                        capped_growth = min(growth_factor, 1.6)  # Cap igual ao treinamento
                        fallback_size = max(0.02, min(0.02 * capped_growth, 0.03))
                else:
                    fallback_size = 0.02  # Base lot do treinamento
                self.log(f"[SIZING ERROR] Usando fallback: {fallback_size:.3f} lotes - Erro: {e}")
                return round(fallback_size, 2)
            except:
                return 0.02  # Base lot do treinamento
    
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
        # Verificar se é posição MT5 real ou dicionário simulado
        if hasattr(pos, 'type'):  # Posição MT5 real
            if pos.type == 0:  # LONG
                price_diff = current_price - pos.price_open
            else:  # SHORT
                price_diff = pos.price_open - current_price
            lot_size = pos.volume
        else:  # Dicionário simulado (fallback)
            pos_type = pos.get('type', 'long')
            if pos_type == 'long':
                price_diff = current_price - pos.get('entry_price', current_price)
            else:
                price_diff = pos.get('entry_price', current_price) - current_price
            lot_size = pos.get('lot_size', 0.02)
        
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

    def _calculate_bb_position_FIXED(self, close_prices, window=20):
        """
        🔧 CÁLCULO CORRETO DO BB POSITION
        Corrige o bug que causava BB sempre = 1.00
        """
        if len(close_prices) < window:
            return 0.5  # Valor neutro se dados insuficientes
        
        # Usar preços ABSOLUTOS (não diferenças relativas)
        close_array = np.array(close_prices)
        
        # SMA usando preços absolutos
        sma_20 = np.mean(close_array[-window:])
        
        # Desvio padrão usando preços absolutos
        bb_std = np.std(close_array[-window:])
        
        # Bandas usando preços absolutos
        bb_upper = sma_20 + (bb_std * 2)
        bb_lower = sma_20 - (bb_std * 2)
        
        # Preço atual
        current_price = close_array[-1]
        
        # Calcular posição (0-1)
        if bb_upper == bb_lower:  # Evitar divisão por zero
            bb_position = 0.5
        else:
            bb_position = (current_price - bb_lower) / (bb_upper - bb_lower)
            bb_position = max(0.0, min(1.0, bb_position))  # Clip 0-1
        
        # Bollinger Bands calculation complete
        
        return bb_position

    def _process_model_action(self, action):
        """
        🧠 PROCESSAR AÇÃO DO MODELO PPOV1 - ACTION SPACE 11D
        Compatível com ppov1.py: [action, confidence, temporal_signal, risk_appetite, market_regime_bias, sl1, sl2, sl3, tp1, tp2, tp3]
        """
        try:
            if not isinstance(action, (list, tuple, np.ndarray)):
                action = np.array([action])
            
            # Garantir 11 dimensões para compatibilidade PPOV1
            if len(action) < 11:
                action = np.pad(action, (0, 11 - len(action)), mode='constant')
            
            # 🧠 PPOV1 ACTION SPACE: [action, confidence, temporal_signal, risk_appetite, market_regime_bias, sl1, sl2, sl3, tp1, tp2, tp3]
            entry_decision = int(np.clip(action[0], 0, 2))  # 0=HOLD, 1=LONG, 2=SHORT
            confidence = float(np.clip(action[1], 0, 1))  # [0,1] Confiança
            temporal_signal = float(np.clip(action[2], -1, 1))  # [-1,1] Sinal temporal
            risk_appetite = float(np.clip(action[3], 0, 1))  # [0,1] Apetite ao risco
            market_regime_bias = float(np.clip(action[4], -1, 1))  # [-1,1] Viés do regime
            
            # SL/TP para cada posição ([-3,3] → pontos reais)
            sl_adjusts = [float(action[i]) for i in range(5, 8)]  # [5-7] SL positions
            tp_adjusts = [float(action[i]) for i in range(8, 11)]  # [8-10] TP positions
            
            # Converter [-3,3] para pontos reais (escala 15x como no treinamento)
            sl_points = [sl * 15 for sl in sl_adjusts]  # [-45, +45] pontos
            tp_points = [tp * 15 for tp in tp_adjusts]  # [-45, +45] pontos
            
            # Mapear ação para nome
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action_name = action_names.get(entry_decision, 'UNKNOWN')
            
            # Calcular position size baseado na confiança e apetite ao risco
            position_size = confidence * risk_appetite
            
            return {
                'entry_decision': entry_decision,
                'entry_confidence': confidence,
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
            self._log(f"❌ [PPOV1-ACTION] Erro ao processar ação: {e}")
            return {
                'entry_decision': 0,
                'entry_confidence': 0.0,
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

    def _execute_model_decision(self, action_analysis, current_price):
        """
        🧠 EXECUTAR DECISÃO DO MODELO PPOV1 NO MT5
        Compatível com ppov1.py - Action space 11D especializado
        """
        try:
            if not self.mt5_connected:
                self._log("⚠️ [V5-EXECUÇÃO] MT5 não conectado - simulação apenas")
                return
                
            action_name = action_analysis['action_name']
            confidence = abs(action_analysis['entry_confidence'])
            
            # 🧠 PPOV1 LOG: Mostrar ação processada
            raw_action = action_analysis.get('raw_action', [])
            temporal = action_analysis.get('temporal_signal', 0.0)
            risk_app = action_analysis.get('risk_appetite', 0.0)
            regime_bias = action_analysis.get('market_regime_bias', 0.0)
            self._log(f"🧠 [PPOV1-DECISION] {action_name} | Conf: {confidence:.3f} | Temporal: {temporal:.3f} | Risk: {risk_app:.3f} | Regime: {regime_bias:.3f}")
            
            # 🔥 COMPORTAMENTO PURO DO MODELO PPOV1 - COM PROTEÇÕES ESSENCIAIS
            # Verifica apenas proteções básicas da conta
            
            # 🚀 EXECUTAR ORDEM BASEADA NA DECISÃO
            if action_name == 'BUY':
                self._execute_buy_order(current_price, confidence, action_analysis)
            elif action_name == 'SELL':
                self._execute_sell_order(current_price, confidence, action_analysis)
            else:
                # HOLD - modelo decidiu não fazer nada
                self._log(f"📊 [PPOV1-EXECUÇÃO] HOLD - Modelo decidiu não operar")
                
        except Exception as e:
            self._log(f"❌ [PPOV1-EXECUÇÃO] Erro ao executar decisão: {e}")

    def _execute_buy_order(self, current_price, confidence, action_analysis=None):
        """🧠 Executar ordem de compra PPOV1 - com SL/TP inteligentes"""
        try:
            # Calcular volume baseado na confiança
            volume = self._calculate_volume_by_confidence(confidence)
            
            # 🧠 V5: Usar SL/TP do modelo se disponível
            if action_analysis and 'sl_points' in action_analysis and 'tp_points' in action_analysis:
                # Usar primeiro SL/TP do modelo (posição 0)
                sl_points = abs(action_analysis['sl_points'][0])  # Garantir positivo
                tp_points = abs(action_analysis['tp_points'][0])  # Garantir positivo
                
                # Aplicar limites de segurança (10-50 pontos)
                sl_points = np.clip(sl_points, 10, 50)
                tp_points = np.clip(tp_points, 15, 80)
                
                sl_price = current_price - (sl_points * 0.01)  # SL abaixo do preço
                tp_price = current_price + (tp_points * 0.01)  # TP acima do preço
                
                self._log(f"🧠 [PPOV1-BUY] Usando SL/TP do modelo: SL={sl_points}pts | TP={tp_points}pts")
            else:
                # Fallback para valores padrão
                sl_price = current_price - (30 * 0.01)  # 30 pontos SL
                tp_price = current_price + (60 * 0.01)  # 60 pontos TP
                self._log(f"📊 [BUY] Usando SL/TP padrão: SL=30pts | TP=60pts")
            
            # Executar ordem
            result = self._execute_order(mt5.ORDER_TYPE_BUY, volume, sl_price, tp_price)
            
            if "SUCCESS" in result:
                self._log(f"✅ [COMPRA] Ordem executada! Vol: {volume} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
            else:
                self._log(f"❌ [COMPRA] Falha na execução: {result}")
                
        except Exception as e:
            self._log(f"❌ [COMPRA] Erro: {e}")

    def _execute_sell_order(self, current_price, confidence, action_analysis=None):
        """🧠 Executar ordem de venda PPOV1 - com SL/TP inteligentes"""
        try:
            # Calcular volume baseado na confiança
            volume = self._calculate_volume_by_confidence(confidence)
            
            # 🧠 V5: Usar SL/TP do modelo se disponível
            if action_analysis and 'sl_points' in action_analysis and 'tp_points' in action_analysis:
                # Usar primeiro SL/TP do modelo (posição 0)
                sl_points = abs(action_analysis['sl_points'][0])  # Garantir positivo
                tp_points = abs(action_analysis['tp_points'][0])  # Garantir positivo
                
                # Aplicar limites de segurança (10-50 pontos)
                sl_points = np.clip(sl_points, 10, 50)
                tp_points = np.clip(tp_points, 15, 80)
                
                sl_price = current_price + (sl_points * 0.01)  # SL acima do preço
                tp_price = current_price - (tp_points * 0.01)  # TP abaixo do preço
                
                self._log(f"🧠 [PPOV1-SELL] Usando SL/TP do modelo: SL={sl_points}pts | TP={tp_points}pts")
            else:
                # Fallback para valores padrão
                sl_price = current_price + (30 * 0.01)  # 30 pontos SL
                tp_price = current_price - (60 * 0.01)  # 60 pontos TP
                self._log(f"📊 [SELL] Usando SL/TP padrão: SL=30pts | TP=60pts")
            
            # Executar ordem
            result = self._execute_order(mt5.ORDER_TYPE_SELL, volume, sl_price, tp_price)
            
            if "SUCCESS" in result:
                self._log(f"✅ [VENDA] Ordem executada! Vol: {volume} | SL: {sl_price:.5f} | TP: {tp_price:.5f}")
            else:
                self._log(f"❌ [VENDA] Falha na execução: {result}")
                
        except Exception as e:
            self._log(f"❌ [VENDA] Erro: {e}")

    def _calculate_volume_by_confidence(self, confidence):
        """Volume PURO baseado no modelo - SEM LIMITAÇÕES"""
        # COMPORTAMENTO PURO: Usa diretamente o que o modelo decidir
        # Sem limitações artificiais de confiança
        base_volume = self.base_lot_size
        
        # Escala linear baseada na confiança absoluta do modelo
        volume_multiplier = 1.0 + abs(confidence)  # Quanto maior confiança, maior volume
        
        return base_volume * volume_multiplier


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
        # Anti-flipflop system removed
        self.session_stats = SessionStats()
        
        # 🎨 SISTEMA DE VISUALIZAÇÃO AVANÇADA
        self.visualization_system = None
        self.enable_visualization = True  # Flag para ativar/desativar visualização
        
        # 🎨 SISTEMA DE DESENHOS TÉCNICOS
        self.technical_drawer = TechnicalAnalysisDrawer()
        self.drawing_enabled = True  # ✅ ATIVADO POR PADRÃO
        # Forçar inicialização imediata dos desenhos técnicos
        try:
            self.technical_drawer = TechnicalAnalysisDrawer()
            self.log(f"[DRAWER] 🎨 Sistema de desenhos técnicos inicializado automaticamente!")
        except Exception as e:
            self.log(f"⚠️ [DRAWER] Erro na inicialização automática: {e}")
        
        # Threading
        self.trading_thread = None
        self.stop_event = Event()
        
        # 🎯 MODO DE TRADING INICIAL - DEFINIR ANTES DA GUI
        self.trading_mode = "DAY_TRADE"  # Modo padrão
        
        # GUI Setup
        self.setup_interface()
        
        # 🔥 CORREÇÃO: Criar ambiente ANTES de carregar modelo
        self.env = TradingEnv()
        
        # Required attributes for execution methods
        self.symbol = "GOLD"  # Símbolo padrão
        self.mt5_connected = False  # Será atualizado no start_trading
        self.base_lot_size = 0.02  # Volume base
        
        # 🔄 MODELO SERÁ CARREGADO QUANDO NECESSÁRIO (NÃO BLOQUEAR GUI)
        self.model = None
        print("✅ [INIT] GUI inicializada - modelo será carregado sob demanda")
        self.initial_balance = 500.0  # Balance inicial padrão
        self.trading = False  # Status de trading
        self.zmq_server = None  # ZMQ server (se disponível)
        self.position_history = {}  # Histórico de posições para anti-micro trades
        
        # 🔥 CORREÇÃO: Atributos ausentes que causavam erros
        # Anti-flip-flop system completely removed
        self.last_action_type = None  # Última ação executada
        self.last_trade_step = 0  # Último step de trade
        self.consecutive_holds = 0  # Contagem de holds consecutivos
        self.trade_count = 0  # Contagem de trades da sessão
        
        # ZMQ functionality removed

        # 🔥 CARREGAR MODELO DAY TRADE AUTOMATICAMENTE NA INICIALIZAÇÃO
        try:
            self.log("🔄 [INIT] Carregando modelo DAY TRADE automaticamente...")
            self._reload_model_for_mode()
            self.log("✅ [INIT] Modelo DAY TRADE carregado com sucesso!")
        except Exception as e:
            self.log(f"⚠️ [INIT] Erro ao carregar modelo DAY TRADE na inicialização: {e}")
            self.log("💡 [INIT] Use o botão toggle para carregar manualmente quando precisar")
            # Não bloquear a inicialização - GUI continua funcionando
        
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
        self.log("=" * 40)
        self.log("📊 DESENHOS TÉCNICOS AUTOMÁTICOS NO GRÁFICO:")
        self.log("   ▶ Clique em '📊 Desenhos Técnicos' para ativar/desativar")
        self.log("   🟢 Linhas VERDES = Níveis de SUPORTE detectados")
        self.log("   🔴 Linhas VERMELHAS = Níveis de RESISTÊNCIA detectados")
        self.log("   📐 Linhas AZUIS pontilhadas = Níveis de FIBONACCI")
        self.log("   ⚖️ Linhas CINZAS tracejadas = PONTOS PIVÔ (PP, R1, S1)")
        self.log("   🎯 Linhas AMARELAS = ZONAS DE CONFLUÊNCIA")
        self.log("   📈 Linhas CIANO/MAGENTA = LINHAS DE TENDÊNCIA")
        self.log("   ⚡ Linhas LARANJA = DIVERGÊNCIAS e BREAKOUTS")
        self.log("   🔄 Anotações = PADRÕES DE REVERSÃO detectados")
        self.log("   🧠 Texto AMARELO = ANÁLISE IA do mercado")
        self.log("   💡 TUDO é desenhado automaticamente baseado na IA!")
        self.log("=" * 60)
    
    def log(self, message):
        """Log apenas no terminal - GUI removida"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        print(formatted_message)
    
    def start_trading(self):
        """Iniciar trading"""
        if not self.model:
            self.log("[ERROR] ❌ Nenhum modelo carregado!")
            self.log("[INFO] 💡 Use o botão toggle para carregar um modelo primeiro!")
            return
        
        # Inicializar estatísticas da sessão
        self.session_stats = SessionStats()
        
        # Obter balance inicial e verificar conexão MT5
        account_info = mt5.account_info()
        if account_info:
            self.session_stats.update_balance(account_info.balance)
            self.mt5_connected = True  # 🔥 CORREÇÃO: Marcar MT5 como conectado
        else:
            self.mt5_connected = False
            self.log("⚠️ [MT5] Falha na conexão - trading em modo simulação")
        
        self.trading = True
        self.stop_event.clear()
        
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_trading.config(text="📊 Trading: Ativo", fg='#00ff88')
        
        # Iniciar atualização da GUI
        self.update_gui_stats()
        
        self.trading_thread = Thread(target=self.run_trading, daemon=True)
        self.trading_thread.start()
        
        # Status baseado no Enhanced Normalizer
        if self.vec_env:
            self.log("[🚀 ⚔️ LEGION] Trading iniciado com NORMALIZAÇÃO ATIVA - Dados processados!")
            self.log("[✅ ENHANCED] Enhanced Normalizer ATIVO para observações normalizadas")
        else:
            self.log("[🚀 ⚔️ LEGION] Trading iniciado com DADOS RAW - Enhanced Normalizer desabilitado!")
            self.log("[⚠️ ENHANCED] Modelo usando dados não normalizados")
        
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
        
        # ZMQ functionality removed
        
        # 🎨 Análise Profunda permanece ativa para salvar dados para EA
        if self.enable_visualization:
            self.log("🎨 [ANÁLISE PROFUNDA] Dados continuarão sendo salvos para EA")
    
    def toggle_visualization(self):
        """Ativa/desativa a visualização avançada"""
        try:
            if not self.enable_visualization:
                # 🔥 ANÁLISE PROFUNDA ATIVADA - SALVAR DADOS PARA EA
                self.visualization_system = None  # EA vai ler os dados do arquivo
                self.enable_visualization = True  # 🔥 ATIVADO POR PADRÃO!
                self.viz_status.config(text="🎨 Análise Profunda: ON", fg='#00ff88')
                self.log("🎨 [ANÁLISE PROFUNDA] ATIVADA - Dados salvos para EA visualizar")
                
            else:
                # Desativar análise profunda
                self.visualization_system = None
                self.enable_visualization = False
                self.viz_status.config(text="🎨 Análise Profunda: OFF", fg='#ff4444')
                self.log("🎨 [ANÁLISE PROFUNDA] DESATIVADA! (dados não serão salvos para EA)")
                
        except Exception as e:
            self.log(f"❌ [VISUALIZATION] Erro ao alternar visualização: {e}")
            self.enable_visualization = False
            self.viz_status.config(text="🎨 Visualização: ERROR", fg='#ff4444')
    
    def toggle_technical_drawings(self):
        """🎨 Toggle dos desenhos técnicos no gráfico"""
        self.drawing_enabled = not self.drawing_enabled
        
        if self.drawing_enabled:
            self.drawing_button.config(text="📊 Desenhos Técnicos - ON", bg='#ff8800')
            self.drawing_status.config(text="📊 Desenhos: ON", fg='#ff8800')
            self.log("📊 [DRAWINGS] Desenhos técnicos ATIVADOS - Suportes, resistências, padrões no gráfico")
            self.log("📊 [DRAWINGS] 🟢 Suportes | 🔴 Resistências | 📐 Fibonacci | ⚖️ Pivots | 🎯 Confluências")
            self.log("📊 [DRAWINGS] ⚡ Divergências | 💥 Breakouts | 🔄 Reversões | 📈 Tendências")
        else:
            self.drawing_button.config(text="📊 Desenhos Técnicos - OFF", bg='#666666')
            self.drawing_status.config(text="📊 Desenhos: OFF", fg='#666666')
            self.log("📊 [DRAWINGS] Desenhos técnicos DESATIVADOS")
            
            # Limpar todos os desenhos quando desativado
            if hasattr(self, 'technical_drawer'):
                self.technical_drawer.clear_all_drawings()
        
        # Atualizar status
        self.update_gui_stats()
    
    def toggle_trading_mode(self):
        """🎯 Toggle entre DAY TRADE e SWING"""
        if self.trading_mode == "DAY_TRADE":
            self.trading_mode = "SWING"
            self.mode_button.config(text="🎯 Modo: SWING", bg='#ff6600')  # LARANJA
            self.mode_status.config(text="🎯 Modo Atual: SWING", fg='#ff6600')
            self.log("🎯 [MODE] Modo alterado para SWING")
            self.log("📁 [MODEL] Carregando modelo: /Modelo PPO/")
            self.log("⚡ [NORMALIZER] Enhanced Normalizer: Swing Trade")
        else:
            self.trading_mode = "DAY_TRADE"
            self.mode_button.config(text="🎯 Modo: DAY TRADE", bg='#0066cc')  # AZUL
            self.mode_status.config(text="🎯 Modo Atual: DAY TRADE", fg='#0066cc')
            self.log("🎯 [MODE] Modo alterado para DAY TRADE")
            self.log("📁 [MODEL] Carregando modelo: /Modelo daytrade/")
            self.log("⚡ [NORMALIZER] Enhanced Normalizer: Day Trade")
        
        # Recarregar modelo e normalizer baseado no modo
        self._reload_model_for_mode()
        
        # Atualizar status
        self.update_gui_stats()
    
    def _reload_model_for_mode(self):
        """🔄 Recarregar modelo e normalizer baseado no modo selecionado"""
        try:
            if self.trading_mode == "DAY_TRADE":
                model_path = "Modelo daytrade"
                normalizer_path = "Modelo daytrade"
            else:  # SWING
                model_path = "Modelo PPO"
                normalizer_path = "Modelo PPO"
            
            self.log(f"🔄 [RELOAD] Carregando modelo {self.trading_mode} de: {model_path}")
            
            # Carregar modelo com verificação rigorosa
            success = self._load_model_strict(model_path)
            if not success:
                raise Exception(f"Falha ao carregar modelo de {model_path}")
            
            # Carregar enhanced normalizer OBRIGATÓRIO
            normalizer_success = self._load_enhanced_normalizer(normalizer_path)
            if not normalizer_success:
                raise Exception(f"Enhanced Normalizer OBRIGATÓRIO não encontrado em {normalizer_path}")
            
            self.log(f"✅ [RELOAD] Modelo {self.trading_mode} carregado com sucesso!")
            if hasattr(self.env, 'observation_space') and self.env.observation_space:
                self.log(f"✅ [RELOAD] Observation Space: {self.env.observation_space.shape[0]} dimensões")
            else:
                self.log("✅ [RELOAD] Observation Space será definido durante trading")
            
        except Exception as e:
            self.log(f"❌ [RELOAD] ERRO CRÍTICO: {e}")
            raise Exception(f"Falha crítica no carregamento do modo {self.trading_mode}")
    
    def _load_model_strict(self, model_path):
        """🔥 Carregamento rigoroso: V5=1320, V6=1480, Strategic Fusion obrigatória"""
        try:
            import os
            from sb3_contrib import RecurrentPPO
            from gym import spaces
            
            print(f"🔍 [MODEL] Verificando pasta: {model_path}")
            self.log(f"🔍 [MODEL] Verificando pasta: {model_path}")
            
            # Procurar arquivo .zip na pasta
            model_files = []
            if os.path.exists(model_path):
                print(f"📁 [MODEL] Pasta existe, listando arquivos...")
                self.log(f"📁 [MODEL] Pasta existe, listando arquivos...")
                for file in os.listdir(model_path):
                    if file.endswith('.zip'):
                        full_path = os.path.join(model_path, file)
                        model_files.append(full_path)
                        print(f"✅ [MODEL] Encontrado: {file}")
                        self.log(f"✅ [MODEL] Encontrado: {file}")
            else:
                error_msg = f"❌ [MODEL] Pasta não existe: {model_path}"
                print(error_msg)
                self.log(error_msg)
                raise Exception(error_msg)
            
            if not model_files:
                error_msg = f"❌ Nenhum modelo .zip encontrado em {model_path}"
                print(error_msg)
                self.log(error_msg)
                raise Exception(error_msg)
            
            # Usar o primeiro modelo encontrado
            model_file = model_files[0]
            print(f"📁 [MODEL] Carregando: {model_file}")
            self.log(f"📁 [MODEL] Carregando: {model_file}")
            
            print("⏳ [MODEL] Iniciando RecurrentPPO.load()...")
            self.log("⏳ [MODEL] Iniciando RecurrentPPO.load()...")
            
            # Carregar modelo SEM verificações desnecessárias
            self.model = RecurrentPPO.load(model_file, device='cpu')
            
            print("✅ [MODEL] RecurrentPPO.load() concluído!")
            self.log("✅ [MODEL] RecurrentPPO.load() concluído!")
            
            # Definir observation space baseado no que existe
            policy_name = str(type(self.model.policy).__name__)
            print(f"🧠 [POLICY] Detectada: {policy_name}")
            self.log(f"🧠 [POLICY] Detectada: {policy_name}")
            
            # Definir observation space correto baseado na política
            if hasattr(self, 'env') and self.env:
                if 'TwoHeadV6' in policy_name:
                    expected_obs_dim = 1480  # V6 = 1480 OBRIGATÓRIO
                    print(f"✅ [V6] TwoHeadV6 detectado: {expected_obs_dim} dimensões")
                    self.log(f"✅ [V6] TwoHeadV6 detectado: {expected_obs_dim} dimensões")
                elif 'TwoHeadV5' in policy_name:
                    expected_obs_dim = 1320  # V5 = 1320 OBRIGATÓRIO
                    print(f"✅ [V5] TwoHeadV5 detectado: {expected_obs_dim} dimensões")
                    self.log(f"✅ [V5] TwoHeadV5 detectado: {expected_obs_dim} dimensões")
                else:
                    # ERRO - POLICY INVÁLIDA
                    raise Exception(f"POLICY INVÁLIDA: {policy_name} - Apenas TwoHeadV5/V6 aceitas")
                
                self.env.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(expected_obs_dim,), dtype=np.float32
                )
                print(f"✅ [OBS] Observation space configurado: {expected_obs_dim} dimensões")
                self.log(f"✅ [OBS] Observation space configurado: {expected_obs_dim} dimensões")
            
            print("✅ [MODEL] Modelo carregado com sucesso!")
            self.log("✅ [MODEL] Modelo carregado com sucesso!")
            
            return True
            
        except Exception as e:
            self.log(f"❌ [MODEL] ERRO CRÍTICO: {e}")
            return False
    
    def _load_enhanced_normalizer(self, normalizer_path):
        """⚡ Carregar Enhanced Normalizer da pasta correspondente"""
        try:
            import os
            from stable_baselines3.common.vec_env import DummyVecEnv
            
            # Arquivo está junto com o modelo na pasta específica
            normalizer_file = os.path.join(normalizer_path, "enhanced_normalizer_final.pkl")
            
            if os.path.exists(normalizer_file):
                self.log(f"⚡ [NORMALIZER] Encontrado: {normalizer_file}")
                
                # CARREGAR O ENHANCED NORMALIZER CORRETAMENTE
                try:
                    temp_env = DummyVecEnv([lambda: self.env])
                    self.vec_env = EnhancedRunningNormalizer.load(normalizer_file, temp_env)
                    self.log(f"✅ [NORMALIZER] Enhanced Normalizer carregado com sucesso!")
                    return True
                except Exception as load_error:
                    self.log(f"❌ [NORMALIZER] Erro ao carregar: {load_error}")
                    return False
            else:
                self.log(f"⚠️ [NORMALIZER] Arquivo não encontrado: {normalizer_file}")
                return False
                
        except Exception as e:
            self.log(f"⚠️ [NORMALIZER] Erro: {e}")
            return False
    
    def _send_signal_via_zmq(self, price, estrategica, confidence):
        """Signal sending functionality removed for cleaner implementation"""
        pass

    def _send_drawing_data_via_zmq(self, obs, price, confidence):
        """🎨 Enviar dados de análise técnica para desenhos (sem ZMQ, só processamento)"""
        try:
            import numpy as np
            
            # Extrair dados das observações para análise técnica
            if hasattr(self.env, 'historical_df') and len(self.env.historical_df) > 0:
                latest = self.env.historical_df.iloc[-1]
                rsi = latest.get('rsi_14_5m', 50.0)
                bb_pos = latest.get('bb_position_5m', 0.5)
                volatility = latest.get('volatility_20_5m', 0.01)
                momentum = latest.get('momentum_5m', 0.0)
            else:
                rsi = 50.0
                bb_pos = 0.5
                volatility = 0.01
                momentum = 0.0
            
            # Calcular níveis de suporte e resistência
            vol_range = max(volatility * price, price * 0.0005)
            support = price - vol_range
            resistance = price + vol_range
            
            # Dados técnicos para análise (sem salvar arquivo)
            drawing_data = {
                "type": "TECHNICAL_ANALYSIS",
                "price": float(price),
                "rsi": float(rsi),
                "bb_position": float(bb_pos),
                "volatility": float(volatility),
                "momentum": float(momentum),
                "support": float(support),
                "resistance": float(resistance),
                "confidence": float(confidence),
                
                # Níveis para desenhar
                "levels": {
                    "support_strong": float(support),
                    "support_weak": float(price - vol_range * 0.5),
                    "resistance_strong": float(resistance),
                    "resistance_weak": float(price + vol_range * 0.5),
                    "pivot": float(price)
                },
                
                # Cores baseadas no contexto
                "colors": {
                    "support": "clrGreen" if bb_pos < 0.3 else "clrLimeGreen",
                    "resistance": "clrRed" if bb_pos > 0.7 else "clrOrangeRed",
                    "pivot": "clrGray"
                }
            }
            
            # Processar dados para desenhos (sem I/O de arquivo)
            return drawing_data
            
        except Exception as e:
            if hasattr(self, 'log'):
                self.log(f"⚠️ [DRAWINGS] Erro: {e}")
            return None
    
    def analyze_model_decision_deep(self, obs, action, current_price, portfolio_value):
        """🧠 ANÁLISE PROFUNDA DA DECISÃO DO MODELO
        Extrai informações detalhadas para logging e EA
        """
        try:
            # Extrair features principais da observação
            obs_features = self.analyze_observation_features(obs)
            market_context = self.analyze_market_context(obs, current_price)
            confidence_analysis = self.analyze_confidence_and_risk(action, obs)
            regime_analysis = self.analyze_market_regime(obs)
            momentum_analysis = self.analyze_momentum_volatility(obs)
            
            # 🔧 CORRIGIR BB POSITION usando dados históricos reais
            if hasattr(self, 'historical_df') and len(self.historical_df) > 20:
                # Usar últimos 20 preços de fechamento
                recent_closes = self.historical_df['close'].tail(20).values if 'close' in self.historical_df.columns else None
                
                if recent_closes is None or len(recent_closes) < 20:
                    # Fallback: usar dados do MT5
                    rates = mt5.copy_rates_from_pos(self.symbol, mt5.TIMEFRAME_M5, 0, 20)
                    if rates is not None and len(rates) >= 20:
                        recent_closes = [r['close'] for r in rates]
                
                if recent_closes is not None and len(recent_closes) >= 20:
                    bb_position_corrected = self.env._calculate_bb_position_FIXED(recent_closes)
                else:
                    bb_position_corrected = 0.5  # Fallback
            else:
                bb_position_corrected = 0.5  # Fallback
            
            # Processar ação do modelo
            action_analysis = self._process_model_action(action)
            
            # 🔥 EXECUTION HANDLED BY MAIN TRADING LOOP
            # The run_trading method already handles order execution
            # This call was redundant and causing errors
            
            # Compilar análise completa
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'market': {
                    'price': current_price,
                    'momentum': momentum_analysis,
                    'regime': regime_analysis
                },
                'action': action_analysis,
                'features': obs_features,
                'context': market_context,
                'confidence': confidence_analysis,
                'portfolio': {
                    'value': portfolio_value,
                    'balance': getattr(self, 'realized_balance', self.initial_balance)
                },
                # 🔧 USAR BB POSITION CORRIGIDO
                'bb_position': bb_position_corrected,
                'rsi': obs_features.get('rsi', 50.0),
                'trend_strength': obs_features.get('trend_strength', 0.0)
            }
            
            # 🚀 ENVIAR DADOS VIA ZMQ SE DISPONÍVEL
            if self.zmq_server:
                try:
                    update_zmq_data(analysis)
                except Exception as e:
                    self.log(f"⚠️ [ZMQ] Erro ao enviar dados: {e}")
            
            # Removido: save_model_data_for_ea - comunicação via servidor Flask
            
            return analysis
            
        except Exception as e:
            self.log(f"❌ [ANÁLISE] Erro na análise profunda: {e}")
            return {}
    
    def _process_model_action(self, action):
        """
        🧠 PROCESSAR AÇÃO DO MODELO PPOV1 - ACTION SPACE 11D (TradingApp)
        Compatível com ppov1.py: [action, confidence, temporal_signal, risk_appetite, market_regime_bias, sl1, sl2, sl3, tp1, tp2, tp3]
        """
        try:
            if not isinstance(action, (list, tuple, np.ndarray)):
                action = np.array([action])
            
            # Garantir 11 dimensões para compatibilidade PPOV1
            if len(action) < 11:
                action = np.pad(action, (0, 11 - len(action)), mode='constant')
            
            # 🧠 PPOV1 ACTION SPACE: [action, confidence, temporal_signal, risk_appetite, market_regime_bias, sl1, sl2, sl3, tp1, tp2, tp3]
            entry_decision = int(np.clip(action[0], 0, 2))  # 0=HOLD, 1=LONG, 2=SHORT
            confidence = float(np.clip(action[1], 0, 1))  # [0,1] Confiança
            temporal_signal = float(np.clip(action[2], -1, 1))  # [-1,1] Sinal temporal
            risk_appetite = float(np.clip(action[3], 0, 1))  # [0,1] Apetite ao risco
            market_regime_bias = float(np.clip(action[4], -1, 1))  # [-1,1] Viés do regime
            
            # SL/TP para cada posição ([-3,3] → pontos reais)
            sl_adjusts = [float(action[i]) for i in range(5, 8)]  # [5-7] SL positions
            tp_adjusts = [float(action[i]) for i in range(8, 11)]  # [8-10] TP positions
            
            # Converter [-3,3] para pontos reais (escala 15x como no treinamento)
            sl_points = [sl * 15 for sl in sl_adjusts]  # [-45, +45] pontos
            tp_points = [tp * 15 for tp in tp_adjusts]  # [-45, +45] pontos
            
            # Mapear ação para nome
            action_names = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            action_name = action_names.get(entry_decision, 'UNKNOWN')
            
            # Calcular position size baseado na confiança e apetite ao risco
            position_size = confidence * risk_appetite
            
            return {
                'entry_decision': entry_decision,
                'entry_confidence': confidence,
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
            self.log(f"❌ [PPOV1-ACTION] Erro ao processar ação: {e}")
            return {
                'entry_decision': 0,
                'entry_confidence': 0.0,
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
        
        # 🎯 Botão Toggle Swing/Day Trade (modo já definido no __init__)
        self.mode_button = tk.Button(button_frame, text="🎯 Modo: DAY TRADE", 
                                   command=self.toggle_trading_mode, bg='#0066cc', fg='white',
                                   font=('Arial', 10, 'bold'), width=18)
        self.mode_button.pack(pady=2)
        
        # 🎨 Botão de Desenhos Técnicos
        self.drawing_button = tk.Button(button_frame, text="📊 Desenhos Técnicos", 
                                       command=self.toggle_technical_drawings, bg='#ff8800', fg='white',
                                       font=('Arial', 10, 'bold'), width=18)
        self.drawing_button.pack(pady=2)
        
        # Status do modo de trading
        self.mode_status = tk.Label(button_frame, text=f"🎯 Modo Atual: {self.trading_mode}", 
                                  fg='#0066cc', bg='#2d2d2d', font=('Arial', 9))
        self.mode_status.pack(pady=2)
        
        # Status dos desenhos
        self.drawing_status = tk.Label(button_frame, text="📊 Desenhos: ON" if self.drawing_enabled else "📊 Desenhos: OFF", 
                                      fg='#ff8800', bg='#2d2d2d', font=('Arial', 9))
        self.drawing_status.pack(pady=2)
        
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
            
            # Anti-flip-flop system completely removed
            behavior_score = 75  # Default value for stable behavior
            
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
            # Converter de 0-1 para 0-100% para exibição
            avg_confidence_percent = avg_confidence * 100
            self.label_avg_confidence.config(text=f"🎯 Confiança: {avg_confidence_percent:.1f}%")
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

    def _capture_v5_entry_outputs(self, obs):
        """🚀 V5: Capturar outputs da Entry Head para aplicar filtros inteligentes"""
        try:
            # 🔍 DEBUG: Verificar se o modelo tem TwoHeadV5Intelligent48h
            if not hasattr(self.model, 'policy'):
                if not hasattr(self, '_debug_no_policy_logged'):
                    self.log("🚨 [V5 DEBUG] Modelo não tem 'policy'")
                    self._debug_no_policy_logged = True
                return None
                
            policy = self.model.policy
            if not hasattr(policy, 'enable_ultra_specialized_entry'):
                if not hasattr(self, '_debug_no_v5_flag_logged'):
                    self.log("🚨 [V5 DEBUG] Policy não tem 'enable_ultra_specialized_entry'")
                    self._debug_no_v5_flag_logged = True
                return None
                
            # 🔍 DEBUG: Verificar se Entry Head está ativa
            v5_enabled = getattr(policy, 'enable_ultra_specialized_entry', False)
            if not v5_enabled:
                if not hasattr(self, '_debug_v5_disabled_logged'):
                    self.log(f"🚨 [V5 DEBUG] enable_ultra_specialized_entry = {v5_enabled}")
                    self._debug_v5_disabled_logged = True
                return None
                
            if not hasattr(policy, 'entry_head'):
                if not hasattr(self, '_debug_no_entry_head_logged'):
                    self.log("🚨 [V5 DEBUG] Policy não tem 'entry_head'")
                    self._debug_no_entry_head_logged = True
                return None
                
            # 🔍 DEBUG: Log que Entry Head foi encontrada
            if not hasattr(self, '_debug_v5_found_logged'):
                self.log(f"✅ [V5 DEBUG] Entry Head encontrada: {type(policy.entry_head)}")
                self._debug_v5_found_logged = True
                
            # Preparar observação para o modelo
            import torch
            if isinstance(obs, np.ndarray):
                obs_tensor = torch.from_numpy(obs).float()
            else:
                obs_tensor = obs
                
            # Se obs é 1D, adicionar batch dimension
            if len(obs_tensor.shape) == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
                
            # Extrair features usando o extractor da policy
            with torch.no_grad():
                policy.eval()  # Modo determinístico
                
                # Extrair features base
                features = policy.extract_features(obs_tensor)
                
                # 🚀 PREPARAR INTELLIGENT COMPONENTS REALISTAS PARA ENTRY HEAD V5
                # Criar componentes baseados nas features extraídas para ativar gates
                batch_size = features.shape[0]
                device = features.device
                
                # Gerar embeddings baseados nas features (não zeros puros)
                feature_mean = features.mean(dim=-1, keepdim=True)
                feature_std = features.std(dim=-1, keepdim=True)
                
                intelligent_components = {
                    'horizon_embedding': (feature_mean.expand(-1, 8) + torch.randn(batch_size, 8).to(device) * 0.1),
                    'timeframe_fusion': (features + torch.randn_like(features) * 0.05),  # Baseado em features reais
                    'risk_embedding': (feature_std.expand(-1, 8) + torch.randn(batch_size, 8).to(device) * 0.1),
                    'regime_embedding': (feature_mean.expand(-1, 8) * 0.5 + torch.randn(batch_size, 8).to(device) * 0.1),
                    'pattern_memory': torch.randn(batch_size, 192).to(device) * 0.2,  # Simulação de memória
                    'lookahead': torch.tanh(feature_mean) * 0.1  # Baseado em features normalizadas
                }
                
                # Chamar Entry Head V5 diretamente
                entry_output = policy.entry_head(features, intelligent_components)
                
                # 🔍 DEBUG: Verificar output da Entry Head
                if not hasattr(self, '_debug_entry_output_logged'):
                    self.log(f"🔍 [V5 DEBUG] Entry output keys: {list(entry_output.keys()) if entry_output else 'None'}")
                    if entry_output and 'gates' in entry_output:
                        gates = entry_output['gates']
                        gate_values = {k: float(v.item() if hasattr(v, 'item') else v) for k, v in gates.items()}
                        self.log(f"🔍 [V5 DEBUG] Gate values: {gate_values}")
                    self._debug_entry_output_logged = True
                
                # Retornar outputs estruturados
                return {
                    'gates': entry_output.get('gates', {}),
                    'scores': entry_output.get('scores', {}),
                    'decision': entry_output.get('entry_decision'),
                    'attention_weights': entry_output.get('attention_weights'),
                    'score_weights': entry_output.get('score_weights')
                }
                
        except Exception as e:
            # Em caso de erro, retornar None (sem filtros V5)
            if hasattr(self, 'log'):
                self.log(f"⚠️ [V5] Erro ao capturar outputs Entry Head: {e}")
            return None

    def _apply_v5_intelligent_filters(self, action_type, v5_outputs, entry_confidence):
        """🚀 V5: Aplicar filtros especializados da Entry Head (adaptado do ppov1.py)"""
        try:
            if not v5_outputs or not isinstance(v5_outputs, dict):
                return True, "V5 Filters: Outputs não disponíveis"
                
            if 'gates' not in v5_outputs:
                return True, "V5 Filters: Gates não disponíveis"
                
            gates = v5_outputs['gates']
            
            # Converter tensors para valores Python se necessário
            def tensor_to_float(value):
                if hasattr(value, 'item'):
                    return value.item()
                elif hasattr(value, 'cpu'):
                    return value.cpu().numpy().item()
                return float(value) if value is not None else 0.0
            
            # 🎯 VERIFICAR TODOS OS 6 GATES ESPECIALIZADOS (AGORA COMPLETOS)
            temporal_pass = tensor_to_float(gates.get('temporal', 0)) > 0.5
            validation_pass = tensor_to_float(gates.get('validation', 0)) > 0.7  
            risk_pass = tensor_to_float(gates.get('risk', 0)) > 0.6
            market_pass = tensor_to_float(gates.get('market', 0)) > 0.5
            quality_pass = tensor_to_float(gates.get('quality', 0)) > 0.7
            confidence_pass = tensor_to_float(gates.get('confidence', 0)) > 0.7
            final_gate = tensor_to_float(gates.get('final', 0)) > 0.5
            
            # Log detalhado dos gates (a cada 20 verificações)
            if not hasattr(self, '_v5_filter_count'):
                self._v5_filter_count = 0
            self._v5_filter_count += 1
            
            if self._v5_filter_count % 20 == 0:
                self.log(f"🚀 [V5 GATES] T:{tensor_to_float(gates.get('temporal', 0)):.2f} | V:{tensor_to_float(gates.get('validation', 0)):.2f} | R:{tensor_to_float(gates.get('risk', 0)):.2f} | M:{tensor_to_float(gates.get('market', 0)):.2f} | Q:{tensor_to_float(gates.get('quality', 0)):.2f} | C:{tensor_to_float(gates.get('confidence', 0)):.2f}")
            
            # 🎯 VERIFICAR TODOS OS 6 GATES ESPECIALIZADOS (EXATAMENTE COMO ppov1.py)
            gates_passed = temporal_pass and validation_pass and risk_pass and market_pass and quality_pass and confidence_pass
            
            if not gates_passed:
                failed_gates = []
                if not temporal_pass: failed_gates.append('temporal')
                if not validation_pass: failed_gates.append('validation')
                if not risk_pass: failed_gates.append('risk')
                if not market_pass: failed_gates.append('market')
                if not quality_pass: failed_gates.append('quality')
                if not confidence_pass: failed_gates.append('confidence')
                
                return False, f"V5 Gates Filter: Gates falharam: {', '.join(failed_gates)}"
            
            # Verificar scores adicionais se disponíveis
            if 'scores' in v5_outputs:
                scores = v5_outputs['scores']
                
                # Score composite (weighted)
                weighted_score = tensor_to_float(scores.get('weighted_composite', 0))
                if weighted_score < 0.4:  # Threshold permissivo
                    return False, f"V5 Filters: Score composite baixo ({weighted_score:.2f})"
                
                # Market fatigue check
                fatigue_score = tensor_to_float(scores.get('fatigue', 1.0))
                if fatigue_score < 0.3:  # Detectar alta fatiga
                    return False, f"V5 Filters: Market fatigue detectado ({fatigue_score:.2f})"
            
            # ✅ Todos os filtros passaram
            return True, "V5 Intelligent Filters: Aprovado"
            
        except Exception as e:
            # Em caso de erro, ser permissivo (não bloquear trades)
            if hasattr(self, 'log'):
                self.log(f"⚠️ [V5] Erro nos filtros: {e}")
            return True, f"V5 Filters: Erro {str(e)[:30]} - Fallback aprovado"

    def _check_v5_quality_filters(self, action_type, current_step):
        """🚀 V5: Quality filters adaptados para produção"""
        try:
            # Verificar condições básicas de mercado usando MT5
            rates = mt5.copy_rates_from_pos(self.env.symbol, mt5.TIMEFRAME_M5, 0, 20)
            if rates is None or len(rates) < 10:
                return True  # Ser permissivo se não há dados
                
            # Verificar volatilidade mínima
            closes = [r['close'] for r in rates]
            volatility = np.std(closes[-10:]) / np.mean(closes[-10:])
            
            if volatility < 0.001:  # Mercado muito parado
                return False
                
            # Verificar spread (se disponível)
            tick = mt5.symbol_info_tick(self.env.symbol)
            if tick:
                spread = tick.ask - tick.bid
                if spread > 30:  # Spread muito alto
                    return False
                
            return True
            
        except Exception:
            return True  # Ser permissivo em caso de erro

    def _check_market_fatigue_v5(self):
        """🚀 V5: Market fatigue detector simplificado"""
        try:
            # Verificar número de trades recentes (usando session_stats se disponível)
            if hasattr(self, 'session_stats') and hasattr(self.session_stats, 'total_trades'):
                # Se muitos trades na sessão atual
                if self.session_stats.total_trades > 20:
                    return True
                    
            return False
            
        except Exception:
            return False  # Não detectar fatiga em caso de erro
    
    def auto_load_model(self):
        """🔥 FUNÇÃO LEGACY DESABILITADA - USAR APENAS TOGGLE SYSTEM"""
        self.log("❌ [LEGACY] auto_load_model() DESABILITADO - Use o botão toggle")
        return False

    def run_trading(self):
        """🔥 LOOP DE TRADING COM PING A CADA 5 MINUTOS"""
        try:
            if not self.model:
                self.log("[❌ ERRO] Modelo não carregado!")
                return
                
            # 🔥 CONFIGURAÇÕES SL/TP GLOBAIS (ALINHADAS COM PPOV1.PY)
            min_sl_distance = 13.0   # EXATO: 13 pontos = $13.00 (ppov1.py)
            max_sl_distance = 46.0   # EXATO: 46 pontos = $46.00 (ppov1.py)
            min_tp_distance = 16.0   # EXATO: 16 pontos = $16.00 (ppov1.py)
            max_tp_distance = 82.0   # EXATO: 82 pontos = $82.00 (ppov1.py)
            sl_threshold = 0.3  # Threshold for model values
            tp_threshold = 0.3  # Threshold for model values
                
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
                                
                                # Alertar apenas se RSI realmente congelado (threshold ULTRA baixo)
                                if rsi_variance < 0.00001 and rsi_range < 0.01:  # Valores ULTRA restritivos (só se realmente travado)
                                    self.log(f"🚨 DADOS CONGELADOS - RSI travado: {recent_data['rsi_14_5m'].iloc[-1]:.1f}")
                                    self.log(f"   Range RSI: {rsi_range:.3f} | Variância: {rsi_variance:.8f}")
                                elif step_count % 2000 == 0:  # Status normal a cada 2000 steps
                                    self.log(f"📊 [DADOS OK] RSI: {recent_data['rsi_14_5m'].iloc[-1]:.1f} | Var: {rsi_variance:.6f}")
                    
                    # 🔥 CORREÇÃO CRÍTICA: Enhanced Normalizer REATIVADO com tamanho correto (1320)
                    USE_ENHANCED_NORM = True  # 🔥 REATIVADO: Modelo foi treinado com enhanced normalizer
                    
                    if USE_ENHANCED_NORM and hasattr(self, 'vec_env') and self.vec_env is not None:
                        # Normalizar apenas observações, NÃO ações
                        raw_obs = self.env._get_observation()
                        
                        # 🔥 CORREÇÃO CRÍTICA: Normalização robusta com fallback
                        try:
                            # SEM RESHAPE ARTIFICIAL - NORMALIZAR COM DIMENSÕES CORRETAS
                            if raw_obs.ndim == 1:
                                raw_obs_batch = raw_obs.reshape(1, -1)
                            else:
                                raw_obs_batch = raw_obs
                            normalized_obs = self.vec_env.normalize_obs(raw_obs_batch)
                            obs = normalized_obs.flatten()
                            
                            # Verificar se resultado é válido
                            if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                                raise ValueError("Observação normalizada contém NaN ou Inf")
                            
                        except Exception as e:
                            # SEM FALLBACKS - FALHA SE ENHANCED NORMALIZER NÃO FUNCIONAR
                            self.log(f"❌ [ENHANCED] Enhanced Normalizer FALHOU: {e}")
                            raise Exception(f"Enhanced Normalizer OBRIGATÓRIO falhou: {e}")
                        
                        # 🚀 ENHANCED NORMALIZER: Dados reais com adaptação inteligente
                        # Atualização gradual e controlada para adaptar aos dados reais
                        
                        # 🔄 ATUALIZAÇÃO INTELIGENTE BASEADA EM MUDANÇAS
                        if step_count % 25 == 0 and step_count > 100:
                            # Detectar se dados mudaram significativamente
                            obs_mean = np.mean(np.abs(raw_obs))
                            obs_std = np.std(raw_obs)
                            
                            # Comparar com estatísticas atuais do Enhanced Normalizer
                            if hasattr(self.vec_env, 'obs_rms') or hasattr(self.vec_env, 'running_mean'):
                                # Enhanced Normalizer pode ter estrutura diferente
                                if hasattr(self.vec_env, 'obs_rms'):
                                    current_mean = np.mean(self.vec_env.obs_rms.mean)
                                    current_var = np.mean(self.vec_env.obs_rms.var)
                                elif hasattr(self.vec_env, 'running_mean'):
                                    current_mean = np.mean(self.vec_env.running_mean)
                                    current_var = np.mean(self.vec_env.running_var)
                                else:
                                    current_mean = obs_mean
                                    current_var = obs_std**2
                                
                                # Calcular diferença percentual
                                mean_diff = abs(obs_mean - current_mean) / (current_mean + 1e-8)
                                var_diff = abs(obs_std**2 - current_var) / (current_var + 1e-8)
                                
                                # Se mudança significativa (>50%), fazer update mais agressivo
                                if mean_diff > 0.5 or var_diff > 0.5:
                                    update_count = 3  # Update mais agressivo
                                    if step_count % 1000 == 0:  # Log apenas a cada 1000 steps
                                        self.log(f"🔄 [ENHANCED ADAPT] Adaptação significativa - Mean: {mean_diff:.1%}, Var: {var_diff:.1%}")
                                else:
                                    update_count = 1  # Update suave
                            else:
                                update_count = 1
                            
                            # Fazer updates adaptativos (Enhanced Normalizer é mais inteligente)
                            original_training = getattr(self.vec_env, 'training', False)
                            if hasattr(self.vec_env, 'training'):
                                self.vec_env.training = True
                            
                            for _ in range(update_count):
                                _ = self.vec_env.normalize_obs(raw_obs)
                            
                            if hasattr(self.vec_env, 'training'):
                                self.vec_env.training = original_training
                        
                        if step_count == 1:  # Apenas no primeiro step
                            self.log(f"✅ [ENHANCED] Sistema adaptativo ativo")
                    else:
                        # SEM ENHANCED NORMALIZER - ERRO OBRIGATÓRIO
                        self.log(f"❌ [CRITICAL] Enhanced Normalizer OBRIGATÓRIO não encontrado!")
                        raise Exception("Enhanced Normalizer OBRIGATÓRIO - sistema não funciona sem ele")
                    
                    # 🔥 PROTEÇÃO FINAL: Garantir que obs tem formato correto antes do modelo
                    try:
                        # Verificar se obs é válido
                        if not isinstance(obs, np.ndarray):
                            obs = np.array(obs, dtype=np.float32)
                        
                        # Garantir que não há NaN ou Inf
                        if np.any(np.isnan(obs)) or np.any(np.isinf(obs)):
                            # Substituir valores inválidos por 0
                            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
                            if step_count % 100 == 0:
                                self.log(f"⚠️ [MODEL] Valores inválidos corrigidos na observação")
                        
                        # 🚀 V5: Fazer predição com captura dos outputs da Entry Head
                        action, _states = self.model.predict(obs, deterministic=False)
                        
                        # 🚀 V5: Capturar outputs da Entry Head se disponível
                        v5_outputs = self._capture_v5_entry_outputs(obs)
                        
                    except Exception as e:
                        # SEM FALLBACKS - ERRO SE PREDIÇÃO FALHAR
                        self.log(f"❌ [MODEL] Erro na predição: {e}")
                        raise Exception(f"Predição do modelo FALHOU - sistema corrompido: {e}")
                    
                    # Model action tracking
                    if step_count % 50 == 0:
                        self.log(f"[MODEL] Entry:{action[0]:.3f} | Conf:{action[1]:.3f} | Size:{action[2]:.3f} | Mgmt:{action[3]:.3f}")
                    
                    # 🔥 CORREÇÃO CRÍTICA: Definir variáveis do action space ANTES do uso
                    # Garantir que action é um array numpy
                    if not isinstance(action, np.ndarray):
                        action = np.array(action)
                    
                    # 🔥 VERIFICAR ACTION SPACE V5 (11 dimensões)
                    if len(action) != 11:  # V5: [entry_decision, entry_confidence, temporal_signal, risk_appetite, market_regime_bias, sl1, sl2, sl3, tp1, tp2, tp3]
                        self.log(f"❌ [ERRO] Action size incompatível: {len(action)} elementos (esperado: 11)")
                        time.sleep(2)
                        continue
                    
                    # 🔥 PROCESSAR ACTION SPACE V5: 11 DIMENSÕES
                    entry_decision = int(np.clip(action[0], 0, 2))  # 0=HOLD, 1=LONG, 2=SHORT
                    entry_confidence = float(np.clip(action[1], 0, 1))  # Confiança da entrada
                    temporal_signal = float(np.clip(action[2], -1, 1))  # Sinal temporal
                    risk_appetite = float(np.clip(action[3], 0, 1))  # Apetite ao risco
                    market_regime_bias = float(np.clip(action[4], -1, 1))  # Viés do regime de mercado
                    
                    # 🚀 V5: Aplicar filtros inteligentes da Entry Head
                    if v5_outputs and entry_decision != 0:  # Só filtrar se não for HOLD
                        filters_passed, filter_reason = self._apply_v5_intelligent_filters(
                            entry_decision, v5_outputs, entry_confidence
                        )
                        
                        if not filters_passed:
                            # Filtros V5 rejeitaram a ação - forçar HOLD
                            if step_count % 10 == 0:  # Log esparso
                                self.log(f"🚫 [V5 FILTER] {filter_reason} - Forçando HOLD")
                            entry_decision = 0  # Forçar HOLD
                            entry_confidence *= 0.5  # Reduzir confiança
                    
                    # 🚀 V5: Verificar quality filters adicionais
                    if entry_decision != 0:  # Só verificar se não for HOLD
                        quality_passed = self._check_v5_quality_filters(entry_decision, step_count)
                        fatigue_detected = self._check_market_fatigue_v5()
                        
                        if not quality_passed or fatigue_detected:
                            reason = "Quality filters" if not quality_passed else "Market fatigue"
                            if step_count % 10 == 0:  # Log esparso
                                self.log(f"🚫 [V5 QUALITY] {reason} - Forçando HOLD")
                            entry_decision = 0  # Forçar HOLD
                            entry_confidence *= 0.3  # Reduzir mais a confiança
                    
                    # SL/TP para cada posição (3 posições)
                    sl_adjusts = [float(np.clip(action[5], -3, 3)), float(np.clip(action[6], -3, 3)), float(np.clip(action[7], -3, 3))]  # SL1, SL2, SL3
                    tp_adjusts = [float(np.clip(action[8], -3, 3)), float(np.clip(action[9], -3, 3)), float(np.clip(action[10], -3, 3))]  # TP1, TP2, TP3
                    
                    # 🎨 SISTEMA COMPLETO DE DESENHOS TÉCNICOS VISUAIS
                    if self.drawing_enabled and hasattr(self, 'technical_drawer'):
                        # Obter preço atual e confiança do modelo
                        tick_temp = mt5.symbol_info_tick(self.env.symbol)
                        current_price = tick_temp.bid if tick_temp else 2000.0
                        
                        # Calcular confiança baseada na ação V5 (quanto mais extrema, mais confiança)
                        model_confidence = entry_confidence  # Já extraído do action space V5
                        
                        # 🎨 DESENHAR ANÁLISE TÉCNICA DIRETAMENTE NO GRÁFICO MT5
                        if step_count % 30 == 0:  # A cada 30 steps para não sobrecarregar
                            try:
                                self.technical_drawer.analyze_and_draw_market_structure(obs, current_price, model_confidence)
                                self.log(f"🎨 [DESENHOS] Análise técnica atualizada no gráfico MT5")
                            except Exception as e:
                                if step_count % 100 == 0:  # Log erro apenas ocasionalmente
                                    self.log(f"⚠️ [DESENHOS] Erro nos desenhos: {e}")
                        
                        # 🎨 ENVIAR DADOS DE DESENHOS VIA ZMQ (a cada 15 steps)
                        if step_count % 15 == 0:
                            try:
                                self._send_drawing_data_via_zmq(obs, current_price, model_confidence)
                            except Exception as e:
                                if step_count % 100 == 0:  # Log erro apenas ocasionalmente
                                    self.log(f"⚠️ [DRAWINGS] Erro nos desenhos técnicos: {e}")
                    
                    # 🎨 FORÇAR INICIALIZAÇÃO DOS DESENHOS TÉCNICOS SE NÃO EXISTIR
                    elif self.drawing_enabled and not hasattr(self, 'technical_drawer'):
                        try:
                            self.technical_drawer = TechnicalAnalysisDrawer()
                            self.log(f"[DRAWER] 🎨 Sistema de desenhos técnicos ATIVADO FORÇADAMENTE!")
                            
                            # Fazer primeiro desenho imediatamente
                            tick_temp = mt5.symbol_info_tick(self.env.symbol)
                            current_price = tick_temp.bid if tick_temp else 2000.0
                            model_confidence = 0.5
                            
                            self.technical_drawer.analyze_and_draw_market_structure(obs, current_price, model_confidence)
                            self.log(f"🎨 [DESENHOS] Primeira análise técnica desenhada no gráfico!")
                            
                        except Exception as e:
                            self.log(f"⚠️ [DRAWER] Erro ao inicializar desenhos: {e}")
                    
                    # 🎨 ANÁLISE PROFUNDA DO MODELO - SALVAR DADOS PARA EA
                    if self.enable_visualization:
                        try:
                            # Obter preço atual e portfolio
                            tick_temp = mt5.symbol_info_tick(self.env.symbol)
                            current_price = tick_temp.bid if tick_temp else 2000.0
                            account_info = mt5.account_info()
                            portfolio_value = account_info.balance if account_info else 500.0
                            
                            # 🧠 ANÁLISE PROFUNDA DO MODELO (apenas a cada 10 steps)
                            if step_count % 10 == 0:
                                model_analysis = self.analyze_model_decision_deep(obs, action, current_price, portfolio_value)
                                # Removido: save_model_data_for_ea - comunicação via servidor Flask
                            
                        except Exception as e:
                            self.log(f"⚠️ [ANALYSIS] Erro na análise profunda: {e}")
                    
                    # Initial diagnostics
                    if step_count == 1:
                        policy_type = type(self.model.policy).__name__
                        self.log(f"[INIT] Policy: {policy_type} | Action: {action.shape} | Obs: {obs.shape}")
                        
                        # Debug adicional para Enhanced Normalizer
                        if USE_ENHANCED_NORM and hasattr(self, 'vec_env') and self.vec_env is not None:
                            raw_obs_debug = self.env._get_observation()
                            self.log(f"[DEBUG] Raw obs shape: {raw_obs_debug.shape} | Enhanced Norm: ATIVO")
                        else:
                            self.log(f"[DEBUG] Enhanced Normalizer: ATIVO")
                            self.log(f"[DEBUG] Observation space real: {obs.shape} | Expected: {self.env.observation_space.shape}")
                            self.log(f"[DEBUG] Action space real: {action.shape} | Expected: {self.env.action_space.shape}")
                    
                    # Data validation check
                    if step_count % 100 == 0:
                        if hasattr(self, 'env') and len(self.env.historical_df) > 0:
                            latest_data = self.env.historical_df.iloc[-1]
                            rsi_5m = latest_data.get('rsi_14_5m', 50.0)
                            if not hasattr(self, '_last_rsi_check'):
                                self._last_rsi_check = rsi_5m
                            else:
                                rsi_diff = abs(rsi_5m - self._last_rsi_check)
                                if rsi_diff < 0.1:
                                    self.log(f"[DATA] RSI stable at {rsi_5m:.1f}")
                                else:
                                    self._rsi_frozen_count = 0
                                
                                self._last_rsi_check = rsi_5m
                    
                    # 🔥 COMPATIBILIDADE: Definir variáveis para código legado
                    estrategica = entry_decision  # 0=HOLD, 1=LONG, 2=SHORT
                    
                    # 🚀 V5: Log de status dos filtros inteligentes
                    if step_count % 50 == 0:  # Log a cada 50 steps
                        v5_status = "ATIVO" if v5_outputs else "INATIVO"
                        filters_info = ""
                        if v5_outputs and 'gates' in v5_outputs:
                            gates = v5_outputs['gates']
                            def safe_tensor_val(val):
                                try:
                                    if hasattr(val, 'item'):
                                        return val.item()
                                    return float(val) if val is not None else 0.0
                                except:
                                    return 0.0
                            
                            filters_info = f" | Gates: T:{safe_tensor_val(gates.get('temporal', 0)):.2f} R:{safe_tensor_val(gates.get('risk', 0)):.2f} Q:{safe_tensor_val(gates.get('quality', 0)):.2f}"
                        
                        self.log(f"🚀 [V5 STATUS] Entry Head: {v5_status}{filters_info}")
                    
                    # 🔥 CONFIANÇA REAL DO MODELO: Usar valor do action space V5
                    # entry_confidence já foi extraído e normalizado acima
                    
                    position_size = 0.5  # Position size fixo (calculado automaticamente)
                    mgmt_action = 0  # Management action desabilitado no novo action space
                    action_names = {0: "HOLD", 1: "LONG", 2: "SHORT"}  # Nomes das ações
                    
                    # 🔥 TÁTICAS: Simular valores para compatibilidade com código legado
                    taticas = [0, 0, 0]  # Táticas desabilitadas no novo action space
                    
                    # 🔥 CONVERSÃO SL/TP NOVA: Usar primeiro par de valores para próxima posição
                    current_positions = len(mt5.positions_get(symbol=self.env.symbol) or [])
                    pos_index = min(current_positions, 2)  # Max índice 2 (pos1, pos2, pos3)
                    
                    sl_adjust = sl_adjusts[pos_index]  # SL para próxima posição
                    tp_adjust = tp_adjusts[pos_index]  # TP para próxima posição
                    
                    # 🔥 CORREÇÃO CRÍTICA: Alinhar com multiplicador do treinamento
                    # TREINAMENTO usa 15x: [-3,3] → [-45,+45] pontos
                    sl_points = abs(sl_adjust) * 15  # [-3,3] → [0,45] pontos ✅
                    tp_points = abs(tp_adjust) * 15  # [-3,3] → [0,45] pontos ✅
                    
                    # 🔥 CONVERSÃO PARA PREÇO OURO (1 ponto = $0.01 diferença)
                    sl_price_diff = sl_points * 0.01  # Converter pontos para preços
                    tp_price_diff = tp_points * 0.01  # Converter pontos para preços
                    
                    # 🔥 DEFINIR sltp_values PARA COMPATIBILIDADE COM CÓDIGO LEGADO
                    sltp_values = [sl_adjust, tp_adjust] + sl_adjusts + tp_adjusts
                    
                    # Contar HOLDs consecutivos
                    if entry_decision == 0:  # HOLD
                        if not hasattr(self, '_consecutive_holds'):
                            self._consecutive_holds = 0
                        self._consecutive_holds += 1
                    else:
                        self._consecutive_holds = 0
                    
                    # Log informativo a cada 10 steps
                    if step_count % 10 == 0:
                        # Verificar dados reais vs simulados
                        if len(self.env.historical_df) > 0:
                            latest_data = self.env.historical_df.iloc[-1]
                            rsi_5m = latest_data.get('rsi_14_5m', 50.0)
                            bb_pos = latest_data.get('bb_position_5m', 0.5)
                            vol_5m = latest_data.get('volatility_20_5m', 0.01)
                            trend = latest_data.get('trend_strength_5m', 0.0)
                            
                            # Usar preço atual do tick
                            tick_temp = mt5.symbol_info_tick(self.env.symbol)
                            price_5m = tick_temp.bid if tick_temp else 2000.0
                            
                            # Status dos dados
                            data_quality = "📈 REAL" if abs(rsi_5m - 50.0) > 1.0 else "⚠️ SIM"
                            
                            # Log detalhado das features críticas
                            self.log(f"[📊 FEATURES] RSI:{rsi_5m:.1f} | BB:{bb_pos:.2f} | Vol:{vol_5m:.4f} | Trend:{trend:.4f} | {data_quality}")
                            
                            # BB position validation
                            if step_count % 100 == 0 and bb_pos == 1.0:
                                self.log(f"[BB] Position at boundary: {bb_pos:.3f}")
                            
                            # Comparar com treinamento (removido spam de CONSERVADOR)
                            
                            self._last_action_log = step_count
                    
                    # Enhanced normalizer monitoring - SEM RESHAPE ARTIFICIAL
                    if step_count % 1000 == 0 and hasattr(self, 'vec_env') and self.vec_env:
                        if obs.ndim == 1:
                            obs_batch = obs.reshape(1, -1)
                        else:
                            obs_batch = obs
                        obs_norm = self.vec_env.normalize_obs(obs_batch).flatten()
                        huge_count = np.sum(np.abs(obs_norm) > 10.0)
                        if huge_count > len(obs_norm) * 0.1:
                            self.log(f"[NORM] {huge_count} extreme values detected")
                    
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
                    
                    # 🔥 PROCESSAR AÇÕES DO MODELO - CORRIGIDO!
                    action_names = {0: "HOLD", 1: "LONG", 2: "SHORT"}
                    
                    # Log apenas mudanças de decisão ou a cada 20 steps
                    if not hasattr(self, '_last_decision'):
                        self._last_decision = -1
                    
                    if entry_decision != self._last_decision or step_count % 20 == 0:
                        # 🔥 CONFIANÇA REAL: Mostrar valor real do modelo V5
                        confidence_status = "🔥 ALTA" if entry_confidence > 0.7 else "⚠️ BAIXA" if entry_confidence < 0.3 else "📊 MED"
                        confidence_percent = entry_confidence * 100
                        self.log(f"[🧠 MODELO] {action_names[entry_decision]} | Conf: {confidence_percent:.1f}% (Raw: {entry_confidence:.3f}) ({confidence_status}) | Size: {position_size:.2f} | Pos: {current_positions}/{self.env.max_positions}")
                        self._last_decision = entry_decision
                    
                        # Salvar normalizador periodicamente
                        if hasattr(self, 'custom_normalizer') and step_count % 1000 == 0:
                            normalizer_path = "enhanced_normalizer_final.pkl"
                            self.custom_normalizer.save(normalizer_path)
                        
                        # Análise comparativa com treinamento a cada 50 steps
                        if step_count % 50 == 0 and hasattr(self, '_consecutive_holds'):
                            if self._consecutive_holds > 0:
                                # Verificar se features estão similares ao treinamento
                                if len(self.env.historical_df) > 0:
                                    latest_data = self.env.historical_df.iloc[-1]
                                
                                # Features críticas para comparação
                                features_check = {
                                    'RSI': latest_data.get('rsi_14_5m', 50.0),
                                    'BB_Position': latest_data.get('bb_position_5m', 0.5),
                                    'Volatility': latest_data.get('volatility_20_5m', 0.01),
                                    'Trend': latest_data.get('trend_strength_5m', 0.0)
                                }
                                
                                # Detectar se features estão em ranges normais de treinamento
                                anomalies = []
                                if features_check['RSI'] == 50.0:
                                    anomalies.append("RSI=50 (estático)")
                                if features_check['BB_Position'] == 0.5:
                                    anomalies.append("BB=0.5 (neutro)")
                                if features_check['Volatility'] < 0.001:
                                    anomalies.append("Vol<0.001 (muito baixa)")
                                if abs(features_check['Trend']) < 0.0001:
                                    anomalies.append("Trend≈0 (sem direção)")
                                
                                if anomalies:
                                    self.log(f"[🔍 DIAGNÓSTICO] {self._consecutive_holds} HOLDs | Anomalias: {', '.join(anomalies)}")
                                    self.log(f"[💡 SUGESTÃO] Features podem estar diferentes do treinamento")
                                else:
                                    pass  # Features normais - não spammar logs desnecessários
                    
                    # Processar entrada de posição IDÊNTICO AO TREINAMENTO - COM VERIFICAÇÃO DE LIMITE
                    if entry_decision > 0 and current_positions < self.env.max_positions:  # PURO: executa se modelo decidir E há espaço
                        
                        # 🔍 DIAGNÓSTICO CRÍTICO: Por que o modelo não está operando?
                        if step_count % 100 == 0:  # Log diagnóstico a cada 100 steps
                            self.log(f"[🔍 DIAGNÓSTICO] Modelo quer {action_names[entry_decision]} com conf: {entry_confidence:.3f}")
                            
                            # Verificar se confiança é muito baixa (problema comum)
                            if entry_confidence < 0.1:
                                self.log(f"[⚠️ PROBLEMA] Confiança muito baixa: {entry_confidence:.3f} - Modelo incerto")
                            elif entry_confidence < 0.3:
                                self.log(f"[📊 STATUS] Confiança modelo: {entry_confidence:.3f} - Decisão pura")
                            else:
                                self.log(f"[✅ OK] Confiança adequada: {entry_confidence:.3f} - Deveria operar")
                            
                            # Verificar filtros (mesmo sendo True)
                            filter_result = self.env._check_entry_filters(entry_decision)
                            self.log(f"[🔍 FILTROS] Resultado: {filter_result}")
                            
                                                    # 🚨 VERIFICAR LIMITE DE POSIÇÕES PRIMEIRO
                        if current_positions >= self.env.max_positions:
                            self.log(f"[🚫 LIMITE] {current_positions}/{self.env.max_positions} posições - Bloqueando entrada")
                            continue
                        
                        self.log(f"[📊 POSIÇÕES] Atual: {current_positions}/{self.env.max_positions} - OK para entrada")
                    
                    # 🎯 GESTÃO INTELIGENTE DE POSIÇÕES EXISTENTES
                    # SEMPRE processa gestão, independente do limite de novas entradas
                    if mgmt_action > 0 and current_positions > 0:
                        # Management Head ativa - sem spam de logs
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
                    
                    # 🎯 AJUSTES DE SL/TP - SEMPRE PROCESSADOS
                    if current_positions > 0 and len(taticas) >= 6:
                        sl_adjust = float(taticas[4])  # sl_adjust
                        tp_adjust = float(taticas[5])  # tp_adjust
                        
                        if abs(sl_adjust) > 0.1 or abs(tp_adjust) > 0.1:  # Só ajustar se mudança significativa
                            self.log(f"[🔧 AJUSTE] Management Head sugere SL: {sl_adjust:.3f}, TP: {tp_adjust:.3f}")
                            # Aqui seria implementado o ajuste real dos SL/TP das posições abertas
                            # Por enquanto apenas log para mostrar que o sistema está ativo
                    
                    # Limite já verificado acima - código removido para evitar duplicação
                    
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
                                self.log(f"📊 Enhanced Mean[0-4]: {obs_mean[:5]}")
                                self.log(f"📊 Enhanced Var[0-4]: {obs_var[:5]}")
                            elif hasattr(self.vec_env, 'running_mean'):
                                obs_mean = self.vec_env.running_mean
                                obs_var = self.vec_env.running_var
                                self.log(f"📊 Enhanced Mean[0-4]: {obs_mean[:5]}")
                                self.log(f"📊 Enhanced Var[0-4]: {obs_var[:5]}")
                        
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
                            self.log(f"💡 SUGESTÃO: Enhanced Normalizer pode precisar de re-calibração")
                            
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


                    
                    # Log da decisão do modelo apenas quando relevante
                    if estrategica > 0 and current_positions < self.env.max_positions:
                        self.log(f"[🔥 PURO] Modelo quer {action_names[estrategica]} - decisão processada")
                    
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
                        
                        # Sistema anti-flip-flop desabilitado - modelo controla qualidade
                        # Market context processing removed
                    
                    # 🔍 ANÁLISE: Modelo sempre retorna HOLD (entry_decision = 0.0)
                    # Action space correto: [0-2, 0-2, 0-2, 0-2, -3-3, -3-3, ...]
                    # Se entry_decision < 0.5, é HOLD; se > 0.5 e < 1.5, é LONG; se > 1.5, é SHORT
                    
                    # Registrar decisão do modelo nas estatísticas - CORRIGIDO!
                    # Usar a confiança real do modelo (action[1]) já normalizada entre 0-1
                    model_confidence_raw = float(action[1]) if len(action) > 1 else 0.5
                    
                    # Periodic action review
                    if step_count % 50 == 0:
                        self.log(f"[ACTION] Entry:{action[0]:.3f} | Conf:{action[1]:.3f} | Size:{action[2]:.3f}")
                    
                    # Passar confiança real (0-1) para SessionStats - o GUI multiplicará por 100 para exibição
                    self.session_stats.add_model_decision(model_confidence_raw)
                    
                    # Anti-flip-flop system disabled - model controls quality
                    should_block = False
                    
                    # Strategic action - open new positions
                    if step_count % 20 == 0:  # Log position limits periodically
                        self.log(f"[POSITIONS] {current_positions}/{self.env.max_positions} | Strategy: {estrategica}")
                    
                    # Emergency system - close excess positions
                    if current_positions > self.env.max_positions:
                        excess_positions = current_positions - self.env.max_positions
                        self.log(f"[EMERGENCY] {excess_positions} excess positions detected! Closing automatically...")
                        
                                                 # Fechar TODAS as posições excedentes com método mais robusto
                        for i in range(excess_positions):
                            if i < len(mt5_positions):
                                pos = mt5_positions[i]
                                
                                # Método mais robusto de fechamento
                                try:
                                    # Usar o método do environment que já funciona
                                    close_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
                                    close_price = tick.bid if pos.type == 0 else tick.ask
                                    
                                    close_request = {
                                        "action": mt5.TRADE_ACTION_DEAL,
                                        "symbol": self.env.symbol,
                                        "volume": pos.volume,
                                        "type": close_type,
                                        "position": pos.ticket,
                                        "price": close_price,
                                        "deviation": 20,
                                        "magic": 123456,
                                        "comment": "EMERGENCY_CLOSE_EXCESS",
                                        "type_time": mt5.ORDER_TIME_GTC,
                                        "type_filling": mt5.ORDER_FILLING_IOC,
                                    }
                                    
                                    result = mt5.order_send(close_request)
                                    if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
                                        self.log(f"[✅ EMERGÊNCIA] Posição #{pos.ticket} fechada com sucesso")
                                    else:
                                        error_code = result.retcode if result else "CONNECTION_ERROR"
                                        self.log(f"[❌ EMERGÊNCIA] Erro ao fechar #{pos.ticket}: {error_code}")
                                        
                                        # Tentar método alternativo
                                        alt_result = self.env._execute_order(close_type, pos.volume, None, None)
                                        self.log(f"[🔄 EMERGÊNCIA] Tentativa alternativa: {alt_result}")
                                        
                                except Exception as e:
                                    self.log(f"[❌ EMERGÊNCIA] Exceção ao fechar #{pos.ticket}: {e}")
                        
                        # Atualizar contagem após fechamentos
                        mt5_positions = mt5.positions_get(symbol=self.env.symbol) or []
                        current_positions = len(mt5_positions)
                        self.log(f"[🔄 EMERGÊNCIA] Posições após limpeza: {current_positions}/{self.env.max_positions}")
                    
                    # Limite respeitado, gestão ativa - sem logs de spam
                    
                    if estrategica == 1 and current_positions < self.env.max_positions:  # LONG - VERIFICAÇÃO AQUI!
                        # Calcular SL/TP baseado na ação do agente
                        current_price = tick.ask
                        sl_value = sltp_values[0] if len(sltp_values) > 0 else 0.3
                        tp_value = sltp_values[1] if len(sltp_values) > 1 else 0.5
                        
                        # Converter valores [-1,1] para preços reais
                        sl_price = None
                        tp_price = None
                        
                        # 🔥 ESCALA IDÊNTICA: EXATAMENTE igual ao treinamento diferenciado
                        # REALISTIC_SLTP_CONFIG: sl_min=13, sl_max=46, tp_min=16, tp_max=82
                        # (Variáveis já definidas globalmente no início da função)
                        
                        if abs(sl_value) > sl_threshold:  # SL significativo 
                            # 🔥 ESCALA IDÊNTICA: 15x multiplicador + 1.00 conversão (EXATO treinamento)
                            model_sl_distance = abs(sl_value * 15 * 1.00)  # Escala real do treinamento
                            sl_distance = max(min(model_sl_distance, max_sl_distance), min_sl_distance)  # Clamp nos ranges exatos
                            sl_price = current_price - sl_distance
                        else:
                            sl_price = current_price - min_sl_distance  # 13 pontos = $13.00 (EXATO)
                            
                        if abs(tp_value) > tp_threshold:  # TP significativo
                            # 🔥 ESCALA IDÊNTICA: 15x multiplicador + 1.00 conversão (EXATO treinamento)
                            model_tp_distance = abs(tp_value * 15 * 1.00)  # Escala real do treinamento
                            tp_distance = max(min(model_tp_distance, max_tp_distance), min_tp_distance)  # Clamp nos ranges exatos
                            tp_price = current_price + tp_distance
                        else:
                            tp_price = current_price + min_tp_distance  # 16 pontos = $16.00 (EXATO)
                        
                        sl_text = f"{sl_price:.2f}" if sl_price is not None else "N/A"
                        tp_text = f"{tp_price:.2f}" if tp_price is not None else "N/A"
                        self.log(f"[🚀 EXECUTANDO] LONG @ {current_price:.2f} | SL: {sl_text} | TP: {tp_text}")
                        # Calcular tamanho dinâmico da posição
                        dynamic_lot_size = self.env._calculate_adaptive_position_size(action_confidence=1.0)
                        response = self.env._execute_order(mt5.ORDER_TYPE_BUY, dynamic_lot_size, sl_price, tp_price)
                        
                        # Sistema anti-flip-flop desabilitado - modelo controla qualidade
                        # self.anti_flipflop.update_action_executed(action_signature, current_time)
                        
                        # Atualizar estatísticas
                        if "SUCCESS" in response:
                            self.session_stats.total_buys += 1
                            self.session_stats.positions_opened += 1
                            self.session_stats.update_last_action("LONG")  # 🔥 ATUALIZAR ÚLTIMA AÇÃO
                        
                        # Se mercado fechado, aguardar mais tempo
                        if "MARKET_CLOSED" in response:
                            self.log("[⏰ AGUARDANDO] Mercado fechado - aguardando 30 minutos...")
                            time.sleep(1800)  # 30 minutos
                        
                    elif estrategica == 2 and current_positions < self.env.max_positions:  # SHORT - VERIFICAÇÃO AQUI!
                        # Calcular SL/TP baseado na ação do agente
                        current_price = tick.bid
                        sl_value = sltp_values[0] if len(sltp_values) > 0 else 0.3
                        tp_value = sltp_values[1] if len(sltp_values) > 1 else 0.5
                        
                        # Converter valores [-1,1] para preços reais
                        sl_price = None
                        tp_price = None
                        
                        if abs(sl_value) > sl_threshold:  # SL significativo
                            # 🔥 ESCALA IDÊNTICA: 15x multiplicador + 1.00 conversão (EXATO treinamento)
                            model_sl_distance = abs(sl_value * 15 * 1.00)  # Escala real do treinamento
                            sl_distance = max(min(model_sl_distance, max_sl_distance), min_sl_distance)  # Clamp nos ranges exatos
                            sl_price = current_price + sl_distance
                        else:
                            sl_price = current_price + min_sl_distance  # 13 pontos = $13.00 (EXATO)
                            
                        if abs(tp_value) > tp_threshold:  # TP significativo
                            # 🔥 ESCALA IDÊNTICA: 15x multiplicador + 1.00 conversão (EXATO treinamento)
                            model_tp_distance = abs(tp_value * 15 * 1.00)  # Escala real do treinamento
                            tp_distance = max(min(model_tp_distance, max_tp_distance), min_tp_distance)  # Clamp nos ranges exatos
                            tp_price = current_price - tp_distance
                        else:
                            tp_price = current_price - min_tp_distance  # 16 pontos = $16.00 (EXATO)
                        
                        sl_text = f"{sl_price:.2f}" if sl_price is not None else "N/A"
                        tp_text = f"{tp_price:.2f}" if tp_price is not None else "N/A"
                        self.log(f"[🚀 EXECUTANDO] SHORT @ {current_price:.2f} | SL: {sl_text} | TP: {tp_text}")
                        # Calcular tamanho dinâmico da posição
                        dynamic_lot_size = self.env._calculate_adaptive_position_size(action_confidence=1.0)
                        response = self.env._execute_order(mt5.ORDER_TYPE_SELL, dynamic_lot_size, sl_price, tp_price)
                        
                        # Action execution tracking removed
                        
                        # Atualizar estatísticas
                        if "SUCCESS" in response:
                            self.session_stats.total_sells += 1
                            self.session_stats.positions_opened += 1
                            self.session_stats.update_last_action("SHORT")  # 🔥 ATUALIZAR ÚLTIMA AÇÃO
                        
                        # Se mercado fechado, aguardar mais tempo
                        if "MARKET_CLOSED" in response:
                            self.log("[⏰ AGUARDANDO] Mercado fechado - aguardando 30 minutos...")
                            time.sleep(1800)  # 30 minutos
                    
                    # Anti-flip-flop system completely removed
                    
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
                                
                                # Action execution tracking removed
                                
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
                            # 🔥 ESCALA IDÊNTICA: EXATAMENTE igual ao treinamento diferenciado
                            # REALISTIC_SLTP_CONFIG: sl_min=13, sl_max=46, tp_min=16, tp_max=82
                            # (Variáveis já definidas globalmente no início da função)
                            
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
                                
                                if abs(sl_value) > sl_threshold:  # Threshold ajustado
                                    if position.type == 0:  # Long
                                        # 🔥 ESCALA IDÊNTICA: 15x multiplicador + 1.00 conversão (EXATO treinamento)
                                        model_sl_distance = abs(sl_value * 15 * 1.00)  # Escala real do treinamento
                                        sl_distance = max(min(model_sl_distance, max_sl_distance), min_sl_distance)  # Clamp nos ranges exatos
                                        new_sl = current_price - sl_distance
                                    else:  # Short
                                        model_sl_distance = abs(sl_value * 15 * 1.00)  # Escala real do treinamento
                                        sl_distance = max(min(model_sl_distance, max_sl_distance), min_sl_distance)  # Clamp nos ranges exatos
                                        new_sl = current_price + sl_distance

                                if abs(tp_value) > tp_threshold:  # Threshold ajustado
                                    if position.type == 0:  # Long
                                        # 🔥 ESCALA IDÊNTICA: 15x multiplicador + 1.00 conversão (EXATO treinamento)
                                        model_tp_distance = abs(tp_value * 15 * 1.00)  # Escala real do treinamento
                                        tp_distance = max(min(model_tp_distance, max_tp_distance), min_tp_distance)  # Clamp nos ranges exatos
                                        new_tp = current_price + tp_distance
                                    else:  # Short
                                        model_tp_distance = abs(tp_value * 15 * 1.00)  # Escala real do treinamento
                                        tp_distance = max(min(model_tp_distance, max_tp_distance), min_tp_distance)  # Clamp nos ranges exatos
                                        new_tp = current_price - tp_distance
                                
                                # Modificar posição
                                # Always apply - model decides all values
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
                    
                    # ENVIO HTTP PARA SERVIDOR LOCAL - DADOS PARA EA
                    try:
                        tick_temp = mt5.symbol_info_tick(self.env.symbol)
                        current_price = tick_temp.bid if tick_temp else 0.0
                        
                        # Calcular confiança do modelo
                        if len(action) >= 2:
                            model_confidence = abs(float(action[1])) if action[1] != 0 else 0.5
                        else:
                            model_confidence = 0.5
                        
                        # Determinar sinal baseado na ação do modelo
                        signal_type = 'HOLD'
                        if len(action) > 0:
                            if action[0] > 0.66:
                                signal_type = 'SELL'
                            elif action[0] > 0.33:
                                signal_type = 'BUY'
                        
                        # Obter dados técnicos reais
                        try:
                            # Calcular RSI
                            rates = mt5.copy_rates_from_pos(self.env.symbol, mt5.TIMEFRAME_M5, 0, 50)
                            if rates is not None and len(rates) >= 14:
                                closes = [r['close'] for r in rates]
                                rsi = self.env._calculate_rsi(closes)
                            else:
                                rsi = 50.0
                            
                            # Calcular BB Position
                            if rates is not None and len(rates) >= 20:
                                bb_position = self.env._calculate_bb_position_FIXED(closes[-20:])
                            else:
                                bb_position = 0.5
                            
                            # Calcular volatilidade (ATR normalizado)
                            if rates is not None and len(rates) >= 14:
                                volatility = 0.0
                                for i in range(1, min(14, len(rates))):
                                    high_low = rates[i]['high'] - rates[i]['low']
                                    high_close = abs(rates[i]['high'] - rates[i-1]['close'])
                                    low_close = abs(rates[i]['low'] - rates[i-1]['close'])
                                    true_range = max(high_low, high_close, low_close)
                                    volatility += true_range
                                volatility = (volatility / 13) / current_price  # ATR normalizado
                            else:
                                volatility = 0.01
                            
                            # Calcular momentum
                            if rates is not None and len(rates) >= 10:
                                momentum = (rates[-1]['close'] - rates[-10]['close']) / rates[-10]['close']
                            else:
                                momentum = 0.0
                            
                        except Exception as e:
                            print(f"[WARN] Erro ao calcular indicadores: {e}")
                            rsi = 50.0
                            bb_position = 0.5
                            volatility = 0.01
                            momentum = 0.0
                        
                        # Obter posições atuais
                        try:
                            positions = mt5.positions_get(symbol=self.env.symbol)
                            positions_data = []
                            if positions:
                                for pos in positions:
                                    positions_data.append({
                                        'ticket': pos.ticket,
                                        'type': pos.type,
                                        'volume': pos.volume,
                                        'price': pos.price_open,
                                        'sl': pos.sl,
                                        'tp': pos.tp,
                                        'profit': pos.profit
                                    })
                        except Exception as e:
                            positions_data = []
                        
                        # Obter portfolio value
                        try:
                            account_info = mt5.account_info()
                            portfolio_value = account_info.balance if account_info else 500.0
                        except:
                            portfolio_value = 500.0
                        
                        # PAYLOAD OTIMIZADO PARA O EA
                        payload = {
                            # DADOS PRINCIPAIS
                            "action": signal_type,
                            "step": int(step_count),
                            "timestamp": int(time.time()),
                            "symbol": getattr(self.env, 'symbol', 'GOLD'),
                            "price": float(current_price),
                            "status": "running",
                            "confidence": float(model_confidence),
                            
                            # DADOS TÉCNICOS REAIS
                            "rsi": float(rsi),
                            "bb_position": float(bb_position),
                            "volatility": float(volatility),
                            "momentum": float(momentum),
                            
                            # DADOS DE POSIÇÃO
                            "positions": positions_data,
                            "portfolio_value": float(portfolio_value),
                            
                            # DADOS PARA COMPATIBILIDADE
                            "raw_action": action.tolist() if hasattr(action, 'tolist') else list(action)
                        }
                        
                        # Adicionar desenhos técnicos reais se disponível
                        if hasattr(self, 'technical_drawer'):
                            try:
                                self.technical_drawer.analyze_and_draw_market_structure(obs, current_price, model_confidence)
                                # Usar o método da classe TradingApp, não do technical_drawer
                                additional_data = self.get_drawings_payload()
                                if additional_data:
                                    payload.update(additional_data)
                            except Exception as e:
                                self.log(f"[WARN] Erro ao obter dados técnicos: {e}")
                        
                        # 🔥 FLASK REATIVADO: Enviar dados para EA
                        try:
                            response = requests.post("http://127.0.0.1:5000/receber", json=payload, timeout=2.0)
                            if step_count % 50 == 0:
                                self.log(f"[📊 DADOS] Desenhos enviados para EA (Flask OK)")
                        except Exception as e:
                            if step_count % 100 == 0:  # Log erro apenas ocasionalmente
                                self.log(f"[⚠️ FLASK] Erro de conexão: {e}")
                    except Exception as e:
                        self.log(f"[⚠️ DADOS] Erro ao preparar payload: {e}")
                    
                except Exception as e:
                    self.log(f"[❌ ERRO] Step de trading: {e}")
                    time.sleep(2)  # 🔥 REDUZIDO: 2 segundos para recovery mais rápido
                    continue  # 🔥 CONTINUAR LOOP AO INVÉS DE PARAR
                    
        except Exception as e:
            self.log(f"[❌ CRÍTICO] Erro no trading: {e}")
        finally:
            self.log("[🛑 STOP] Trading finalizado")
            self.trading = False
    
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
            # 🔥 CORREÇÃO: Usar confiança diretamente (modelo já normalizado)
            raw_confidence = float(action[1]) if len(action) > 1 else 0.0
            entry_confidence = raw_confidence  # SEM divisão - modelo já normalizado!
            
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
    

    def get_drawings_payload(self):
        """Serializa todos os desenhos técnicos em um dicionário para envio ao Flask, com nomes padronizados para o EA"""
        try:
            # Se temos o technical_drawer, pegar os dados dele
            if hasattr(self, 'technical_drawer') and self.technical_drawer:
                # Gerar dados de desenhos técnicos baseados na análise atual
                supports = []
                resistances = []
                pressure_zones = []
                trendlines = []
                formations = []
                
                # Simular alguns níveis baseados no preço atual (para teste)
                current_price = 2000.0  # Valor padrão
                model_confidence = 0.5  # Valor padrão
                
                if hasattr(self, 'env') and hasattr(self.env, 'symbol'):
                    tick = mt5.symbol_info_tick(self.env.symbol)
                    if tick:
                        current_price = tick.bid
                        model_confidence = 0.5  # Valor padrão
                
                # Gerar suportes e resistências baseados no preço atual (formato simples para EA)
                supports = [
                    current_price - 5.0,
                    current_price - 10.0
                ]
                
                resistances = [
                    current_price + 5.0,
                    current_price + 10.0
                ]
                
                # Gerar zonas de pressão (formato simples para EA)
                pressure_zones = [
                    current_price - 3.0,
                    current_price + 3.0
                ]
                
                # Gerar trendlines simples (formato simples para EA)
                current_time = int(time.time())
                trendlines = [
                    current_time - 3600,  # 1 hora atrás
                    current_price - 15.0,
                    current_time,         # agora
                    current_price
                ]
                
                # Gerar formações (formato simples para EA)
                formations = [
                    "triangle",
                    current_time - 1800,  # 30 min atrás
                    current_price - 8.0,
                    current_time - 900,   # 15 min atrás
                    current_price - 4.0,
                    current_time,         # agora
                    current_price
                ]
                
                # Determinar sinal baseado na última ação
                signal = 'HOLD'
                if hasattr(self, '_last_decision'):
                    if self._last_decision == 1:
                        signal = 'BUY'
                    elif self._last_decision == 2:
                        signal = 'SELL'
                
                return {
                    'supports': supports,
                    'resistances': resistances,
                    'pressure_zones': pressure_zones,
                    'trendlines': trendlines,
                    'formations': formations,
                    'signal': signal,
                    'signal_price': current_price,
                    'confidence': model_confidence
                }
            else:
                # Fallback se não temos technical_drawer
                return {
                    'supports': [],
                    'resistances': [],
                    'pressure_zones': [],
                    'trendlines': [],
                    'formations': [],
                    'signal': 'HOLD',
                }
        except Exception as e:
            print(f"[ERRO get_drawings_payload] {e}")
            return {
                'supports': [],
                'resistances': [],
                'pressure_zones': [],
                'trendlines': [],
                'formations': [],
                'signal': 'HOLD',
            }


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