import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional, List
import logging
from datetime import datetime

class MarketRegimeDetector:
    """
    🧠 DETECTOR INTELIGENTE DE REGIME DE MERCADO
    Identifica automaticamente: LATERAL, TENDÊNCIA ALTA, TENDÊNCIA BAIXA
    E adapta estratégias: Topos/Fundos vs Seguir Tendência
    """
    
    def __init__(self):
        # 🎯 CONFIGURAÇÕES DE DETECÇÃO
        self.lookback_periods = 50  # Períodos para análise
        self.trend_threshold = 0.015  # 1.5% para considerar tendência forte
        self.sideways_threshold = 0.008  # 0.8% para mercado lateral
        self.volatility_periods = 20
        
        # 📊 HISTÓRICO DE REGIMES
        self.regime_history = []
        self.current_regime = "SIDEWAYS"  # SIDEWAYS, UPTREND, DOWNTREND
        self.regime_strength = 0.0  # 0-1, força do regime
        self.regime_duration = 0  # Há quantos steps no regime atual
        
        # 🎯 NÍVEIS DE SUPORTE/RESISTÊNCIA (para lateral)
        self.support_levels = []
        self.resistance_levels = []
        
        # 📈 MÉTRICAS DE TENDÊNCIA
        self.trend_slope = 0.0
        self.trend_r_squared = 0.0  # Qualidade da tendência linear
        self.momentum_score = 0.0
        
    def detect_market_regime(self, env) -> Dict[str, Any]:
        """🔍 DETECÇÃO PRINCIPAL DO REGIME DE MERCADO"""
        try:
            if not hasattr(env, 'df') or not hasattr(env, 'current_step'):
                return self._default_regime()
            
            current_idx = min(env.current_step, len(env.df) - 1)
            if current_idx < self.lookback_periods:
                return self._default_regime()
            
            # 📊 EXTRAIR DADOS HISTÓRICOS
            start_idx = max(0, current_idx - self.lookback_periods)
            price_data = env.df['close_5m'].iloc[start_idx:current_idx+1]
            
            # 1. 📈 ANÁLISE DE TENDÊNCIA LINEAR
            trend_analysis = self._analyze_trend_strength(price_data)
            
            # 2. 🌊 ANÁLISE DE VOLATILIDADE E RANGE
            volatility_analysis = self._analyze_volatility_pattern(price_data)
            
            # 3. 🎯 DETECÇÃO DE SUPORTE/RESISTÊNCIA
            sr_analysis = self._detect_support_resistance(price_data)
            
            # 4. 📊 ANÁLISE DE MOMENTUM
            momentum_analysis = self._analyze_momentum_pattern(env, current_idx)
            
            # 5. 🧠 DECISÃO FINAL DO REGIME
            regime_decision = self._determine_regime(
                trend_analysis, volatility_analysis, sr_analysis, momentum_analysis
            )
            
            return regime_decision
            
        except Exception as e:
            return self._default_regime(error=str(e))
    
    def _analyze_trend_strength(self, prices) -> Dict[str, float]:
        """📈 Análise de força da tendência usando regressão linear"""
        try:
            x = np.arange(len(prices))
            y = prices.values
            
            # Regressão linear simples
            if len(x) > 1:
                slope = np.polyfit(x, y, 1)[0]
                
                # Calcular R² manualmente
                y_mean = np.mean(y)
                ss_tot = np.sum((y - y_mean) ** 2)
                y_pred = slope * x + np.mean(y)
                ss_res = np.sum((y - y_pred) ** 2)
                r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                
                # Calcular mudança percentual total
                total_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]
                
                # Força da tendência baseada em R² e mudança percentual
                trend_strength = abs(r_squared) * abs(total_change) * 10
                
                self.trend_slope = slope
                self.trend_r_squared = r_squared
                
                return {
                    'slope': slope,
                    'r_squared': r_squared,
                    'total_change': total_change,
                    'trend_strength': trend_strength
                }
            
        except Exception:
            pass
            
        return {'slope': 0, 'r_squared': 0, 'total_change': 0, 'trend_strength': 0}
    
    def _analyze_volatility_pattern(self, prices) -> Dict[str, float]:
        """🌊 Análise do padrão de volatilidade"""
        try:
            returns = prices.pct_change().dropna()
            
            if len(returns) > self.volatility_periods:
                recent_returns = returns.iloc[-self.volatility_periods:]
                current_vol = recent_returns.std()
                avg_vol = returns.std()
                vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
            else:
                current_vol = returns.std() if len(returns) > 0 else 0
                avg_vol = current_vol
                vol_ratio = 1
            
            price_range = (prices.max() - prices.min()) / prices.mean() if prices.mean() > 0 else 0
            
            return {
                'current_volatility': current_vol,
                'avg_volatility': avg_vol,
                'volatility_ratio': vol_ratio,
                'price_range': price_range
            }
            
        except Exception:
            return {'current_volatility': 0, 'avg_volatility': 0, 'volatility_ratio': 1, 'price_range': 0}
    
    def _detect_support_resistance(self, prices) -> Dict[str, Any]:
        """🎯 Detecção inteligente de suporte e resistência"""
        try:
            highs = []
            lows = []
            
            for i in range(2, len(prices) - 2):
                # Máximo local
                if (prices.iloc[i] > prices.iloc[i-1] and prices.iloc[i] > prices.iloc[i-2] and
                    prices.iloc[i] > prices.iloc[i+1] and prices.iloc[i] > prices.iloc[i+2]):
                    highs.append(prices.iloc[i])
                
                # Mínimo local
                if (prices.iloc[i] < prices.iloc[i-1] and prices.iloc[i] < prices.iloc[i-2] and
                    prices.iloc[i] < prices.iloc[i+1] and prices.iloc[i] < prices.iloc[i+2]):
                    lows.append(prices.iloc[i])
            
            resistance_level = np.mean(highs) if highs else prices.max()
            support_level = np.mean(lows) if lows else prices.min()
            
            self.resistance_levels = highs[-5:] if len(highs) >= 5 else highs
            self.support_levels = lows[-5:] if len(lows) >= 5 else lows
            
            sr_range = (resistance_level - support_level) / prices.mean() if prices.mean() > 0 else 0
            
            return {
                'resistance_level': resistance_level,
                'support_level': support_level,
                'sr_range': sr_range,
                'num_highs': len(highs),
                'num_lows': len(lows)
            }
            
        except Exception:
            return {'resistance_level': 0, 'support_level': 0, 'sr_range': 0, 'num_highs': 0, 'num_lows': 0}
    
    def _analyze_momentum_pattern(self, env, current_idx: int) -> Dict[str, float]:
        """📊 Análise do padrão de momentum"""
        try:
            momentum_score = 0.0
            
            # RSI momentum
            if 'rsi_14' in env.df.columns:
                current_rsi = env.df['rsi_14'].iloc[current_idx]
                if current_rsi > 70:
                    momentum_score += 0.3
                elif current_rsi < 30:
                    momentum_score -= 0.3
            
            # Moving Average momentum
            if 'sma_20' in env.df.columns and 'sma_50' in env.df.columns:
                sma_20 = env.df['sma_20'].iloc[current_idx]
                sma_50 = env.df['sma_50'].iloc[current_idx]
                current_price = env.df['close_5m'].iloc[current_idx]
                
                if current_price > sma_20 > sma_50:
                    momentum_score += 0.4
                elif current_price < sma_20 < sma_50:
                    momentum_score -= 0.4
            
            self.momentum_score = momentum_score
            
            return {
                'momentum_score': momentum_score,
                'momentum_direction': 'UP' if momentum_score > 0.2 else 'DOWN' if momentum_score < -0.2 else 'NEUTRAL'
            }
            
        except Exception:
            return {'momentum_score': 0, 'momentum_direction': 'NEUTRAL'}
    
    def _determine_regime(self, trend_analysis, volatility_analysis, sr_analysis, momentum_analysis) -> Dict[str, Any]:
        """🧠 DECISÃO FINAL DO REGIME DE MERCADO"""
        
        trend_strength = trend_analysis['trend_strength']
        total_change = trend_analysis['total_change']
        r_squared = trend_analysis['r_squared']
        sr_range = sr_analysis['sr_range']
        momentum_score = momentum_analysis['momentum_score']
        
        # 🎯 LÓGICA DE DECISÃO INTELIGENTE
        regime = "SIDEWAYS"
        confidence = 0.5
        strategy = "RANGE_TRADE"
        
        # 1. TENDÊNCIA FORTE
        if (r_squared > 0.7 and abs(total_change) > self.trend_threshold and 
            abs(momentum_score) > 0.3):
            
            if total_change > 0 and momentum_score > 0:
                regime = "UPTREND"
                strategy = "TREND_FOLLOW"
                confidence = min(0.9, r_squared + abs(momentum_score))
            elif total_change < 0 and momentum_score < 0:
                regime = "DOWNTREND" 
                strategy = "TREND_FOLLOW"
                confidence = min(0.9, r_squared + abs(momentum_score))
        
        # 2. MERCADO LATERAL
        elif (abs(total_change) < self.sideways_threshold and sr_range > 0.01 and
              r_squared < 0.5):
            regime = "SIDEWAYS"
            strategy = "RANGE_TRADE"
            confidence = 0.8 - r_squared
        
        # 3. BREAKOUT POTENCIAL
        elif (volatility_analysis['volatility_ratio'] > 1.5 and 
              abs(momentum_score) > 0.4):
            regime = "BREAKOUT"
            strategy = "BREAKOUT"
            confidence = min(0.8, abs(momentum_score))
        
        # Atualizar estado interno
        if regime != self.current_regime:
            self.regime_duration = 0
            self.current_regime = regime
        else:
            self.regime_duration += 1
        
        self.regime_strength = confidence
        
        return {
            'regime': regime,
            'strategy': strategy,
            'confidence': confidence,
            'duration': self.regime_duration,
            'trend_strength': trend_strength,
            'momentum_score': momentum_score,
            'support_level': sr_analysis['support_level'],
            'resistance_level': sr_analysis['resistance_level']
        }
    
    def _default_regime(self, error=None):
        """🔄 Regime padrão em caso de erro"""
        return {
            'regime': 'SIDEWAYS',
            'strategy': 'RANGE_TRADE', 
            'confidence': 0.5,
            'duration': 0,
            'trend_strength': 0,
            'momentum_score': 0,
            'support_level': 0,
            'resistance_level': 0,
            'error': error
        }
    
    def get_trading_signals(self, env, current_price: float, regime_info: Dict) -> Dict[str, Any]:
        """🎯 SINAIS DE TRADING BASEADOS NO REGIME DETECTADO"""
        signals = {
            'entry_signal': 'HOLD',
            'confidence': 0.5,
            'strategy_type': regime_info['strategy'],
            'target_sl': 20,
            'target_tp': 30,
            'position_size_multiplier': 1.0,
            'reasoning': ''
        }
        
        try:
            regime = regime_info['regime']
            strategy = regime_info['strategy']
            support = regime_info.get('support_level', current_price * 0.99)
            resistance = regime_info.get('resistance_level', current_price * 1.01)
            
            if strategy == "RANGE_TRADE":
                # 🎯 ESTRATÉGIA LATERAL: Comprar suporte, vender resistência
                distance_to_support = abs(current_price - support) / current_price
                distance_to_resistance = abs(current_price - resistance) / current_price
                
                if distance_to_support < 0.003:  # Próximo ao suporte
                    signals['entry_signal'] = 'BUY'
                    signals['confidence'] = 0.8
                    signals['target_sl'] = 15
                    signals['target_tp'] = int((resistance - current_price) * 10000 * 0.7)
                    signals['reasoning'] = f'Range: Comprando suporte {support:.2f}'
                    
                elif distance_to_resistance < 0.003:  # Próximo à resistência
                    signals['entry_signal'] = 'SELL'
                    signals['confidence'] = 0.8
                    signals['target_sl'] = 15
                    signals['target_tp'] = int((current_price - support) * 10000 * 0.7)
                    signals['reasoning'] = f'Range: Vendendo resistência {resistance:.2f}'
            
            elif strategy == "TREND_FOLLOW":
                # 🚀 ESTRATÉGIA DE TENDÊNCIA
                momentum = regime_info.get('momentum_score', 0)
                
                if regime == "UPTREND" and momentum > 0.2:
                    signals['entry_signal'] = 'BUY'
                    signals['confidence'] = min(0.9, regime_info['confidence'] + 0.1)
                    signals['target_sl'] = 25
                    signals['target_tp'] = 50
                    signals['position_size_multiplier'] = 1.2
                    signals['reasoning'] = f'Trend: Seguindo uptrend (momentum: {momentum:.2f})'
                    
                elif regime == "DOWNTREND" and momentum < -0.2:
                    signals['entry_signal'] = 'SELL'
                    signals['confidence'] = min(0.9, regime_info['confidence'] + 0.1)
                    signals['target_sl'] = 25
                    signals['target_tp'] = 50
                    signals['position_size_multiplier'] = 1.2
                    signals['reasoning'] = f'Trend: Seguindo downtrend (momentum: {momentum:.2f})'
            
            elif strategy == "BREAKOUT":
                # 💥 ESTRATÉGIA DE BREAKOUT
                momentum = regime_info.get('momentum_score', 0)
                
                if abs(momentum) > 0.4:
                    signals['entry_signal'] = 'BUY' if momentum > 0 else 'SELL'
                    signals['confidence'] = 0.75
                    signals['target_sl'] = 20
                    signals['target_tp'] = 60
                    signals['position_size_multiplier'] = 0.8
                    signals['reasoning'] = f'Breakout: Rompimento detectado (momentum: {momentum:.2f})'
            
        except Exception as e:
            signals['reasoning'] = f'Erro na geração de sinais: {str(e)}'
        
        return signals

class SimpleRewardCalculatorWithExecution:
    """
    🔥 SISTEMA COMPLETO DE REWARDS + MÉTRICAS DE TRADING
    Centraliza toda lógica de trading, métricas e recompensas em um só lugar
    """
    
    def __init__(self, initial_balance: float = 500.0):
        self.logger = logging.getLogger(__name__)
        
        # 💰 CONFIGURAÇÕES DO PORTFÓLIO
        self.initial_balance = initial_balance
        self.initial_portfolio = initial_balance
        self.base_lot_size = 0.02
        self.max_lot_size = 0.03
        
        # 📊 ESTADO COMPLETO DO TRADING
        self.positions = {}  # Posições abertas
        self.trades = []     # Histórico completo de trades
        self.trade_count = 0
        
        # 💰 MÉTRICAS DE PORTFOLIO
        self.portfolio_value = initial_balance
        self.realized_balance = initial_balance
        self.unrealized_pnl = 0.0
        self.peak_portfolio = initial_balance
        self.peak_portfolio_value = initial_balance
        self.current_drawdown = 0.0
        self.peak_drawdown = 0.0
        
        # 📈 MÉTRICAS DE PERFORMANCE
        self.total_pnl = 0.0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_trades = 0
        self.largest_win = 0.0
        self.largest_loss = 0.0
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        
        # ⏰ MÉTRICAS TEMPORAIS
        self.daily_pnl = {}
        self.daily_trades = {}  # Trades por dia
        self.last_trade_timestamp = None
        self.trading_days = 0
        self.episode_trades = []  # Trades do episódio atual
        self.episode_drawdowns = []  # Drawdowns do episódio
        
        # 📊 MÉTRICAS AVANÇADAS
        self.recent_trades_100d = []  # Últimos 100 dias de trades
        self.recent_pnl_100d = []    # Últimos 100 dias de PnL
        
        # 🔧 CONFIGURAÇÕES DE TRADE
        self.spread = 0.00020  # 2 pips spread
        self.commission = 0.7  # $0.7 por lote
        
        # 🧠 NOVO: DETECTOR DE REGIME DE MERCADO
        self.market_regime_detector = MarketRegimeDetector()
        self.adaptive_trading_enabled = True  # Pode ser desabilitado se necessário
        
        # 🔧 PESOS BALANCEADOS PARA CONVERGÊNCIA ESTÁVEL
        self.weights = {
            # 💰 MONETIZAÇÃO DIRETA - BALANCEADA (35% do peso)
            "pnl_direct": 1.0,                    # 🔧 REDUZIDO 15x: De 15.0 para 1.0
            "win_bonus": 2.0,                     # 🔧 REDUZIDO 10x: De 20.0 para 2.0
            "loss_penalty": -1.0,                 # 🔧 AUMENTADO 2x: Mais feedback de perda
            "excellent_trade_bonus": 3.0,        # 🔧 REDUZIDO 10x: De 30.0 para 3.0
            "quality_trade_bonus": 2.0,          # 🔧 REDUZIDO 7.5x: De 15.0 para 2.0
            
            # 🎯 CONSEQUÊNCIAS FUTURAS - BALANCEADAS (30% do peso)
            "future_consequence_bonus": 2.5,     # 🔧 AUMENTADO: De 2.0 para 2.5
            "pattern_recognition_bonus": 2.0,    # 🔧 AUMENTADO: De 1.5 para 2.0
            "timing_mastery_bonus": 3.0,         # 🔧 AUMENTADO: De 2.5 para 3.0
            "patience_wisdom_bonus": 2.0,        # 🔧 AUMENTADO: De 1.5 para 2.0
            
            # 🚀 EXPLORAÇÃO INTELIGENTE - BALANCEADA (20% do peso)
            "exploration_bonus": 1.5,            # 🔧 AUMENTADO: De 1.0 para 1.5
            "innovation_bonus": 2.5,             # 🔧 AUMENTADO: De 2.0 para 2.5
            "adaptation_bonus": 1.5,             # 🔧 AUMENTADO: De 1.0 para 1.5
            
            # 🔄 HOLD INTELIGENTE - BALANCEADO (15% do peso)
            "smart_hold_bonus": 0.5,             # 🔧 AUMENTADO: De 0.3 para 0.5
            "perfect_wait_bonus": 1.5,           # 🔧 AUMENTADO: De 1.0 para 1.5
            "discipline_bonus": 1.2,             # 🔧 AUMENTADO: De 0.8 para 1.2
            
            # 🚨 PENALIDADES EDUCATIVAS - BALANCEADAS (10% do peso)
            "overtrading_education": -2.0,       # 🔧 REDUZIDO 2.5x: De -5.0 para -2.0
            "micro_trading_hint": -1.0,          # 🔧 REDUZIDO 2x: De -2.0 para -1.0
            "flip_flop_guidance": -1.5,          # 🔧 REDUZIDO 2x: De -3.0 para -1.5
            "extreme_overtrading": -8.0,         # 🔧 REDUZIDO 2.5x: De -20.0 para -8.0
            
            # 🎯 QUALIDADE SOBRE QUANTIDADE - BALANCEADA
            "natural_quality_bonus": 2.0,        # 🔧 REDUZIDO 4x: De 8.0 para 2.0
            "flow_state_bonus": 2.5,             # 🔧 REDUZIDO 4x: De 10.0 para 2.5
            "mastery_bonus": 3.0,                # 🔧 REDUZIDO 5x: De 15.0 para 3.0
            
            # 🎯 BÔNUS ANTI-OVERTRADING - BALANCEADOS
            "daily_discipline_bonus": 5.0,       # 🔧 REDUZIDO 5x: De 25.0 para 5.0
            "patience_mastery_bonus": 6.0,       # 🔧 REDUZIDO 5x: De 30.0 para 6.0
            "quality_over_quantity": 8.0,        # 🔧 REDUZIDO 4.4x: De 35.0 para 8.0
        }
        
        # 🧠 PARÂMETROS INTELIGENTES - BALANCEADOS PARA TREINAMENTO
        self.min_steps_between_trades = 5       # 🔥 REDUZIDO: 25min entre trades (era 12 = 1h)
        self.max_trades_per_day = 35            # 🔥 AUMENTADO: Limite mais permissivo (era 25)
        self.quality_guidance_threshold = 30    # 🔥 AUMENTADO: Orientação mais tardia (era 20)
        self.exploration_encouragement = 40     # 🔥 AUMENTADO: Mais exploração (era 25)
        self.last_trade_step = 0               # Tracking do último trade
        
        # 🚨 EMERGÊNCIA: COMPLETAMENTE DESABILITADO PARA FORÇAR EXECUÇÃO
        self.emergency_daily_limit = 99999      # 🔥 ILIMITADO: Sem limites diários
        self.emergency_cooldown = 0             # 🔥 ZERO: Sem cooldown
        self.emergency_mode = False             # 🔥 DESABILITADO: FORÇAR EXECUÇÃO
        
        # 🎯 PARÂMETROS DE QUALIDADE NATURAL
        self.min_trade_duration = 1            # 🧠 LIVRE: Sem duração mínima artificial
        self.quality_profit_threshold = 3.0    # 🧠 REALISTA: $3 para trade de qualidade
        self.excellent_profit_threshold = 10.0 # 🧠 REALISTA: $10 para trade excelente
        self.mastery_profit_threshold = 20.0   # 🧠 NOVO: $20 para demonstrar maestria
        self.pattern_recognition_window = 10   # 🧠 NOVO: 10 steps para reconhecer padrões
        
        # Sistema inicializado silenciosamente
    
    def calculate_reward_and_info(self, env, action: np.ndarray, current_price: float, 
                                 timestamp: Any) -> Tuple[float, Dict[str, Any], bool]:
        """
        🎯 SISTEMA V2 OTIMIZADO COM FOCO NOS TARGETS ESPECÍFICOS
        
        FÓRMULA PRINCIPAL:
        Reward = PnL_real(60%) + Expert_SL/TP(25%) + Atividade_Guiada(15%) + Penalidades_Moderadas
        """
        
        reward = 0.0
        trade_executed = False
        info = {"components": {}, "target_analysis": {}, "behavioral_analysis": {}}
        
        # 🧠 NOVO: DETECÇÃO INTELIGENTE DE REGIME DE MERCADO
        market_regime_reward = 0.0
        regime_info = {}
        trading_signals = {}
        
        if hasattr(self, 'market_regime_detector') and self.adaptive_trading_enabled:
            try:
                # 1. 🔍 DETECTAR REGIME ATUAL
                regime_info = self.market_regime_detector.detect_market_regime(env)
                
                # 2. 🎯 GERAR SINAIS BASEADOS NO REGIME
                trading_signals = self.market_regime_detector.get_trading_signals(
                    env, current_price, regime_info
                )
                
                # 3. 💰 RECOMPENSAR TRADES ALINHADOS COM O REGIME
                # 🚨 CORREÇÃO: Converter Box action para discrete
                if action[0] < 0.5:
                    entry_decision = 0  # Hold
                elif action[0] < 1.5:
                    entry_decision = 1  # Buy
                else:
                    entry_decision = 2  # Sell
                    
                if entry_decision > 0:  # Se está tentando fazer trade
                    
                    # Verificar se o trade está alinhado com o regime
                    regime_signal = trading_signals.get('entry_signal', 'HOLD')
                    regime_confidence = trading_signals.get('confidence', 0.5)
                    
                    # BÔNUS por trade alinhado com regime
                    if ((entry_decision == 1 and regime_signal == 'BUY') or 
                        (entry_decision == 2 and regime_signal == 'SELL')):
                        
                        # Bônus escalado pela confiança do regime
                        alignment_bonus = 5.0 * regime_confidence
                        market_regime_reward += alignment_bonus
                        info["components"]["regime_alignment_bonus"] = alignment_bonus
                        info["components"]["regime_reasoning"] = trading_signals.get('reasoning', '')
                    
                    # PENALIDADE por trade contra o regime (moderada)
                    elif ((entry_decision == 1 and regime_signal == 'SELL') or 
                          (entry_decision == 2 and regime_signal == 'BUY')):
                        
                        # Penalidade moderada por ir contra o regime
                        misalignment_penalty = -2.0 * regime_confidence
                        market_regime_reward += misalignment_penalty
                        info["components"]["regime_misalignment_penalty"] = misalignment_penalty
                        info["components"]["regime_reasoning"] = trading_signals.get('reasoning', '')
                
                # Adicionar informações do regime ao info
                info["market_regime"] = {
                    "regime": regime_info.get('regime', 'UNKNOWN'),
                    "strategy": regime_info.get('strategy', 'UNKNOWN'),
                    "confidence": regime_info.get('confidence', 0),
                    "duration": regime_info.get('duration', 0),
                    "support_level": regime_info.get('support_level', 0),
                    "resistance_level": regime_info.get('resistance_level', 0),
                    "momentum_score": regime_info.get('momentum_score', 0)
                }
                
                info["trading_signals"] = trading_signals
                reward += market_regime_reward
                
            except Exception as e:
                # Em caso de erro, continuar sem o sistema (não quebrar)
                info["market_regime_error"] = str(e)
                pass
        
        # 🔥 FILTRO DE QUALIDADE INTELIGENTE (mantido para compatibilidade)
        quality_reward = 0.0
        if hasattr(self, 'quality_filter') and getattr(self, 'quality_filter_enabled', False):
            try:
                allow_trade, quality_score, quality_info = self.quality_filter.should_allow_trade(env, action)
                
                # Adicionar informações de qualidade
                info["quality_filter"] = quality_info
                
                # Se não permitir trade, aplicar penalidade moderada
                if not allow_trade:
                    quality_reward = -5.0  # Penalidade moderada por trade de baixa qualidade
                    info["components"]["quality_filter_penalty"] = quality_reward
                else:
                    # Bônus por trade de alta qualidade
                    if quality_score > 90:
                        quality_reward = 3.0  # Bônus por trade excelente
                        info["components"]["quality_excellent_bonus"] = quality_reward
                    elif quality_score > 80:
                        quality_reward = 1.5  # Bônus por trade de qualidade
                        info["components"]["quality_good_bonus"] = quality_reward
                
                reward += quality_reward
                
            except Exception as e:
                # Em caso de erro no filtro, continuar sem ele (não quebrar)
                info["quality_filter_error"] = str(e)
                pass
        
        # 🚨 CORREÇÃO CRÍTICA: Converter Box actions para discrete actions
        # Action[0] vem como float 0-2, converter para discrete
        if action[0] < 0.5:
            entry_decision = 0  # Hold
        # 🎯 DECODIFICAR AÇÃO SIMPLIFICADA - DUAS CABEÇAS
        trade_action = int(action[0])  # 0=Hold, 1=Long, 2=Short, 3=Close_All
        sl_adjust = float(action[1])   # -3 a +3 para ajuste SL  
        tp_adjust = float(action[2])   # -3 a +3 para ajuste TP
        
        # 🧠 MAPEAR TRADE_ACTION PARA VARIÁVEIS ANTIGAS (compatibilidade)
        if trade_action == 0:      # Hold
            entry_decision = 0
            mgmt_action = 0
        elif trade_action == 1:    # Long
            entry_decision = 1
            mgmt_action = 0
        elif trade_action == 2:    # Short
            entry_decision = 2
            mgmt_action = 0
        elif trade_action == 3:    # Close All
            entry_decision = 0
            mgmt_action = 2  # Fechar todas as posições
        else:
            entry_decision = 0
            mgmt_action = 0
        
        # 🎯 POSITION SIZE AUTOMÁTICO - CALCULADO PELO AMBIENTE
        entry_confidence = 0.8  # Confiança padrão alta (não mais usado)
        
        # 💡 POSITION SIZE AUTOMÁTICO: Será calculado pelo ambiente baseado em gestão de risco
        # O modelo não decide mais o tamanho - isso é gestão de risco automática
        position_size = None  # Será calculado automaticamente
        
        # 💰 COMPONENTE PRINCIPAL: PnL DIRETO (60% do peso)
        old_trades_count = len(self.trades)
        
        # 🚨 VERIFICAÇÃO CRÍTICA: CONTROLE MÍNIMO PARA EVITAR OVERTRADING EXTREMO
        if self.emergency_mode and entry_decision > 0:
            current_step = getattr(env, 'current_step', 0)
            trades_today = self._get_trades_today_corrected(env)
            
            # Registrar contagem no info se próximo do limite
            if trades_today >= 40:
                info.setdefault("components", {})["quota_warning"] = f"Próximo do limite: {trades_today}/50"
            
            # Verificar se deve bloquear ANTES de qualquer processamento
            cooldown_violated = (current_step - self.last_trade_step) < self.emergency_cooldown
            quota_exceeded = trades_today >= self.emergency_daily_limit
            
            if cooldown_violated or quota_exceeded:
                # BLOQUEIO IMEDIATO - sem processamento adicional
                severe_penalty = -20.0 if quota_exceeded else -10.0  # 🔥 PENALIDADES REDUZIDAS
                reward += severe_penalty
                
                # Registrar bloqueios no info
                if cooldown_violated:
                    info["components"]["emergency_cooldown_block"] = severe_penalty
                    info["emergency_reason"] = f"BLOQUEADO - Cooldown: {current_step - self.last_trade_step} < {self.emergency_cooldown} steps"
                else:
                    info["components"]["emergency_quota_block"] = severe_penalty
                    info["emergency_reason"] = f"BLOQUEADO - Quota: {trades_today} >= {self.emergency_daily_limit} trades/dia"
                
                # Retornar imediatamente
                reward = np.clip(reward, -100.0, 100.0)
                complete_info = self._get_complete_info(entry_decision, mgmt_action, False)
                info.update(complete_info)
                return reward, info, False
        
        # 1. 🚨 EXECUÇÃO DE TRADE FORÇADA (SEMPRE PERMITIR)
        if entry_decision > 0:  # 🔥 REMOVIDO: Restrição de posições múltiplas
            current_step = getattr(env, 'current_step', 0)
            
            # 🔥 EXECUÇÃO FORÇADA: Sempre executar trades para descongelar portfolio
            trade_executed = self._execute_entry_order(
                entry_decision, entry_confidence, position_size, current_price, timestamp, env
            )
            if trade_executed:
                self.last_trade_step = current_step
                info["components"]["trade_executed"] = "FORCED_EXECUTION"
        
        # 2. 🎛️ GESTÃO DE POSIÇÕES EXISTENTES
        if len(self.positions) > 0:
            management_reward = self._execute_management(
                mgmt_action, sl_adjust, tp_adjust, current_price, timestamp
            )
            reward += management_reward
        
        # 3. 📊 ATUALIZAR PnL DAS POSIÇÕES ABERTAS
        self._update_positions_pnl(current_price)
        
        # 4. 🎯 VERIFICAR SL/TP AUTOMÁTICO
        closed_pnl = self._check_automatic_closes(current_price, timestamp)
        
        # 5. 📈 ANÁLISE DE TRADES FECHADOS
        current_trades_count = len(self.trades)
        if current_trades_count > old_trades_count:
            # Trade fechado - análise completa V2
            last_trade = self.trades[-1]
            pnl = last_trade.get('pnl_usd', 0.0)
            
            # 💰 PnL DIRETO - COMPONENTE DOMINANTE INTELIGENTE
            pnl_reward = pnl * self.weights.get("pnl_direct", 5.0)
            reward += pnl_reward
            info["components"]["pnl_direct"] = pnl_reward
            
            # 🧠 WIN/LOSS INTELIGENTE
            if pnl > 0:
                win_bonus = self.weights.get("win_bonus", 8.0)
                reward += win_bonus
                info["components"]["win_bonus"] = win_bonus
                
                # 🚨 NOVO: BÔNUS EXTRA POR QUALIDADE SOBRE QUANTIDADE
                trades_today = self._get_trades_today_corrected(env)
                # 🚨 REMOVIDO: Sem bônus por target de trades - anti-overtrading
                if False:  # Desabilitado
                    quality_bonus = self.weights.get("quality_over_quantity", 30.0)
                    reward += quality_bonus
                    info["components"]["quality_over_quantity_bonus"] = quality_bonus
                    info["quality_reason"] = f"Trade lucrativo dentro do limite: {trades_today}/18"
                
                # 🎯 QUALIDADE ESCALADA INTELIGENTE
                if pnl > self.mastery_profit_threshold:  # Trade de maestria
                    excellent_bonus = self.weights.get("excellent_trade_bonus", 25.0)
                    reward += excellent_bonus
                    info["components"]["mastery_trade_bonus"] = excellent_bonus
                elif pnl > self.excellent_profit_threshold:  # Trade excelente
                    excellent_bonus = self.weights.get("excellent_trade_bonus", 25.0) * 0.6
                    reward += excellent_bonus
                    info["components"]["excellent_trade_bonus"] = excellent_bonus
                elif pnl > self.quality_profit_threshold:  # Trade de qualidade
                    quality_bonus = self.weights.get("quality_trade_bonus", 12.0)
                    reward += quality_bonus
                    info["components"]["quality_trade_bonus"] = quality_bonus
                
                # 🎯 SEQUÊNCIA INTELIGENTE
                consecutive_wins = self._count_consecutive_wins(self.trades)
                if consecutive_wins > 1:
                    streak_bonus = self.weights.get("future_consequence_bonus", 8.0) * min(consecutive_wins - 1, 5)
                    reward += streak_bonus
                    info["components"]["win_sequence_bonus"] = streak_bonus
                
                # 🎯 TIMING PERFEITO
                trade_duration = last_trade.get('duration_steps', 0)
                if 5 <= trade_duration <= 50 and pnl > 2.0:
                    timing_bonus = self.weights.get("timing_mastery_bonus", 10.0) * 0.5
                    reward += timing_bonus
                    info["components"]["timing_mastery_bonus"] = timing_bonus
                    
            else:
                loss_penalty = self.weights.get("loss_penalty", -0.5)
                reward += loss_penalty
                info["components"]["gentle_loss_feedback"] = loss_penalty
            
            # 🧠 EXPERT SL/TP SIMPLIFICADO (25% do peso)
            expert_sltp_reward = self._analyze_expert_sltp_v2(env, last_trade)
            reward += expert_sltp_reward
            info["target_analysis"]["expert_sltp"] = expert_sltp_reward
            
            # 🔍 ANÁLISE COMPORTAMENTAL
            behavioral_reward = self._analyze_trade_behavior(env, last_trade, entry_decision)
            reward += behavioral_reward
            info["behavioral_analysis"]["trade_behavior"] = behavioral_reward
        
        # 6. 🧠 HOLD INTELIGENTE (Incentiva paciência quando apropriado)
        hold_reward = 0.0
        if entry_decision == 0:  # Se está em hold
            # 🧠 BÔNUS BASE POR PACIÊNCIA INTELIGENTE - AMPLIFICADO PARA ANTI-OVERTRADING
            hold_reward += self.weights.get("smart_hold_bonus", 5.0)  # 0.2 → 5.0
            
            # 🧠 BÔNUS INTELIGENTE: Hold em momento ruim para trade
            if hasattr(env, 'current_step') and env.current_step > 50:
                try:
                    # Verificar se condições de mercado não são ideais para trading
                    current_idx = min(env.current_step, len(env.df) - 1)
                    
                    # Volatilidade muito baixa ou muito alta
                    volatility_5m = env.df.get('volatility_20_5m', pd.Series([0.001])).iloc[current_idx]
                    price_5m = env.df['close_5m'].iloc[current_idx]
                    vol_ratio = volatility_5m / price_5m if price_5m > 0 else 0
                    
                    # RSI neutro (não extremo)
                    rsi_5m = env.df.get('rsi_14_5m', pd.Series([50])).iloc[current_idx]
                    
                    # Se volatilidade inadequada OU RSI neutro = bom momento para hold
                    if (vol_ratio < 0.002 or vol_ratio > 0.02) or (35 < rsi_5m < 65):
                        hold_reward += self.weights.get("perfect_wait_bonus", 15.0)  # 2.5 → 15.0
                        info["components"]["perfect_wait_reason"] = f"Vol:{vol_ratio:.4f}, RSI:{rsi_5m:.1f}"
                    
                    # 🚨 REMOVIDO: Discipline bonus removido - hold reward já é suficiente
                    # trades_today = self._get_trades_today_corrected(env)
                    # if trades_today > 10:  # DESABILITADO
                    #     hold_reward += self.weights.get("discipline_bonus", 10.0)
                    #     info["components"]["discipline_reason"] = f"Trades hoje: {trades_today}"
                        
                except Exception:
                    # Em caso de erro, dar bônus básico de paciência
                    hold_reward += self.weights.get("smart_hold_bonus", 5.0)
            
            info["components"]["hold_bonus"] = hold_reward
        
        reward += hold_reward
        
        # 7. 🚨 ATIVIDADE GUIADA COM ANTI-OVERTRADING SEVERO
        trades_today = self._get_trades_today_corrected(env)
        activity_reward = self._calculate_activity_reward_v3(trades_today, entry_decision)
        reward += activity_reward
        info["target_analysis"]["activity"] = activity_reward
        info["target_analysis"]["trades_today"] = trades_today
        
        # 8. 🛡️ PENALIDADES DISCIPLINARES
        discipline_penalty = self._calculate_discipline_penalties(env, action, entry_decision)
        reward += discipline_penalty
        info["behavioral_analysis"]["discipline"] = discipline_penalty
        
        # 9. 🎯 GESTÃO DE RISCO E DRAWDOWN
        risk_reward = self._calculate_risk_management_reward(env)
        reward += risk_reward
        info["target_analysis"]["risk_management"] = risk_reward
        
        # 10. 📈 ATUALIZAR TODAS AS MÉTRICAS
        self._update_all_metrics(timestamp)
        
        # 11. 📊 SINCRONIZAR COM AMBIENTE
        self._sync_with_environment(env)
        
        # 🎯 LIMITES BALANCEADOS - CLIPPING RIGOROSO PARA ESTABILIDADE
        reward = np.clip(reward, -5.0, 10.0)   # 🔧 MAIS RIGOROSO: De -10/+20 para -5/+10
        
        # Info completo para debug
        complete_info = self._get_complete_info(entry_decision, mgmt_action, trade_executed)
        info.update(complete_info)
        
        return reward, info, trade_executed
    
    def _execute_entry_order(self, decision: int, confidence: float, size: float, 
                           price: float, timestamp: Any, env=None) -> bool:
        """🚀 Executa ordem de entrada com POSITION SIZING AUTOMÁTICO"""
        if decision == 0:
            return False
        
        # 🎯 POSITION SIZING AUTOMÁTICO - Calculado pelo ambiente
        if env and hasattr(env, '_calculate_adaptive_position_size'):
            # Usar position sizing automático do ambiente
            lot_size = env._calculate_adaptive_position_size()
        else:
            # Fallback para size fixo se ambiente não suportar
            lot_size = self.base_lot_size if size is None else self.base_lot_size + (size * (self.max_lot_size - self.base_lot_size))
        
        lot_size = min(lot_size, self.max_lot_size)
        
        # Definir direção
        direction = 'BUY' if decision == 1 else 'SELL'
        
        # Calcular SL/TP baseado na confiança
        pip_value = 0.0001
        sl_distance = max(10, int(30 * (1 - confidence))) * pip_value
        tp_distance = max(15, int(45 * confidence)) * pip_value
        
        if direction == 'BUY':
            entry_price = price + self.spread/2
            sl_price = entry_price - sl_distance
            tp_price = entry_price + tp_distance
        else:
            entry_price = price - self.spread/2
            sl_price = entry_price + sl_distance
            tp_price = entry_price - tp_distance
        
        # 🔥 OBTER STEP ATUAL CORRETAMENTE
        current_step = 0
        if env and hasattr(env, 'current_step'):
            current_step = env.current_step
        elif hasattr(timestamp, 'current_step'):
            current_step = timestamp.current_step
        
        # Criar posição
        position_id = f"{direction}_{self.trade_count}"
        
        self.positions[position_id] = {
            'id': position_id,
            'direction': direction,
            'lot_size': lot_size,
            'entry_price': entry_price,
            'sl_price': sl_price,
            'tp_price': tp_price,
            'entry_timestamp': timestamp,
            'entry_step': current_step,  # 🔥 TRACKING CORRETO
            'pnl': 0.0,
            'pnl_usd': 0.0
        }
        
        self.trade_count += 1
        self.last_trade_timestamp = timestamp
        
        return True
    
    def _execute_management(self, action: int, sl_adjust: float, tp_adjust: float, 
                          current_price: float, timestamp: Any) -> float:
        """🧠 GESTÃO INTELIGENTE - APENAS SL/TP, SEM FECHAMENTOS MANUAIS"""
        reward = 0.0
        
        # 🚫 DESABILITADO: Fechamentos manuais removidos para evitar overtrading
        # O modelo agora só pode ajustar SL/TP e deixar o mercado decidir
        
        # ✅ PERMITIDO: Apenas ajustes inteligentes de SL/TP
        if abs(sl_adjust) > 0.1 or abs(tp_adjust) > 0.1:
            adjustment_reward = self._adjust_sl_tp_intelligent(sl_adjust, tp_adjust, current_price)
            reward += adjustment_reward
        
        return reward
    
    def _close_profitable_positions(self, current_price: float, timestamp: Any) -> float:
        """💰 Fecha posições lucrativas"""
        total_pnl = 0.0
        positions_to_close = []
        
        for pos_id, pos in self.positions.items():
            pnl_usd = self._calculate_position_pnl_usd(pos, current_price)
            if pnl_usd > 0:
                total_pnl += pnl_usd
                positions_to_close.append(pos_id)
        
        # Fechar posições lucrativas silenciosamente
        for pos_id in positions_to_close:
            self._close_position(pos_id, current_price, timestamp, "PROFITABLE_CLOSE")
        
        return total_pnl * 0.2
    
    def _close_all_positions(self, current_price: float, timestamp: Any) -> float:
        """🔄 Fecha todas as posições"""
        total_pnl = 0.0
        
        for pos_id in list(self.positions.keys()):
            pnl_usd = self._close_position(pos_id, current_price, timestamp, "MANUAL_CLOSE")
            total_pnl += pnl_usd
        
        return total_pnl * 0.1
    
    def _close_position(self, position_id: str, exit_price: float, timestamp: Any, 
                       reason: str = "MANUAL") -> float:
        """💼 Fecha uma posição específica e registra no histórico"""
        if position_id not in self.positions:
            return 0.0
        
        pos = self.positions[position_id]
        pnl_usd = self._calculate_position_pnl_usd(pos, exit_price)
        
        # Registrar trade no histórico
        trade_record = {
            'id': position_id,
            'direction': pos['direction'],
            'lot_size': pos['lot_size'],
            'entry_price': pos['entry_price'],
            'exit_price': exit_price,
            'entry_timestamp': pos['entry_timestamp'],
            'exit_timestamp': timestamp,
            'entry_step': pos.get('entry_step', 0),  # 🔥 CORREÇÃO CRÍTICA: Transferir entry_step
            'pnl_usd': pnl_usd,
            'reason': reason,
            'sl_price': pos['sl_price'],
            'tp_price': pos['tp_price']
        }
        
        self.trades.append(trade_record)
        
        # Atualizar estatísticas
        self.total_trades += 1
        self.total_pnl += pnl_usd
        self.realized_balance += pnl_usd
        
        if pnl_usd > 0:
            self.winning_trades += 1
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.largest_win = max(self.largest_win, pnl_usd)
            self.max_consecutive_wins = max(self.max_consecutive_wins, self.consecutive_wins)
        else:
            self.losing_trades += 1
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.largest_loss = min(self.largest_loss, pnl_usd)
            self.max_consecutive_losses = max(self.max_consecutive_losses, self.consecutive_losses)
        
        # 📊 ATUALIZAR MÉTRICAS DIÁRIAS E DO EPISÓDIO
        if timestamp and hasattr(timestamp, 'date'):
            date_key = timestamp.date()
            if date_key in self.daily_pnl:
                self.daily_pnl[date_key] += pnl_usd
                self.daily_trades[date_key] += 1
        
        # Adicionar trade ao episódio atual
        self.episode_trades.append({
            'pnl_usd': pnl_usd,
            'timestamp': timestamp,
            'reason': reason
        })
        
        # Debug removido - posição fechada silenciosamente
        
        # Remover posição
        del self.positions[position_id]
        
        return pnl_usd
    
    def _check_automatic_closes(self, current_price: float, timestamp: Any) -> float:
        """🎯 Verifica e executa fechamentos automáticos por SL/TP"""
        total_pnl = 0.0
        positions_to_close = []
        
        for pos_id, pos in self.positions.items():
            close_reason = None
            
            if pos['direction'] == 'BUY':
                if current_price <= pos['sl_price']:
                    close_reason = "SL"
                elif current_price >= pos['tp_price']:
                    close_reason = "TP"
            else:  # SELL
                if current_price >= pos['sl_price']:
                    close_reason = "SL"
                elif current_price <= pos['tp_price']:
                    close_reason = "TP"
            
            if close_reason:
                positions_to_close.append((pos_id, close_reason))
        
        # Executar fechamentos
        for pos_id, reason in positions_to_close:
            pnl_usd = self._close_position(pos_id, current_price, timestamp, f"{reason}")
            total_pnl += pnl_usd

        
        return total_pnl
    
    def _update_positions_pnl(self, current_price: float):
        """📊 Atualiza PnL das posições abertas"""
        self.unrealized_pnl = 0.0
        
        for pos in self.positions.values():
            pnl_usd = self._calculate_position_pnl_usd(pos, current_price)
            pos['pnl_usd'] = pnl_usd
            self.unrealized_pnl += pnl_usd
    
    def _calculate_position_pnl_usd(self, position: Dict, current_price: float) -> float:
        """💰 Calcula PnL em USD de uma posição"""
        entry_price = position['entry_price']
        lot_size = position['lot_size']
        
        if position['direction'] == 'BUY':
            price_diff = current_price - entry_price
        else:
            price_diff = entry_price - current_price
        
        # 🔥 CORREÇÃO ESCALA OURO: 1 ponto = $1 para 0.01 lotes
        # Ouro: 1500.00 -> 1501.00 = 1 ponto = $1 para 0.01 lotes
        # Para lot_size = 0.02: 1 ponto = $2
        pnl_usd = price_diff * (lot_size / 0.01)
        
        # Subtrair comissão
        pnl_usd -= self.commission * lot_size
        
        return pnl_usd
    
    def _update_all_metrics(self, timestamp: Any):
        """📈 Atualiza todas as métricas de performance"""
        # Portfolio total
        self.portfolio_value = self.realized_balance + self.unrealized_pnl
        
        # Atualizar pico de portfolio
        if self.portfolio_value > self.peak_portfolio:
            self.peak_portfolio = self.portfolio_value
            self.peak_portfolio_value = self.portfolio_value
        
        # Calcular drawdown atual
        if self.peak_portfolio > 0:
            self.current_drawdown = max(0, (self.peak_portfolio - self.portfolio_value) / self.peak_portfolio)
            if self.current_drawdown > self.peak_drawdown:
                self.peak_drawdown = self.current_drawdown
        
        # Atualizar métricas diárias
        if timestamp and hasattr(timestamp, 'date'):
            date_key = timestamp.date()
            if date_key not in self.daily_pnl:
                self.daily_pnl[date_key] = 0.0
                self.daily_trades[date_key] = 0
                self.trading_days += 1
        
        # Adicionar drawdown atual ao histórico do episódio
        self.episode_drawdowns.append(self.current_drawdown)
    
    def _sync_with_environment(self, env):
        """🔄 Sincroniza métricas com o ambiente de treinamento"""
        # 🚨 DEBUG: Verificar se sincronização está funcionando
        old_portfolio = getattr(env, 'portfolio_value', 0)
        
        # 🔥 FORÇAR SINCRONIZAÇÃO COMPLETA
        env.portfolio_value = self.portfolio_value
        env.realized_balance = self.realized_balance
        env.trades = self.trades.copy()
        env.peak_portfolio = self.peak_portfolio
        env.peak_portfolio_value = self.peak_portfolio_value
        env.current_drawdown = self.current_drawdown
        env.peak_drawdown = self.peak_drawdown
        
        # 🔥 FORÇAR TAMBÉM POSIÇÕES SE EXISTIREM
        if hasattr(env, 'positions'):
            # Converter posições do reward system para formato do env
            env_positions = []
            for pos_id, pos in self.positions.items():
                env_position = {
                    'type': 'long' if pos['direction'] == 'BUY' else 'short',
                    'entry_price': pos['entry_price'],
                    'lot_size': pos['lot_size'],
                    'entry_step': pos['entry_step'],
                    'sl': pos.get('sl_price', 0),
                    'tp': pos.get('tp_price', 0)
                }
                env_positions.append(env_position)
            env.positions = env_positions
        

    
    def _calculate_advanced_metrics(self) -> Dict[str, float]:
        """📊 Calcula métricas avançadas solicitadas"""
        # 📈 TRADES/DIA DOS ÚLTIMOS 100 DIAS
        recent_days = sorted(self.daily_trades.keys())[-100:] if self.daily_trades else []
        trades_last_100d = sum(self.daily_trades[day] for day in recent_days) if recent_days else 0
        avg_trades_per_day_100d = trades_last_100d / max(len(recent_days), 1)
        
        # 💰 LUCRO MÉDIO/DIA DOS ÚLTIMOS 100 DIAS
        recent_pnl_days = sorted(self.daily_pnl.keys())[-100:] if self.daily_pnl else []
        pnl_last_100d = sum(self.daily_pnl[day] for day in recent_pnl_days) if recent_pnl_days else 0
        avg_profit_per_day_100d = pnl_last_100d / max(len(recent_pnl_days), 1)
        
        # 📉 DRAWDOWN DO EPISÓDIO (MÁXIMO)
        episode_max_dd = max(self.episode_drawdowns) * 100 if self.episode_drawdowns else 0.0
        
        # 📉 DRAWDOWN MÉDIO DO EPISÓDIO
        episode_avg_dd = (sum(self.episode_drawdowns) / len(self.episode_drawdowns) * 100) if self.episode_drawdowns else 0.0
        
        # 📊 TRADES/DIA GERAL (CORRIGIDO)
        trades_per_day_total = self.total_trades / max(self.trading_days, 1)
        
        return {
            'trades_per_day_100d': avg_trades_per_day_100d,
            'profit_per_day_100d': avg_profit_per_day_100d,
            'episode_max_drawdown': episode_max_dd,
            'episode_avg_drawdown': episode_avg_dd,
            'trades_per_day_corrected': trades_per_day_total
        }
    
    def _get_complete_info(self, entry_decision: int, mgmt_action: int, trade_executed: bool) -> Dict[str, Any]:
        """📊 Retorna informações completas do sistema"""
        win_rate = (self.winning_trades / max(1, self.total_trades)) * 100
        avg_pnl_per_trade = self.total_pnl / max(1, self.total_trades)
        trades_per_day = self.total_trades / max(1, self.trading_days)
        profit_per_day = self.total_pnl / max(1, self.trading_days)
        
        # 🔥 MÉTRICAS AVANÇADAS
        advanced_metrics = self._calculate_advanced_metrics()
        
        return {
            # 💰 Portfolio
            'portfolio_value': self.portfolio_value,
            'realized_balance': self.realized_balance,
            'unrealized_pnl': self.unrealized_pnl,
            
            # 📉 Drawdown
            'current_drawdown': self.current_drawdown * 100,
            'peak_drawdown': self.peak_drawdown * 100,
            'peak_portfolio': self.peak_portfolio,
            
            # 📈 Trades
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'positions_open': len(self.positions),
            
            # 💵 PnL
            'total_pnl': self.total_pnl,
            'avg_pnl_per_trade': avg_pnl_per_trade,
            'largest_win': self.largest_win,
            'largest_loss': self.largest_loss,
            
            # ⏰ Performance
            'trades_per_day': advanced_metrics['trades_per_day_corrected'],  # 🔥 CORRIGIDO
            'profit_per_day': profit_per_day,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses,
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            
            # 🔥 MÉTRICAS AVANÇADAS SOLICITADAS
            'trades_per_day_100d': advanced_metrics['trades_per_day_100d'],
            'profit_per_day_100d': advanced_metrics['profit_per_day_100d'],
            'episode_max_drawdown': advanced_metrics['episode_max_drawdown'],
            'episode_avg_drawdown': advanced_metrics['episode_avg_drawdown'],
            
            # 🎯 Estado atual
            'entry_decision': entry_decision,
            'mgmt_action': mgmt_action,
            'trade_executed': trade_executed,
            'trading_days': self.trading_days
        }
    
    def _adjust_sl_tp_intelligent(self, sl_adjust: float, tp_adjust: float, current_price: float) -> float:
        """🧠 AJUSTE INTELIGENTE DE SL/TP COM RECOMPENSAS"""
        reward = 0.0
        pip_value = 0.0001
        
        for pos in self.positions.values():
            current_pnl = self._calculate_position_pnl_usd(pos, current_price)
            
            # 🎯 AJUSTE DE STOP LOSS INTELIGENTE
            if abs(sl_adjust) > 0.1:
                adjustment = sl_adjust * 15 * pip_value  # ±15 pips por unidade
                old_sl = pos['sl_price']
                
                if pos['direction'] == 'BUY':
                    new_sl = max(pos['sl_price'] + adjustment, current_price - 50*pip_value)
                    if new_sl > old_sl and current_pnl > 5.0:  # Trailing stop em lucro
                        pos['sl_price'] = new_sl
                        reward += self.weights.get("timing_mastery_bonus", 10.0) * 0.3
                    elif new_sl > old_sl:  # Proteção preventiva
                        pos['sl_price'] = new_sl
                        reward += self.weights.get("pattern_recognition_bonus", 6.0) * 0.2
                else:
                    new_sl = min(pos['sl_price'] - adjustment, current_price + 50*pip_value)
                    if new_sl < old_sl and current_pnl > 5.0:  # Trailing stop em lucro
                        pos['sl_price'] = new_sl
                        reward += self.weights.get("timing_mastery_bonus", 10.0) * 0.3
                    elif new_sl < old_sl:  # Proteção preventiva
                        pos['sl_price'] = new_sl
                        reward += self.weights.get("pattern_recognition_bonus", 6.0) * 0.2
            
            # 🎯 AJUSTE DE TAKE PROFIT INTELIGENTE
            if abs(tp_adjust) > 0.1:
                adjustment = tp_adjust * 25 * pip_value  # ±25 pips por unidade
                
                if pos['direction'] == 'BUY':
                    old_tp = pos['tp_price']
                    pos['tp_price'] += adjustment
                    if current_pnl > 0 and adjustment > 0:  # Expandindo TP em lucro
                        reward += self.weights.get("patience_wisdom_bonus", 5.0) * 0.4
                else:
                    old_tp = pos['tp_price']
                    pos['tp_price'] -= adjustment
                    if current_pnl > 0 and adjustment > 0:  # Expandindo TP em lucro
                        reward += self.weights.get("patience_wisdom_bonus", 5.0) * 0.4
        
        return reward
    
    def _adjust_sl_tp(self, sl_adjust: float, tp_adjust: float, current_price: float):
        """🎯 Método compatível - chama versão inteligente"""
        return self._adjust_sl_tp_intelligent(sl_adjust, tp_adjust, current_price)
    
    def _calculate_base_reward(self, current_price: float) -> float:
        """Calcula recompensa base simples"""
        # Recompensa baseada no portfolio atual vs inicial
        portfolio_change = (self.portfolio_value - self.initial_balance) / self.initial_balance
        return portfolio_change * 0.1  # 10% da mudança percentual
    
    def reset(self):
        """🔥 RESET COMPLETO COM CORREÇÃO DE BUGS"""
        # 🚨 RESET CRÍTICO: Limpar TUDO para evitar portfolio congelado
        self.positions = {}
        self.trades = []
        self.episode_trades = []
        
        # 🔥 RESET DE BALANÇO FORÇADO
        self.realized_balance = self.initial_balance  # SEMPRE voltar ao inicial
        self.total_pnl = 0.0
        
        # 🔥 RESET DE MÉTRICAS CRÍTICAS
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        # 🔥 RESET DE DRAWDOWN (corrigir matemática absurda)
        self.peak_balance = self.initial_balance
        self.current_drawdown = 0.0
        self.max_drawdown = 0.0
        
        # 🔥 RESET DE SEQUÊNCIAS
        self.consecutive_wins = 0
        self.consecutive_losses = 0
        self.max_consecutive_wins = 0
        self.max_consecutive_losses = 0
        
        # 🔥 RESET DE EXTREMOS
        self.largest_win = 0.0
        self.largest_loss = 0.0
        
        # 🔥 RESET DE TIMESTAMPS
        self.last_trade_timestamp = None
        self.last_trade_step = -999  # Reset para valor inicial
        
        # 🔥 RESET DE MÉTRICAS DIÁRIAS
        self.daily_pnl = {}
        self.daily_trades = {}
        
        # 🔥 RESET DE CONTROLE DE EMERGENCY
        self.emergency_mode = True  # Manter ativo
        self.emergency_cooldown = 5  # Reduzido para mais atividade
        self.emergency_daily_limit = 50  # Limite realista
        
        print(f"🔥 RESET FORÇADO: Portfolio=${self.initial_balance}, Drawdown=0%, Trades=0")
    
    def _analyze_expert_sltp_v2(self, env, trade: Dict) -> float:
        """🧠 EXPERT SL/TP INTELIGENTE (25% do peso)"""
        reward = 0.0
        
        try:
            # Extrair dados do trade
            entry_price = trade.get('entry_price', 0)
            sl_price = trade.get('sl_price', 0)
            tp_price = trade.get('tp_price', 0)
            direction = trade.get('direction', 'BUY')
            
            if entry_price > 0 and sl_price > 0 and tp_price > 0:
                # Calcular SL/TP em pontos
                if direction == 'BUY':
                    sl_points = abs(entry_price - sl_price) * 10000
                    tp_points = abs(tp_price - entry_price) * 10000
                else:
                    sl_points = abs(sl_price - entry_price) * 10000
                    tp_points = abs(entry_price - tp_price) * 10000
                
                # 🎯 RANGES INTELIGENTES (10-50 SL, 15-80 TP)
                if 10 <= sl_points <= 50 and 15 <= tp_points <= 80:
                    reward += self.weights.get("timing_mastery_bonus", 10.0)
                
                # 🎯 RATIO INTELIGENTE
                if sl_points > 0:
                    risk_reward_ratio = tp_points / sl_points
                    if 1.4 <= risk_reward_ratio <= 2.5:  # Ratio ótimo
                        reward += self.weights.get("pattern_recognition_bonus", 6.0)
                    elif 1.0 <= risk_reward_ratio <= 3.0:  # Aceitável
                        reward += self.weights.get("natural_quality_bonus", 5.0) * 0.5
                
                # 🎯 TIMING PERFEITO
                trade_duration = trade.get('duration_steps', 0)
                if 5 <= trade_duration <= 200:  # Duração flexível
                    reward += self.weights.get("patience_wisdom_bonus", 5.0)
                
        except Exception:
            pass
        
        return reward
    
    def _calculate_activity_reward_v3(self, trades_today: int, entry_decision: int) -> float:
        """🧠 ATIVIDADE INTELIGENTE - CORRIGIDA PARA 24H REAIS"""
        reward = 0.0
        
        # 🚨 THRESHOLDS CORRIGIDOS PARA 24H (não mais 2h artificiais)
        if trades_today > 60:  # >60 trades/dia - extremo overtrading
            reward += self.weights.get("extreme_overtrading", -20.0) * 2  # PENALIDADE SEVERA
        elif trades_today > 40:  # >40 trades/dia - overtrading alto
            reward += self.weights.get("overtrading_education", -5.0) * 1.5  # PENALIDADE MODERADA
        elif trades_today > 30:  # >30 trades/dia - overtrading moderado
            reward += self.weights.get("overtrading_education", -5.0) * 0.8  # PENALIDADE LEVE
        
        # 🎯 ZONA ALVO REALISTA (para 24h completas)
        elif 12 <= trades_today <= 25:  # 12-25 trades/dia = atividade ideal
            reward += self.weights.get("natural_quality_bonus", 8.0)  # INCENTIVO FORTE
        elif 8 <= trades_today <= 35:  # 8-35 trades/dia = atividade aceitável
            reward += self.weights.get("exploration_bonus", 5.0)  # INCENTIVO MODERADO
        
        # 🧠 ORIENTAÇÃO EDUCATIVA
        elif trades_today > 50:  # >50 trades/dia - educação necessária
            reward += self.weights.get("overtrading_education", -5.0)  # PENALIDADE EDUCATIVA
        elif trades_today < 5:  # <5 trades/dia - inatividade
            reward += self.weights.get("smart_hold_bonus", 0.2) * 3  # Incentiva atividade mínima
        
        # 🚀 BÔNUS POR EXPLORAÇÃO INTELIGENTE
        if entry_decision > 0 and 8 <= trades_today <= 40:
            reward += self.weights.get("exploration_bonus", 5.0) * 0.6
        
        # 🎯 MAESTRIA: Bônus especial para atividade equilibrada
        if 15 <= trades_today <= 25:  # ZONA DE MAESTRIA IDEAL
            reward += self.weights.get("mastery_bonus", 15.0) * 0.7
        
        return reward
    
    def _calculate_activity_reward(self, trades_today: int, entry_decision: int) -> float:
        """Método legado - usar _calculate_activity_reward_v3"""
        return self._calculate_activity_reward_v3(trades_today, entry_decision)
    
    def _analyze_trade_behavior(self, env, trade: Dict, entry_decision: int) -> float:
        """🔍 ANÁLISE COMPORTAMENTAL"""
        reward = 0.0
        
        try:
            # 🧠 CONSISTÊNCIA INTELIGENTE
            if len(self.trades) >= 5:
                recent_trades = self.trades[-5:]
                win_rate = sum(1 for t in recent_trades if t.get('pnl_usd', 0) > 0) / len(recent_trades)
                if win_rate >= 0.6:  # 60% win rate
                    reward += self.weights.get("mastery_bonus", 12.0) * 0.4
            
        except Exception:
            pass
        
        return reward
    
    def _calculate_discipline_penalties(self, env, action: np.ndarray, entry_decision: int) -> float:
        """🧠 ORIENTAÇÃO DISCIPLINAR EDUCATIVA"""
        penalty = 0.0
        
        try:
            # 🧠 FLIP-FLOP GUIDANCE (orientação gentil)
            if len(self.trades) >= 2:
                last_trade = self.trades[-1]
                second_last = self.trades[-2]
                
                last_type = 1 if last_trade.get('direction') == 'BUY' else 2
                second_type = 1 if second_last.get('direction') == 'BUY' else 2
                
                if last_type != second_type and \
                   last_trade.get('duration_steps', 100) < 10:  # Mudança rápida
                    penalty += self.weights.get("flip_flop_guidance", -2.0)
            
            # 🧠 MICRO-TRADING HINT (dica gentil)
            if len(self.trades) > 0:
                last_trade = self.trades[-1]
                if last_trade.get('duration_steps', 100) < 3:
                    penalty += self.weights.get("micro_trading_hint", -1.0)
            
            # 🧠 SL/TP EDUCATION (educação sobre ranges)
            if len(action) >= 3:  # 🔥 CORRIGIDO: action space agora tem 3 dimensões
                sl_val = abs(float(action[1]))  # 🔥 CORRIGIDO: índice 1 para SL
                tp_val = abs(float(action[2]))  # 🔥 CORRIGIDO: índice 2 para TP
                if sl_val > 3.0 or tp_val > 3.0:  # Valores extremos
                    penalty += self.weights.get("overtrading_education", -3.0) * 0.5
            
        except Exception:
            pass
        
        return penalty
    
    def _calculate_risk_management_reward(self, env) -> float:
        """🧠 GESTÃO DE RISCO INTELIGENTE"""
        reward = 0.0
        
        try:
            # 🧠 ORIENTAÇÃO SOBRE DRAWDOWN (educativa)
            if self.current_drawdown > 0.08:  # >8% - orientação gentil
                excess_dd = (self.current_drawdown - 0.08) * 50  # Reduzido
                reward += self.weights.get("overtrading_education", -3.0) * excess_dd
            
            # 🎯 BÔNUS POR EXCELENTE GESTÃO
            if self.current_drawdown < 0.03 and len(self.trades) > 5:  # <3% DD com atividade
                reward += self.weights.get("mastery_bonus", 12.0) * 0.5
            
            # 🚀 BÔNUS POR CONTROLE DE RISCO
            if self.current_drawdown < 0.05 and len(self.trades) > 10:  # <5% DD com boa atividade
                reward += self.weights.get("natural_quality_bonus", 5.0)
            
        except Exception:
            pass
        
        return reward
    
    def _get_trades_today_corrected(self, env) -> int:
        """🚨 CONTA TRADES DO DIA ATUAL - CORRIGIDO PARA DIA COMPLETO"""
        try:
            if hasattr(env, 'current_step'):
                # 🔥 CORREÇÃO CRÍTICA: 1 dia = 288 steps (5min * 288 = 24h)
                current_step = env.current_step
                
                # 🎯 CORREÇÃO: Contar trades do DIA COMPLETO (288 steps = 24h)
                # Não mais janela artificial de 2h que distorce contagem
                cutoff_step = max(0, current_step - 288)  # Últimas 24h (dia completo)
                
                trades_today = 0
                for trade in self.trades:
                    trade_step = trade.get('entry_step', 0)
                    if trade_step >= cutoff_step:
                        trades_today += 1
                
                return trades_today
        except Exception:
            pass
        
        # Fallback: contar últimos trades com limite realista
        return min(len(self.trades), 25)  # Máximo 25 trades/dia como fallback
    
    def _get_trades_today(self, env) -> int:
        """Método legado - usar _get_trades_today_corrected"""
        return self._get_trades_today_corrected(env)
    
    def _count_consecutive_wins(self, trades: List[Dict]) -> int:
        """Conta wins consecutivos"""
        if not trades:
            return 0
        
        consecutive = 0
        for trade in reversed(trades):
            if trade.get('pnl_usd', 0) > 0:
                consecutive += 1
            else:
                break
        
        return consecutive
    
    def _calculate_drawdown_metrics(self) -> Dict[str, float]:
        """🔥 CORREÇÃO CRÍTICA: Cálculo de drawdown realista COM PROTEÇÃO ANTI-BUG"""
        current_balance = self.get_portfolio_value()
        
        # 🚨 PROTEÇÃO: Se valores absurdos, resetar tudo
        if current_balance < 1.0 or current_balance > 50000.0:
            print(f"🚨 PORTFOLIO ABSURDO: ${current_balance:.2f} - RESET FORÇADO")
            current_balance = self.initial_balance
            self.realized_balance = self.initial_balance
            self.peak_balance = self.initial_balance
            self.current_drawdown = 0.0
            self.max_drawdown = 0.0
            
            return {
                'current_drawdown': 0.0,
                'max_drawdown': 0.0,
                'peak_balance': self.initial_balance,
                'current_balance': self.initial_balance
            }
        
        # 🔥 CORREÇÃO: Atualizar peak apenas se realmente maior
        if current_balance > self.peak_balance:
            self.peak_balance = current_balance
        
        # 🔥 CORREÇÃO: Drawdown como percentual REALISTA
        if self.peak_balance > 0:
            # Drawdown = (Peak - Current) / Peak * 100
            drawdown_amount = max(0, self.peak_balance - current_balance)
            current_drawdown_pct = (drawdown_amount / self.peak_balance) * 100.0
            
            # 🚨 LIMITE REALISTA: Drawdown máximo 100% (não pode ser >100%)
            current_drawdown_pct = max(0.0, min(100.0, current_drawdown_pct))
        else:
            current_drawdown_pct = 0.0
        
        # 🔥 CORREÇÃO: Atualizar max drawdown apenas se maior
        if current_drawdown_pct > self.max_drawdown:
            self.max_drawdown = current_drawdown_pct
        
        # 🔥 LIMITE FINAL: Garantir que nunca exceda 100%
        self.max_drawdown = min(100.0, self.max_drawdown)
        current_drawdown_pct = min(100.0, current_drawdown_pct)
        
        # 🚨 VERIFICAÇÃO FINAL: Se drawdown ainda absurdo, resetar
        if current_drawdown_pct > 100.0 or self.max_drawdown > 100.0:
            print(f"🚨 DRAWDOWN ABSURDO DETECTADO: {current_drawdown_pct:.1f}% - RESET FORÇADO")
            current_drawdown_pct = 0.0
            self.max_drawdown = 0.0
            self.peak_balance = current_balance
        
        # 🔥 ATUALIZAR VALORES CORRIGIDOS
        self.current_drawdown = current_drawdown_pct / 100.0  # Converter para decimal
        
        return {
            'current_drawdown': current_drawdown_pct,
            'max_drawdown': self.max_drawdown,
            'peak_balance': self.peak_balance,
            'current_balance': current_balance
        }
    
    def get_portfolio_value(self) -> float:
        """🔥 CORREÇÃO CRÍTICA: Portfolio value sempre atualizado COM PROTEÇÃO ANTI-BUG"""
        # 🔥 BALANÇO BASE: Sempre usar realized_balance como base
        base_balance = max(0.0, self.realized_balance)
        
        # 🚨 PROTEÇÃO ANTI-BUG: Se balance absurdo, resetar para inicial
        if base_balance < 1.0 or base_balance > 10000.0:  # Limites realistas
            print(f"🚨 BALANCE ABSURDO DETECTADO: ${base_balance:.2f} - RESETANDO PARA ${self.initial_balance}")
            base_balance = self.initial_balance
            self.realized_balance = self.initial_balance
            self.peak_balance = self.initial_balance
            self.current_drawdown = 0.0
            self.max_drawdown = 0.0
        
        # 🔥 PnL NÃO REALIZADO: Calcular das posições abertas
        unrealized_pnl = 0.0
        if self.positions:
            for symbol, position in self.positions.items():
                if position.get('is_open', False):
                    # Calcular PnL não realizado baseado no preço atual
                    entry_price = position.get('entry_price', 0.0)
                    current_price = position.get('current_price', entry_price)
                    size = position.get('size', 0.0)
                    side = position.get('side', 'long')
                    
                    if entry_price > 0 and current_price > 0 and size > 0:
                        if side == 'long':
                            pnl = (current_price - entry_price) * size
                        else:  # short
                            pnl = (entry_price - current_price) * size
                        
                        # 🚨 PROTEÇÃO: PnL não pode ser absurdo
                        if abs(pnl) > base_balance * 2:  # PnL máximo 200% do balance
                            pnl = 0.0
                            
                        unrealized_pnl += pnl
        
        # 🔥 PORTFOLIO TOTAL: Base + PnL não realizado
        total_portfolio = base_balance + unrealized_pnl
        
        # 🔥 PROTEÇÃO FINAL: Valores realistas
        if total_portfolio < 1.0:
            total_portfolio = 1.0
        elif total_portfolio > 50000.0:  # Limite máximo realista
            total_portfolio = base_balance  # Ignorar PnL absurdo
        
        return total_portfolio


def create_simple_execution_system(initial_balance: float = 500.0):
    """🔥 FACTORY FUNCTION PARA CRIAR SISTEMA DE EXECUÇÃO SIMPLES"""
    return SimpleRewardCalculatorWithExecution(initial_balance) 