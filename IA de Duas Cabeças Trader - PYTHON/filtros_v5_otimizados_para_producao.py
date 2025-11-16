# 🚀 FILTROS V5 OTIMIZADOS PARA PRODUÇÃO/TRADING REAL
# Integração dos 6 Gates e 10 Scores da TwoHeadV5 no ambiente de trading

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import torch

class TradingFilterV5:
    """
    🎯 SISTEMA DE FILTROS V5 OTIMIZADO PARA TRADING REAL
    
    Integra os 6 Gates e 10 Scores da TwoHeadV5Intelligent48h
    para ambiente de produção com thresholds adaptativos
    """
    
    def __init__(self):
        # 🎯 THRESHOLDS ADAPTATIVOS PARA PRODUÇÃO
        self.adaptive_thresholds = {
            'temporal_gate': {
                'conservative': 0.65,   # Mercado volátil
                'moderate': 0.50,       # Mercado normal
                'aggressive': 0.35      # Mercado calmo
            },
            'validation_gate': {
                'conservative': 0.80,
                'moderate': 0.70,
                'aggressive': 0.60
            },
            'risk_gate': {
                'conservative': 0.75,
                'moderate': 0.60,
                'aggressive': 0.45
            },
            'market_gate': {
                'conservative': 0.60,
                'moderate': 0.45,
                'aggressive': 0.30
            },
            'quality_gate': {
                'conservative': 0.85,
                'moderate': 0.75,
                'aggressive': 0.65
            },
            'confidence_gate': {
                'conservative': 0.80,
                'moderate': 0.70,
                'aggressive': 0.60
            }
        }
        
        # 🎯 CONFIGURAÇÃO DINÂMICA BASEADA EM CONDIÇÕES DE MERCADO
        self.current_regime = 'moderate'  # conservative, moderate, aggressive
        self.market_fatigue_memory = []
        self.recent_performance = []
        
        # 🎯 CONTADORES PARA AJUSTE DINÂMICO
        self.consecutive_losses = 0
        self.trades_last_hour = 0
        self.drawdown_current = 0.0
        
    def get_market_regime(self, volatility: float, volume_ratio: float, trend_strength: float) -> str:
        """
        🎯 DETECTOR DE REGIME DE MERCADO PARA AJUSTE AUTOMÁTICO
        
        Returns: 'conservative', 'moderate', 'aggressive'
        """
        # Score composto para determinar regime
        regime_score = 0.0
        
        # Volatilidade (peso 40%)
        if volatility > 0.015:      # Alta volatilidade
            regime_score += 0.4 * 1.0  # Conservative
        elif volatility > 0.008:    # Volatilidade média
            regime_score += 0.4 * 0.5  # Moderate
        else:                       # Baixa volatilidade
            regime_score += 0.4 * 0.0  # Aggressive
            
        # Volume (peso 30%)
        if volume_ratio > 1.5:      # Alto volume
            regime_score += 0.3 * 0.0  # Aggressive (mais oportunidades)
        elif volume_ratio > 0.8:    # Volume normal
            regime_score += 0.3 * 0.5  # Moderate
        else:                       # Baixo volume
            regime_score += 0.3 * 1.0  # Conservative
            
        # Força da tendência (peso 30%)
        if abs(trend_strength) > 0.7:  # Tendência forte
            regime_score += 0.3 * 0.0  # Aggressive
        elif abs(trend_strength) > 0.3:  # Tendência moderada
            regime_score += 0.3 * 0.5  # Moderate
        else:                          # Mercado lateral
            regime_score += 0.3 * 1.0  # Conservative
            
        # Determinar regime
        if regime_score > 0.7:
            return 'conservative'
        elif regime_score > 0.3:
            return 'moderate'
        else:
            return 'aggressive'
    
    def apply_v5_filters(self, 
                        v5_scores: Dict[str, float], 
                        market_context: Dict[str, float],
                        trading_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        🎯 APLICAR FILTROS V5 COMPLETOS PARA TRADING REAL
        
        Args:
            v5_scores: Scores da TwoHeadV5 (10 scores)
            market_context: Contexto de mercado (volatilidade, volume, etc.)
            trading_state: Estado atual do trading (DD, trades recentes, etc.)
            
        Returns:
            Dict com decisão final e detalhes dos filtros
        """
        
        # 1. ATUALIZAR REGIME DE MERCADO
        self.current_regime = self.get_market_regime(
            market_context.get('volatility', 0.01),
            market_context.get('volume_ratio', 1.0),
            market_context.get('trend_strength', 0.0)
        )
        
        # 2. OBTER THRESHOLDS ADAPTATIVOS
        thresholds = {
            gate: self.adaptive_thresholds[gate][self.current_regime]
            for gate in self.adaptive_thresholds.keys()
        }
        
        # 3. APLICAR AJUSTES DINÂMICOS
        thresholds = self._apply_dynamic_adjustments(thresholds, trading_state)
        
        # 4. AVALIAR CADA GATE
        gate_results = {}
        
        # TEMPORAL GATE - Análise de horizonte temporal
        temporal_score = v5_scores.get('temporal', 0.0)
        gate_results['temporal'] = {
            'passed': temporal_score > thresholds['temporal_gate'],
            'score': temporal_score,
            'threshold': thresholds['temporal_gate'],
            'confidence': min(temporal_score / thresholds['temporal_gate'], 2.0)
        }
        
        # VALIDATION GATE - Multi-timeframe + Pattern validation
        validation_score = v5_scores.get('validation', 0.0)
        gate_results['validation'] = {
            'passed': validation_score > thresholds['validation_gate'],
            'score': validation_score,
            'threshold': thresholds['validation_gate'],
            'confidence': min(validation_score / thresholds['validation_gate'], 2.0)
        }
        
        # RISK GATE - Dynamic risk + Market regime
        risk_score = v5_scores.get('risk', 0.0)
        gate_results['risk'] = {
            'passed': risk_score > thresholds['risk_gate'],
            'score': risk_score,
            'threshold': thresholds['risk_gate'],
            'confidence': min(risk_score / thresholds['risk_gate'], 2.0)
        }
        
        # MARKET GATE - Lookahead + Market fatigue
        market_score = v5_scores.get('market', 0.0)
        gate_results['market'] = {
            'passed': market_score > thresholds['market_gate'],
            'score': market_score,
            'threshold': thresholds['market_gate'],
            'confidence': min(market_score / thresholds['market_gate'], 2.0)
        }
        
        # QUALITY GATE - Momentum + Volatility + Volume + Trend
        quality_score = v5_scores.get('quality', 0.0)
        gate_results['quality'] = {
            'passed': quality_score > thresholds['quality_gate'],
            'score': quality_score,
            'threshold': thresholds['quality_gate'],
            'confidence': min(quality_score / thresholds['quality_gate'], 2.0)
        }
        
        # CONFIDENCE GATE - Overall confidence
        confidence_score = v5_scores.get('confidence', 0.0)
        gate_results['confidence'] = {
            'passed': confidence_score > thresholds['confidence_gate'],
            'score': confidence_score,
            'threshold': thresholds['confidence_gate'],
            'confidence': min(confidence_score / thresholds['confidence_gate'], 2.0)
        }
        
        # 5. DECISÃO FINAL
        all_gates_passed = all(gate['passed'] for gate in gate_results.values())
        
        # 6. CALCULAR SCORE COMPOSTO
        composite_score = np.mean([gate['score'] for gate in gate_results.values()])
        
        # 7. MARKET FATIGUE DETECTOR AVANÇADO
        fatigue_analysis = self._analyze_market_fatigue(
            v5_scores.get('fatigue', 0.0),
            trading_state
        )
        
        return {
            'entry_allowed': all_gates_passed and not fatigue_analysis['critical_fatigue'],
            'composite_score': composite_score,
            'gates': gate_results,
            'market_regime': self.current_regime,
            'thresholds_used': thresholds,
            'fatigue_analysis': fatigue_analysis,
            'recommendations': self._generate_recommendations(gate_results, fatigue_analysis)
        }
    
    def _apply_dynamic_adjustments(self, thresholds: Dict, trading_state: Dict) -> Dict:
        """
        🎯 AJUSTES DINÂMICOS BASEADOS NO ESTADO DE TRADING
        """
        adjusted = thresholds.copy()
        
        # AJUSTE POR DRAWDOWN
        current_dd = trading_state.get('current_drawdown', 0.0)
        if current_dd > 0.15:  # DD > 15%
            # Tornar mais conservador
            for gate in adjusted:
                adjusted[gate] *= 1.2  # +20% nos thresholds
        elif current_dd > 0.08:  # DD > 8%
            # Tornar ligeiramente mais conservador
            for gate in adjusted:
                adjusted[gate] *= 1.1  # +10% nos thresholds
        
        # AJUSTE POR SEQUÊNCIA DE PERDAS
        consecutive_losses = trading_state.get('consecutive_losses', 0)
        if consecutive_losses >= 3:
            for gate in adjusted:
                adjusted[gate] *= 1.15  # +15% nos thresholds
        
        # AJUSTE POR ATIVIDADE EXCESSIVA
        trades_last_hour = trading_state.get('trades_last_hour', 0)
        if trades_last_hour > 5:  # Muitos trades
            adjusted['market_gate'] *= 1.3  # Aumentar threshold do market gate
            adjusted['quality_gate'] *= 1.2  # Aumentar threshold do quality gate
        
        return adjusted
    
    def _analyze_market_fatigue(self, fatigue_score: float, trading_state: Dict) -> Dict:
        """
        🎯 MARKET FATIGUE DETECTOR AVANÇADO PARA TRADING 24/7
        """
        # Histórico de fadiga
        self.market_fatigue_memory.append(fatigue_score)
        if len(self.market_fatigue_memory) > 20:
            self.market_fatigue_memory.pop(0)
        
        # Análise de tendência de fadiga
        if len(self.market_fatigue_memory) >= 5:
            recent_avg = np.mean(self.market_fatigue_memory[-5:])
            older_avg = np.mean(self.market_fatigue_memory[-10:-5]) if len(self.market_fatigue_memory) >= 10 else recent_avg
            fatigue_trend = recent_avg - older_avg
        else:
            fatigue_trend = 0.0
        
        # Classificação de fadiga
        if fatigue_score > 0.8:
            fatigue_level = 'critical'
        elif fatigue_score > 0.6:
            fatigue_level = 'high'
        elif fatigue_score > 0.4:
            fatigue_level = 'moderate'
        else:
            fatigue_level = 'low'
        
        # Fatores adicionais
        trades_last_2h = trading_state.get('trades_last_2h', 0)
        avg_trade_duration = trading_state.get('avg_trade_duration', 60)  # minutos
        
        # Fatiga crítica (bloquear entradas)
        critical_fatigue = (
            fatigue_score > 0.85 or
            (fatigue_score > 0.7 and fatigue_trend > 0.1) or
            trades_last_2h > 8 or
            avg_trade_duration < 15  # Trades muito rápidos
        )
        
        return {
            'fatigue_score': fatigue_score,
            'fatigue_level': fatigue_level,
            'fatigue_trend': fatigue_trend,
            'critical_fatigue': critical_fatigue,
            'trades_last_2h': trades_last_2h,
            'avg_trade_duration': avg_trade_duration,
            'recommendation': self._get_fatigue_recommendation(fatigue_level, critical_fatigue)
        }
    
    def _get_fatigue_recommendation(self, level: str, critical: bool) -> str:
        """Recomendações baseadas no nível de fadiga"""
        if critical:
            return "PARAR TRADING - Fadiga crítica detectada"
        elif level == 'high':
            return "REDUZIR ATIVIDADE - Alta fadiga"
        elif level == 'moderate':
            return "CAUTELA - Fadiga moderada"
        else:
            return "NORMAL - Baixa fadiga"
    
    def _generate_recommendations(self, gates: Dict, fatigue: Dict) -> List[str]:
        """
        🎯 GERADOR DE RECOMENDAÇÕES BASEADO NOS FILTROS
        """
        recommendations = []
        
        # Análise dos gates
        failed_gates = [name for name, result in gates.items() if not result['passed']]
        
        if not failed_gates:
            recommendations.append("✅ TODOS OS FILTROS PASSARAM - Trade de alta qualidade")
        else:
            recommendations.append(f"❌ FILTROS REPROVADOS: {', '.join(failed_gates)}")
            
            # Recomendações específicas por gate
            if 'temporal' in failed_gates:
                recommendations.append("⏰ AGUARDAR - Timing temporal inadequado")
            if 'validation' in failed_gates:
                recommendations.append("📊 VALIDAR - Padrões multi-timeframe inconsistentes")
            if 'risk' in failed_gates:
                recommendations.append("⚠️ RISCO ALTO - Condições desfavoráveis")
            if 'market' in failed_gates:
                recommendations.append("📈 MERCADO - Lookahead ou fadiga problemática")
            if 'quality' in failed_gates:
                recommendations.append("🎯 QUALIDADE BAIXA - Momentum/volatilidade inadequados")
            if 'confidence' in failed_gates:
                recommendations.append("🤔 BAIXA CONFIANÇA - Incerteza do modelo")
        
        # Fadiga
        if fatigue['critical_fatigue']:
            recommendations.append(f"🚨 {fatigue['recommendation']}")
        elif fatigue['fatigue_level'] in ['high', 'moderate']:
            recommendations.append(f"😴 {fatigue['recommendation']}")
        
        return recommendations

class QualityFiltersV5:
    """
    🎯 QUALITY FILTERS ESPECIALIZADOS PARA DIFERENTES CONDIÇÕES DE MERCADO
    """
    
    def __init__(self):
        self.market_conditions = {
            'trending': {
                'momentum_weight': 0.4,
                'volatility_weight': 0.2,
                'volume_weight': 0.2,
                'trend_weight': 0.2
            },
            'ranging': {
                'momentum_weight': 0.2,
                'volatility_weight': 0.3,
                'volume_weight': 0.3,
                'trend_weight': 0.2
            },
            'volatile': {
                'momentum_weight': 0.3,
                'volatility_weight': 0.4,
                'volume_weight': 0.2,
                'trend_weight': 0.1
            },
            'calm': {
                'momentum_weight': 0.3,
                'volatility_weight': 0.1,
                'volume_weight': 0.4,
                'trend_weight': 0.2
            }
        }
    
    def get_market_condition(self, volatility: float, trend_strength: float, volume_ratio: float) -> str:
        """Detectar condição de mercado atual"""
        if volatility > 0.015:
            return 'volatile'
        elif abs(trend_strength) > 0.6:
            return 'trending'
        elif volatility < 0.005:
            return 'calm'
        else:
            return 'ranging'
    
    def calculate_quality_score(self, 
                               momentum_score: float,
                               volatility_score: float, 
                               volume_score: float,
                               trend_score: float,
                               market_context: Dict) -> Dict:
        """
        🎯 CALCULAR SCORE DE QUALIDADE ADAPTATIVO
        """
        condition = self.get_market_condition(
            market_context.get('volatility', 0.01),
            market_context.get('trend_strength', 0.0),
            market_context.get('volume_ratio', 1.0)
        )
        
        weights = self.market_conditions[condition]
        
        quality_score = (
            momentum_score * weights['momentum_weight'] +
            volatility_score * weights['volatility_weight'] +
            volume_score * weights['volume_weight'] +
            trend_score * weights['trend_weight']
        )
        
        return {
            'quality_score': quality_score,
            'market_condition': condition,
            'weights_used': weights,
            'component_scores': {
                'momentum': momentum_score,
                'volatility': volatility_score,
                'volume': volume_score,
                'trend': trend_score
            }
        }

# 🎯 EXEMPLO DE INTEGRAÇÃO NO PPOV1.PY
class V5FilterIntegration:
    """
    🎯 CLASSE PARA INTEGRAÇÃO DOS FILTROS V5 NO AMBIENTE DE TRADING
    """
    
    def __init__(self, env):
        self.env = env
        self.trading_filter = TradingFilterV5()
        self.quality_filter = QualityFiltersV5()
        
    def should_allow_entry(self, action, v5_output=None) -> Tuple[bool, Dict]:
        """
        🎯 DECISÃO FINAL DE ENTRADA BASEADA NOS FILTROS V5
        
        Para ser usado no método step() do ambiente
        """
        if v5_output is None or 'scores' not in v5_output:
            # Fallback para filtros básicos se V5 não disponível
            return self._basic_filters(action)
        
        # Preparar contexto
        market_context = {
            'volatility': self._get_current_volatility(),
            'volume_ratio': self._get_volume_ratio(),
            'trend_strength': self._get_trend_strength()
        }
        
        trading_state = {
            'current_drawdown': self.env.current_drawdown,
            'consecutive_losses': getattr(self.env, 'consecutive_losses', 0),
            'trades_last_hour': self._count_recent_trades(60),  # 60 minutos
            'trades_last_2h': self._count_recent_trades(120),   # 120 minutos
            'avg_trade_duration': self._get_avg_trade_duration()
        }
        
        # Aplicar filtros V5
        filter_result = self.trading_filter.apply_v5_filters(
            v5_output['scores'],
            market_context,
            trading_state
        )
        
        return filter_result['entry_allowed'], filter_result
    
    def _basic_filters(self, action) -> Tuple[bool, Dict]:
        """Filtros básicos de fallback"""
        # Implementar filtros básicos aqui
        return True, {'type': 'basic_fallback'}
    
    def _get_current_volatility(self) -> float:
        """Calcular volatilidade atual"""
        if hasattr(self.env, 'df') and len(self.env.df) > 20:
            recent_prices = self.env.df['close_5m'].iloc[-20:].values
            returns = np.diff(np.log(recent_prices))
            return np.std(returns) * np.sqrt(252 * 24 * 12)  # Anualizada
        return 0.01
    
    def _get_volume_ratio(self) -> float:
        """Calcular ratio de volume atual vs média"""
        if hasattr(self.env, 'df') and len(self.env.df) > 20:
            current_volume = self.env.df['volume_5m'].iloc[-1]
            avg_volume = self.env.df['volume_5m'].iloc[-20:].mean()
            return current_volume / avg_volume if avg_volume > 0 else 1.0
        return 1.0
    
    def _get_trend_strength(self) -> float:
        """Calcular força da tendência"""
        if hasattr(self.env, 'df') and len(self.env.df) > 20:
            prices = self.env.df['close_5m'].iloc[-20:].values
            sma_short = np.mean(prices[-5:])
            sma_long = np.mean(prices[-20:])
            return (sma_short - sma_long) / sma_long
        return 0.0
    
    def _count_recent_trades(self, minutes: int) -> int:
        """Contar trades recentes"""
        if not hasattr(self.env, 'trades') or not self.env.trades:
            return 0
        
        current_time = self.env.current_step
        cutoff_time = current_time - (minutes // 5)  # Converter para steps de 5min
        
        recent_trades = 0
        for trade in self.env.trades:
            if trade.get('entry_step', 0) >= cutoff_time:
                recent_trades += 1
        
        return recent_trades
    
    def _get_avg_trade_duration(self) -> float:
        """Calcular duração média dos trades"""
        if not hasattr(self.env, 'trades') or not self.env.trades:
            return 60.0
        
        durations = []
        for trade in self.env.trades[-10:]:  # Últimos 10 trades
            if 'entry_step' in trade and 'exit_step' in trade:
                duration = (trade['exit_step'] - trade['entry_step']) * 5  # minutos
                durations.append(duration)
        
        return np.mean(durations) if durations else 60.0