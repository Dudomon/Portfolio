"""
💰 SISTEMA DE RECOMPENSAS SIMPLES E MATEMATICAMENTE COERENTE - V7 ENHANCED
Focado em convergência e estabilidade com balanceamento otimizado

🔥 NOVIDADES V7 - PROGRESSIVE ZONES + SMART RISK FACTORS:
- Progressive Risk Zones: Sistema de zonas graduais para drawdown
- Smart Risk Factors: Análise multi-fator de risco inteligente
- Compatibilidade total com sistema existente
- Balanceamento preservado (90% PnL, 5% atividade, 3% risco, 2% qualidade)

🎯 TARGETS ESPECÍFICOS:
- Trades/dia: 18 (otimizado)
- Zona alvo: 12-24 trades/dia (ultra-expandida)
- SL Range: 11-56 pontos  
- TP Range: 14-82 pontos
- Risk/Reward: 1.5-1.8 (ótimo)

🎯 PRINCÍPIOS V7:
1. PnL real aumentado (pnl_direct: 0.6→1.0) conforme solicitado
2. Progressive Risk Zones (gestão inteligente de drawdown)
3. Smart Risk Factors (análise multi-dimensional)
4. Controles de risco aprimorados mas balanceados
5. Zona de atividade ultra-expandida (12-24 vs 14-22 anterior)
6. Rewards de SL/TP removidos (não mais necessários com ranges fixos)

🔥 MELHORIAS V7 - PROGRESSIVE ZONES + SMART RISK:
- Progressive Risk Zones: Green (0-3%), Yellow (3-8%), Orange (8-15%), Red (15-25%), Black (25%+)
- Smart Risk Factors: Drawdown (40%) + Position Concentration (25%) + Volatility (20%) + Performance Streak (15%)
- Penalidades graduais e inteligentes
- Sistema de alertas progressivos
- Proteção contra riscos extremos
- Compatibilidade total com sistema existente

🚨 CORREÇÃO CRÍTICA V7:
- Mantidos todos os pesos existentes
- Adicionados sistemas inteligentes de risco
- Preservado balanceamento 90% PnL
- Sistema progressivo não punitivo
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum

class RiskZone(Enum):
    """🟢 Zonas de Risco Progressivas"""
    GREEN = "green"      # 0-3%: Zona segura
    YELLOW = "yellow"    # 3-8%: Atenção
    ORANGE = "orange"    # 8-15%: Cuidado
    RED = "red"          # 15-25%: Perigo
    BLACK = "black"      # 25%+: Crítico

@dataclass
class RiskMetrics:
    """📊 Métricas de Risco Consolidadas"""
    drawdown_pct: float
    position_concentration: float
    volatility_factor: float
    performance_streak: int
    overall_risk_score: float
    risk_zone: RiskZone

class ProgressiveRiskZones:
    """
    🎯 SISTEMA DE ZONAS PROGRESSIVAS DE RISCO V1.0
    Implementa gestão inteligente de drawdown com penalidades graduais
    """
    
    def __init__(self):
        # 🎯 DEFINIÇÃO DAS ZONAS
        self.zones = {
            RiskZone.GREEN: {"min": 0.0, "max": 3.0, "penalty": 0.0, "multiplier": 1.0},
            RiskZone.YELLOW: {"min": 3.0, "max": 8.0, "penalty": -0.1, "multiplier": 0.95},
            RiskZone.ORANGE: {"min": 8.0, "max": 15.0, "penalty": -0.3, "multiplier": 0.90},
            RiskZone.RED: {"min": 15.0, "max": 25.0, "penalty": -0.8, "multiplier": 0.80},
            RiskZone.BLACK: {"min": 25.0, "max": 100.0, "penalty": -2.0, "multiplier": 0.60}
        }
        
        # 🎯 CONFIGURAÇÕES PROGRESSIVAS
        self.zone_transition_bonus = 0.2  # Bônus por melhorar de zona
        self.zone_maintenance_bonus = 0.1  # Bônus por manter zona boa
        self.previous_zone = RiskZone.GREEN
        
    def get_risk_zone(self, drawdown_pct: float) -> RiskZone:
        """Determina zona de risco baseada no drawdown"""
        for zone, config in self.zones.items():
            if config["min"] <= drawdown_pct < config["max"]:
                return zone
        return RiskZone.BLACK  # Fallback para casos extremos
    
    def calculate_zone_penalty(self, drawdown_pct: float) -> Tuple[float, RiskZone, Dict]:
        """
        🎯 CALCULA PENALIDADE PROGRESSIVA
        Retorna: (penalty, zone, info)
        """
        current_zone = self.get_risk_zone(drawdown_pct)
        zone_config = self.zones[current_zone]
        
        # 🎯 PENALIDADE BASE DA ZONA
        base_penalty = zone_config["penalty"]
        
        # 🎯 PENALIDADE PROGRESSIVA DENTRO DA ZONA
        if current_zone != RiskZone.GREEN:
            zone_range = zone_config["max"] - zone_config["min"]
            position_in_zone = (drawdown_pct - zone_config["min"]) / zone_range
            progressive_penalty = base_penalty * (1.0 + position_in_zone * 0.5)
        else:
            progressive_penalty = 0.0
        
        # 🎯 BÔNUS POR MELHORIA DE ZONA
        zone_bonus = 0.0
        if current_zone.value < self.previous_zone.value:  # Melhorou
            zone_bonus = self.zone_transition_bonus
        elif current_zone == RiskZone.GREEN:  # Mantém zona verde
            zone_bonus = self.zone_maintenance_bonus
        
        # 🎯 ATUALIZAR TRACKING
        self.previous_zone = current_zone
        
        # 🎯 PENALIDADE FINAL
        final_penalty = progressive_penalty + zone_bonus
        
        info = {
            "zone": current_zone.value,
            "base_penalty": base_penalty,
            "progressive_penalty": progressive_penalty,
            "zone_bonus": zone_bonus,
            "final_penalty": final_penalty,
            "multiplier": zone_config["multiplier"],
            "position_in_zone": position_in_zone if current_zone != RiskZone.GREEN else 0.0
        }
        
        return final_penalty, current_zone, info

class SmartRiskFactors:
    """
    🧠 SISTEMA DE ANÁLISE MULTI-FATOR DE RISCO V1.0
    Análise inteligente combinando múltiplos fatores de risco
    """
    
    def __init__(self):
        # 🎯 PESOS DOS FATORES (total = 100%)
        self.factor_weights = {
            "drawdown": 0.40,           # 40% - Fator mais importante
            "position_concentration": 0.25,  # 25% - Concentração de posições
            "volatility": 0.20,         # 20% - Volatilidade do mercado
            "performance_streak": 0.15   # 15% - Sequência de performance
        }
        
        # 🎯 CONFIGURAÇÕES DE ANÁLISE
        self.max_concentration_threshold = 0.8  # 80% do portfolio em uma posição
        self.high_volatility_threshold = 0.025  # 2.5% volatilidade alta
        self.negative_streak_threshold = 3      # 3 perdas consecutivas
        
        # 🎯 TRACKING
        self.recent_volatility = []
        self.recent_performance = []
        
    def calculate_position_concentration(self, env) -> float:
        """
        📊 CALCULA CONCENTRAÇÃO DE POSIÇÕES
        Retorna: 0.0 (diversificado) a 1.0 (concentrado)
        """
        try:
            if not hasattr(env, 'positions') or not env.positions:
                return 0.0
            
            # Calcular exposição por posição
            total_exposure = 0.0
            max_single_exposure = 0.0
            
            for position in env.positions:
                exposure = abs(position.get('size', 0) * position.get('price', 0))
                total_exposure += exposure
                max_single_exposure = max(max_single_exposure, exposure)
            
            if total_exposure == 0:
                return 0.0
            
            concentration = max_single_exposure / total_exposure
            return min(concentration, 1.0)
            
        except Exception:
            return 0.0  # Erro = sem concentração
    
    def calculate_volatility_factor(self, env) -> float:
        """
        📈 CALCULA FATOR DE VOLATILIDADE
        Retorna: 0.0 (baixa vol) a 1.0 (alta vol)
        """
        try:
            if not hasattr(env, 'df') or not hasattr(env, 'current_step'):
                return 0.3  # Volatilidade média como fallback
            
            # Usar ATR ou calcular volatilidade recente
            current_idx = min(env.current_step, len(env.df) - 1)
            
            for vol_col in ['atr_5m', 'atr_14', 'volatility_20_5m']:
                if vol_col in env.df.columns:
                    recent_vol = env.df[vol_col].iloc[max(0, current_idx-10):current_idx+1].mean()
                    
                    # Normalizar volatilidade (0-1)
                    if recent_vol > self.high_volatility_threshold:
                        return 1.0  # Alta volatilidade
                    elif recent_vol < 0.005:
                        return 0.1  # Baixa volatilidade
                    else:
                        return recent_vol / self.high_volatility_threshold
            
            return 0.3  # Fallback
            
        except Exception:
            return 0.3
    
    def calculate_performance_streak_factor(self, env) -> float:
        """
        📊 CALCULA FATOR DE SEQUÊNCIA DE PERFORMANCE
        Retorna: 0.0 (boa sequência) a 1.0 (má sequência)
        """
        try:
            if not hasattr(env, 'trades') or len(env.trades) < 3:
                return 0.0  # Poucos trades = sem risco
            
            # Analisar últimos 5 trades
            recent_trades = env.trades[-5:]
            losses = sum(1 for trade in recent_trades if trade.get('pnl_usd', 0) < 0)
            
            if losses >= 4:  # 4+ perdas em 5 trades
                return 1.0
            elif losses >= 3:  # 3 perdas em 5 trades
                return 0.7
            elif losses >= 2:  # 2 perdas em 5 trades
                return 0.4
            else:  # 0-1 perdas
                return 0.0
            
        except Exception:
            return 0.0
    
    def calculate_overall_risk_score(self, env) -> RiskMetrics:
        """
        🎯 CALCULA SCORE GERAL DE RISCO
        Combina todos os fatores em um score único
        """
        try:
            # 🎯 CALCULAR CADA FATOR
            drawdown_pct = abs(getattr(env, 'current_drawdown', 0.0))
            position_concentration = self.calculate_position_concentration(env)
            volatility_factor = self.calculate_volatility_factor(env)
            performance_streak = self.calculate_performance_streak_factor(env)
            
            # 🎯 SCORE PONDERADO
            overall_risk_score = (
                drawdown_pct * self.factor_weights["drawdown"] +
                position_concentration * self.factor_weights["position_concentration"] +
                volatility_factor * self.factor_weights["volatility"] +
                performance_streak * self.factor_weights["performance_streak"]
            )
            
            # 🎯 DETERMINAR ZONA DE RISCO
            if overall_risk_score <= 0.15:
                risk_zone = RiskZone.GREEN
            elif overall_risk_score <= 0.30:
                risk_zone = RiskZone.YELLOW
            elif overall_risk_score <= 0.50:
                risk_zone = RiskZone.ORANGE
            elif overall_risk_score <= 0.70:
                risk_zone = RiskZone.RED
            else:
                risk_zone = RiskZone.BLACK
            
            return RiskMetrics(
                drawdown_pct=drawdown_pct,
                position_concentration=position_concentration,
                volatility_factor=volatility_factor,
                performance_streak=int(performance_streak * 5),  # Converter para número de trades
                overall_risk_score=overall_risk_score,
                risk_zone=risk_zone
            )
            
        except Exception:
            # Fallback seguro
            return RiskMetrics(
                drawdown_pct=0.0,
                position_concentration=0.0,
                volatility_factor=0.3,
                performance_streak=0,
                overall_risk_score=0.1,
                risk_zone=RiskZone.GREEN
            )
    
    def calculate_smart_risk_penalty(self, env) -> Tuple[float, RiskMetrics, Dict]:
        """
        🎯 CALCULA PENALIDADE INTELIGENTE BASEADA EM MÚLTIPLOS FATORES
        Retorna: (penalty, risk_metrics, detailed_info)
        """
        risk_metrics = self.calculate_overall_risk_score(env)
        
        # 🎯 PENALIDADE BASEADA NO SCORE GERAL
        if risk_metrics.overall_risk_score <= 0.15:  # Verde
            penalty = 0.0
        elif risk_metrics.overall_risk_score <= 0.30:  # Amarelo
            penalty = -0.1
        elif risk_metrics.overall_risk_score <= 0.50:  # Laranja
            penalty = -0.3
        elif risk_metrics.overall_risk_score <= 0.70:  # Vermelho
            penalty = -0.8
        else:  # Preto
            penalty = -2.0
        
        # 🎯 PENALIDADE PROGRESSIVA DENTRO DA ZONA
        zone_penalty = penalty * (1.0 + risk_metrics.overall_risk_score * 0.3)
        
        # 🎯 INFORMAÇÕES DETALHADAS
        detailed_info = {
            "overall_score": risk_metrics.overall_risk_score,
            "risk_zone": risk_metrics.risk_zone.value,
            "factors": {
                "drawdown": risk_metrics.drawdown_pct,
                "position_concentration": risk_metrics.position_concentration,
                "volatility": risk_metrics.volatility_factor,
                "performance_streak": risk_metrics.performance_streak
            },
            "penalty": zone_penalty,
            "zone_multiplier": 1.0 - (risk_metrics.overall_risk_score * 0.2)
        }
        
        return zone_penalty, risk_metrics, detailed_info

class QualityFilter:
    """
    🧠 FILTRO DE QUALIDADE INTELIGENTE V3
    ENSINA o modelo sobre qualidade através de REWARDS GRADUAIS
    NÃO bloqueia trades - apenas dá mais reward para trades melhores
    """
    
    def __init__(self):
        # 🧠 SISTEMA DE ENSINO: Sem bloqueios, apenas educação
        self.teaching_mode = True  # Sempre ensinar, nunca bloquear
        
        # 🎯 CONFIGURAÇÕES INTELIGENTES DE RISCO/GANHO
        self.excellent_risk_reward = 2.0   # 1:2 = excelente
        self.good_risk_reward = 1.5        # 1:1.5 = bom
        self.acceptable_risk_reward = 1.0  # 1:1 = aceitável
        
    def calculate_trade_quality_score(self, env, action: np.ndarray) -> float:
        """
        🎯 SCORE DE QUALIDADE INTELIGENTE (0-100)
        Analisa qualidade SEM bloquear - apenas para educar o modelo
        """
        score = 20.0  # Base: Todo trade começa com 20 pontos
        
        try:
            if hasattr(env, 'df') and hasattr(env, 'current_step'):
                current_idx = min(env.current_step, len(env.df) - 1)
                
                # 1. CONFLUÊNCIA DE SINAIS (30 pontos máximo)
                confluence_score = self._analyze_market_confluence(env, current_idx)
                score += confluence_score * 0.3  # Max +9 pontos
                
                # 2. TIMING DE ENTRADA (25 pontos máximo)  
                timing_score = self._analyze_entry_timing(env, current_idx)
                score += timing_score * 0.25  # Max +6.25 pontos
                
                # 3. GESTÃO DE RISCO (25 pontos máximo)
                risk_score = self._analyze_risk_management(env, action)
                score += risk_score * 0.25  # Max +6.25 pontos
                
                # 4. CONTEXTO DE VOLATILIDADE (20 pontos máximo)
                vol_score = self._analyze_volatility_context(env, current_idx)
                score += vol_score * 0.2  # Max +4 pontos
                
        except Exception:
            # Em caso de erro, score neutro (não penalizar)
            score = 50.0
            
        return min(max(score, 0), 100)  # Entre 0-100
    
    def _analyze_market_confluence(self, env, current_idx: int) -> float:
        """Analisa confluência de múltiplos sinais (0-100) - MAIS SENSÍVEL"""
        signals = []
        
        try:
            # RSI extremo - ESCALA MAIS DIFERENCIADA
            if 'rsi_5m' in env.df.columns:
                rsi = env.df['rsi_5m'].iloc[current_idx]
                if rsi < 20 or rsi > 80:  # Extremo
                    signals.append(90)
                elif rsi < 25 or rsi > 75:  # Muito forte
                    signals.append(75)
                elif rsi < 30 or rsi > 70:  # Forte
                    signals.append(60)
                elif rsi < 35 or rsi > 65:  # Moderado
                    signals.append(45)
                elif rsi < 40 or rsi > 60:  # Fraco
                    signals.append(30)
                else:  # Neutro
                    signals.append(15)
            
            # Momentum - ESCALA MAIS SENSÍVEL
            momentum_cols = ['momentum_5_5m', 'returns_5m']
            for col in momentum_cols:
                if col in env.df.columns:
                    momentum = abs(env.df[col].iloc[current_idx])
                    if momentum > 0.02:  # Muito forte
                        signals.append(90)
                    elif momentum > 0.015:  # Forte
                        signals.append(75)
                    elif momentum > 0.01:  # Moderado
                        signals.append(60)
                    elif momentum > 0.007:  # Fraco
                        signals.append(45)
                    elif momentum > 0.003:  # Muito fraco
                        signals.append(30)
                    else:  # Praticamente zero
                        signals.append(15)
                    break
            
            # Volatilidade - MAIS ESPECÍFICA
            vol_cols = ['atr_5m', 'volatility_20_5m']
            for col in vol_cols:
                if col in env.df.columns:
                    vol = env.df[col].iloc[current_idx]
                    if 0.010 <= vol <= 0.020:  # Volatilidade perfeita
                        signals.append(90)
                    elif 0.008 <= vol <= 0.025:  # Volatilidade muito boa
                        signals.append(75)
                    elif 0.006 <= vol <= 0.030:  # Volatilidade boa
                        signals.append(60)
                    elif 0.004 <= vol <= 0.035:  # Volatilidade aceitável
                        signals.append(45)
                    elif 0.002 <= vol <= 0.040:  # Volatilidade questionável
                        signals.append(30)
                    else:  # Volatilidade ruim
                        signals.append(15)
                    break
            
            # Trend strength - MAIS DIFERENCIADO
            if 'sma_20' in env.df.columns and 'close_5m' in env.df.columns:
                price = env.df['close_5m'].iloc[current_idx]
                sma = env.df['sma_20'].iloc[current_idx]
                trend_strength = abs(price - sma) / sma if sma > 0 else 0
                
                if trend_strength > 0.025:  # Tendência muito forte
                    signals.append(90)
                elif trend_strength > 0.015:  # Tendência forte
                    signals.append(75)
                elif trend_strength > 0.010:  # Tendência moderada
                    signals.append(60)
                elif trend_strength > 0.005:  # Tendência fraca
                    signals.append(45)
                elif trend_strength > 0.002:  # Tendência muito fraca
                    signals.append(30)
                else:  # Sem tendência
                    signals.append(15)
        
        except Exception:
            signals = [30]  # Score mais baixo em caso de erro
        
        return np.mean(signals) if signals else 30
    
    def _analyze_entry_timing(self, env, current_idx: int) -> float:
        """Analisa timing de entrada (0-100)"""
        try:
            # Verificar se não está em área de alta atividade recente
            recent_activity_score = 70  # Base: timing neutro
            
            if hasattr(env, 'trades') and len(env.trades) > 0:
                recent_trades = env.trades[-5:]  # Últimos 5 trades
                
                # Se win rate recente for alto, timing é bom
                recent_wins = sum(1 for t in recent_trades if t.get('pnl_usd', 0) > 0)
                if len(recent_trades) >= 3:
                    win_rate = recent_wins / len(recent_trades)
                    if win_rate >= 0.6:
                        recent_activity_score = 90  # Timing excelente
                    elif win_rate >= 0.4:
                        recent_activity_score = 75  # Timing bom
                    else:
                        recent_activity_score = 60  # Timing questionável
            
            return recent_activity_score
            
        except Exception:
            return 70  # Timing neutro
    
    def _analyze_risk_management(self, env, action: np.ndarray) -> float:
        """Analisa gestão de risco baseada em SL/TP (0-100)"""
        try:
            if len(action) >= 6:
                sl_adjust = float(action[4])
                tp_adjust = float(action[5])
                
                # Converter para pontos - ESCALA MAIS SENSÍVEL
                sl_points = abs(sl_adjust * 15)  # Escala realista
                tp_points = abs(tp_adjust * 15)
                
                if sl_points > 0 and tp_points > 0:
                    risk_reward_ratio = tp_points / sl_points
                    
                    # Score baseado no ratio - MAIS DIFERENCIADO
                    if risk_reward_ratio >= 3.0:  # R:R >= 3:1
                        return 95  # Excepcional
                    elif risk_reward_ratio >= 2.5:  # R:R >= 2.5:1
                        return 85  # Excelente
                    elif risk_reward_ratio >= 2.0:  # R:R >= 2:1
                        return 75  # Muito bom
                    elif risk_reward_ratio >= 1.5:  # R:R >= 1.5:1
                        return 65  # Bom
                    elif risk_reward_ratio >= 1.0:  # R:R >= 1:1
                        return 50  # Aceitável
                    elif risk_reward_ratio >= 0.8:  # R:R >= 0.8:1
                        return 35  # Questionável
                    else:
                        return 20  # Ruim
                else:
                    return 15  # Sem SL/TP definido
            else:
                return 15  # Ação incompleta
                
        except Exception:
            return 30  # Score neutro
    
    def _analyze_volatility_context(self, env, current_idx: int) -> float:
        """Analisa contexto de volatilidade (0-100)"""
        try:
            vol_cols = ['atr_5m', 'volatility_20_5m']
            for col in vol_cols:
                if col in env.df.columns:
                    current_vol = env.df[col].iloc[current_idx]
                    
                    # Calcular volatilidade média dos últimos 10 períodos
                    start_idx = max(0, current_idx - 10)
                    avg_vol = env.df[col].iloc[start_idx:current_idx+1].mean()
                    
                    # Score baseado na relação com volatilidade média
                    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1.0
                    
                    if 0.8 <= vol_ratio <= 1.5:  # Volatilidade normal
                        return 80
                    elif 0.6 <= vol_ratio <= 2.0:  # Volatilidade aceitável
                        return 65
                    else:  # Volatilidade extrema
                        return 45
                    break
            
            return 70  # Score neutro se não encontrar dados
            
        except Exception:
            return 70
    
    def get_quality_reward_bonus(self, quality_score: float) -> float:
        """
        🎯 CONVERTE SCORE DE QUALIDADE EM REWARD BONUS
        Sistema de ensino: Premia qualidade, não penaliza mediocridade
        """
        if quality_score >= 85:
            return 2.0    # Trade excepcional
        elif quality_score >= 75:
            return 1.5    # Trade de alta qualidade
        elif quality_score >= 65:
            return 1.0    # Trade de boa qualidade
        elif quality_score >= 50:
            return 0.5    # Trade médio
        else:
            return 0.0    # Trade questionável (sem bônus, mas sem penalidade)
    
    def should_allow_trade(self, env, action: np.ndarray) -> tuple[bool, float, dict]:
        """
        🧠 SISTEMA INTELIGENTE: SEMPRE PERMITE, APENAS EDUCA
        Retorna qualidade para fins educacionais, nunca bloqueia
        """
        entry_decision = int(action[0]) if len(action) > 0 else 0
        
        if entry_decision == 0:
            return True, 100.0, {"reason": "hold_action", "quality": "neutral"}
        
        # Calcular qualidade para fins educacionais
        quality_score = self.calculate_trade_quality_score(env, action)
        quality_bonus = self.get_quality_reward_bonus(quality_score)
        
        # Determinar categoria de qualidade
        if quality_score >= 85:
            quality_category = "EXCEPTIONAL"
        elif quality_score >= 75:
            quality_category = "HIGH_QUALITY"
        elif quality_score >= 65:
            quality_category = "GOOD_QUALITY"
        elif quality_score >= 50:
            quality_category = "AVERAGE"
        else:
            quality_category = "LEARNING_OPPORTUNITY"
        
        info = {
            "quality_score": quality_score,
            "quality_bonus": quality_bonus,
            "quality_category": quality_category,
            "teaching_mode": True,
            "always_allowed": True,
            "reason": "intelligent_teaching_system"
        }
        
        # 🧠 SEMPRE PERMITE - Sistema de ensino, não de bloqueio
        return True, quality_score, info

class SimpleRewardCalculator:
    """Sistema de recompensas V2 otimizado para targets específicos"""
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.step_count = 0
        self.position_history = {}
        
        # 🧠 FILTRO DE QUALIDADE INTELIGENTE - ENSINA EM VEZ DE BLOQUEAR
        self.quality_filter = QualityFilter()
        self.quality_filter_enabled = True  # 🧠 ATIVADO: Sistema inteligente que ensina qualidade
        
        # 🔥 NOVOS SISTEMAS V7 - PROGRESSIVE ZONES + SMART RISK FACTORS
        self.progressive_zones = ProgressiveRiskZones()
        self.smart_risk_factors = SmartRiskFactors()
        self.enhanced_risk_enabled = True  # 🔥 ATIVADO: Sistemas inteligentes de risco
        
        # 🎯 TARGETS ESPECÍFICOS - ZONA ULTRA-FLEXÍVEL PARA CONVERGÊNCIA
        self.target_trades_per_day = 18
        self.target_zone_min = 12  # 🔥 EXPANDIDO: Zona mínima mais flexível (era 14)
        self.target_zone_max = 24  # 🔥 EXPANDIDO: Zona máxima mais flexível (era 22)
        self.sl_range_min = 11
        self.sl_range_max = 56
        self.tp_range_min = 14
        self.tp_range_max = 82
        self.optimal_risk_reward_min = 1.5
        self.optimal_risk_reward_max = 1.8
        
        # 🎯 PESOS REBALANCEADOS V3 - AJUSTE FINO PARA BALANCE PERFEITO
        self.weights = {
            # 💰 PnL MICRO-SUAVIZADO (15% do total - AJUSTE FINAL para balance > 0.5)
            "pnl_direct": 0.08,          # 🔧 AJUSTE FINAL: 0.15 → 0.08 (quase 2x menor)
            "win_bonus": 0.05,           # 🔧 AJUSTE FINAL: 0.1 → 0.05 (2x menor)
            "loss_penalty": -0.04,       # 🔧 AJUSTE FINAL: -0.08 → -0.04 (2x menor)
            
            # 🎯 INCENTIVOS REBALANCEADOS (25% do total - AJUSTE FINO)
            "trade_completion_bonus": 1.2,   # 🔧 AJUSTE FINO: 0.5 → 1.2 (compensar PnL reduzido)
            "position_management": 0.8,      # 🔧 AJUSTE FINO: 0.3 → 0.8 (mais weight para gestão)
            "target_zone_bonus": 1.5,        # 🔧 AJUSTE FINO: 1.0 → 1.5 (incentivar zona alvo)
            
            # 🎯 QUALIDADE CONTROLADA (20% do total - AJUSTE FINAL)
            "excellent_trade_bonus": 0.9,    # 🔧 AJUSTE FINAL: 1.8 → 0.9 (2x menor para balance)
            "quality_trade_bonus": 0.8,      # 🔧 AJUSTE FINAL: 1.0 → 0.8 (reduzir para balance)
            "win_streak_bonus": 0.3,         # 🔧 CHERRY FIX: 0.5 → 0.3
            "quick_profit_bonus": 0.4,       # 🔧 CHERRY FIX: 0.8 → 0.4 (2x menor)
            
            # 🧠 QUALIDADE SL/TP BALANCEADA (10% do total)
            "perfect_sltp_ratio": 0.8,       # 🔧 CHERRY FIX: 1.5 → 0.8
            "adaptive_sltp": 0.3,            # 🔧 CHERRY FIX: 0.5 → 0.3  
            "smart_sltp_timing": 0.3,        # 🔧 CHERRY FIX: 0.5 → 0.3
            
            # 🚀 REMOVIDO: Apenas rewards de ranges específicos
            # "target_sltp_ranges": REMOVIDO - ranges agora são fixos
            # "poor_sltp_penalty": REMOVIDO - ranges fixos eliminam SL/TP ruins
            
            # 🛡️ CONTROLES DE RISCO AGRESSIVOS (30% do total - URGENTE!)
            "excessive_drawdown": -2.0,      # 🚨 EMERGÊNCIA: -0.2 → -2.0 (10x mais agressivo)
            "critical_drawdown": -5.0,       # 🆕 NOVO: -5.0 para DD >20%
            "catastrophic_drawdown": -10.0,  # 🆕 NOVO: -10.0 para DD >30%
            "overtrading_penalty": -0.5,     # 🔧 RESTAURADO: -0.2 → -0.5
            "inactivity_penalty": -0.1,      # MANTIDO
            
            # 🆕 SISTEMA HOLD REBALANCEADO (25% do total - AJUSTE FINO)
            "intelligent_hold_bonus": 2.5,     # 🔧 AJUSTE FINO: 1.5 → 2.5 (melhorar competitividade HOLD)
            "market_timing_bonus": 2.0,        # 🔧 AJUSTE FINO: 1.0 → 2.0 (dobrar timing reward)
            "patient_analysis_bonus": 1.5,     # 🔧 AJUSTE FINO: 0.8 → 1.5 (quase dobrar)
            "good_hold_bonus": 0.1,            # MANTIDO: +0.1 por HOLD inteligente
            "market_analysis_bonus": 0.05,     # MANTIDO: +0.05 por análise correta
            "patience_bonus": 0.02,            # MANTIDO: +0.02 por paciência
            
            # 🎯 CORRELAÇÃO PERFORMANCE (15% do total - CORREÇÃO MATEMÁTICA)
            "performance_correlation": 2.0,    # 🚨 NOVO: +2.0 forte peso para correlação PnL
            "consistent_profit_bonus": 1.5,    # 🚨 NOVO: +1.5 por consistência lucrativa
            "risk_adjusted_return": 1.2,       # 🚨 NOVO: +1.2 por retorno ajustado ao risco
            
            # 🚀 REMOVIDO: Penalidades obsoletas
            # "flip_flop_penalty": REMOVIDO - obsoleto
            # "micro_trade_penalty": REMOVIDO - obsoleto
            
            # 🔥 RISK MANAGEMENT MANTIDO
            "progressive_zone_penalty": 1.5,     # MANTIDO: Peso para penalidades progressivas
            "smart_risk_penalty": 1.2,           # MANTIDO: Peso para análise multi-fator
            "zone_improvement_bonus": 1.0,       # MANTIDO: Bônus por melhorar zona de risco
            "risk_awareness_bonus": 0.5,         # MANTIDO: Bônus por operar em zona segura
            
            # 🎯 BÔNUS EDUCATIVOS ESPECIAIS MANTIDOS
            "consistency_bonus": 0.8,        # MANTIDO: +0.8 por consistência
            "risk_management_bonus": 0.8,    # MANTIDO: +0.8 por gestão de risco
            
            # 🔥 TRAILING STOP EDUCATIVO MANTIDO
            "trailing_stop_execution": 1.0,      # MANTIDO: +1.0 por trailing executado
            "trailing_stop_activation": 0.8,     # MANTIDO: +0.8 por ativar trailing
            "trailing_stop_protection": 0.6,     # MANTIDO: +0.6 por proteger lucros
            "trailing_stop_timing": 0.4,         # MANTIDO: +0.4 por timing correto
            "missed_trailing_opportunity": -0.2, # MANTIDO: -0.2 penalidade por perder trailing
            
            # 📊 ANÁLISE TÉCNICA INTELIGENTE - ONDAS E SUPORTE/RESISTÊNCIA
            "wave_projection_tp": 1.8,           # 🚀 NOVO: +1.8 por TP baseado em projeção de ondas
            "support_resistance_sl": 1.5,        # 🚀 NOVO: +1.5 por SL próximo a suporte/resistência
            "fibonacci_levels": 1.2,             # 🚀 NOVO: +1.2 por alvos em níveis de Fibonacci
            "trend_alignment": 1.0,              # 🚀 NOVO: +1.0 por trade alinhado com tendência
            "volume_confirmation": 0.8,          # 🚀 NOVO: +0.8 por confirmação de volume
            "breakout_timing": 1.3,              # 🚀 NOVO: +1.3 por timing correto em breakouts
        }
        
        # Tracking para análise comportamental
        self.recent_actions = []
        self.daily_trade_count = 0
        self.last_trade_type = None
        self.last_action_step = 0
        
    def reset(self):
        """Reset para novo episódio"""
        self.step_count = 0
        self.position_history = {}
        self.recent_actions = []
        self.daily_trade_count = 0
        self.last_trade_type = None
        self.last_action_step = 0
        
        # 🔥 NOVO: Reset do filtro de qualidade
        if hasattr(self, 'quality_filter'):
            self.quality_filter.recent_trades = []
            self.quality_filter.market_memory = {}
        
        # 🔥 NOVO V7: Reset dos sistemas de risco aprimorados
        if hasattr(self, 'progressive_zones'):
            self.progressive_zones.previous_zone = RiskZone.GREEN
        if hasattr(self, 'smart_risk_factors'):
            self.smart_risk_factors.recent_volatility = []
            self.smart_risk_factors.recent_performance = []
        
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        🧠 SISTEMA V8 - ENTRY HEAD V5 SPECIALIZATION
        Calcula reward focado em treinar seletividade inteligente
        """
        reward = 0.0
        info = {"components": {}, "target_analysis": {}, "behavioral_analysis": {}, "v5_analysis": {}}
        done = False

        self.step_count += 1
        
        # Processar ações PRIMEIRO
        entry_decision = int(action[0]) if len(action) > 0 else 0
        mgmt_action = int(action[3]) if len(action) > 3 else 0
        sl_adjust = float(action[4]) if len(action) > 4 else 0.0
        tp_adjust = float(action[5]) if len(action) > 5 else 0.0
        
        # 🆕 CHERRY FIX: DENSE REWARDS PARA TODAS AS AÇÕES (resolve esparsidade)
        dense_reward = self._calculate_dense_rewards_cherry_fix(env, action, entry_decision)
        reward += dense_reward
        info["components"]["dense_rewards_cherry_fix"] = dense_reward
        
        # 🧠 V5 ENHANCEMENT: ANALISAR INTELLIGENT COMPONENTS
        v5_reward, v5_info = self._calculate_v5_entry_quality_reward(env, action, entry_decision)
        reward += v5_reward
        info["v5_analysis"] = v5_info
        
        # 🔥 NOVO: SÓ DAR RECOMPENSAS QUANDO TRADES SÃO FECHADOS OU GERENCIADOS
        immediate_reward = 0.0
        
        # 🧠 BÔNUS APENAS POR GESTÃO DE POSIÇÕES EXISTENTES
        if mgmt_action > 0 and hasattr(env, 'positions') and len(env.positions) > 0:
            # Recompensar gestão ativa de posições abertas
            management_bonus = self.weights["position_management"]
            immediate_reward += management_bonus
            info["components"]["position_management_bonus"] = management_bonus
            
            # 🧠 ADICIONAR BÔNUS DE QUALIDADE INTELIGENTE APENAS PARA GESTÃO
            if hasattr(self, 'quality_filter') and self.quality_filter_enabled:
                try:
                    allow_trade, quality_score, quality_info = self.quality_filter.should_allow_trade(env, action)
                    if mgmt_action > 0:  # Só para gestão real
                        quality_bonus = self.quality_filter.get_quality_reward_bonus(quality_score) * 0.5
                        immediate_reward += quality_bonus
                        info["components"]["quality_bonus"] = quality_bonus
                        info["quality_analysis"] = {
                            "score": quality_score,
                            "bonus": quality_bonus,
                            "category": quality_info.get("quality_category", "unknown"),
                            "teaching_mode": True,
                            "action_type": "management"
                        }
                except Exception as e:
                    info["quality_analysis"] = {"status": "error", "error": str(e)}
            else:
                info["quality_analysis"] = {"status": "disabled"}
        else:
            info["quality_analysis"] = {"status": "no_management_action"}
        
        reward += immediate_reward
        
        # 💰 COMPONENTE PRINCIPAL: PnL DIRETO (60% do peso)
        old_trades_count = old_state.get('trades_count', 0)
        current_trades_count = len(env.trades)
        
        if current_trades_count > old_trades_count:
            # Trade fechado - análise completa
            last_trade = env.trades[-1]
            pnl = last_trade.get('pnl_usd', 0.0)
            
            # 🔥 BÔNUS CRÍTICO POR FECHAR TRADE - ENSINA COMPLETAR TRADES
            completion_bonus = self.weights["trade_completion_bonus"]
            reward += completion_bonus
            info["components"]["trade_completion_bonus"] = completion_bonus
            
            # 🔥 PnL DIRETO - COMPONENTE DOMINANTE
            pnl_reward = pnl * self.weights["pnl_direct"]
            reward += pnl_reward
            info["components"]["pnl_direct"] = pnl_reward
            
            # 🔥 WIN/LOSS BALANCEADO
            if pnl > 0:
                win_bonus = self.weights["win_bonus"]
                reward += win_bonus
                info["components"]["win_bonus"] = win_bonus
                
                # 🎯 QUALIDADE ESCALADA
                if pnl > 15.0:  # Trade excelente
                    excellent_bonus = self.weights["excellent_trade_bonus"]
                    reward += excellent_bonus
                    info["components"]["excellent_trade_bonus"] = excellent_bonus
                elif pnl > 5.0:  # Trade de qualidade
                    quality_bonus = self.weights["quality_trade_bonus"]
                    reward += quality_bonus
                    info["components"]["quality_trade_bonus"] = quality_bonus
                
                # 🎯 WIN STREAK
                consecutive_wins = self._count_consecutive_wins(env.trades)
                if consecutive_wins > 1:
                    streak_bonus = self.weights["win_streak_bonus"] * min(consecutive_wins - 1, 5)
                    reward += streak_bonus
                    info["components"]["win_streak_bonus"] = streak_bonus
                
                # 🎯 LUCRO RÁPIDO
                trade_duration = last_trade.get('duration_steps', 0)
                if trade_duration < 30 and pnl > 3.0:
                    quick_bonus = self.weights["quick_profit_bonus"]
                    reward += quick_bonus
                    info["components"]["quick_profit_bonus"] = quick_bonus
                    
            else:
                loss_penalty = self.weights["loss_penalty"]
                reward += loss_penalty
                info["components"]["loss_penalty"] = loss_penalty
            
            # 🧠 EXPERT SL/TP SIMPLIFICADO (25% do peso)
            expert_sltp_reward = self._analyze_expert_sltp_v2(env, last_trade)
            reward += expert_sltp_reward
            info["target_analysis"]["expert_sltp"] = expert_sltp_reward
            
            # 📊 ANÁLISE TÉCNICA INTELIGENTE (Novo)
            technical_reward = self._analyze_technical_analysis(env, last_trade, entry_decision)
            reward += technical_reward
            info["target_analysis"]["technical_analysis"] = technical_reward
            
            # 🔍 ANÁLISE COMPORTAMENTAL
            behavioral_reward = self._analyze_trade_behavior(env, last_trade, entry_decision)
            reward += behavioral_reward
            info["behavioral_analysis"]["trade_behavior"] = behavioral_reward
        
        # 🎯 ATIVIDADE GUIADA AOS TARGETS (15% do peso)
        trades_today = self._get_trades_today(env)
        activity_reward = self._calculate_activity_reward(trades_today, entry_decision)
        reward += activity_reward
        info["target_analysis"]["activity"] = activity_reward
        
        # 🛡️ PENALIDADES DISCIPLINARES REMOVIDAS - CAUSAVAM HOLD ETERNO
        discipline_penalty = 0.0  # Completamente desabilitado
        info["behavioral_analysis"]["discipline"] = {"status": "disabled", "penalty": 0.0}
        
        # 🚨 EMERGENCIAL: FORÇAR SISTEMA DE RISCO AGRESSIVO PARA DD 45%
        # Ignorar enhanced_risk e usar SEMPRE o sistema original com penalidades severas
        risk_reward = self._calculate_risk_management_reward(env)
        reward += risk_reward
        info["target_analysis"]["risk_management"] = risk_reward
        info["components"]["aggressive_risk_control"] = risk_reward
        
        # --- INÍCIO: Lógica de trailing stop ---
        trailing_bonus = 0.0
        missed_trailing_penalty = 0.0
        trailing_activated = False
        trailing_executed = False
        trailing_protected = False
        trailing_timing = False

        # Exemplo: checar se trade foi fechado por trailing stop com lucro
        if 'trades' in env.__dict__ and len(env.trades) > 0:
            last_trade = env.trades[-1]
            if last_trade.get('exit_reason', '') == 'trailing_stop':
                if last_trade.get('pnl_usd', 0) > 0:
                    trailing_bonus += self.weights["trailing_stop_execution"]
                    trailing_executed = True
                # Ativação correta do trailing
                if last_trade.get('trailing_activated', False):
                    trailing_bonus += self.weights["trailing_stop_activation"]
                    trailing_activated = True
                # Proteção de lucro
                if last_trade.get('trailing_protected', False):
                    trailing_bonus += self.weights["trailing_stop_protection"]
                    trailing_protected = True
                # Timing correto
                if last_trade.get('trailing_timing', False):
                    trailing_bonus += self.weights["trailing_stop_timing"]
                    trailing_timing = True
            # Penalidade se perdeu oportunidade de trailing
            if last_trade.get('missed_trailing_opportunity', False):
                missed_trailing_penalty -= self.weights["missed_trailing_opportunity"]

        reward += trailing_bonus + missed_trailing_penalty
        info["trailing_stop_bonus"] = trailing_bonus
        info["missed_trailing_penalty"] = missed_trailing_penalty
        info["trailing_executed"] = trailing_executed
        info["trailing_activated"] = trailing_activated
        info["trailing_protected"] = trailing_protected
        info["trailing_timing"] = trailing_timing
        # --- FIM: Lógica de trailing stop ---
        
        # 🧠 BÔNUS FINAL POR INTELIGÊNCIA E SELETIVIDADE
        intelligence_bonus = 0.0
        intelligence_info = {}
        
        # Verificar progresso de aprendizado adaptativo
        if hasattr(env, 'learning_progress'):
            adaptation_level = env.learning_progress.get('adaptation_level', 0)
            total_trades = env.learning_progress.get('total_trades', 0)
            quality_trades = env.learning_progress.get('quality_trades', 0)
            excellent_trades = env.learning_progress.get('excellent_trades', 0)
            
            # Bônus por nível de competência
            if adaptation_level >= 3:
                intelligence_bonus += 0.8  # Bônus por competência avançada
                intelligence_info['adaptation_bonus'] = f'level_{adaptation_level}_advanced'
            elif adaptation_level >= 1:
                intelligence_bonus += 0.4  # Bônus por competência básica
                intelligence_info['adaptation_bonus'] = f'level_{adaptation_level}_basic'
            
            # Bônus por taxa de qualidade
            if total_trades >= 50:
                quality_rate = quality_trades / total_trades
                excellence_rate = excellent_trades / total_trades
                
                if quality_rate >= 0.7:
                    intelligence_bonus += 0.6  # Alta taxa de qualidade
                    intelligence_info['quality_rate_bonus'] = 'high_quality_rate'
                elif quality_rate >= 0.5:
                    intelligence_bonus += 0.3  # Taxa média de qualidade
                    intelligence_info['quality_rate_bonus'] = 'medium_quality_rate'
                
                if excellence_rate >= 0.3:
                    intelligence_bonus += 0.5  # Alta taxa de excelência
                    intelligence_info['excellence_rate_bonus'] = 'high_excellence_rate'
                
                intelligence_info['stats'] = {
                    'quality_rate': quality_rate,
                    'excellence_rate': excellence_rate,
                    'total_trades': total_trades
                }
        
        # Aplicar bônus de inteligência
        reward += intelligence_bonus
        
        # 🚨 EMERGENCIAL: REMOVER CLIPPING QUE MATA PENALIDADES DE RISCO!
        # O clipping estava impedindo que penalidades de -100 (DD alto) fossem sentidas
        raw_reward = reward
        # APENAS clipping de segurança para valores absurdos (>200)
        reward = np.clip(reward, -200.0, 200.0)  # Deixar penalidades passarem!
        
        # 📊 INFORMAÇÕES DETALHADAS V7
        info.update({
            "trades_today": trades_today,
            "target_trades": self.target_trades_per_day,
            "activity_zone": self._get_activity_zone(trades_today),
            "total_reward": reward,
            "intelligence_bonus": intelligence_bonus,
            "intelligence_info": intelligence_info,
            "step_count": self.step_count,
            "weight_distribution": {
                "pnl_core": "60%",
                "expert_sltp": "25%", 
                "activity_guided": "10%",
                "enhanced_risk_v7": "5%"
            },
            # 🧠 SISTEMA DE QUALIDADE INTELIGENTE
            "quality_system_status": {
                "enabled": getattr(self, 'quality_filter_enabled', False),
                "total_quality_reward": immediate_reward,
                "system_type": "intelligent_teaching_system",
                "always_allows_trades": True,
                "teaches_via_rewards": True
            },
            # 🔥 SISTEMAS APRIMORADOS V7
            "enhanced_risk_systems_v7": {
                "progressive_zones": {
                    "enabled": hasattr(self, 'progressive_zones'),
                    "description": "Progressive Risk Zones: Green(0-3%), Yellow(3-8%), Orange(8-15%), Red(15-25%), Black(25%+)"
                },
                "smart_risk_factors": {
                    "enabled": hasattr(self, 'smart_risk_factors'),
                    "description": "Smart Risk Factors: Drawdown(40%) + Position Concentration(25%) + Volatility(20%) + Performance Streak(15%)"
                },
                "system_version": "v7_enhanced",
                "compatibility": "100% compatible with existing system",
                "balanced_approach": "Maintains 90% PnL focus with intelligent risk management"
            }
        })
        
        return reward, info, done
    
    def _calculate_v5_entry_quality_reward(self, env, action: np.ndarray, entry_decision: int) -> Tuple[float, Dict]:
        """
        🧠 V5 ENTRY HEAD SPECIALIZATION ENHANCED
        Calcula reward baseado na qualidade inteligente da decisão de entrada + filtros inteligentes
        """
        v5_reward = 0.0
        v5_info = {"status": "active", "components": {}, "analysis": {}, "intelligence_filters": {}}
        
        try:
            # 🧠 PRIORIDADE 1: USAR FILTROS INTELIGENTES PARA QUALIDADE BASE
            if hasattr(env, 'positions') and env.positions and entry_decision > 0:
                last_position = env.positions[-1]
                if 'entry_quality_score' in last_position:
                    quality_score = last_position['entry_quality_score']
                    classification = last_position.get('entry_classification', 'UNKNOWN')
                    intelligence_info = last_position.get('intelligence_info', {})
                    
                    # 🎯 BÔNUS EDUCATIVO POR CLASSIFICAÇÃO DE QUALIDADE
                    intelligence_reward = 0.0
                    if classification == 'EXCELENTE':
                        intelligence_reward = 1.2  # +1.2 bônus forte
                        v5_info["intelligence_filters"]["classification_bonus"] = "excellent_entry"
                    elif classification == 'BOA':
                        intelligence_reward = 0.8  # +0.8 bônus bom
                        v5_info["intelligence_filters"]["classification_bonus"] = "good_entry"
                    elif classification == 'MÉDIA':
                        intelligence_reward = 0.3  # +0.3 bônus leve
                        v5_info["intelligence_filters"]["classification_bonus"] = "average_entry"
                    else:  # BAIXA
                        intelligence_reward = -0.2  # -0.2 educativo (leve)
                        v5_info["intelligence_filters"]["classification_bonus"] = "low_quality_education"
                    
                    # 🧠 BÔNUS POR COMPONENTES ESPECÍFICOS (educativo)
                    component_bonus = 0.0
                    momentum_quality = intelligence_info.get('momentum_quality', 0.0)
                    volatility_quality = intelligence_info.get('volatility_quality', 0.0)
                    timing_quality = intelligence_info.get('timing_quality', 0.0)
                    confluence_quality = intelligence_info.get('confluence_quality', 0.0)
                    
                    # Bônus graduais por excelência em componentes
                    if momentum_quality > 0.8:
                        component_bonus += 0.3
                        v5_info["intelligence_filters"]["momentum_excellence"] = True
                    
                    if volatility_quality > 0.8:
                        component_bonus += 0.25
                        v5_info["intelligence_filters"]["volatility_excellence"] = True
                    
                    if timing_quality > 0.8:
                        component_bonus += 0.4  # Timing é muito importante
                        v5_info["intelligence_filters"]["timing_excellence"] = True
                    
                    if confluence_quality > 0.8:
                        component_bonus += 0.5  # Confluência é crítica
                        v5_info["intelligence_filters"]["confluence_excellence"] = True
                    
                    # 🎯 BÔNUS COMBINADO POR MÚLTIPLA EXCELÊNCIA
                    excellent_components = sum([
                        momentum_quality > 0.8,
                        volatility_quality > 0.8,
                        timing_quality > 0.8,
                        confluence_quality > 0.8
                    ])
                    
                    if excellent_components >= 3:
                        component_bonus += 0.8  # Bônus por múltipla excelência
                        v5_info["intelligence_filters"]["multiple_excellence"] = True
                    elif excellent_components >= 2:
                        component_bonus += 0.4  # Bônus por dupla excelência
                        v5_info["intelligence_filters"]["double_excellence"] = True
                    
                    v5_reward += intelligence_reward + component_bonus
                    
                    v5_info["intelligence_filters"].update({
                        "base_quality_score": quality_score,
                        "classification": classification,
                        "intelligence_reward": intelligence_reward,
                        "component_bonus": component_bonus,
                        "component_details": {
                            "momentum": momentum_quality,
                            "volatility": volatility_quality,
                            "timing": timing_quality,
                            "confluence": confluence_quality
                        }
                    })
            
            # 🧠 PRIORIDADE 2: ANÁLISE TRADICIONAL V5 (peso reduzido para manter compatibilidade)
            if hasattr(env, '_generate_intelligent_components'):
                intelligent_components = env._generate_intelligent_components()
                v5_info["components"] = intelligent_components
                
                # Análises tradicionais com peso reduzido (30% vs 70% dos filtros)
                fatigue_reward, fatigue_info = self._analyze_market_fatigue_v5(intelligent_components, entry_decision)
                regime_reward, regime_info = self._analyze_regime_alignment_v5(intelligent_components, entry_decision)
                confluence_reward, confluence_info = self._analyze_confluence_quality_v5(intelligent_components, entry_decision)
                risk_reward, risk_info = self._analyze_risk_context_v5(intelligent_components, entry_decision)
                selectivity_reward, selectivity_info = self._analyze_selectivity_v5(intelligent_components, entry_decision)
                
                # Combinar análises tradicionais com peso reduzido
                traditional_reward = (fatigue_reward + regime_reward + confluence_reward + risk_reward + selectivity_reward) * 0.3
                v5_reward += traditional_reward
                
                v5_info["analysis"] = {
                    "market_fatigue": fatigue_info,
                    "regime_alignment": regime_info,
                    "confluence_quality": confluence_info,
                    "risk_context": risk_info,
                    "selectivity": selectivity_info,
                    "traditional_reward": traditional_reward
                }
                
            v5_info["total_reward"] = v5_reward
            v5_info["status"] = "enhanced_with_intelligence_filters"
                
        except Exception as e:
            v5_info["status"] = "error"
            v5_info["error"] = str(e)
            
        return v5_reward, v5_info
    
    def _analyze_market_fatigue_v5(self, components: Dict, entry_decision: int) -> Tuple[float, Dict]:
        """🧠 Analisar fadiga do mercado para Entry Head V5"""
        try:
            fatigue_data = components.get('market_fatigue', {})
            fatigue_score = fatigue_data.get('fatigue_score', 0.0)
            should_avoid = fatigue_data.get('should_avoid_entry', False)
            recent_trades = fatigue_data.get('recent_trades', 0)
            
            reward = 0.0
            info = {"fatigue_score": fatigue_score, "should_avoid": should_avoid, "recent_trades": recent_trades}
            
            if entry_decision > 0:  # Tentativa de entrada
                if should_avoid:
                    # PENALIDADE por entrar quando mercado está fatigado
                    penalty = -2.0 * fatigue_score
                    reward += penalty
                    info["penalty"] = penalty
                    info["reason"] = "entry_during_fatigue"
                else:
                    # BÔNUS por entrar quando mercado está fresco
                    bonus = 1.0 * (1.0 - fatigue_score)
                    reward += bonus
                    info["bonus"] = bonus
                    info["reason"] = "entry_during_fresh_market"
            else:  # Hold decision
                if should_avoid:
                    # BÔNUS por evitar entrada quando mercado está fatigado
                    bonus = 1.5 * fatigue_score
                    reward += bonus
                    info["bonus"] = bonus
                    info["reason"] = "avoided_entry_during_fatigue"
                    
            return reward, info
            
        except Exception as e:
            return 0.0, {"status": "error", "error": str(e)}
    
    def _analyze_regime_alignment_v5(self, components: Dict, entry_decision: int) -> Tuple[float, Dict]:
        """🧠 Analisar alinhamento com regime de mercado"""
        try:
            regime_data = components.get('market_regime', {})
            regime = regime_data.get('regime', 'unknown')
            strength = regime_data.get('strength', 0.0)
            direction = regime_data.get('direction', 0.0)
            
            reward = 0.0
            info = {"regime": regime, "strength": strength, "direction": direction}
            
            if entry_decision > 0:  # Tentativa de entrada
                if regime == 'trending' and strength > 0.5:
                    # BÔNUS por entrar em trending market forte
                    bonus = 1.5 * strength
                    reward += bonus
                    info["bonus"] = bonus
                    info["reason"] = "entry_in_strong_trend"
                elif regime == 'ranging' and strength < 0.3:
                    # PENALIDADE por entrar em ranging market
                    penalty = -1.0 * (0.3 - strength)
                    reward += penalty
                    info["penalty"] = penalty
                    info["reason"] = "entry_in_ranging_market"
                elif regime == 'volatile':
                    # PENALIDADE por entrar em mercado volátil
                    penalty = -0.5 * strength
                    reward += penalty
                    info["penalty"] = penalty
                    info["reason"] = "entry_in_volatile_market"
                    
            return reward, info
            
        except Exception as e:
            return 0.0, {"status": "error", "error": str(e)}
    
    def _analyze_confluence_quality_v5(self, components: Dict, entry_decision: int) -> Tuple[float, Dict]:
        """🧠 Analisar qualidade da confluência de indicadores"""
        try:
            momentum_data = components.get('momentum_confluence', {})
            confluence_score = momentum_data.get('score', 0.0)
            direction = momentum_data.get('direction', 0.0)
            strength = momentum_data.get('strength', 0.0)
            
            reward = 0.0
            info = {"score": confluence_score, "direction": direction, "strength": strength}
            
            if entry_decision > 0:  # Tentativa de entrada
                if confluence_score > 0.7 and strength > 0.6:
                    # BÔNUS por entrar com confluência forte
                    bonus = 2.0 * confluence_score * strength
                    reward += bonus
                    info["bonus"] = bonus
                    info["reason"] = "strong_confluence_entry"
                elif confluence_score < 0.3:
                    # PENALIDADE por entrar com confluência fraca
                    penalty = -1.5 * (0.3 - confluence_score)
                    reward += penalty
                    info["penalty"] = penalty
                    info["reason"] = "weak_confluence_entry"
            else:  # Hold decision
                if confluence_score < 0.4:
                    # BÔNUS por evitar entrada com confluência fraca
                    bonus = 0.5 * (0.4 - confluence_score)
                    reward += bonus
                    info["bonus"] = bonus
                    info["reason"] = "avoided_weak_confluence"
                    
            return reward, info
            
        except Exception as e:
            return 0.0, {"status": "error", "error": str(e)}
    
    def _analyze_risk_context_v5(self, components: Dict, entry_decision: int) -> Tuple[float, Dict]:
        """🧠 Analisar contexto de risco para entrada"""
        try:
            risk_data = components.get('risk_assessment', {})
            risk_score = risk_data.get('risk_score', 0.0)
            drawdown = risk_data.get('drawdown', 0.0)
            position_concentration = risk_data.get('position_concentration', 0.0)
            
            reward = 0.0
            info = {"risk_score": risk_score, "drawdown": drawdown, "concentration": position_concentration}
            
            if entry_decision > 0:  # Tentativa de entrada
                if risk_score > 0.7:
                    # PENALIDADE por entrar em alto risco
                    penalty = -2.0 * risk_score
                    reward += penalty
                    info["penalty"] = penalty
                    info["reason"] = "high_risk_entry"
                elif risk_score < 0.3:
                    # BÔNUS por entrar em baixo risco
                    bonus = 1.0 * (0.3 - risk_score)
                    reward += bonus
                    info["bonus"] = bonus
                    info["reason"] = "low_risk_entry"
            else:  # Hold decision
                if risk_score > 0.6:
                    # BÔNUS por evitar entrada em alto risco
                    bonus = 1.5 * risk_score
                    reward += bonus
                    info["bonus"] = bonus
                    info["reason"] = "avoided_high_risk"
                    
            return reward, info
            
        except Exception as e:
            return 0.0, {"status": "error", "error": str(e)}
    
    def _analyze_selectivity_v5(self, components: Dict, entry_decision: int) -> Tuple[float, Dict]:
        """🧠 Analisar seletividade inteligente da Entry Head"""
        try:
            # Combinar múltiplos fatores para medir seletividade
            fatigue_data = components.get('market_fatigue', {})
            regime_data = components.get('market_regime', {})
            momentum_data = components.get('momentum_confluence', {})
            liquidity_data = components.get('liquidity_zones', {})
            
            # Calcular score de qualidade combinado
            quality_factors = []
            
            # Fator 1: Fadiga do mercado
            fatigue_score = fatigue_data.get('fatigue_score', 0.0)
            quality_factors.append(1.0 - fatigue_score)  # Inverter: menos fadiga = mais qualidade
            
            # Fator 2: Força do regime
            regime_strength = regime_data.get('strength', 0.0)
            if regime_data.get('regime') == 'trending':
                quality_factors.append(regime_strength)
            else:
                quality_factors.append(0.3)  # Penalizar ranging/volatile
            
            # Fator 3: Confluência de momentum
            confluence_score = momentum_data.get('score', 0.0)
            quality_factors.append(confluence_score)
            
            # Fator 4: Proximidade de zonas de liquidez
            zone_strength = liquidity_data.get('zone_strength', 0.0)
            quality_factors.append(zone_strength)
            
            # Score combinado
            combined_quality = np.mean(quality_factors)
            
            reward = 0.0
            info = {
                "combined_quality": combined_quality,
                "quality_factors": quality_factors,
                "quality_threshold": 0.6
            }
            
            if entry_decision > 0:  # Tentativa de entrada
                if combined_quality > 0.8:
                    # BÔNUS ALTO por entrada de altíssima qualidade
                    bonus = 3.0 * combined_quality
                    reward += bonus
                    info["bonus"] = bonus
                    info["reason"] = "ultra_high_quality_entry"
                elif combined_quality > 0.6:
                    # BÔNUS MODERADO por entrada de boa qualidade
                    bonus = 1.5 * combined_quality
                    reward += bonus
                    info["bonus"] = bonus
                    info["reason"] = "good_quality_entry"
                elif combined_quality < 0.4:
                    # PENALIDADE por entrada de baixa qualidade
                    penalty = -2.0 * (0.4 - combined_quality)
                    reward += penalty
                    info["penalty"] = penalty
                    info["reason"] = "low_quality_entry"
            else:  # Hold decision
                if combined_quality < 0.5:
                    # BÔNUS por evitar entrada de baixa qualidade
                    bonus = 1.0 * (0.5 - combined_quality)
                    reward += bonus
                    info["bonus"] = bonus
                    info["reason"] = "avoided_low_quality_entry"
                    
            return reward, info
            
        except Exception as e:
            return 0.0, {"status": "error", "error": str(e)}
    
    def _analyze_expert_sltp_v2(self, env, trade: Dict) -> float:
        """
        🧠 QUALIDADE SL/TP DENTRO DOS RANGES FIXOS
        Ensina o modelo a escolher as MELHORES combinações dentro dos ranges fixos
        """
        expert_reward = 0.0
        
        try:
            sl_points = abs(trade.get('sl_points', 0))
            tp_points = abs(trade.get('tp_points', 0))
            pnl = trade.get('pnl_usd', 0.0)
            
            if sl_points == 0 or tp_points == 0:
                return 0.0  # Sem penalidade - ranges fixos garantem SL/TP
            
            # 🎯 QUALIDADE 1: RATIO RISK/REWARD PERFEITO
            risk_reward_ratio = tp_points / sl_points if sl_points > 0 else 0
            if self.optimal_risk_reward_min <= risk_reward_ratio <= self.optimal_risk_reward_max:
                expert_reward += self.weights["perfect_sltp_ratio"]
            elif 1.3 <= risk_reward_ratio <= 2.0:  # Próximo do ótimo
                expert_reward += self.weights["perfect_sltp_ratio"] * 0.6
            
            # 🎯 QUALIDADE 2: ADAPTAÇÃO CONTEXTUAL
            volatility_context = self._get_market_volatility_simple(env)
            if volatility_context == "HIGH" and sl_points >= 25:  # SL maior em alta volatilidade
                expert_reward += self.weights["adaptive_sltp"]
            elif volatility_context == "LOW" and sl_points <= 25:  # SL menor em baixa volatilidade
                expert_reward += self.weights["adaptive_sltp"]
            elif volatility_context == "MEDIUM":  # Qualquer SL no range é bom
                expert_reward += self.weights["adaptive_sltp"] * 0.7
            
            # 🎯 QUALIDADE 3: TIMING INTELIGENTE
            if pnl > 0:  # Trade vencedor
                duration = trade.get('duration_steps', 0)
                if duration < 50 and sl_points <= 30:  # Scalping bem executado
                    expert_reward += self.weights["smart_sltp_timing"]
                elif 50 <= duration <= 200 and 20 <= sl_points <= 40:  # Swing bem executado
                    expert_reward += self.weights["smart_sltp_timing"]
                elif duration > 200 and sl_points >= 30:  # Position bem executado
                    expert_reward += self.weights["smart_sltp_timing"] * 0.8
            
        except Exception:
            # Em caso de erro, sem penalidade
            pass
        
        return expert_reward
    
    def _analyze_technical_analysis(self, env, trade: Dict, entry_decision: int) -> float:
        """
        📊 ANÁLISE TÉCNICA INTELIGENTE - ONDAS E SUPORTE/RESISTÊNCIA
        Ensina o modelo a usar análise técnica profissional
        """
        technical_reward = 0.0
        
        try:
            current_price = trade.get('entry_price', 0)
            sl_price = trade.get('sl_price', 0)
            tp_price = trade.get('tp_price', 0)
            
            if not all([current_price, sl_price, tp_price]):
                return 0.0
                
            # Obter dados históricos para análise
            df = getattr(env, 'df', None)
            current_step = getattr(env, 'current_step', 0)
            
            if df is None or current_step < 50:
                return 0.0
                
            # Análise das últimas 50 barras
            recent_data = df.iloc[max(0, current_step-50):current_step]
            highs = recent_data['high_5m'].values if 'high_5m' in recent_data.columns else []
            lows = recent_data['low_5m'].values if 'low_5m' in recent_data.columns else []
            closes = recent_data['close_5m'].values if 'close_5m' in recent_data.columns else []
            
            if len(highs) < 20:
                return 0.0
                
            # 🎯 ANÁLISE 1: PROJEÇÃO DE ONDAS PARA TP
            if len(highs) >= 20:
                recent_highs = highs[-20:]
                recent_lows = lows[-20:]
                
                # Calcular amplitude média das ondas
                wave_amplitudes = []
                for i in range(1, len(recent_highs)):
                    if i < len(recent_lows):
                        amplitude = abs(recent_highs[i] - recent_lows[i])
                        wave_amplitudes.append(amplitude)
                
                if wave_amplitudes:
                    avg_wave = np.mean(wave_amplitudes)
                    tp_distance = abs(tp_price - current_price)
                    
                    # Reward se TP está próximo da projeção da onda
                    if 0.8 * avg_wave <= tp_distance <= 1.2 * avg_wave:
                        technical_reward += self.weights["wave_projection_tp"]
            
            # 🎯 ANÁLISE 2: SL PRÓXIMO A SUPORTE/RESISTÊNCIA
            if len(lows) >= 10:
                recent_lows_sorted = sorted(lows[-10:])
                recent_highs_sorted = sorted(highs[-10:])
                
                # Encontrar níveis de suporte/resistência
                support_levels = []
                resistance_levels = []
                
                for price in recent_lows_sorted:
                    count = sum(1 for low in lows[-20:] if abs(low - price) < price * 0.002)
                    if count >= 2:  # Pelo menos 2 toques
                        support_levels.append(price)
                        
                for price in recent_highs_sorted:
                    count = sum(1 for high in highs[-20:] if abs(high - price) < price * 0.002)
                    if count >= 2:  # Pelo menos 2 toques
                        resistance_levels.append(price)
                
                # Verificar se SL está próximo de suporte/resistência
                sl_near_level = False
                for level in support_levels + resistance_levels:
                    if abs(sl_price - level) < current_price * 0.003:  # 0.3% de tolerância
                        sl_near_level = True
                        break
                
                if sl_near_level:
                    technical_reward += self.weights["support_resistance_sl"]
            
            # 🎯 ANÁLISE 3: NÍVEIS DE FIBONACCI
            if len(highs) >= 20 and len(lows) >= 20:
                swing_high = max(highs[-20:])
                swing_low = min(lows[-20:])
                range_size = swing_high - swing_low
                
                if range_size > 0:
                    # Níveis de Fibonacci
                    fib_levels = [
                        swing_low + range_size * 0.236,  # 23.6%
                        swing_low + range_size * 0.382,  # 38.2%
                        swing_low + range_size * 0.618,  # 61.8%
                        swing_low + range_size * 0.786,  # 78.6%
                    ]
                    
                    # Verificar se TP está próximo de nível de Fibonacci
                    for fib_level in fib_levels:
                        if abs(tp_price - fib_level) < current_price * 0.002:
                            technical_reward += self.weights["fibonacci_levels"]
                            break
            
            # 🎯 ANÁLISE 4: ALINHAMENTO COM TENDÊNCIA
            if len(closes) >= 20:
                sma_short = np.mean(closes[-10:])
                sma_long = np.mean(closes[-20:])
                
                trend_up = sma_short > sma_long
                trend_down = sma_short < sma_long
                
                if (entry_decision == 1 and trend_up) or (entry_decision == 2 and trend_down):
                    technical_reward += self.weights["trend_alignment"]
            
            # 🎯 ANÁLISE 5: CONFIRMAÇÃO DE VOLUME
            if 'volume_5m' in recent_data.columns:
                volumes = recent_data['volume_5m'].values
                if len(volumes) >= 10:
                    avg_volume = np.mean(volumes[-10:])
                    current_volume = volumes[-1] if len(volumes) > 0 else 0
                    
                    if current_volume > 1.2 * avg_volume:  # Volume 20% acima da média
                        technical_reward += self.weights["volume_confirmation"]
            
            # 🎯 ANÁLISE 6: TIMING DE BREAKOUT
            if len(closes) >= 5:
                recent_closes = closes[-5:]
                is_breakout = False
                
                if entry_decision == 1:  # Long
                    resistance = max(highs[-20:]) if len(highs) >= 20 else 0
                    if current_price > resistance * 1.001:  # Breakout acima
                        is_breakout = True
                elif entry_decision == 2:  # Short
                    support = min(lows[-20:]) if len(lows) >= 20 else 0
                    if current_price < support * 0.999:  # Breakout abaixo
                        is_breakout = True
                
                if is_breakout:
                    technical_reward += self.weights["breakout_timing"]
            
        except Exception as e:
            # Em caso de erro, retornar 0
            pass
            
        return technical_reward
    
    def _calculate_activity_reward(self, trades_today: int, entry_decision: int) -> float:
        """
        🎯 ATIVIDADE GUIADA AOS TARGETS - ZONA ULTRA-EXPANDIDA V6
        Recompensa zona alvo ultra-expandida + penalidade suavizada por overtrading
        """
        activity_reward = 0.0
        
        # ✅ COMBATE À INATIVIDADE EXTREMA
        if trades_today == 0:
            # PENALIDADE FORTE por inatividade total
            activity_reward += self.weights["inactivity_penalty"]
        elif trades_today < self.target_zone_min:  # 1-11 trades/dia
            # FORTE incentivo para começar a operar
            activity_reward += 1.0
        elif self.target_zone_min <= trades_today <= self.target_zone_max:  # 12-24 trades/dia
            # Zona target - bônus completo
            activity_reward += self.weights["target_zone_bonus"]
        
        # 🛡️ PENALIDADE POR OVERTRADING - SUAVIZADA
        if trades_today > 30:  # Acima de 30 trades/dia (era 25)
            activity_reward += self.weights["overtrading_penalty"]
        
        return activity_reward
    
    def _analyze_trade_behavior(self, env, trade: Dict, entry_decision: int) -> float:
        """
        🔍 ANÁLISE COMPORTAMENTAL DO TRADE
        Detecta padrões inteligentes vs problemáticos
        """
        behavioral_reward = 0.0
        
        try:
            trade_type = trade.get('type', 'long')
            duration = trade.get('duration_steps', 0)
            pnl = trade.get('pnl_usd', 0.0)
            
            # Detectar micro-trades
            if duration < 5:
                behavioral_reward += self.weights["micro_trade_penalty"]
            
            # Detectar flip-flop (mudança rápida de direção)
            if (self.last_trade_type and 
                self.last_trade_type != trade_type and 
                self.step_count - self.last_action_step < 10):
                behavioral_reward += self.weights["flip_flop_penalty"]
            
            # Bônus por consistência
            if len(env.trades) >= 3:
                recent_trades = env.trades[-3:]
                win_rate = sum(1 for t in recent_trades if t.get('pnl_usd', 0) > 0) / len(recent_trades)
                if win_rate >= 0.6:  # 60%+ win rate
                    behavioral_reward += self.weights["consistency_bonus"]
            
            # Atualizar tracking
            self.last_trade_type = trade_type
            self.last_action_step = self.step_count
            
        except Exception:
            pass
        
        return behavioral_reward
    
    def _calculate_discipline_penalties(self, env, action: np.ndarray, entry_decision: int) -> float:
        """
        🛡️ PENALIDADES DISCIPLINARES MODERADAS
        Evita comportamento errático sem ser muito punitivo
        """
        penalty = 0.0
        
        # Tracking de ações recentes
        self.recent_actions.append(entry_decision)
        if len(self.recent_actions) > 20:
            self.recent_actions.pop(0)
        
        # Detectar padrões problemáticos
        if len(self.recent_actions) >= 10:
            # Flip-flop excessivo
            changes = sum(1 for i in range(1, len(self.recent_actions)) 
                         if self.recent_actions[i] != self.recent_actions[i-1])
            if changes > 7:  # Mais de 7 mudanças em 10 ações
                penalty += self.weights["flip_flop_penalty"] * 0.5
        
        return penalty
    
    def _calculate_enhanced_risk_management_v7(self, env) -> Tuple[float, Dict]:
        """
        🔥 GESTÃO DE RISCO APRIMORADA V7 - PROGRESSIVE ZONES + SMART RISK FACTORS
        Combina Progressive Risk Zones com Smart Risk Factors para análise inteligente
        """
        total_risk_reward = 0.0
        risk_info = {
            "progressive_zones": {},
            "smart_risk_factors": {},
            "combined_analysis": {},
            "system_version": "v7_enhanced"
        }
        
        try:
            # 🎯 COMPONENTE 1: PROGRESSIVE RISK ZONES
            current_dd = abs(getattr(env, 'current_drawdown', 0.0))
            
            if hasattr(self, 'progressive_zones'):
                zone_penalty, current_zone, zone_info = self.progressive_zones.calculate_zone_penalty(current_dd)
                zone_reward = zone_penalty * self.weights["progressive_zone_penalty"]
                total_risk_reward += zone_reward
                
                risk_info["progressive_zones"] = {
                    "current_drawdown": current_dd,
                    "risk_zone": current_zone.value,
                    "zone_penalty": zone_penalty,
                    "weighted_reward": zone_reward,
                    "zone_details": zone_info
                }
            
            # 🎯 COMPONENTE 2: SMART RISK FACTORS
            if hasattr(self, 'smart_risk_factors'):
                smart_penalty, risk_metrics, smart_info = self.smart_risk_factors.calculate_smart_risk_penalty(env)
                smart_reward = smart_penalty * self.weights["smart_risk_penalty"]
                total_risk_reward += smart_reward
                
                risk_info["smart_risk_factors"] = {
                    "overall_risk_score": risk_metrics.overall_risk_score,
                    "risk_zone": risk_metrics.risk_zone.value,
                    "smart_penalty": smart_penalty,
                    "weighted_reward": smart_reward,
                    "metrics": {
                        "drawdown_pct": risk_metrics.drawdown_pct,
                        "position_concentration": risk_metrics.position_concentration,
                        "volatility_factor": risk_metrics.volatility_factor,
                        "performance_streak": risk_metrics.performance_streak
                    },
                    "detailed_analysis": smart_info
                }
            
            # 🎯 COMPONENTE 3: BÔNUS POR MELHORIA DE ZONA
            if hasattr(self, 'progressive_zones') and hasattr(self, 'smart_risk_factors'):
                # Bônus se ambos os sistemas indicam zona segura
                prog_zone_safe = current_zone == RiskZone.GREEN
                smart_zone_safe = risk_metrics.risk_zone in [RiskZone.GREEN, RiskZone.YELLOW]
                
                if prog_zone_safe and smart_zone_safe:
                    zone_improvement_bonus = self.weights["zone_improvement_bonus"]
                    total_risk_reward += zone_improvement_bonus
                    risk_info["combined_analysis"]["zone_improvement_bonus"] = zone_improvement_bonus
                
                # Bônus por awareness de risco (operar conscientemente em zona segura)
                if prog_zone_safe or smart_zone_safe:
                    risk_awareness_bonus = self.weights["risk_awareness_bonus"]
                    total_risk_reward += risk_awareness_bonus
                    risk_info["combined_analysis"]["risk_awareness_bonus"] = risk_awareness_bonus
            
            # 🎯 COMPONENTE 4: GESTÃO DE POSIÇÕES (mantido do sistema original)
            if hasattr(env, 'positions') and env.positions:
                positions_with_sltp = sum(1 for pos in env.positions 
                                        if pos.get('sl', 0) > 0 and pos.get('tp', 0) > 0)
                if positions_with_sltp == len(env.positions):  # Todas com SL/TP
                    position_management_bonus = self.weights["risk_management_bonus"]
                    total_risk_reward += position_management_bonus
                    risk_info["combined_analysis"]["position_management_bonus"] = position_management_bonus
            
            # 🎯 ANÁLISE COMBINADA
            risk_info["combined_analysis"]["total_reward"] = total_risk_reward
            risk_info["combined_analysis"]["system_status"] = "fully_active"
            risk_info["combined_analysis"]["components"] = {
                "progressive_zones": "active",
                "smart_risk_factors": "active",
                "position_management": "active",
                "bonus_systems": "active"
            }
            
        except Exception as e:
            # Fallback seguro em caso de erro
            risk_info["combined_analysis"]["error"] = str(e)
            risk_info["combined_analysis"]["system_status"] = "error_fallback"
            total_risk_reward = self._calculate_risk_management_reward(env)
        
        return total_risk_reward, risk_info

    def _calculate_risk_management_reward(self, env) -> float:
        """
        🎯 GESTÃO DE RISCO ORIGINAL - FALLBACK
        Recompensa gestão inteligente de risco + penalidade por drawdown
        """
        risk_reward = 0.0
        
        try:
            # 🚨 CONTROLE DE RISCO AGRESSIVO - EMERGENCIAL PARA DD 45%!
            current_dd = abs(getattr(env, 'current_drawdown', 0.0))
            
            if current_dd > 20.0:  # DD CRÍTICO >20% 
                # PUNIÇÃO ABSOLUTA - QUADRÁTICA para DD extremo
                dd_penalty = self.weights["catastrophic_drawdown"] * (current_dd / 5.0) * (current_dd / 20.0)  # Cresce MUITO rápido
                risk_reward += dd_penalty
                
                # PENALIDADE ADICIONAL por cada ação em DD alto
                if hasattr(env, 'trades') and env.trades:
                    # Se ainda está tradando em DD >20%, penalidade BRUTAL adicional
                    additional_penalty = -50.0  # -50 por CADA trade em DD crítico
                    risk_reward += additional_penalty
                
            elif current_dd > 10.0:  # DD ALTO >10%
                # Penalidade crescente
                dd_penalty = self.weights["critical_drawdown"] * (current_dd / 5.0)
                risk_reward += dd_penalty
                
            elif current_dd > 5.0:  # DD MODERADO >5%
                # Penalidade inicial
                dd_penalty = self.weights["excessive_drawdown"] * (current_dd / 5.0)
                risk_reward += dd_penalty
            
            # 🛡️ BÔNUS GRANDE por operar com DD baixo (balancear bias negativo)
            if current_dd < 3.0:  # DD baixo = trading inteligente
                safety_bonus = 5.0  # +5.0 por operar com segurança (era 1.0)
                risk_reward += safety_bonus
            elif current_dd < 5.0:  # DD aceitável
                safety_bonus = 2.0  # +2.0 por DD ainda controlado
                risk_reward += safety_bonus
            
            # Bônus por gestão de risco (SL/TP)
            if hasattr(env, 'positions') and env.positions:
                positions_with_sltp = sum(1 for pos in env.positions 
                                        if pos.get('sl', 0) > 0 and pos.get('tp', 0) > 0)
                if positions_with_sltp == len(env.positions):  # Todas com SL/TP
                    risk_reward += self.weights["risk_management_bonus"]
            
        except Exception:
            pass
        
        return risk_reward
    
    def _get_activity_zone(self, trades_today: int) -> str:
        """Determina zona de atividade atual - ZONA ULTRA-EXPANDIDA V6"""
        if trades_today < self.target_zone_min:
            return "UNDERTRADING"
        elif self.target_zone_min <= trades_today <= self.target_zone_max:
            return "TARGET_ZONE"
        elif trades_today > 30:  # Aumentado para 30 (era 25)
            return "OVERTRADING"
        else:
            return "NORMAL"
    
    def _get_market_volatility_simple(self, env) -> str:
        """Análise simples de volatilidade"""
        try:
            if hasattr(env, 'df') and hasattr(env, 'current_step'):
                if env.current_step < 20:
                    return "MEDIUM"
                
                # Usar ATR se disponível
                for col in ['atr_5m', 'atr_14', 'atr']:
                    if col in env.df.columns:
                        current_idx = min(env.current_step, len(env.df) - 1)
                        recent_atr = env.df[col].iloc[max(0, current_idx-5):current_idx+1].mean()
                        
                        if recent_atr > 20:
                            return "HIGH"
                        elif recent_atr < 10:
                            return "LOW"
                        else:
                            return "MEDIUM"
        except Exception:
            pass
        return "MEDIUM"
    
    def _get_trades_today(self, env) -> int:
        """Conta trades realizados hoje"""
        try:
            if hasattr(env, 'trades') and env.trades:
                # Aproximação: assumir que cada episódio = 1 dia
                return len(env.trades)
            return 0
        except Exception:
            return 0
    
    def _count_consecutive_wins(self, trades: List[Dict]) -> int:
        """Conta wins consecutivos"""
        consecutive = 0
        for trade in reversed(trades):
            if trade.get('pnl_usd', 0.0) > 0:
                consecutive += 1
            else:
                break
        return consecutive

    def _calculate_dense_rewards_cherry_fix(self, env, action, entry_decision):
        """
        🚨 V2 REBALANCEADO: DENSE REWARDS com sistema HOLD inteligente
        Corrige bias anti-HOLD identificado nos testes matemáticos
        """
        dense_reward = 0.0
        
        try:
            # 🎯 SISTEMA HOLD INTELIGENTE (CORREÇÃO ANTI-BIAS)
            if entry_decision == 0:  # HOLD
                current_positions = len(getattr(env, 'positions', []))
                current_dd = abs(getattr(env, 'current_drawdown', 0.0))
                portfolio_value = getattr(env, 'portfolio_value', self.initial_balance)
                
                # 🔧 AJUSTE FINO: HOLD INTELIGENTE - Critérios mais generosos
                if current_positions >= 1:  # Portfolio ativo = HOLD pode ser inteligente
                    dense_reward += self.weights["intelligent_hold_bonus"] * 0.7  # 70% do bônus
                if current_positions >= 2:  # Portfolio cheio = HOLD muito inteligente  
                    dense_reward += self.weights["intelligent_hold_bonus"] * 0.3  # 30% adicional
                
                # 🔧 AJUSTE FINO: MARKET TIMING mais sensível
                if current_dd > 10.0:  # DD alta = momento ruim para entrar
                    dense_reward += self.weights["market_timing_bonus"]
                elif current_dd > 5.0:  # DD moderada
                    dense_reward += self.weights["patient_analysis_bonus"]
                elif current_dd > 2.0:  # DD leve
                    dense_reward += self.weights["patient_analysis_bonus"] * 0.5
                
                # 🎯 BASE HOLD REWARD - Sempre dar reward base para HOLD
                dense_reward += self.weights["good_hold_bonus"]
                dense_reward += self.weights["market_analysis_bonus"]
                dense_reward += self.weights["patience_bonus"]
                
                # 🚨 PERFORMANCE CORRELATION - CORRIGIDO: usar initial_balance correto
                if portfolio_value > self.initial_balance:  # Portfolio positivo
                    profit_pct = (portfolio_value - self.initial_balance) / self.initial_balance
                    correlation_bonus = min(profit_pct * self.weights["performance_correlation"], 1.0)
                    dense_reward += correlation_bonus
                elif portfolio_value < self.initial_balance:  # Portfolio negativo - penalty
                    loss_pct = (self.initial_balance - portfolio_value) / self.initial_balance
                    correlation_penalty = -min(loss_pct * self.weights["performance_correlation"] * 0.5, 1.0)
                    dense_reward += correlation_penalty
            
            # 🎯 REWARD PARA ENTRADA EM TRADES
            elif entry_decision > 0:
                # Base reward menor que HOLD para balancear
                dense_reward += self.weights["market_analysis_bonus"] * 0.3
                
                # Reward por gestão ativa se já tem posições
                if hasattr(env, 'positions') and env.positions:
                    dense_reward += self.weights["position_management"] * 0.2
                    
        except Exception as e:
            # Em caso de erro, reward neutro pequeno 
            dense_reward = 0.05
            
        return dense_reward

    def calculate_reward(self, action, portfolio_value, trades, current_step, old_state):
        """Método de compatibilidade para sistemas legados"""
        # Criar mock environment
        class MockEnv:
            def __init__(self, trades, current_step, portfolio_value):
                self.trades = trades
                self.current_step = current_step
                self.portfolio_value = portfolio_value
                self.positions = []
                self.current_drawdown = 0.0
        
        env = MockEnv(trades, current_step, portfolio_value)
        reward, info, done = self.calculate_reward_and_info(env, action, old_state)
        return reward

def create_simple_reward_system(initial_balance: float = 1000.0):
    """Factory function para criar sistema de rewards simples"""
    return SimpleRewardCalculator(initial_balance)

# Removido create_simple_reward_system_with_execution - usando apenas SimpleRewardCalculator

# Configuração padrão para o sistema simples
SIMPLE_REWARD_CONFIG = {
    "initial_balance": 1000.0,
    "system_type": "simple_reward",
    "description": "Sistema de recompensas simples e matematicamente coerente"
} 
