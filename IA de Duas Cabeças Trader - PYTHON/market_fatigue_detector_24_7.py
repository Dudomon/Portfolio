# 🚀 MARKET FATIGUE DETECTOR 24/7 OTIMIZADO PARA TRADING REAL
# Sistema avançado de detecção de fadiga de mercado para operação contínua

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import time
from collections import deque
import datetime

class FatigueLevel(Enum):
    """📊 Níveis de Fadiga de Mercado"""
    FRESH = "fresh"           # 0-20%: Mercado descansado
    NORMAL = "normal"         # 20-40%: Atividade normal
    MODERATE = "moderate"     # 40-60%: Fadiga moderada
    HIGH = "high"            # 60-80%: Alta fadiga
    CRITICAL = "critical"    # 80-100%: Fadiga crítica

class MarketSession(Enum):
    """🌍 Sessões de Mercado Global"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP_LONDON_NY = "overlap_london_ny"
    OVERLAP_NY_ASIAN = "overlap_ny_asian"

@dataclass
class FatigueMetrics:
    """📈 Métricas de Fadiga Consolidadas"""
    overall_fatigue: float        # Score geral 0-1
    trading_frequency: float      # Frequência de trades
    market_efficiency: float      # Eficiência do mercado
    volatility_fatigue: float     # Fadiga por volatilidade
    volume_fatigue: float         # Fadiga por volume
    pattern_degradation: float    # Degradação de padrões
    session_fatigue: float        # Fadiga específica da sessão
    
    # Recomendações
    fatigue_level: FatigueLevel
    should_reduce_activity: bool
    should_pause_trading: bool
    recommended_break_minutes: int

class MarketFatigueDetector:
    """
    🎯 DETECTOR DE FADIGA DE MERCADO 24/7
    
    Sistema avançado que monitora:
    - Frequência e qualidade dos trades
    - Padrões de volatilidade e volume
    - Eficiência do mercado por sessão
    - Degradação de sinais técnicos
    - Ciclos naturais de mercado
    """
    
    def __init__(self, lookback_hours: int = 8):
        self.lookback_hours = lookback_hours
        self.lookback_minutes = lookback_hours * 60
        
        # 🎯 MEMÓRIA CIRCULAR PARA DADOS HISTÓRICOS
        max_memory_size = self.lookback_minutes // 5  # 5min bars
        self.trade_memory = deque(maxlen=max_memory_size)
        self.price_memory = deque(maxlen=max_memory_size)
        self.volume_memory = deque(maxlen=max_memory_size)
        self.volatility_memory = deque(maxlen=max_memory_size)
        
        # 🎯 CARACTERÍSTICAS POR SESSÃO
        self.session_characteristics = {
            MarketSession.ASIAN: {
                'typical_trades_per_hour': 2.5,
                'volatility_baseline': 0.008,
                'volume_baseline': 0.8,
                'efficiency_threshold': 0.6
            },
            MarketSession.LONDON: {
                'typical_trades_per_hour': 4.5,
                'volatility_baseline': 0.015,
                'volume_baseline': 1.2,
                'efficiency_threshold': 0.7
            },
            MarketSession.NEW_YORK: {
                'typical_trades_per_hour': 5.5,
                'volatility_baseline': 0.018,
                'volume_baseline': 1.4,
                'efficiency_threshold': 0.75
            },
            MarketSession.OVERLAP_LONDON_NY: {
                'typical_trades_per_hour': 7.0,
                'volatility_baseline': 0.025,
                'volume_baseline': 1.8,
                'efficiency_threshold': 0.8
            },
            MarketSession.OVERLAP_NY_ASIAN: {
                'typical_trades_per_hour': 3.5,
                'volatility_baseline': 0.012,
                'volume_baseline': 1.0,
                'efficiency_threshold': 0.65
            }
        }
        
        # 🎯 CONFIGURAÇÕES DE FADIGA
        self.fatigue_config = {
            'trade_frequency_weight': 0.25,    # Peso da frequência de trades
            'market_efficiency_weight': 0.20,  # Peso da eficiência
            'volatility_weight': 0.20,         # Peso da volatilidade
            'volume_weight': 0.15,             # Peso do volume
            'pattern_weight': 0.20,            # Peso dos padrões
            
            # Thresholds críticos
            'critical_trades_per_hour': 10,    # Trades/hora crítico
            'efficiency_decline_threshold': 0.3, # Declínio de eficiência
            'volatility_spike_threshold': 2.5,   # Spike de volatilidade
            'volume_exhaustion_threshold': 0.3   # Exaustão de volume
        }
        
        # 🎯 HISTÓRICO DE PERFORMANCE POR SESSÃO
        self.session_performance = {}
        for session in MarketSession:
            self.session_performance[session] = {
                'trades': [],
                'avg_duration': [],
                'win_rates': [],
                'volatility_levels': [],
                'volume_levels': []
            }
        
        # 🎯 CONTADORES INTERNOS
        self.last_reset_time = time.time()
        self.session_start_time = time.time()
        self.current_session = self._detect_current_session()
        
    def _detect_current_session(self) -> MarketSession:
        """🕐 Detectar sessão atual baseada no horário UTC"""
        utc_hour = datetime.datetime.utcnow().hour
        
        if 0 <= utc_hour < 6:          # Asian session
            return MarketSession.ASIAN
        elif 6 <= utc_hour < 8:        # Asian-London transition
            return MarketSession.ASIAN
        elif 8 <= utc_hour < 13:       # London session
            return MarketSession.LONDON
        elif 13 <= utc_hour < 16:      # London-NY overlap
            return MarketSession.OVERLAP_LONDON_NY
        elif 16 <= utc_hour < 21:      # New York session
            return MarketSession.NEW_YORK
        else:                          # NY-Asian overlap
            return MarketSession.OVERLAP_NY_ASIAN
    
    def update_market_data(self, 
                          timestamp: float,
                          price: float,
                          volume: float,
                          volatility: float):
        """
        📊 ATUALIZAR DADOS DE MERCADO
        """
        # Adicionar aos buffers circulares
        self.price_memory.append({
            'timestamp': timestamp,
            'price': price
        })
        
        self.volume_memory.append({
            'timestamp': timestamp,
            'volume': volume
        })
        
        self.volatility_memory.append({
            'timestamp': timestamp,
            'volatility': volatility
        })
        
        # Detectar mudança de sessão
        new_session = self._detect_current_session()
        if new_session != self.current_session:
            self._handle_session_change(new_session)
    
    def update_trade_data(self,
                         trade_entry_time: float,
                         trade_exit_time: float,
                         trade_pnl: float,
                         trade_type: str):
        """
        💰 ATUALIZAR DADOS DE TRADING
        """
        trade_duration = (trade_exit_time - trade_entry_time) / 60  # minutos
        
        trade_record = {
            'entry_time': trade_entry_time,
            'exit_time': trade_exit_time,
            'duration': trade_duration,
            'pnl': trade_pnl,
            'type': trade_type,
            'session': self.current_session,
            'is_win': trade_pnl > 0
        }
        
        self.trade_memory.append(trade_record)
        
        # Atualizar histórico da sessão
        session_data = self.session_performance[self.current_session]
        session_data['trades'].append(trade_record)
        session_data['avg_duration'].append(trade_duration)
        
        # Manter apenas últimas 50 operações por sessão
        for key in session_data:
            if len(session_data[key]) > 50:
                session_data[key] = session_data[key][-50:]
    
    def calculate_fatigue_metrics(self) -> FatigueMetrics:
        """
        🎯 CALCULAR MÉTRICAS COMPLETAS DE FADIGA
        """
        current_time = time.time()
        
        # 1. FADIGA POR FREQUÊNCIA DE TRADING
        trading_frequency_fatigue = self._calculate_trading_frequency_fatigue(current_time)
        
        # 2. FADIGA POR EFICIÊNCIA DE MERCADO
        market_efficiency_fatigue = self._calculate_market_efficiency_fatigue()
        
        # 3. FADIGA POR VOLATILIDADE
        volatility_fatigue = self._calculate_volatility_fatigue()
        
        # 4. FADIGA POR VOLUME
        volume_fatigue = self._calculate_volume_fatigue()
        
        # 5. FADIGA POR DEGRADAÇÃO DE PADRÕES
        pattern_fatigue = self._calculate_pattern_degradation()
        
        # 6. FADIGA ESPECÍFICA DA SESSÃO
        session_fatigue = self._calculate_session_fatigue()
        
        # 7. SCORE GERAL DE FADIGA
        weights = self.fatigue_config
        overall_fatigue = (
            trading_frequency_fatigue * weights['trade_frequency_weight'] +
            market_efficiency_fatigue * weights['market_efficiency_weight'] +
            volatility_fatigue * weights['volatility_weight'] +
            volume_fatigue * weights['volume_weight'] +
            pattern_fatigue * weights['pattern_weight']
        )
        
        # Ajustar por fadiga da sessão
        overall_fatigue = min(overall_fatigue + session_fatigue * 0.1, 1.0)
        
        # 8. DETERMINAR NÍVEL DE FADIGA
        fatigue_level = self._determine_fatigue_level(overall_fatigue)
        
        # 9. GERAR RECOMENDAÇÕES
        should_reduce, should_pause, break_minutes = self._generate_recommendations(
            overall_fatigue, fatigue_level
        )
        
        return FatigueMetrics(
            overall_fatigue=overall_fatigue,
            trading_frequency=trading_frequency_fatigue,
            market_efficiency=market_efficiency_fatigue,
            volatility_fatigue=volatility_fatigue,
            volume_fatigue=volume_fatigue,
            pattern_degradation=pattern_fatigue,
            session_fatigue=session_fatigue,
            fatigue_level=fatigue_level,
            should_reduce_activity=should_reduce,
            should_pause_trading=should_pause,
            recommended_break_minutes=break_minutes
        )
    
    def _calculate_trading_frequency_fatigue(self, current_time: float) -> float:
        """📊 Calcular fadiga por frequência de trading"""
        if not self.trade_memory:
            return 0.0
        
        # Contar trades na última hora
        one_hour_ago = current_time - 3600
        recent_trades = [t for t in self.trade_memory 
                        if t['exit_time'] >= one_hour_ago]
        
        trades_per_hour = len(recent_trades)
        
        # Obter baseline da sessão atual
        session_baseline = self.session_characteristics[self.current_session]['typical_trades_per_hour']
        
        # Calcular fadiga (0-1)
        if trades_per_hour <= session_baseline:
            return 0.0  # Não há fadiga
        
        excess_ratio = trades_per_hour / session_baseline
        
        if excess_ratio < 1.5:
            return 0.2 * (excess_ratio - 1.0) / 0.5  # Fadiga leve
        elif excess_ratio < 2.0:
            return 0.2 + 0.3 * (excess_ratio - 1.5) / 0.5  # Fadiga moderada
        elif excess_ratio < 3.0:
            return 0.5 + 0.3 * (excess_ratio - 2.0) / 1.0  # Fadiga alta
        else:
            return 0.8 + 0.2 * min((excess_ratio - 3.0) / 2.0, 1.0)  # Fadiga crítica
    
    def _calculate_market_efficiency_fatigue(self) -> float:
        """📈 Calcular fadiga por eficiência de mercado"""
        if len(self.trade_memory) < 5:
            return 0.0
        
        # Calcular win rate recente (últimos 10 trades)
        recent_trades = list(self.trade_memory)[-10:]
        win_rate = sum(1 for t in recent_trades if t['is_win']) / len(recent_trades)
        
        # Calcular profit factor
        wins = [t['pnl'] for t in recent_trades if t['pnl'] > 0]
        losses = [abs(t['pnl']) for t in recent_trades if t['pnl'] < 0]
        
        if not losses:
            profit_factor = 10.0  # Sem perdas
        else:
            profit_factor = sum(wins) / sum(losses) if wins else 0.0
        
        # Baseline de eficiência da sessão
        session_baseline = self.session_characteristics[self.current_session]['efficiency_threshold']
        
        # Score de eficiência atual
        efficiency_score = (win_rate * 0.6) + (min(profit_factor / 2.0, 1.0) * 0.4)
        
        # Calcular fadiga por declínio de eficiência
        if efficiency_score >= session_baseline:
            return 0.0  # Eficiência boa
        
        decline = (session_baseline - efficiency_score) / session_baseline
        return min(decline * 2.0, 1.0)  # Amplificar declínio
    
    def _calculate_volatility_fatigue(self) -> float:
        """📊 Calcular fadiga por volatilidade excessiva"""
        if len(self.volatility_memory) < 10:
            return 0.0
        
        recent_vol = [v['volatility'] for v in list(self.volatility_memory)[-20:]]
        current_vol = np.mean(recent_vol[-5:])  # Últimos 5 períodos
        baseline_vol = np.mean(recent_vol)      # Média geral
        
        session_baseline = self.session_characteristics[self.current_session]['volatility_baseline']
        
        # Calcular spike de volatilidade
        vol_ratio = current_vol / max(session_baseline, 0.001)
        
        if vol_ratio < 1.2:
            return 0.0  # Volatilidade normal
        elif vol_ratio < 2.0:
            return 0.3 * (vol_ratio - 1.2) / 0.8  # Fadiga leve
        elif vol_ratio < 3.0:
            return 0.3 + 0.4 * (vol_ratio - 2.0) / 1.0  # Fadiga moderada
        else:
            return 0.7 + 0.3 * min((vol_ratio - 3.0) / 2.0, 1.0)  # Fadiga alta
    
    def _calculate_volume_fatigue(self) -> float:
        """📊 Calcular fadiga por padrões de volume"""
        if len(self.volume_memory) < 10:
            return 0.0
        
        recent_volumes = [v['volume'] for v in list(self.volume_memory)[-20:]]
        current_vol = np.mean(recent_volumes[-5:])
        baseline_vol = np.mean(recent_volumes)
        
        session_baseline = self.session_characteristics[self.current_session]['volume_baseline']
        
        # Detectar exaustão de volume (volume muito baixo)
        vol_ratio = current_vol / max(session_baseline, 0.1)
        
        if vol_ratio >= 0.7:
            return 0.0  # Volume saudável
        elif vol_ratio >= 0.5:
            return 0.2 * (0.7 - vol_ratio) / 0.2  # Fadiga leve
        elif vol_ratio >= 0.3:
            return 0.2 + 0.3 * (0.5 - vol_ratio) / 0.2  # Fadiga moderada
        else:
            return 0.5 + 0.5 * (0.3 - vol_ratio) / 0.3  # Volume criticamente baixo
    
    def _calculate_pattern_degradation(self) -> float:
        """📊 Calcular degradação de padrões técnicos"""
        if len(self.trade_memory) < 5:
            return 0.0
        
        recent_trades = list(self.trade_memory)[-10:]
        
        # Analisar duração dos trades (padrões mais rápidos = degradação)
        durations = [t['duration'] for t in recent_trades]
        avg_duration = np.mean(durations)
        
        # Analisar consistência dos resultados
        pnls = [t['pnl'] for t in recent_trades]
        pnl_volatility = np.std(pnls) / (abs(np.mean(pnls)) + 0.001)
        
        # Score de degradação
        duration_score = max(0, 1.0 - avg_duration / 30.0)  # 30min baseline
        volatility_score = min(pnl_volatility / 2.0, 1.0)   # Normalizar
        
        return (duration_score * 0.6 + volatility_score * 0.4)
    
    def _calculate_session_fatigue(self) -> float:
        """🕐 Calcular fadiga específica da sessão atual"""
        session_duration = (time.time() - self.session_start_time) / 3600  # horas
        
        # Fadiga cresce com duração da sessão
        if session_duration < 2:
            return 0.0
        elif session_duration < 4:
            return 0.1 * (session_duration - 2) / 2
        elif session_duration < 6:
            return 0.1 + 0.2 * (session_duration - 4) / 2
        else:
            return 0.3 + 0.4 * min((session_duration - 6) / 4, 1.0)
    
    def _determine_fatigue_level(self, overall_fatigue: float) -> FatigueLevel:
        """📊 Determinar nível de fadiga"""
        if overall_fatigue < 0.2:
            return FatigueLevel.FRESH
        elif overall_fatigue < 0.4:
            return FatigueLevel.NORMAL
        elif overall_fatigue < 0.6:
            return FatigueLevel.MODERATE
        elif overall_fatigue < 0.8:
            return FatigueLevel.HIGH
        else:
            return FatigueLevel.CRITICAL
    
    def _generate_recommendations(self, 
                                fatigue_score: float, 
                                fatigue_level: FatigueLevel) -> Tuple[bool, bool, int]:
        """🎯 Gerar recomendações baseadas na fadiga"""
        should_reduce = fatigue_score >= 0.4
        should_pause = fatigue_score >= 0.7
        
        # Calcular tempo de pausa recomendado
        if fatigue_level == FatigueLevel.CRITICAL:
            break_minutes = 60  # 1 hora
        elif fatigue_level == FatigueLevel.HIGH:
            break_minutes = 30  # 30 minutos
        elif fatigue_level == FatigueLevel.MODERATE:
            break_minutes = 15  # 15 minutos
        else:
            break_minutes = 0
        
        return should_reduce, should_pause, break_minutes
    
    def _handle_session_change(self, new_session: MarketSession):
        """🔄 Lidar com mudança de sessão"""
        print(f"🕐 Mudança de sessão: {self.current_session.value} → {new_session.value}")
        
        self.current_session = new_session
        self.session_start_time = time.time()
        
        # Reset de contadores específicos da sessão se necessário
        # (manter memória geral mas resetar contadores temporais)
    
    def get_fatigue_report(self) -> Dict:
        """📋 Relatório completo de fadiga"""
        metrics = self.calculate_fatigue_metrics()
        
        return {
            'timestamp': time.time(),
            'session': self.current_session.value,
            'metrics': {
                'overall_fatigue': f"{metrics.overall_fatigue:.2%}",
                'trading_frequency': f"{metrics.trading_frequency:.2%}",
                'market_efficiency': f"{metrics.market_efficiency:.2%}",
                'volatility_fatigue': f"{metrics.volatility_fatigue:.2%}",
                'volume_fatigue': f"{metrics.volume_fatigue:.2%}",
                'pattern_degradation': f"{metrics.pattern_degradation:.2%}",
                'session_fatigue': f"{metrics.session_fatigue:.2%}"
            },
            'status': {
                'fatigue_level': metrics.fatigue_level.value,
                'should_reduce_activity': metrics.should_reduce_activity,
                'should_pause_trading': metrics.should_pause_trading,
                'recommended_break_minutes': metrics.recommended_break_minutes
            },
            'session_info': self.session_characteristics[self.current_session],
            'recent_activity': {
                'trades_last_hour': len([t for t in self.trade_memory 
                                       if t['exit_time'] >= time.time() - 3600]),
                'avg_trade_duration_minutes': np.mean([t['duration'] for t in list(self.trade_memory)[-10:]]) if self.trade_memory else 0,
                'recent_win_rate': sum(1 for t in list(self.trade_memory)[-10:] if t['is_win']) / max(len(list(self.trade_memory)[-10:]), 1)
            }
        }

# 🎯 EXEMPLO DE INTEGRAÇÃO
def exemplo_integracao_fatigue_detector():
    """Exemplo de como integrar no sistema de trading"""
    
    # Inicializar detector
    fatigue_detector = MarketFatigueDetector(lookback_hours=6)
    
    # Simular dados de mercado
    import time
    current_time = time.time()
    
    # Atualizar dados de mercado
    fatigue_detector.update_market_data(
        timestamp=current_time,
        price=1950.50,
        volume=1.2,  # Ratio vs baseline
        volatility=0.015
    )
    
    # Simular alguns trades
    for i in range(5):
        entry_time = current_time - (i * 600)  # A cada 10 min
        exit_time = entry_time + 300  # 5 min duration
        pnl = 15.0 if i % 2 == 0 else -8.0  # Alternando win/loss
        
        fatigue_detector.update_trade_data(
            trade_entry_time=entry_time,
            trade_exit_time=exit_time,
            trade_pnl=pnl,
            trade_type="long"
        )
    
    # Obter métricas de fadiga
    metrics = fatigue_detector.calculate_fatigue_metrics()
    
    # Gerar relatório
    report = fatigue_detector.get_fatigue_report()
    
    print("🚀 RELATÓRIO DE FADIGA DE MERCADO")
    print("=" * 50)
    print(f"Sessão Atual: {report['session'].upper()}")
    print(f"Nível de Fadiga: {metrics.fatigue_level.value.upper()}")
    print(f"Score Geral: {metrics.overall_fatigue:.1%}")
    print()
    print("📊 COMPONENTES:")
    for component, value in report['metrics'].items():
        print(f"  {component.replace('_', ' ').title()}: {value}")
    print()
    print("🎯 RECOMENDAÇÕES:")
    if metrics.should_pause_trading:
        print(f"  🚨 PAUSAR TRADING por {metrics.recommended_break_minutes} minutos")
    elif metrics.should_reduce_activity:
        print(f"  ⚠️ REDUZIR ATIVIDADE de trading")
    else:
        print(f"  ✅ CONTINUAR trading normal")
    
    return metrics

if __name__ == "__main__":
    exemplo_integracao_fatigue_detector()