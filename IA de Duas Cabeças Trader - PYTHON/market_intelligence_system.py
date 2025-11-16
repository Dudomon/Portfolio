#!/usr/bin/env python3
"""
🧠 MARKET INTELLIGENCE SYSTEM
============================

Sistema para adicionar inteligência real ao modelo:
- ✅ Microstructura de mercado
- ✅ Timing awareness
- ✅ Execution conditions
- ✅ Risk regime detection
"""

import numpy as np
import pandas as pd
from datetime import datetime, time
from typing import Dict, Tuple

class MarketIntelligenceSystem:
    """
    🧠 Sistema que adiciona inteligência de mercado real ao modelo
    """

    def __init__(self):
        self.min_volume_threshold = 1000
        self.max_spread_threshold = 3.0
        self.session_multipliers = {
            'london_open': 1.5,     # 08:00-12:00 UTC
            'us_open': 1.3,         # 13:30-17:00 UTC
            'overlap': 1.8,         # 13:30-16:00 UTC (overlap)
            'asian': 0.7,           # 00:00-08:00 UTC
            'dead': 0.3             # 17:00-00:00 UTC
        }

    def should_trade_now(self, market_data: Dict, action_confidence: float) -> Tuple[bool, float, str]:
        """
        🎯 DECISÃO INTELIGENTE: Deve tradear agora?

        Returns:
            should_trade (bool): Se deve executar o trade
            confidence_multiplier (float): Multiplicador de confiança
            reason (str): Razão da decisão
        """

        # 1. 📊 VOLUME ANALYSIS
        volume_ok, volume_reason = self._check_volume_conditions(market_data)
        if not volume_ok:
            return False, 0.0, f"Volume inadequado: {volume_reason}"

        # 2. 💰 SPREAD ANALYSIS
        spread_ok, spread_reason = self._check_spread_conditions(market_data)
        if not spread_ok:
            return False, 0.0, f"Spread muito alto: {spread_reason}"

        # 3. 🕐 SESSION TIMING
        session_mult, session_name = self._get_session_multiplier()

        # 4. 📈 VOLATILITY REGIME
        vol_mult, vol_reason = self._check_volatility_regime(market_data)

        # 5. 🚨 NEWS/EVENT PROXIMITY
        news_ok, news_reason = self._check_news_proximity()
        if not news_ok:
            return False, 0.0, f"Evento próximo: {news_reason}"

        # 6. 🎯 FINAL DECISION
        final_multiplier = session_mult * vol_mult

        # Só tradear se confiança ajustada > threshold
        adjusted_confidence = action_confidence * final_multiplier

        if adjusted_confidence > 0.6:  # Threshold inteligente
            return True, final_multiplier, f"Condições adequadas: {session_name}, vol_ok"
        else:
            return False, final_multiplier, f"Confiança baixa após ajustes: {adjusted_confidence:.2f}"

    def _check_volume_conditions(self, market_data: Dict) -> Tuple[bool, str]:
        """📊 ANÁLISE DE VOLUME"""
        try:
            current_volume = market_data.get('volume', 0)
            avg_volume = market_data.get('avg_volume_20', current_volume)

            # Volume muito baixo = não tradear
            if current_volume < self.min_volume_threshold:
                return False, f"Volume {current_volume} < threshold {self.min_volume_threshold}"

            # Volume muito abaixo da média = aguardar
            if current_volume < avg_volume * 0.3:
                return False, f"Volume {current_volume} << média {avg_volume:.0f}"

            return True, "Volume adequado"

        except:
            return True, "Volume check failed - assumindo OK"

    def _check_spread_conditions(self, market_data: Dict) -> Tuple[bool, str]:
        """💰 ANÁLISE DE SPREAD"""
        try:
            spread = market_data.get('spread', 1.0)

            if spread > self.max_spread_threshold:
                return False, f"Spread {spread} > {self.max_spread_threshold}"

            # Spread muito alto vs. volatilidade = esperar
            volatility = market_data.get('atr_14', 10.0)
            if spread > volatility * 0.3:  # Spread > 30% da volatilidade
                return False, f"Spread {spread} alto vs volatilidade {volatility:.1f}"

            return True, "Spread adequado"

        except:
            return True, "Spread check failed - assumindo OK"

    def _get_session_multiplier(self) -> Tuple[float, str]:
        """🕐 MULTIPLICADOR POR SESSÃO"""
        try:
            now_utc = datetime.utcnow().time()

            # Londres: 08:00-12:00 UTC
            if time(8, 0) <= now_utc <= time(12, 0):
                return self.session_multipliers['london_open'], "London Open"

            # Overlap Londres-US: 13:30-16:00 UTC (MELHOR PERÍODO)
            elif time(13, 30) <= now_utc <= time(16, 0):
                return self.session_multipliers['overlap'], "London-US Overlap"

            # US: 13:30-17:00 UTC
            elif time(13, 30) <= now_utc <= time(17, 0):
                return self.session_multipliers['us_open'], "US Open"

            # Asiático: 00:00-08:00 UTC
            elif time(0, 0) <= now_utc <= time(8, 0):
                return self.session_multipliers['asian'], "Asian Session"

            # Dead zone: 17:00-00:00 UTC
            else:
                return self.session_multipliers['dead'], "Dead Zone"

        except:
            return 1.0, "Unknown Session"

    def _check_volatility_regime(self, market_data: Dict) -> Tuple[float, str]:
        """📈 ANÁLISE DE REGIME DE VOLATILIDADE"""
        try:
            atr = market_data.get('atr_14', 10.0)
            atr_avg = market_data.get('atr_avg', atr)

            vol_ratio = atr / atr_avg if atr_avg > 0 else 1.0

            if vol_ratio > 2.0:  # Volatilidade extrema
                return 0.5, "Volatilidade extrema - reduzir posições"
            elif vol_ratio > 1.5:  # Alta volatilidade
                return 0.8, "Alta volatilidade - cuidado"
            elif vol_ratio < 0.5:  # Baixa volatilidade
                return 0.7, "Baixa volatilidade - sinais fracos"
            else:  # Volatilidade normal
                return 1.0, "Volatilidade normal"

        except:
            return 1.0, "Volatility check failed"

    def _check_news_proximity(self) -> Tuple[bool, str]:
        """🚨 PROXIMIDADE DE EVENTOS/NEWS"""
        try:
            now = datetime.utcnow()
            minute = now.minute
            hour = now.hour

            # Evitar alguns minutos específicos (NFP, FOMC, etc.)
            # Horários típicos de news: :30, :00 de algumas horas
            news_minutes = [0, 30]
            critical_hours = [8, 9, 10, 13, 14, 15]  # UTC

            if hour in critical_hours and minute in news_minutes:
                return False, f"Possível horário de news: {hour:02d}:{minute:02d} UTC"

            # Sexta-feira após 15:00 UTC (fim de semana se aproximando)
            if now.weekday() == 4 and hour >= 15:  # Friday
                return False, "Sexta-feira tarde - evitar novas posições"

            return True, "Sem eventos próximos"

        except:
            return True, "News check failed - assumindo OK"

    def adjust_position_size(self, base_size: float, market_conditions: Dict) -> float:
        """
        📏 AJUSTAR TAMANHO DA POSIÇÃO BASEADO EM CONDIÇÕES
        """
        try:
            # Base adjustment factors
            vol_factor = 1.0
            session_factor = 1.0

            # Reduce size in high volatility
            atr = market_conditions.get('atr_14', 10.0)
            atr_avg = market_conditions.get('atr_avg', atr)
            vol_ratio = atr / atr_avg if atr_avg > 0 else 1.0

            if vol_ratio > 2.0:
                vol_factor = 0.5  # Halve size in extreme volatility
            elif vol_ratio > 1.5:
                vol_factor = 0.7  # Reduce size in high volatility

            # Adjust for session
            _, session_mult = self._get_session_multiplier()
            session_factor = min(session_mult, 1.0)  # Only reduce, never increase

            # Apply adjustments
            adjusted_size = base_size * vol_factor * session_factor

            # Ensure minimum size
            return max(adjusted_size, base_size * 0.3)  # Never go below 30%

        except:
            return base_size  # Fallback to original size

def create_market_intelligence_system():
    """Factory function"""
    return MarketIntelligenceSystem()

# Test function
if __name__ == "__main__":
    mi = MarketIntelligenceSystem()

    test_data = {
        'volume': 5000,
        'avg_volume_20': 3000,
        'spread': 2.0,
        'atr_14': 12.0,
        'atr_avg': 10.0
    }

    should_trade, multiplier, reason = mi.should_trade_now(test_data, 0.8)
    print(f"Should trade: {should_trade}")
    print(f"Multiplier: {multiplier:.2f}")
    print(f"Reason: {reason}")