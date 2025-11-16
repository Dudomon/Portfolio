#!/usr/bin/env python3
"""
🧠 INTELLIGENT EXECUTION FILTER
==============================

Filtro inteligente para integrar com SILUS e evitar trades burros
"""

import sys
sys.path.append("D:/Projeto")

from market_intelligence_system import MarketIntelligenceSystem
import numpy as np

class IntelligentExecutionFilter:
    """
    🧠 Filtro que decide se o modelo deve executar a ação ou não
    """

    def __init__(self):
        self.market_intelligence = MarketIntelligenceSystem()
        self.recent_bad_trades = []
        self.max_bad_trades_memory = 5

    def should_execute_action(self, action, env, market_data=None) -> dict:
        """
        🎯 DECISÃO FINAL: Executar ação do modelo?

        Args:
            action: Ação do modelo [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
            env: Environment do SILUS
            market_data: Dados de mercado atual

        Returns:
            dict: {
                'execute': bool,
                'modified_action': np.array,
                'reason': str,
                'confidence_adjustment': float
            }
        """

        entry_decision = float(action[0]) if len(action) > 0 else 0.0
        confidence = float(action[1]) if len(action) > 1 else 0.5

        # 🚫 1. FILTRO BÁSICO: Só analisar se modelo quer tradear
        if abs(entry_decision) < 0.33:  # HOLD
            return {
                'execute': True,
                'modified_action': action,
                'reason': 'HOLD action - no filtering needed',
                'confidence_adjustment': 1.0
            }

        # 📊 2. PREPARAR DADOS DE MERCADO
        if market_data is None:
            market_data = self._extract_market_data(env)

        # 🧠 3. ANÁLISE INTELIGENTE
        should_trade, multiplier, reason = self.market_intelligence.should_trade_now(
            market_data, confidence
        )

        # 🚨 4. DECISÃO FINAL
        if not should_trade:
            # ❌ BLOQUEAR TRADE
            modified_action = action.copy()
            modified_action[0] = 0.0  # Force HOLD

            # Track bad timing
            self._track_bad_trade_attempt(reason)

            return {
                'execute': False,
                'modified_action': modified_action,
                'reason': f"🚫 BLOCKED: {reason}",
                'confidence_adjustment': 0.0
            }

        else:
            # ✅ PERMITIR TRADE COM AJUSTES
            modified_action = action.copy()

            # Ajustar confiança baseado em condições
            adjusted_confidence = min(confidence * multiplier, 1.0)
            modified_action[1] = adjusted_confidence

            # Ajustar tamanho implicitamente via confidence
            # (SILUS usa confidence para position sizing)

            return {
                'execute': True,
                'modified_action': modified_action,
                'reason': f"✅ ALLOWED: {reason} (conf: {confidence:.2f}→{adjusted_confidence:.2f})",
                'confidence_adjustment': multiplier
            }

    def _extract_market_data(self, env) -> dict:
        """📊 EXTRAIR DADOS DE MERCADO DO ENVIRONMENT"""
        try:
            current_step = getattr(env, 'current_step', 0)
            df = getattr(env, 'df', None)

            if df is None or current_step >= len(df):
                return self._get_default_market_data()

            # Extract current market data
            current_data = {
                'volume': df.get('volume_1m', pd.Series([1000])).iloc[current_step] if 'volume_1m' in df.columns else 1000,
                'close': df.get('close_1m', pd.Series([2000])).iloc[current_step] if 'close_1m' in df.columns else 2000,
                'high': df.get('high_1m', pd.Series([2010])).iloc[current_step] if 'high_1m' in df.columns else 2010,
                'low': df.get('low_1m', pd.Series([1990])).iloc[current_step] if 'low_1m' in df.columns else 1990,
            }

            # Calculate derived metrics
            if current_step >= 20:
                # Average volume (20 periods)
                vol_series = df.get('volume_1m', pd.Series([1000] * len(df)))
                current_data['avg_volume_20'] = vol_series.iloc[current_step-20:current_step].mean()

                # ATR estimation
                high_series = df.get('high_1m', pd.Series([2010] * len(df)))
                low_series = df.get('low_1m', pd.Series([1990] * len(df)))
                close_series = df.get('close_1m', pd.Series([2000] * len(df)))

                tr = np.maximum(
                    high_series.iloc[current_step-20:current_step] - low_series.iloc[current_step-20:current_step],
                    np.maximum(
                        np.abs(high_series.iloc[current_step-20:current_step] - close_series.iloc[current_step-21:current_step-1]),
                        np.abs(low_series.iloc[current_step-20:current_step] - close_series.iloc[current_step-21:current_step-1])
                    )
                )
                current_data['atr_14'] = tr.mean()
                current_data['atr_avg'] = current_data['atr_14']  # Simplified
            else:
                current_data['avg_volume_20'] = current_data['volume']
                current_data['atr_14'] = 10.0
                current_data['atr_avg'] = 10.0

            # Estimate spread (simplified)
            current_data['spread'] = max(1.0, (current_data['high'] - current_data['low']) * 0.1)

            return current_data

        except Exception as e:
            print(f"⚠️ [INTELLIGENCE] Erro extraindo dados: {e}")
            return self._get_default_market_data()

    def _get_default_market_data(self) -> dict:
        """📊 DADOS PADRÃO SE EXTRAÇÃO FALHAR"""
        return {
            'volume': 2000,
            'avg_volume_20': 2000,
            'spread': 2.0,
            'atr_14': 10.0,
            'atr_avg': 10.0,
            'close': 2000,
            'high': 2010,
            'low': 1990
        }

    def _track_bad_trade_attempt(self, reason: str):
        """📝 RASTREAR TENTATIVAS DE TRADE RUINS"""
        self.recent_bad_trades.append({
            'reason': reason,
            'timestamp': pd.Timestamp.now()
        })

        # Manter apenas os últimos N
        if len(self.recent_bad_trades) > self.max_bad_trades_memory:
            self.recent_bad_trades.pop(0)

    def get_intelligence_stats(self) -> dict:
        """📊 ESTATÍSTICAS DO SISTEMA DE INTELIGÊNCIA"""
        return {
            'recent_blocks': len(self.recent_bad_trades),
            'block_reasons': [trade['reason'] for trade in self.recent_bad_trades[-3:]],
            'market_intelligence_active': True
        }

def create_intelligent_execution_filter():
    """Factory function"""
    return IntelligentExecutionFilter()

# Test
if __name__ == "__main__":
    import pandas as pd

    filter_system = IntelligentExecutionFilter()

    # Test action
    test_action = np.array([0.8, 0.7, 0.2, -0.1])  # Strong long signal

    # Mock env
    class MockEnv:
        def __init__(self):
            self.current_step = 50
            self.df = pd.DataFrame({
                'volume_1m': np.random.randint(1000, 5000, 100),
                'close_1m': 2000 + np.random.randn(100) * 10,
                'high_1m': 2000 + np.random.randn(100) * 10 + 5,
                'low_1m': 2000 + np.random.randn(100) * 10 - 5,
            })

    result = filter_system.should_execute_action(test_action, MockEnv())

    print(f"Execute: {result['execute']}")
    print(f"Reason: {result['reason']}")
    print(f"Original action: {test_action}")
    print(f"Modified action: {result['modified_action']}")