#!/usr/bin/env python3
"""
🔧 CORREÇÃO DEFINITIVA DO REWARD SYSTEM V3
Implementação das 3 correções fundamentais identificadas
"""

import sys
sys.path.append("D:/Projeto")
import numpy as np
from typing import Dict, List, Any
from trading_framework.rewards.reward_daytrade_v2 import BalancedDayTradingRewardCalculator

class FixedRewardCalculator(BalancedDayTradingRewardCalculator):
    """Reward calculator com correções fundamentais"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 🔧 CORREÇÃO 1: ANTI-MICRO-FARMING
        self.min_trade_size_threshold = 10.0  # Trades < $10 são penalizados
        self.overtrading_penalty_threshold = 20  # 20+ trades por episódio = overtrading
        self.trade_quality_bonus_multiplier = 2.0  # Premia trades grandes
        
        # 🔧 CORREÇÃO 2: REBALANCEAMENTO DE COMPONENTES
        self.activity_bonus_weight = 0.15  # Reduzido de dominante para 15%
        self.base_reward_weight = 0.60     # Aumentado para 60% (componente principal)
        self.risk_reward_weight = 0.20     # 20% para risk management
        self.quality_bonus_weight = 0.05   # 5% para qualidade de trades
        
        # 🔧 CORREÇÃO 3: QUEBRA DE DEPENDÊNCIA CIRCULAR
        self.episode_trade_count = 0
        self.episode_volume_threshold = 25  # Máximo 25 trades por episódio sem penalidade
        
    def calculate_reward_and_info(self, env, action, old_state):
        """Reward calculation com correções aplicadas"""
        
        # Reset episode counter se necessário
        if hasattr(env, 'current_step') and env.current_step == 0:
            self.episode_trade_count = 0
            
        # Get base reward first
        base_reward, info, done = super().calculate_reward_and_info(env, action, old_state)
        
        # Aplicar correções
        corrected_reward = self._apply_corrections(base_reward, env, action, info)
        
        # Update info with corrections
        info['corrections_applied'] = self._get_correction_details(env)
        info['original_reward'] = base_reward
        info['corrected_reward'] = corrected_reward
        
        return corrected_reward, info, done
    
    def _apply_corrections(self, base_reward: float, env, action, info: Dict) -> float:
        """Aplicar todas as correções ao reward base"""
        
        corrected_reward = base_reward
        recent_trades = getattr(env, 'trades', [])
        
        if not recent_trades:
            return corrected_reward
            
        # 🔧 CORREÇÃO 1: Anti-Micro-Farming
        corrected_reward = self._apply_anti_microfarming(corrected_reward, recent_trades)
        
        # 🔧 CORREÇÃO 2: Rebalanceamento de componentes
        corrected_reward = self._apply_component_rebalancing(corrected_reward, env, info)
        
        # 🔧 CORREÇÃO 3: Anti-Overtrading
        corrected_reward = self._apply_anti_overtrading(corrected_reward, env)
        
        return corrected_reward
        
    def _apply_anti_microfarming(self, reward: float, trades: List) -> float:
        """Correção 1: Penalizar micro-farming"""
        
        if not trades:
            return reward
            
        recent_trades = trades[-10:] if len(trades) >= 10 else trades
        small_trades = 0
        total_volume = 0
        
        for trade in recent_trades:
            trade_pnl = abs(trade.get('pnl_usd', 0))
            total_volume += trade_pnl
            
            if trade_pnl < self.min_trade_size_threshold:
                small_trades += 1
        
        # Penalidade por trades muito pequenos
        if small_trades > 0:
            small_trade_ratio = small_trades / len(recent_trades)
            micro_penalty = small_trade_ratio * 0.3  # Até 30% de penalidade
            reward *= (1 - micro_penalty)
            
        # Bonus por trades de qualidade (grandes)
        if len(recent_trades) > 0:
            avg_trade_size = total_volume / len(recent_trades)
            if avg_trade_size > 50:  # Trades grandes
                quality_bonus = min(0.5, avg_trade_size / 200)  # Até 50% bonus
                reward *= (1 + quality_bonus * self.trade_quality_bonus_multiplier)
                
        return reward
    
    def _apply_component_rebalancing(self, reward: float, env, info: Dict) -> float:
        """Correção 2: Rebalancear componentes do reward"""
        
        components = info.get('reward_components', {})
        
        # Recalcular com novos pesos
        rebalanced_reward = 0.0
        
        # Base reward (60% do peso)
        base_component = components.get('base_reward', 0)
        rebalanced_reward += base_component * self.base_reward_weight
        
        # Activity bonus (15% do peso - reduzido)
        activity_component = components.get('activity_bonus', 0)
        rebalanced_reward += activity_component * self.activity_bonus_weight
        
        # Risk reward (20% do peso)
        risk_component = components.get('risk_reward', 0)
        rebalanced_reward += risk_component * self.risk_reward_weight
        
        # Quality bonus (5% do peso)
        quality_component = self._calculate_quality_component(env)
        rebalanced_reward += quality_component * self.quality_bonus_weight
        
        return rebalanced_reward
    
    def _apply_anti_overtrading(self, reward: float, env) -> float:
        """Correção 3: Penalizar overtrading"""
        
        total_trades = len(getattr(env, 'trades', []))
        
        # Penalidade progressiva por overtrading
        if total_trades > self.episode_volume_threshold:
            excess_trades = total_trades - self.episode_volume_threshold
            overtrading_penalty = min(0.8, excess_trades * 0.02)  # Até 80% de penalidade
            reward *= (1 - overtrading_penalty)
            
        return reward
        
    def _calculate_quality_component(self, env) -> float:
        """Calcular componente de qualidade baseado em métricas reais"""
        
        trades = getattr(env, 'trades', [])
        if not trades:
            return 0.0
            
        recent_trades = trades[-5:] if len(trades) >= 5 else trades
        
        # Métricas de qualidade
        avg_pnl = np.mean([trade.get('pnl_usd', 0) for trade in recent_trades])
        win_rate = len([t for t in recent_trades if t.get('pnl_usd', 0) > 0]) / len(recent_trades)
        
        # Score de qualidade baseado em PnL e win rate balanceados
        quality_score = 0.0
        
        if avg_pnl > 20 and win_rate >= 0.4:  # Trades rentáveis com win rate razoável
            quality_score = min(1.0, avg_pnl / 100) * win_rate
            
        return quality_score
        
    def _get_correction_details(self, env) -> Dict:
        """Detalhes das correções aplicadas para debugging"""
        
        trades = getattr(env, 'trades', [])
        recent_trades = trades[-10:] if len(trades) >= 10 else trades
        
        return {
            'total_trades': len(trades),
            'recent_trades_count': len(recent_trades),
            'small_trades_ratio': len([t for t in recent_trades if abs(t.get('pnl_usd', 0)) < self.min_trade_size_threshold]) / max(1, len(recent_trades)),
            'overtrading_detected': len(trades) > self.episode_volume_threshold,
            'avg_trade_size': np.mean([abs(t.get('pnl_usd', 0)) for t in recent_trades]) if recent_trades else 0
        }

def test_corrections():
    """Testar as correções implementadas"""
    print("🔧 TESTANDO CORREÇÕES DO REWARD SYSTEM")
    print("=" * 50)
    
    # Mock environment
    class MockEnv:
        def __init__(self):
            self.trades = []
            self.current_step = 0
            self.balance = 1000
            self.realized_balance = 1000
            self.portfolio_value = 1000
            
        def add_trade(self, pnl):
            self.trades.append({'pnl_usd': pnl, 'pnl': pnl})
            
    # Test 1: Micro-farming scenario
    print("\n🧪 TESTE 1: MICRO-FARMING")
    env1 = MockEnv()
    calc = FixedRewardCalculator()
    
    # Add many small trades
    total_reward_old = 0
    for i in range(50):
        env1.add_trade(2.0)  # Small $2 trades
        
    # Get reward
    action = np.array([0.6, 0.2, 0, 0, 0, 0, 0, 0])
    old_state = {}
    
    reward_new, info, _ = calc.calculate_reward_and_info(env1, action, old_state)
    
    print(f"   Trades: 50 × $2")
    print(f"   New Reward: {reward_new:.6f}")
    print(f"   Corrections: {info['corrections_applied']}")
    
    # Test 2: Quality trading scenario
    print("\n🧪 TESTE 2: QUALITY TRADING")
    env2 = MockEnv()
    
    # Add fewer but larger trades
    for i in range(5):
        env2.add_trade(80.0)  # Large $80 trades
        
    reward_new2, info2, _ = calc.calculate_reward_and_info(env2, action, old_state)
    
    print(f"   Trades: 5 × $80")
    print(f"   New Reward: {reward_new2:.6f}")
    print(f"   Corrections: {info2['corrections_applied']}")
    
    print(f"\n🎯 COMPARAÇÃO:")
    print(f"   Micro-farming (50×$2): {reward_new:.6f}")
    print(f"   Quality trading (5×$80): {reward_new2:.6f}")
    print(f"   Quality ratio: {reward_new2/reward_new:.2f}x melhor")
    
if __name__ == "__main__":
    test_corrections()