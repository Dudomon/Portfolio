"""
⚡ REWARD SYSTEM ULTRA SIMPLES E RÁPIDO
Sistema minimalista focado apenas no essencial para detectar se é problema de performance
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any

class SimpleFastReward:
    """
    Sistema de reward ULTRA SIMPLES - apenas PnL direto
    Zero complexidade, máxima performance
    """
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        Reward ULTRA SIMPLES: apenas PnL direto normalizado
        Zero processamento extra, máxima velocidade
        """
        
        try:
            # 1. PnL direto apenas
            realized_pnl = getattr(env, 'total_realized_pnl', 0.0)
            unrealized_pnl = getattr(env, 'total_unrealized_pnl', 0.0)
            
            # 2. PnL total simples
            total_pnl = realized_pnl + (unrealized_pnl * 0.5)
            pnl_percent = total_pnl / self.initial_balance
            
            # 3. Reward = PnL normalizado
            reward = pnl_percent * 10.0  # Scale para range adequado
            
            # 4. Clipping simples
            reward = np.clip(reward, -1.0, 1.0)
            
            # 5. Early termination simples
            portfolio_value = getattr(env, 'current_balance', self.initial_balance)
            done = portfolio_value < (self.initial_balance * 0.5)  # -50% stop
            
            # 6. Info mínimo
            info = {
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl,
                'reward_type': 'simple_fast'
            }
            
            return reward, info, done
            
        except Exception as e:
            # Fallback seguro
            return 0.0, {'error': str(e)}, False

# Factory function para compatibilidade
def create_simple_fast_reward_system(initial_balance: float = 1000.0):
    """Factory function para o sistema simples"""
    return SimpleFastReward(initial_balance)