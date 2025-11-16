#!/usr/bin/env python3
"""
💰 SIMPLE MONEY REWARD
Reward system que REALMENTE ensina o modelo a fazer dinheiro
Sem bullshit acadêmico, sem over-engineering
"""

import numpy as np

class SimpleMoneyReward:
    """
    Reward system brutalmente simples:
    - PnL = reward direto
    - Perdas grandes = dor multiplicada
    - Sem métricas acadêmicas inúteis
    """
    
    def __init__(self, pain_multiplier: float = 3.0):
        self.pain_multiplier = pain_multiplier
        
    def calculate_reward(self, trade_pnl: float, portfolio_value: float, initial_balance: float) -> float:
        """
        Calcula reward baseado EXCLUSIVAMENTE em fazer dinheiro
        
        Args:
            trade_pnl: PnL do trade (positivo = lucro, negativo = prejuízo)
            portfolio_value: Valor atual do portfolio
            initial_balance: Balance inicial
            
        Returns:
            reward: Valor direto proporcional ao PnL
        """
        # PnL como % do balance inicial (scale realista)
        pnl_percent = trade_pnl / initial_balance
        
        # Reward base = PnL direto amplificado para RL
        reward = pnl_percent * 100.0  # Scale para [-10, +10] range típico
        
        # CRÍTICO: Amplificar DOR de perdas grandes
        if pnl_percent < -0.05:  # Perda > 5%
            # Perdas grandes doem MUITO mais
            reward *= self.pain_multiplier
            
        # OPCIONAL: Bonus pequeno para consistency em lucros
        if pnl_percent > 0.02:  # Lucro > 2%
            reward *= 1.1  # Pequeno bonus
            
        # Clipping para estabilidade RL
        return np.clip(reward, -20.0, 20.0)
    
    def calculate_continuous_reward(self, unrealized_pnl: float, portfolio_value: float, initial_balance: float) -> float:
        """
        Reward contínuo baseado em PnL não realizado
        (para feedback durante trades abertos)
        """
        pnl_percent = unrealized_pnl / initial_balance
        
        # Reward menor para PnL não realizado (desconto de risco)
        reward = pnl_percent * 50.0  # Metade do peso do PnL realizado
        
        # Mesmo sistema de pain para perdas não realizadas
        if pnl_percent < -0.05:
            reward *= self.pain_multiplier
            
        return np.clip(reward, -10.0, 10.0)


def test_simple_reward():
    """Teste para verificar que o sistema funciona"""
    reward_system = SimpleMoneyReward()
    
    # Cenários de teste
    scenarios = [
        ("Lucro pequeno 1%", 100, 10000, 10000),      # +1% = reward ~1.0
        ("Lucro grande 5%", 500, 10500, 10000),       # +5% = reward ~5.5 (com bonus)
        ("Perda pequena 2%", -200, 9800, 10000),      # -2% = reward ~-2.0
        ("Perda grande 8%", -800, 9200, 10000),       # -8% = reward ~-24.0 (pain!)
        ("Break even", 0, 10000, 10000),              # 0% = reward 0.0
    ]
    
    print("🧪 TESTE SIMPLE MONEY REWARD")
    print("=" * 50)
    
    for scenario, pnl, portfolio, initial in scenarios:
        reward = reward_system.calculate_reward(pnl, portfolio, initial)
        pnl_pct = (pnl / initial) * 100
        
        print(f"{scenario}")
        print(f"  PnL: ${pnl} ({pnl_pct:+.1f}%)")
        print(f"  Reward: {reward:+.2f}")
        print()
    
    print("🎯 OBJETIVO: Reward deve ser PROPORCIONAL ao PnL real")
    print("💥 PAIN: Perdas grandes devem DOER muito mais que lucros pequenos")


if __name__ == "__main__":
    test_simple_reward()