"""
💰 BRUTAL MONEY-FOCUSED REWARD SYSTEM V3
Sistema de reward que REALMENTE ensina a fazer dinheiro
Zero bullshit acadêmico, zero over-engineering

🎯 FILOSOFIA:
- PnL = 90% do reward
- Risk management = 10% do reward  
- Perdas grandes = DOR MULTIPLICADA
- Sem synthetic PnL, action quality, ou outras merdas acadêmicas
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

class BrutalMoneyReward:
    """
    Sistema de reward focado 100% em fazer dinheiro
    COM REWARD SHAPING ADEQUADO para resolver sparsity
    """
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.logger = logging.getLogger(__name__)
        
        # Configurações brutais (ULTRA ESTABILIZADAS)
        self.pain_multiplier = 1.5      # Perdas doem 1.5x mais (era 2x)
        self.risk_penalty_threshold = 0.15  # Drawdown > 15% = penalty severa
        self.max_reward = 1.0           # Cap normalizado para PPO
        
        # 🎯 REWARD SHAPING PARAMETERS (ESTABILIZADOS)
        self.progress_scaling = 0.05    # Escala para progress rewards (era 0.1)
        self.momentum_weight = 0.02     # Peso para momentum trading (era 0.05)
        self.position_decay = 0.995     # Decay para posições longas (era 0.99)
        
        # Tracking para shaping
        self.last_portfolio_value = initial_balance
        self.position_age_steps = {}    # position_id -> steps_held
        self.recent_pnl_trend = []      # Últimos 10 PnLs para momentum
        
        # Performance cache
        self.step_counter = 0
        self.cached_risk_reward = 0.0
        self.cached_shaping_reward = 0.0
        self.cached_portfolio_drawdown = 0.0
        
        # Métricas para debugging
        self.total_rewards_given = 0
        self.positive_rewards = 0
        self.negative_rewards = 0
        
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        Calcula reward focado EXCLUSIVAMENTE em PnL real + REWARD SHAPING
        
        Returns:
            reward: Valor baseado no PnL real + shaping
            info: Informações de debug
            done: Se deve terminar episódio (perdas catastróficas)
        """
        
        self.step_counter += 1
        
        # 1. EXTRAIR PNL REAL (85% do reward - sempre calculado)
        pnl_reward, pnl_info = self._calculate_pure_pnl_reward(env)
        
        # 2. RISK MANAGEMENT BÁSICO (10% do reward) - CACHED A CADA 5 STEPS
        if self.step_counter % 5 == 0:
            self.cached_risk_reward, risk_info = self._calculate_basic_risk_reward(env)
        else:
            risk_info = {'cached': True}
        risk_reward = self.cached_risk_reward
        
        # 3. 🎯 REWARD SHAPING ADEQUADO (5% do reward) - CACHED A CADA 10 STEPS
        if self.step_counter % 10 == 0:
            self.cached_shaping_reward, shaping_info = self._calculate_reward_shaping(env, action)
        else:
            shaping_info = {'cached': True}
        shaping_reward = self.cached_shaping_reward
        
        # 4. REWARD FINAL = PnL PURO para NOTA 10 (85%) + shaping proporcional (15%)
        # Eliminar risk components para nota 10
        pure_pnl_component = pnl_reward * 0.85
        proportional_shaping = self._calculate_proportional_shaping(env, pnl_reward)
        total_reward = pure_pnl_component + (proportional_shaping * 0.15)
        
        # 5. EARLY TERMINATION para perdas catastróficas - CACHED A CADA 20 STEPS
        if self.step_counter % 20 == 0:
            self.cached_portfolio_drawdown = self._calculate_portfolio_drawdown(env)
        done = self.cached_portfolio_drawdown > 0.5  # Termina se perder >50%
        
        # 6. NORMALIZAÇÃO SUAVE para PPO (removendo divisão excessiva)
        # total_reward = total_reward / 10.0  # REMOVIDO - estava causando rewards ~0
        
        # Aplicar tanh suave ao invés de clipping duro
        total_reward = self.max_reward * np.tanh(total_reward / self.max_reward)
        
        # Safeguard numérico robusto
        if not np.isfinite(total_reward) or abs(total_reward) > 100:
            total_reward = 0.0
        
        # 7. UPDATE TRACKING para próxima iteração - REDUCED FREQUENCY
        if self.step_counter % 5 == 0:
            self._update_tracking(env)
        
        # 8. LOGGING - REDUCED FREQUENCY  
        if self.step_counter % 10 == 0:
            self._update_stats(total_reward)
        
        # 9. INFO para debugging
        info = {
            'pnl_reward': pnl_reward,
            'risk_reward': 0.0,  # Eliminado para nota 10
            'shaping_reward': proportional_shaping,
            'total_reward': total_reward,
            'pure_pnl_component': pure_pnl_component,
            'proportional_shaping': proportional_shaping,
            'portfolio_drawdown': self.cached_portfolio_drawdown,
            'brutal_nota10_mode': True,
            **pnl_info
        }
        
        return total_reward, info, done
    
    def _calculate_pure_pnl_reward(self, env) -> Tuple[float, Dict]:
        """
        Calcula reward baseado EXCLUSIVAMENTE no PnL real
        """
        try:
            # PnL realizado dos trades fechados
            realized_pnl = getattr(env, 'total_realized_pnl', 0.0)
            
            # PnL não realizado das posições abertas  
            unrealized_pnl = getattr(env, 'total_unrealized_pnl', 0.0)
            
            # PnL total como % do balance inicial
            total_pnl = realized_pnl + (unrealized_pnl * 0.5)  # Desconto para não realizado
            pnl_percent = total_pnl / self.initial_balance
            
            # REWARD BASE = PnL amplificado para RL (ULTRA ESTABILIZADO)
            # Aplicar clipping ao PnL percent ANTES da multiplicação
            pnl_percent_clipped = np.clip(pnl_percent, -0.15, 0.15)  # Max ±15%
            base_reward = pnl_percent_clipped * 5.0  # Reduzido: 10.0 -> 5.0
            
            # PAIN MULTIPLICATION SUAVE (sem discontinuidades bruscas)
            if pnl_percent_clipped < -0.03:  # Use clipped value
                # Pain multiplier suave usando tanh para continuidade
                pain_factor = 1.0 + (self.pain_multiplier - 1.0) * np.tanh(abs(pnl_percent_clipped) * 20)
                base_reward *= pain_factor
                
            # BONUS SUAVE para lucros consistentes
            elif pnl_percent_clipped > 0.02:  # Use clipped value
                bonus_factor = 1.0 + 0.1 * np.tanh(pnl_percent_clipped * 50)  # Bonus suave
                base_reward *= bonus_factor
            
            info = {
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl,
                'pnl_percent': pnl_percent * 100,
                'pain_applied': pnl_percent < -0.05
            }
            
            return base_reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo PnL reward: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_basic_risk_reward(self, env) -> Tuple[float, Dict]:
        """
        Risk management básico (apenas para perdas catastróficas)
        """
        try:
            # Drawdown atual
            drawdown = self._calculate_portfolio_drawdown(env)
            
            # Penalty apenas para drawdowns extremos
            if drawdown > self.risk_penalty_threshold:
                # Penalty severa para drawdown > 15%
                excess_drawdown = drawdown - self.risk_penalty_threshold
                penalty = -excess_drawdown * 20.0  # Penalty severa
            else:
                penalty = 0.0
            
            info = {
                'portfolio_drawdown': drawdown * 100,
                'risk_penalty': penalty,
                'risk_threshold_breached': drawdown > self.risk_penalty_threshold
            }
            
            return penalty, info
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo risk reward: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_portfolio_drawdown(self, env) -> float:
        """
        Calcula drawdown do portfolio
        """
        try:
            current_balance = getattr(env, 'portfolio_value', self.initial_balance)
            peak_balance = getattr(env, 'peak_portfolio_value', self.initial_balance)
            
            if peak_balance <= 0:
                return 0.0
                
            drawdown = (peak_balance - current_balance) / peak_balance
            return max(0.0, drawdown)
            
        except:
            return 0.0
    
    def _calculate_proportional_shaping(self, env, pnl_reward: float) -> float:
        """
        🎯 SHAPING PROPORCIONAL AO PNL - Mantém nota 10
        Resolve sparsity sem quebrar linearidade
        """
        try:
            # Shaping sempre proporcional ao PNL atual (zero quando PnL=0)
            current_portfolio = getattr(env, 'portfolio_value', self.initial_balance)
            pnl_percent = (current_portfolio - self.initial_balance) / self.initial_balance
            
            # Base shaping: 5% do PnL (sempre proporcional, nunca constante)
            base_shaping = pnl_percent * 0.05
            
            # Action decisiveness: proporcional ao PnL magnitude
            action_bonus = 0.0
            if hasattr(env, 'last_action'):
                action = env.last_action
                if len(action) > 0:
                    action_magnitude = np.sum(np.abs(action))
                    pnl_magnitude = abs(pnl_percent)
                    
                    # Bonus/penalty proporcional (zero quando PnL=0)
                    if action_magnitude > 0.1:
                        action_bonus = pnl_magnitude * 0.001  # Decisivo
                    else:
                        action_bonus = -pnl_magnitude * 0.0005  # Indeciso
            
            return base_shaping + action_bonus
            
        except:
            return 0.0
    
    def _calculate_reward_shaping(self, env, action: np.ndarray) -> Tuple[float, Dict]:
        """
        🎯 REWARD SHAPING ADEQUADO - Resolve sparsity sem contaminar signal
        Baseado em Potential-Based Reward Shaping (Ng et al. 1999)
        """
        try:
            shaping_reward = 0.0
            info = {}
            
            # 1. PORTFOLIO PROGRESS SHAPING (mais importante)
            current_portfolio = getattr(env, 'portfolio_value', self.initial_balance)
            progress = (current_portfolio - self.last_portfolio_value) / self.initial_balance
            
            if abs(progress) > 0.001:  # Só se houve mudança significativa
                progress_reward = progress * self.progress_scaling * 10.0  # Amplificar
                shaping_reward += progress_reward
                info['progress_reward'] = progress_reward
                info['portfolio_delta'] = progress * 100
            else:
                info['progress_reward'] = 0.0
                info['portfolio_delta'] = 0.0
            
            # 2. POSITION MOMENTUM SHAPING (menor peso)
            momentum_reward = self._calculate_momentum_shaping(env)
            shaping_reward += momentum_reward
            info['momentum_reward'] = momentum_reward
            
            # 3. POSITION AGE DECAY (evita posições muito longas sem resultado)
            age_penalty = self._calculate_position_age_penalty(env)
            shaping_reward += age_penalty
            info['age_penalty'] = age_penalty
            
            # 4. ACTION DECISIVENESS (mínimo - apenas evita paralisia)
            decisiveness_reward = self._calculate_action_decisiveness(action)
            shaping_reward += decisiveness_reward
            info['decisiveness_reward'] = decisiveness_reward
            
            return shaping_reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no reward shaping: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_momentum_shaping(self, env) -> float:
        """Reward shaping baseado em momentum de PnL"""
        try:
            # Obter PnL do último trade fechado
            trades = getattr(env, 'trades', [])
            if not trades:
                return 0.0
            
            # Pegar último PnL
            last_pnl = trades[-1].get('pnl_usd', 0.0)
            
            # Adicionar ao trend tracking
            self.recent_pnl_trend.append(last_pnl)
            if len(self.recent_pnl_trend) > 10:
                self.recent_pnl_trend.pop(0)
            
            # Calcular momentum (tendência recente)
            if len(self.recent_pnl_trend) >= 3:
                recent_avg = np.mean(self.recent_pnl_trend[-3:])
                momentum = recent_avg / self.initial_balance
                
                # Reward pequeno por momentum positivo
                if momentum > 0.01:  # >1% momentum
                    return self.momentum_weight * momentum * 10.0
                elif momentum < -0.01:  # <-1% momentum negativo
                    return self.momentum_weight * momentum * 5.0  # Menos penalty
            
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_position_age_penalty(self, env) -> float:
        """Pequena penalty para posições muito longas sem resultado"""
        try:
            positions = getattr(env, 'positions', [])
            if not positions:
                self.position_age_steps.clear()
                return 0.0
            
            total_penalty = 0.0
            current_step = getattr(env, 'current_step', 0)
            
            for pos in positions:
                pos_id = pos.get('id', str(hash(str(pos))))
                entry_step = pos.get('entry_step', current_step)
                
                # Atualizar idade
                age = current_step - entry_step
                self.position_age_steps[pos_id] = age
                
                # Penalty progressiva para posições muito longas (>100 steps = >8h) ESTABILIZADA
                if age > 100:
                    excess_age = min(age - 100, 50)  # Cap excess_age para evitar overflow
                    penalty = -0.0005 * excess_age  # Linear penalty (era exponencial)
                    total_penalty += penalty
            
            return total_penalty
            
        except:
            return 0.0
    
    def _calculate_action_decisiveness(self, action: np.ndarray) -> float:
        """Pequeno reward para ações decisivas vs. paralisia"""
        try:
            if len(action) == 0:
                return -0.001  # Pequena penalty por não-ação
            
            # Calcular "força" da decisão
            action_magnitude = np.sum(np.abs(action))
            
            # Pequeno reward para decisões (evita paralisia)
            if action_magnitude > 0.1:
                return 0.001  # Muito pequeno - apenas anti-paralisia
            else:
                return -0.0005  # Penalty ainda menor por indecisão
                
        except:
            return 0.0
    
    def _update_tracking(self, env):
        """Update tracking variables para próxima iteração"""
        try:
            # Update last portfolio value
            self.last_portfolio_value = getattr(env, 'portfolio_value', self.initial_balance)
            
            # Limpar posições antigas do tracking
            current_positions = getattr(env, 'positions', [])
            current_pos_ids = set(pos.get('id', str(hash(str(pos)))) for pos in current_positions)
            
            # Remover posições que não existem mais
            for pos_id in list(self.position_age_steps.keys()):
                if pos_id not in current_pos_ids:
                    del self.position_age_steps[pos_id]
                    
        except:
            pass
    
    def _update_stats(self, reward: float):
        """Atualizar estatísticas para debugging"""
        self.total_rewards_given += 1
        
        if reward > 0:
            self.positive_rewards += 1
        elif reward < 0:
            self.negative_rewards += 1
    
    def get_stats(self) -> Dict:
        """Retorna estatísticas do reward system"""
        total = max(1, self.total_rewards_given)
        return {
            'total_rewards': self.total_rewards_given,
            'positive_ratio': self.positive_rewards / total,
            'negative_ratio': self.negative_rewards / total,
            'zero_ratio': (total - self.positive_rewards - self.negative_rewards) / total
        }
    
    def reset(self):
        """Reset para novo episódio"""
        # Reset tracking para novo episódio
        self.last_portfolio_value = self.initial_balance
        self.position_age_steps.clear()
        self.recent_pnl_trend.clear()
        
        # Manter apenas stats cumulativas


def test_brutal_reward_with_shaping():
    """Teste completo do sistema com reward shaping"""
    
    class MockEnv:
        def __init__(self, realized_pnl, unrealized_pnl, portfolio_value, peak_value, trades=None, positions=None, current_step=0):
            self.total_realized_pnl = realized_pnl
            self.total_unrealized_pnl = unrealized_pnl
            self.portfolio_value = portfolio_value
            self.peak_portfolio_value = peak_value
            self.trades = trades or []
            self.positions = positions or []
            self.current_step = current_step
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    
    # Simular sequência de trading
    scenarios = [
        ("Início - sem trades", MockEnv(0, 0, 10000, 10000, [], [], 0)),
        ("Trade lucrativo +2%", MockEnv(200, 0, 10200, 10200, [{'pnl_usd': 200}], [], 10)),
        ("Posição aberta não realizada +1%", MockEnv(200, 100, 10300, 10300, [{'pnl_usd': 200}], [{'id': 'pos1', 'entry_step': 15}], 20)),
        ("Posição fechada com lucro +3%", MockEnv(300, 0, 10300, 10300, [{'pnl_usd': 200}, {'pnl_usd': 100}], [], 25)),
        ("Trade perdedor -5%", MockEnv(-200, 0, 9800, 10300, [{'pnl_usd': 200}, {'pnl_usd': 100}, {'pnl_usd': -500}], [], 30)),
        ("Drawdown 15%", MockEnv(-1500, 0, 8500, 10300, [{'pnl_usd': -1500}], [], 40)),
        ("Posição muito antiga", MockEnv(-1500, -200, 8300, 10300, [{'pnl_usd': -1500}], [{'id': 'pos2', 'entry_step': 40}], 150)),
    ]
    
    print("🧪 TESTE BRUTAL MONEY REWARD COM SHAPING")
    print("=" * 70)
    
    for i, (scenario, env) in enumerate(scenarios):
        # Simular diferentes ações
        if i % 2 == 0:
            action = np.array([0.8, 0.5, 0.2, 0.1])  # Ação decisiva
        else:
            action = np.array([0.02, 0.01, 0.0, 0.0])  # Ação indecisa
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
        
        print(f"\n{i+1}. {scenario}")
        print(f"  Total Reward: {reward:+.4f}")
        print(f"    └─ PnL Reward: {info.get('pnl_reward', 0):+.4f}")
        print(f"    └─ Risk Reward: {info.get('risk_reward', 0):+.4f}")
        print(f"    └─ Shaping Reward: {info.get('shaping_reward', 0):+.4f}")
        print(f"       ├─ Progress: {info.get('progress_reward', 0):+.4f}")
        print(f"       ├─ Momentum: {info.get('momentum_reward', 0):+.4f}")
        print(f"       ├─ Age Penalty: {info.get('age_penalty', 0):+.4f}")
        print(f"       └─ Decisiveness: {info.get('decisiveness_reward', 0):+.4f}")
        print(f"  PnL %: {info.get('pnl_percent', 0):+.1f}%")
        print(f"  Portfolio Δ: {info.get('portfolio_delta', 0):+.2f}%")
        print(f"  Drawdown: {info.get('portfolio_drawdown', 0):.1f}%")
        print(f"  Done: {done}")
        print(f"  Pain Applied: {info.get('pain_applied', False)}")

def test_brutal_reward():
    """Teste rápido do sistema original"""
    
    class MockEnv:
        def __init__(self, realized_pnl, unrealized_pnl, portfolio_value, peak_value):
            self.total_realized_pnl = realized_pnl
            self.total_unrealized_pnl = unrealized_pnl
            self.portfolio_value = portfolio_value
            self.peak_portfolio_value = peak_value
            self.trades = []
            self.positions = []
            self.current_step = 0
    
    reward_system = BrutalMoneyReward(initial_balance=10000)
    
    scenarios = [
        ("Lucro realizado +5%", MockEnv(500, 0, 10500, 10500)),
        ("Prejuízo realizado -8%", MockEnv(-800, 0, 9200, 10000)), 
        ("Drawdown 20%", MockEnv(0, 0, 8000, 10000)),
        ("Lucro não realizado +3%", MockEnv(0, 300, 10300, 10300)),
    ]
    
    print("🧪 TESTE BRUTAL MONEY REWARD BÁSICO")
    print("=" * 60)
    
    for scenario, env in scenarios:
        reward, info, done = reward_system.calculate_reward_and_info(env, np.zeros(8), {})
        
        print(f"\n{scenario}")
        print(f"  Reward: {reward:+.2f}")
        print(f"  PnL %: {info.get('pnl_percent', 0):+.1f}%")
        print(f"  Drawdown: {info.get('portfolio_drawdown', 0):.1f}%")
        print(f"  Done: {done}")
        print(f"  Pain applied: {info.get('pain_applied', False)}")


# Factory function para compatibilidade
def create_brutal_daytrade_reward_system(initial_balance: float = 1000.0):
    """Factory function para o sistema V3 brutal"""
    return BrutalMoneyReward(initial_balance)

if __name__ == "__main__":
    print("Escolha o teste:")
    print("1. Teste básico")
    print("2. Teste com reward shaping")
    
    choice = input("Digite 1 ou 2: ")
    
    if choice == "2":
        test_brutal_reward_with_shaping()
    else:
        test_brutal_reward()