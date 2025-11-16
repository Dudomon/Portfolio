"""
🚀 INNOVATIVE MONEY-FOCUSED REWARD SYSTEM V4
Sistema de reward otimizado que REALMENTE ensina a fazer dinheiro
Baseado no V3 Brutal com melhorias para performance trading

🎯 FILOSOFIA V4 INNO:
- PnL = 70% do reward (dominante)
- Shaping inteligente = 20% do reward  
- Trade activity bonus = 10% do reward
- Perdas grandes = DOR MULTIPLICADA
- Unrealized PnL valorizado (0.8x)
- Action decisiveness amplificado
- Position performance reward (não age penalty)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging

class InnovativeMoneyReward:
    """
    Sistema de reward focado 100% em fazer dinheiro
    COM REWARD SHAPING INTELIGENTE para resolver sparsity
    E OTIMIZAÇÕES para aumentar atividade de trading
    """
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.logger = logging.getLogger(__name__)
        
        # Configurações V4 INNO (OTIMIZADAS)
        self.pain_multiplier = 1.5      # Perdas doem 1.5x mais
        self.risk_penalty_threshold = 0.15  # Drawdown > 15% = penalty severa
        self.max_reward = 2.0           # Cap aumentado para reduzir compressão tanh
        
        # 🚀 V4 INNO PARAMETERS (OTIMIZADOS PARA PERFORMANCE)
        self.progress_scaling = 0.1     # Aumentado: 0.05 -> 0.1
        self.momentum_weight = 0.05     # Aumentado: 0.02 -> 0.05
        self.position_performance_weight = 0.02  # Novo: reward por performance
        self.action_decisiveness_weight = 0.05   # Aumentado: 50x
        
        # Tracking para shaping
        self.last_portfolio_value = initial_balance
        self.position_performance = {}  # position_id -> performance_tracking
        self.recent_pnl_trend = []      # Últimos 10 PnLs para momentum
        
        # Performance cache
        self.step_counter = 0
        self.cached_risk_reward = 0.0
        self.cached_shaping_reward = 0.0
        self.cached_portfolio_drawdown = 0.0
        self.cached_activity_bonus = 0.0
        
        # Métricas para debugging
        self.total_rewards_given = 0
        self.positive_rewards = 0
        self.negative_rewards = 0
        
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        Calcula reward V4 INNO:
        70% PnL + 20% Shaping + 10% Activity
        
        Returns:
            reward: Valor otimizado para performance
            info: Informações de debug
            done: Se deve terminar episódio (perdas catastróficas)
        """
        
        self.step_counter += 1
        
        # 1. EXTRAIR PNL REAL (70% do reward - OTIMIZADO)
        pnl_reward, pnl_info = self._calculate_pure_pnl_reward_v4(env)
        
        # 2. REWARD SHAPING INTELIGENTE (20% do reward) - CACHED A CADA 5 STEPS
        if self.step_counter % 5 == 0:
            self.cached_shaping_reward, shaping_info = self._calculate_intelligent_shaping(env, action)
        else:
            shaping_info = {'cached': True}
        shaping_reward = self.cached_shaping_reward
        
        # 3. TRADE ACTIVITY BONUS (10% do reward) - SEMPRE CALCULADO para teste
        self.cached_activity_bonus, activity_info = self._calculate_activity_bonus(env)
        activity_bonus = self.cached_activity_bonus
        
        # 4. 🚀 REWARD FINAL V4 INNO = 70% PnL + 20% Shaping + 10% Activity
        pure_pnl_component = pnl_reward * 0.70
        shaping_component = shaping_reward * 0.20
        activity_component = activity_bonus * 0.10
        total_reward = pure_pnl_component + shaping_component + activity_component
        
        # 5. EARLY TERMINATION DESABILITADO - Permitir episódios completos
        # if self.step_counter % 20 == 0:
        #     self.cached_portfolio_drawdown = self._calculate_portfolio_drawdown(env)
        # done = self.cached_portfolio_drawdown > 0.5  # DESABILITADO: Terminava se perder >50%
        done = False  # 🚀 NUNCA terminar episódio por reward system
        
        # 6. NORMALIZAÇÃO LINEAR para PPO (remover compressão tanh)
        total_reward = np.clip(total_reward, -self.max_reward, self.max_reward)
        
        # Safeguard numérico robusto
        if not np.isfinite(total_reward) or abs(total_reward) > 100:
            total_reward = 0.0
        
        # 7. UPDATE TRACKING para próxima iteração - REDUCED FREQUENCY
        if self.step_counter % 5 == 0:
            self._update_tracking_v4(env)
        
        # 8. LOGGING - REDUCED FREQUENCY  
        if self.step_counter % 10 == 0:
            self._update_stats(total_reward)
        
        # 9. INFO para debugging V4
        info = {
            'pnl_reward': pnl_reward,
            'shaping_reward': shaping_reward,
            'activity_bonus': activity_bonus,
            'total_reward': total_reward,
            'pure_pnl_component': pure_pnl_component,
            'shaping_component': shaping_component,
            'activity_component': activity_component,
            'portfolio_drawdown': self.cached_portfolio_drawdown,
            'v4_inno_mode': True,
            **pnl_info,
            **shaping_info,
            **activity_info
        }
        
        return total_reward, info, done
    
    def _calculate_pure_pnl_reward_v4(self, env) -> Tuple[float, Dict]:
        """
        V4 INNO: PnL reward otimizado com unrealized PnL valorizado
        """
        try:
            # PnL realizado dos trades fechados
            realized_pnl = getattr(env, 'total_realized_pnl', 0.0)
            
            # PnL não realizado das posições abertas (VALORIZADO V4)
            unrealized_pnl = getattr(env, 'total_unrealized_pnl', 0.0)
            
            # 🚀 V4 INNO: Valorizar unrealized PnL (0.5 -> 0.8)
            total_pnl = realized_pnl + (unrealized_pnl * 0.8)
            pnl_percent = total_pnl / self.initial_balance
            
            # 🚀 V4 INNO: REWARD BASE amplificado e balanceado (5.0 -> 12.0)
            pnl_percent_clipped = np.clip(pnl_percent, -0.15, 0.15)  # Max ±15%
            base_reward = pnl_percent_clipped * 12.0  # 2.4x mais forte, balanceado
            
            # PAIN MULTIPLICATION SUAVE (sem discontinuidades bruscas)
            if pnl_percent_clipped < -0.03:
                # Pain multiplier suave usando tanh para continuidade
                pain_factor = 1.0 + (self.pain_multiplier - 1.0) * np.tanh(abs(pnl_percent_clipped) * 20)
                base_reward *= pain_factor
                
            # BONUS SUAVE para lucros consistentes
            elif pnl_percent_clipped > 0.02:
                bonus_factor = 1.0 + 0.2 * np.tanh(pnl_percent_clipped * 50)  # Bonus aumentado
                base_reward *= bonus_factor
            
            info = {
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl,
                'pnl_percent': pnl_percent * 100,
                'pain_applied': pnl_percent < -0.05,
                'v4_unrealized_factor': 0.8,
                'v4_pnl_multiplier': 12.0
            }
            
            return base_reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no cálculo PnL reward V4: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_intelligent_shaping(self, env, action: np.ndarray) -> Tuple[float, Dict]:
        """
        🎯 V4 INNO: SHAPING INTELIGENTE - Resolve sparsity + aumenta atividade
        """
        try:
            shaping_reward = 0.0
            info = {}
            
            # 1. PORTFOLIO PROGRESS SHAPING (mais importante - AMPLIFICADO)
            current_portfolio = getattr(env, 'portfolio_value', self.initial_balance)
            progress = (current_portfolio - self.last_portfolio_value) / self.initial_balance
            
            if abs(progress) > 0.001:  # Só se houve mudança significativa
                progress_reward = progress * self.progress_scaling * 20.0  # 2x amplificado
                shaping_reward += progress_reward
                info['progress_reward'] = progress_reward
                info['portfolio_delta'] = progress * 100
            else:
                info['progress_reward'] = 0.0
                info['portfolio_delta'] = 0.0
            
            # 2. POSITION MOMENTUM SHAPING (peso aumentado)
            momentum_reward = self._calculate_momentum_shaping_v4(env)
            shaping_reward += momentum_reward
            info['momentum_reward'] = momentum_reward
            
            # 3. 🚀 V4 INNO: POSITION PERFORMANCE REWARD (substitui age penalty)
            performance_reward = self._calculate_position_performance_reward(env)
            shaping_reward += performance_reward
            info['performance_reward'] = performance_reward
            
            # 4. ACTION DECISIVENESS AMPLIFICADO (50x mais forte)
            decisiveness_reward = self._calculate_action_decisiveness_v4(action, env)
            shaping_reward += decisiveness_reward
            info['decisiveness_reward'] = decisiveness_reward
            
            return shaping_reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no shaping inteligente V4: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_position_performance_reward(self, env) -> float:
        """
        🚀 V4 INNO: Reward baseado em PERFORMANCE, não em idade
        """
        try:
            positions = getattr(env, 'positions', [])
            if not positions:
                self.position_performance.clear()
                return 0.0
            
            total_reward = 0.0
            
            for pos in positions:
                pos_id = pos.get('id', str(hash(str(pos))))
                unrealized_pnl = pos.get('unrealized_pnl', 0.0)
                pnl_percent = unrealized_pnl / self.initial_balance
                
                # Reward por posições lucrativas
                if pnl_percent > 0.01:  # >1% profit
                    performance_bonus = self.position_performance_weight * pnl_percent * 50.0
                    total_reward += performance_bonus
                elif pnl_percent < -0.02:  # <-2% loss
                    performance_penalty = -self.position_performance_weight * abs(pnl_percent) * 25.0
                    total_reward += performance_penalty
                
                # Track performance
                self.position_performance[pos_id] = pnl_percent
            
            return total_reward
            
        except:
            return 0.0
    
    def _calculate_momentum_shaping_v4(self, env) -> float:
        """V4 INNO: Momentum reward amplificado"""
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
                
                # 🚀 V4 INNO: Momentum reward amplificado
                if momentum > 0.01:  # >1% momentum
                    return self.momentum_weight * momentum * 25.0  # 2.5x amplificado
                elif momentum < -0.01:  # <-1% momentum negativo
                    return self.momentum_weight * momentum * 10.0  # 2x amplificado
            
            return 0.0
            
        except:
            return 0.0
    
    def _calculate_action_decisiveness_v4(self, action: np.ndarray, env) -> float:
        """🚀 V4 INNO: Action decisiveness AMPLIFICADO (50x)"""
        try:
            if len(action) == 0:
                return -0.02  # Penalty aumentada por não-ação
            
            # Calcular "força" da decisão
            action_magnitude = np.sum(np.abs(action))
            
            # 🚀 V4 INNO: Action bonus AMPLIFICADO
            if action_magnitude > 0.1:
                # Bonus proporcional ao PnL atual
                current_portfolio = getattr(env, 'portfolio_value', self.initial_balance)
                pnl_percent = (current_portfolio - self.initial_balance) / self.initial_balance
                pnl_magnitude = abs(pnl_percent)
                
                # Amplificado: 0.001 -> 0.05 (50x)
                return self.action_decisiveness_weight * (1.0 + pnl_magnitude * 10.0)
            else:
                # Penalty amplificada por indecisão: 0.0005 -> 0.02 (40x)
                return -self.action_decisiveness_weight * 0.4
                
        except:
            return 0.0
    
    def _calculate_activity_bonus(self, env) -> Tuple[float, Dict]:
        """
        🚀 V4 INNO: Bonus por atividade de trading saudável
        """
        try:
            trades = getattr(env, 'trades', [])
            trades_count = len(trades)
            
            # Estimar trades por episódio (assumindo 500 steps = 1 episódio)
            current_step = getattr(env, 'current_step', self.step_counter)
            if current_step > 0:
                estimated_episodes = max(1, current_step / 500.0)
                trades_per_episode = trades_count / estimated_episodes
            else:
                trades_per_episode = 0
            
            # 🚀 V4 INNO: Target 3-12 trades por episódio
            if 3 <= trades_per_episode <= 12:
                bonus = 0.08  # Bonus significativo para range ideal
                status = 'optimal'
            elif trades_per_episode < 3:
                # Penalty por poucos trades (aumenta atividade)
                shortage = 3 - trades_per_episode
                bonus = -0.04 * shortage  # Penalty progressiva
                status = 'too_low'
            else:
                # Penalty leve por overtrading
                excess = trades_per_episode - 12
                bonus = -0.02 * excess
                status = 'too_high'
            
            # Bonus adicional por consistência
            if len(trades) >= 2:
                # Calcular volatilidade dos PnLs
                pnls = [trade.get('pnl_usd', 0) for trade in trades[-5:]]  # Últimos 5 trades
                if len(pnls) >= 3:
                    pnl_std = np.std(pnls)
                    avg_pnl = np.mean(pnls)
                    if avg_pnl > 0 and pnl_std < abs(avg_pnl) * 2:  # Baixa volatilidade com lucro médio
                        bonus += 0.02  # Bonus por consistência
                        status += '_consistent'
            
            info = {
                'trades_count': trades_count,
                'trades_per_episode': trades_per_episode,
                'activity_bonus': bonus,
                'activity_status': status,
                'target_range': '3-12 trades/episode'
            }
            
            return bonus, info
            
        except Exception as e:
            self.logger.error(f"Erro no activity bonus: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_portfolio_drawdown(self, env) -> float:
        """
        Calcula drawdown do portfolio (mantido do V3)
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
    
    def _update_tracking_v4(self, env):
        """V4 INNO: Update tracking variables otimizado"""
        try:
            # Update last portfolio value
            self.last_portfolio_value = getattr(env, 'portfolio_value', self.initial_balance)
            
            # Limpar posições antigas do tracking
            current_positions = getattr(env, 'positions', [])
            current_pos_ids = set(pos.get('id', str(hash(str(pos)))) for pos in current_positions)
            
            # Remover posições que não existem mais
            for pos_id in list(self.position_performance.keys()):
                if pos_id not in current_pos_ids:
                    del self.position_performance[pos_id]
                    
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
        """Retorna estatísticas do reward system V4"""
        total = max(1, self.total_rewards_given)
        return {
            'total_rewards': self.total_rewards_given,
            'positive_ratio': self.positive_rewards / total,
            'negative_ratio': self.negative_rewards / total,
            'zero_ratio': (total - self.positive_rewards - self.negative_rewards) / total,
            'version': 'V4_INNO'
        }
    
    def reset(self):
        """Reset para novo episódio"""
        # Reset tracking para novo episódio
        self.last_portfolio_value = self.initial_balance
        self.position_performance.clear()
        self.recent_pnl_trend.clear()
        
        # Manter apenas stats cumulativas


def test_reward_balance():
    """🧪 TESTE DE BALANCE - V3 vs V4 COMPARAÇÃO"""
    
    class MockEnv:
        def __init__(self, realized_pnl, unrealized_pnl, portfolio_value, peak_value, trades=None, positions=None, current_step=0):
            self.total_realized_pnl = realized_pnl
            self.total_unrealized_pnl = unrealized_pnl
            self.portfolio_value = portfolio_value
            self.peak_portfolio_value = peak_value
            self.trades = trades or []
            self.positions = positions or []
            self.current_step = current_step
    
    # Importar V3 para comparação
    import sys
    sys.path.append('D:/Projeto/trading_framework/rewards/')
    from reward_daytrade_v3_brutal import BrutalMoneyReward
    
    reward_v3 = BrutalMoneyReward(initial_balance=10000)
    reward_v4 = InnovativeMoneyReward(initial_balance=10000)
    
    # Cenários críticos para testar balance
    test_scenarios = [
        ("Neutro - sem PnL", MockEnv(0, 0, 10000, 10000, [], [], 100)),
        ("Pequeno lucro +1%", MockEnv(100, 0, 10100, 10100, [{'pnl_usd': 100}], [], 200)),
        ("Lucro médio +3%", MockEnv(300, 0, 10300, 10300, [{'pnl_usd': 300}], [], 300)),
        ("Grande lucro +8%", MockEnv(800, 0, 10800, 10800, [{'pnl_usd': 800}], [], 400)),
        ("Pequena perda -2%", MockEnv(-200, 0, 9800, 10000, [{'pnl_usd': -200}], [], 500)),
        ("Perda média -5%", MockEnv(-500, 0, 9500, 10000, [{'pnl_usd': -500}], [], 600)),
        ("Grande perda -10%", MockEnv(-1000, 0, 9000, 10000, [{'pnl_usd': -1000}], [], 700)),
        ("Posição aberta lucrativa +2%", MockEnv(0, 200, 10200, 10200, [], [{'id': 'pos1', 'unrealized_pnl': 200}], 800)),
        ("Posição aberta perdedora -3%", MockEnv(0, -300, 9700, 10000, [], [{'id': 'pos2', 'unrealized_pnl': -300}], 900)),
    ]
    
    print("🧪 TESTE DE BALANCE V3 BRUTAL vs V4 INNO")
    print("=" * 80)
    print(f"{'CENÁRIO':<25} {'V3 REWARD':<12} {'V4 REWARD':<12} {'DIFERENÇA':<12} {'RATIO':<8}")
    print("-" * 80)
    
    action_decisive = np.array([0.8, 0.6, 0.4, 0.2])  # Ação decisiva padrão
    
    for scenario, env in test_scenarios:
        # Testar V3
        reward_v3_val, info_v3, _ = reward_v3.calculate_reward_and_info(env, action_decisive, {})
        
        # Testar V4
        reward_v4_val, info_v4, _ = reward_v4.calculate_reward_and_info(env, action_decisive, {})
        
        # Calcular diferença
        diff = reward_v4_val - reward_v3_val
        ratio = reward_v4_val / reward_v3_val if reward_v3_val != 0 else float('inf') if reward_v4_val != 0 else 1.0
        
        print(f"{scenario:<25} {reward_v3_val:+8.4f}    {reward_v4_val:+8.4f}    {diff:+8.4f}    {ratio:6.2f}x")
    
    print("\n" + "=" * 80)
    print("📊 ANÁLISE DETALHADA - CASOS ESPECÍFICOS")
    print("=" * 80)
    
    # Teste detalhado em casos críticos
    critical_cases = [
        ("Lucro pequeno +1%", MockEnv(100, 0, 10100, 10100, [{'pnl_usd': 100}], [], 100)),
        ("Unrealized +2%", MockEnv(0, 200, 10200, 10200, [], [{'id': 'pos1', 'unrealized_pnl': 200}], 200)),
        ("Perda -5%", MockEnv(-500, 0, 9500, 10000, [{'pnl_usd': -500}], [], 300)),
    ]
    
    for case_name, env in critical_cases:
        print(f"\n🔍 {case_name}")
        print("-" * 40)
        
        # V3 Detalhado
        reward_v3_val, info_v3, _ = reward_v3.calculate_reward_and_info(env, action_decisive, {})
        print(f"V3 Brutal:")
        print(f"  Total: {reward_v3_val:+.4f}")
        print(f"  PnL: {info_v3.get('pnl_reward', 0):+.4f}")
        print(f"  Shaping: {info_v3.get('proportional_shaping', 0):+.4f}")
        print(f"  PnL %: {info_v3.get('pnl_percent', 0):+.1f}%")
        
        # V4 Detalhado
        reward_v4_val, info_v4, _ = reward_v4.calculate_reward_and_info(env, action_decisive, {})
        print(f"V4 Inno:")
        print(f"  Total: {reward_v4_val:+.4f}")
        print(f"  PnL Component (80%): {info_v4.get('pure_pnl_component', 0):+.4f}")
        print(f"  Shaping Component (20%): {info_v4.get('shaping_component', 0):+.4f}")
        print(f"  Raw PnL Reward: {info_v4.get('pnl_reward', 0):+.4f}")
        print(f"  PnL %: {info_v4.get('pnl_percent', 0):+.1f}%")
        print(f"  Unrealized Factor: {info_v4.get('v4_unrealized_factor', 'N/A')}")
        print(f"  PnL Multiplier: {info_v4.get('v4_pnl_multiplier', 'N/A')}")
        
        # Comparação
        improvement = ((reward_v4_val - reward_v3_val) / abs(reward_v3_val) * 100) if reward_v3_val != 0 else 0
        print(f"📈 Melhoria V4: {improvement:+.1f}%")


def test_innovative_reward_v4():
    """Teste básico do sistema V4 INNO"""
    
    class MockEnv:
        def __init__(self, realized_pnl, unrealized_pnl, portfolio_value, peak_value, trades=None, positions=None, current_step=0):
            self.total_realized_pnl = realized_pnl
            self.total_unrealized_pnl = unrealized_pnl
            self.portfolio_value = portfolio_value
            self.peak_portfolio_value = peak_value
            self.trades = trades or []
            self.positions = positions or []
            self.current_step = current_step
    
    reward_system = InnovativeMoneyReward(initial_balance=10000)
    
    # Simular sequência de trading V4
    scenarios = [
        ("Início - sem trades", MockEnv(0, 0, 10000, 10000, [], [], 0)),
        ("Poucos trades (2)", MockEnv(200, 0, 10200, 10200, [{'pnl_usd': 100}, {'pnl_usd': 100}], [], 250)),  # 2 trades em 250 steps = 4 trades/episódio
        ("Atividade ótima (6 trades)", MockEnv(500, 0, 10500, 10500, [{'pnl_usd': 80} for _ in range(6)], [], 500)),  # 6 trades em 500 steps = 6 trades/episódio
        ("Overtrading (20 trades)", MockEnv(400, 0, 10400, 10400, [{'pnl_usd': 20} for _ in range(20)], [], 500)),  # 20 trades em 500 steps = 20 trades/episódio
        ("Trades consistentes", MockEnv(400, 0, 10400, 10400, [{'pnl_usd': 80}, {'pnl_usd': 85}, {'pnl_usd': 75}, {'pnl_usd': 90}, {'pnl_usd': 70}], [], 500)),  # Trades consistentes
    ]
    
    print("🚀 TESTE INNOVATIVE MONEY REWARD V4")
    print("=" * 70)
    
    for i, (scenario, env) in enumerate(scenarios):
        # Simular diferentes ações
        if i % 3 == 0:
            action = np.array([0.9, 0.7, 0.5, 0.3])  # Ação muito decisiva
        elif i % 3 == 1:
            action = np.array([0.3, 0.2, 0.1, 0.1])  # Ação moderada
        else:
            action = np.array([0.01, 0.005, 0.0, 0.0])  # Ação indecisa
        
        reward, info, done = reward_system.calculate_reward_and_info(env, action, {})
        
        print(f"\n{i+1}. {scenario}")
        print(f"  🎯 Total Reward: {reward:+.4f}")
        print(f"    ├─ PnL Component (70%): {info.get('pure_pnl_component', 0):+.4f}")
        print(f"    ├─ Shaping Component (20%): {info.get('shaping_component', 0):+.4f}")
        print(f"    └─ Activity Component (10%): {info.get('activity_component', 0):+.4f}")
        print(f"  📈 PnL Details:")
        print(f"    ├─ PnL %: {info.get('pnl_percent', 0):+.1f}%")
        print(f"    ├─ Unrealized Factor: {info.get('v4_unrealized_factor', 0)}")
        print(f"    └─ PnL Multiplier: {info.get('v4_pnl_multiplier', 0)}")
        print(f"  🎭 Shaping Details:")
        print(f"    ├─ Progress: {info.get('progress_reward', 0):+.4f}")
        print(f"    ├─ Momentum: {info.get('momentum_reward', 0):+.4f}")
        print(f"    ├─ Performance: {info.get('performance_reward', 0):+.4f}")
        print(f"    └─ Decisiveness: {info.get('decisiveness_reward', 0):+.4f}")
        print(f"  📊 Activity Details:")
        print(f"    ├─ Trades/Episode: {info.get('trades_per_episode', 0):.1f}")
        print(f"    ├─ Status: {info.get('activity_status', 'N/A')}")
        print(f"    └─ Target: {info.get('target_range', 'N/A')}")
        print(f"  ⚠️ Risk: Drawdown {info.get('portfolio_drawdown', 0):.1f}% | Done: {done}")


# Factory function para compatibilidade
def create_innovative_daytrade_reward_system(initial_balance: float = 1000.0):
    """Factory function para o sistema V4 INNO"""
    return InnovativeMoneyReward(initial_balance)

if __name__ == "__main__":
    print("Escolha o teste:")
    print("1. Teste básico V4")
    print("2. Teste de balance V3 vs V4")
    
    choice = input("Digite 1 ou 2: ")
    
    if choice == "2":
        test_reward_balance()
    else:
        test_innovative_reward_v4()