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

# 🎯 SEVENTEEN: Import Entry Timing Rewards
from .entry_timing_rewards import EntryTimingRewards

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
        
        # 🚀 CACHE AGRESSIVO para funções custosas
        self.cached_trailing_reward = 0.0
        self.cached_sltp_reward = 0.0

        # 📊 EXPONENTIAL SMOOTHING para estabilizar explained_variance
        self.smoothing_alpha = 1.0      # DESABILITADO: estava destruindo variabilidade (0.1=muito suave, 0.5=responsivo)
        self.smooth_reward = 0.0        # Reward suavizado acumulado
        self.smoothing_initialized = False  # Flag para primeira execução
        self.last_trades_hash = None  # Hash dos trades para detectar mudanças

        # 🎯 NOVO: Tracking de estado anterior para SL/TP improvement rewards
        self.previous_sltp_state = {}  # position_id -> {'sl': float, 'tp': float, 'rr_ratio': float}
        self.training_progress = 0.0   # Progresso do treinamento (0.0 a 1.0)

        # 🎯 TRACKING PARA TP HIT / SL NEAR-MISS REWARDS
        self.previous_trades_count = 0  # Número de trades fechados no step anterior
        self.last_closed_trades = []    # Trades fechados recentemente

        # Métricas para debugging
        self.total_rewards_given = 0
        self.positive_rewards = 0
        self.negative_rewards = 0

        # 🎯 SEVENTEEN: Entry Timing Rewards System
        self.entry_timing_system = EntryTimingRewards()
        
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
        
        # 2. RISK MANAGEMENT BÁSICO (10% do reward) - CACHED A CADA 2 STEPS
        if self.step_counter % 2 == 0:
            self.cached_risk_reward, risk_info = self._calculate_basic_risk_reward(env)
        else:
            risk_info = {'cached': True}
        risk_reward = self.cached_risk_reward
        
        # 3. 🎯 REWARD SHAPING ADEQUADO (5% do reward) - CACHED A CADA 5 STEPS
        if self.step_counter % 5 == 0:
            self.cached_shaping_reward, shaping_info = self._calculate_reward_shaping(env, action)
        else:
            shaping_info = {'cached': True}
        shaping_reward = self.cached_shaping_reward
        
        # 4. REWARD FINAL = PnL PURO (85%) + shaping proporcional (15%)
        # DISTRIBUIÇÃO MATEMÁTICA EXATA: Garantir 85%/15% split
        
        # Calcular raw shaping
        raw_shaping = self._calculate_proportional_shaping(env, pnl_reward)
        
        if abs(pnl_reward) > 1e-8:
            # 🔥 DISTRIBUIÇÃO ATUALIZADA: 70% PnL / 30% Shaping
            # Aumenta peso do SL/TP management para forçar aprendizado de gestão

            # Step 1: PnL component será 70% de si mesmo
            pure_pnl_component = pnl_reward * 0.70

            # Step 2: Shaping deve ter magnitude que resulte em 30% do total
            # Mas queremos preservar a direção do raw_shaping
            shaping_direction = 1.0 if raw_shaping >= 0 else -1.0
            if raw_shaping == 0:
                shaping_direction = 1.0  # Default to positive if shaping is zero

            # Shaping component deve ser 30% da magnitude do PnL reward original
            shaping_component = shaping_direction * abs(pnl_reward) * 0.30

            # Para debugging, calcular o scaled_shaping baseado no peso aplicado
            scaled_shaping = shaping_component / 0.30 if 0.30 != 0 else shaping_component

        else:
            # Edge case: PnL é zero
            pure_pnl_component = 0.0
            shaping_component = raw_shaping * 0.30  # Small shaping contribution
            scaled_shaping = raw_shaping
        
        total_reward = pure_pnl_component + shaping_component
        
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
        if self.step_counter % 2 == 0:
            self._update_tracking(env)
        
        # 8. LOGGING - REDUCED FREQUENCY
        if self.step_counter % 5 == 0:
            self._update_stats(total_reward)
            
        # Debug logging DESABILITADO para máxima performance
        # if self.step_counter % 100 == 0 and abs(total_reward) > 0.01:
        #     pnl_pct = (pure_pnl_component / (abs(total_reward) + 1e-6)) * 100
        #     shaping_pct = (abs(shaping_component) / (abs(total_reward) + 1e-6)) * 100
        #     print(f"[V3-BRUTAL] PnL: {pnl_pct:.1f}%, Shaping: {shaping_pct:.1f}% (includes SL/TP+Trailing) | Total: {total_reward:.4f}")
        
        # 9. 📊 EXPONENTIAL SMOOTHING para estabilizar explained_variance
        raw_reward = total_reward  # Salvar reward original para info

        if not self.smoothing_initialized:
            # Primeira execução: inicializar com o valor atual
            self.smooth_reward = total_reward
            self.smoothing_initialized = True
            smoothed_reward = total_reward
        else:
            # Aplicar smoothing exponencial: reward_t = α * raw_t + (1-α) * smooth_{t-1}
            smoothed_reward = (self.smoothing_alpha * total_reward +
                             (1 - self.smoothing_alpha) * self.smooth_reward)
            self.smooth_reward = smoothed_reward

        # 10. INFO para debugging (incluindo métricas de smoothing)
        info = {
            'pnl_reward': pnl_reward,
            'risk_reward': 0.0,  # Eliminado para nota 10
            'shaping_reward': scaled_shaping,
            'total_reward': smoothed_reward,    # Reward final suavizado
            'raw_reward': raw_reward,           # Reward original (antes do smoothing)
            'smoothing_applied': smoothed_reward != raw_reward,
            'smoothing_alpha': self.smoothing_alpha,
            'pure_pnl_component': pure_pnl_component,
            'shaping_component': shaping_component,
            'proportional_shaping': scaled_shaping,
            'raw_shaping': raw_shaping,  # Original shaping before scaling
            'portfolio_drawdown': self.cached_portfolio_drawdown,
            'brutal_nota10_mode': True,
            'eighteen_entry_timing_v2_enabled': True,  # 🎯 EIGHTEEN FLAG
            **pnl_info,
            **shaping_info  # Adicionar info detalhado do shaping (inclui entry timing)
        }

        return smoothed_reward, info, done
    
    def _calculate_pure_pnl_reward(self, env) -> Tuple[float, Dict]:
        """
        Calcula reward baseado EXCLUSIVAMENTE no PnL real
        """
        try:
            # PnL realizado dos trades fechados
            realized_pnl = getattr(env, 'total_realized_pnl', 0.0)

            # PnL não realizado das posições abertas
            unrealized_pnl = getattr(env, 'total_unrealized_pnl', 0.0)

            # 🔥 FIX SHORT BIAS: Remover desconto de 50% que favorecia SHORTs rápidos
            # PnL total SEM desconto - tratar unrealized com mesmo peso que realized
            total_pnl = realized_pnl + unrealized_pnl  # SEM DESCONTO!
            pnl_percent = total_pnl / self.initial_balance
            
            # REWARD BASE = PnL amplificado para RL (ULTRA ESTABILIZADO)
            # Aplicar clipping ao PnL percent ANTES da multiplicação
            pnl_percent_clipped = np.clip(pnl_percent, -0.15, 0.15)  # Max ±15%
            base_reward = pnl_percent_clipped * 5.0  # Reduzido: 10.0 -> 5.0
            
            # 🔥 FIX SHORT BIAS: Pain multiplier SIMÉTRICO
            # Amplificar tanto perdas quanto ganhos para não favorecer estratégias defensivas
            if abs(pnl_percent_clipped) > 0.02:  # Qualquer PnL significativo (±2%)
                if pnl_percent_clipped < 0:
                    # PAIN para perdas: amplifica 1.5x
                    pain_factor = 1.0 + (self.pain_multiplier - 1.0) * np.tanh(abs(pnl_percent_clipped) * 20)
                    base_reward *= pain_factor
                else:
                    # BONUS para ganhos: amplifica proporcionalmente
                    # Se pain = 1.5x para perdas, ganhos devem ter boost equivalente
                    bonus_factor = 1.0 + (self.pain_multiplier - 1.0) * np.tanh(pnl_percent_clipped * 20)
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
            
            # Base shaping: escalado para ser mais significativo
            base_shaping = pnl_percent * 0.5  # Aumentado de 0.05 para 0.5
            
            # Action decisiveness: proporcional ao PnL magnitude
            action_bonus = 0.0
            if hasattr(env, 'last_action'):
                action = env.last_action
                if len(action) > 0:
                    action_magnitude = np.sum(np.abs(action))
                    pnl_magnitude = abs(pnl_percent)
                    
                    # Bonus/penalty proporcional escalado (zero quando PnL=0)
                    if action_magnitude > 0.1:
                        action_bonus = pnl_magnitude * 0.1  # Decisivo - escalado de 0.001 para 0.1
                    else:
                        action_bonus = -pnl_magnitude * 0.05  # Indeciso - escalado de 0.0005 para 0.05
            
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
            
            # 5. 🎯 TRAILING STOP REWARDS - CACHED A CADA 25 STEPS (PERFORMANCE)
            if self.step_counter % 25 == 0:
                self.cached_trailing_reward = self._calculate_trailing_stop_rewards(env)
            trailing_reward = self.cached_trailing_reward
            shaping_reward += trailing_reward
            info['trailing_reward'] = trailing_reward
            
            # 6. 🎯 DYNAMIC SL/TP ADJUSTMENT REWARDS - CACHED A CADA 25 STEPS (PERFORMANCE)
            if self.step_counter % 25 == 0:
                self.cached_sltp_reward = self._calculate_dynamic_sltp_rewards(env)
            sltp_reward = self.cached_sltp_reward
            shaping_reward += sltp_reward
            info['sltp_reward'] = sltp_reward

            # 7. 🚨 ANTI-GAMING: Penalidade por SL mínimo + TP máximo
            if self.step_counter % 25 == 0:
                self.cached_gaming_penalty = self._calculate_sltp_gaming_penalty(env)
            gaming_penalty = getattr(self, 'cached_gaming_penalty', 0.0)
            shaping_reward += gaming_penalty
            info['sltp_gaming_penalty'] = gaming_penalty

            # 8. 🎯 TP REALISM: Bonificar TP que mira em zonas realistas
            if self.step_counter % 25 == 0:
                self.cached_tp_realism = self._calculate_tp_realism_bonus(env)
            tp_realism = getattr(self, 'cached_tp_realism', 0.0)
            shaping_reward += tp_realism
            info['tp_realism_bonus'] = tp_realism

            # 9. 🎯 TP HIT EXPERT: Reward MASSIVO por TP hit (SEMPRE calculado)
            tp_hit_reward = self._calculate_tp_hit_expert_reward(env)
            shaping_reward += tp_hit_reward
            info['tp_hit_expert_reward'] = tp_hit_reward

            # 10. 🎯 SL NEAR-MISS: Reward por SL bem posicionado (CACHED a cada 10 steps)
            if self.step_counter % 10 == 0:
                self.cached_sl_near_miss = self._calculate_sl_near_miss_reward(env)
            sl_near_miss = getattr(self, 'cached_sl_near_miss', 0.0)
            shaping_reward += sl_near_miss
            info['sl_near_miss_reward'] = sl_near_miss

            # 11. 🎯 TRAILING TIMING: Reward por trailing no momento certo (SEMPRE calculado)
            trailing_timing = self._calculate_trailing_timing_reward(env)
            shaping_reward += trailing_timing
            info['trailing_timing_reward'] = trailing_timing

            # 12. 🎯 TREND FOLLOWING: Reward por seguir a tendência (CACHED a cada 5 steps)
            if self.step_counter % 5 == 0:
                self.cached_trend_following = self._calculate_trend_following_reward(env)
            trend_following = getattr(self, 'cached_trend_following', 0.0)
            shaping_reward += trend_following
            info['trend_following_reward'] = trend_following

            # 13. 🎯 EIGHTEEN: ENTRY TIMING REWARDS V2 (PESO DOBRADO: 6% → 12%)
            entry_decision = self._extract_entry_decision(action)
            entry_timing_reward = 0.0
            if entry_decision in [1, 2]:  # BUY ou SELL
                entry_timing_reward, entry_timing_info = self.entry_timing_system.calculate_entry_timing_rewards(
                    env, entry_decision, action
                )
                # 🔥 EIGHTEEN: DOBRAR PESO (6% → 12% do reward total)
                entry_timing_reward_scaled = entry_timing_reward * 2.0
                shaping_reward += entry_timing_reward_scaled
                info['entry_timing_reward'] = entry_timing_reward
                info['entry_timing_reward_scaled'] = entry_timing_reward_scaled
                info.update(entry_timing_info)

            return shaping_reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no reward shaping: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_trailing_stop_rewards(self, env) -> float:
        """
        🎯 SISTEMA HÍBRIDO DE SL/TP MANAGEMENT REWARDS
        Combina heurísticas, improvement tracking e curriculum learning
        """
        try:
            # Obter progresso do treinamento do environment (se disponível)
            self.training_progress = getattr(env, 'training_progress', 0.0)

            # 1. Heurísticas de RR ratio e caps
            heuristic_reward = self._calculate_smart_sltp_heuristics(env)

            # 2. Recompensa por melhorias vs estado anterior
            improvement_reward = self._calculate_sltp_improvement_reward(env)

            # 3. Aplicar curriculum weight (guidance decai com treino)
            guidance_weight = self._get_sltp_guidance_weight(self.training_progress)

            # Combinação final: 70% heurísticas + 30% improvement
            total_reward = (heuristic_reward * 0.7 + improvement_reward * 0.3) * guidance_weight

            return total_reward
        except Exception as e:
            self.logger.debug(f"Erro em trailing_stop_rewards: {e}")
            return 0.0
    
    def _calculate_dynamic_sltp_rewards(self, env) -> float:
        """
        🧠 ALIAS para _calculate_trailing_stop_rewards (unificado)
        Mantém compatibilidade com código existente
        """
        # Sistema unificado agora (trailing + sltp management)
        return 0.0  # Já incluído em _calculate_trailing_stop_rewards

    def _sl_adjustment_was_smart(self, trade) -> bool:
        """
        🧠 AVALIA se ajuste de SL foi inteligente
        """
        # Não pode ter movido contra breakeven
        if self._sl_moved_against_breakeven(trade):
            return False

        # Se protegeu lucro = inteligente
        pnl = trade.get('pnl', 0)
        return pnl >= 0

    def _tp_adjustment_was_smart(self, trade) -> bool:
        """
        🧠 AVALIA se ajuste de TP foi inteligente
        """
        pnl = trade.get('pnl', 0)
        close_reason = trade.get('close_reason', '')

        # Se TP foi hit com lucro = inteligente
        if close_reason == 'TP hit' and pnl > 25:
            return True

        # Se contribuiu para resultado positivo = inteligente
        return pnl > 0
    
    def test_trailing_sltp_rewards(self):
        """
        🧪 Teste dos novos rewards para SL/TP dinâmico e trailing stops
        """
        print("🧪 [V3-BRUTAL] Testando novos rewards SL/TP + Trailing Stop...")
        
        # Mock environment com trades de exemplo
        class MockEnv:
            def __init__(self, trades):
                self.trades = trades
        
        # Cenários de teste
        test_scenarios = [
            {
                'name': 'Trailing Stop Bem-Sucedido',
                'trades': [{
                    'trailing_activated': True,
                    'trailing_protected': True, 
                    'close_reason': 'trailing_stop',
                    'pnl': 45.0,
                    'pnl_percent': 0.02,
                    'trailing_moves': 2
                }]
            },
            {
                'name': 'SL/TP Dynamic Adjustment',
                'trades': [{
                    'sl_adjusted': True,
                    'tp_adjusted': True,
                    'close_reason': 'TP hit',
                    'pnl': 60.0
                }]
            },
            {
                'name': 'Oportunidade Perdida',
                'trades': [{
                    'missed_trailing_opportunity': True,
                    'pnl': -25.0
                }]
            },
            {
                'name': 'Combo Completo',
                'trades': [{
                    'trailing_activated': True,
                    'tp_adjusted': True,
                    'pnl': 75.0,
                    'trailing_moves': 1
                }]
            }
        ]
        
        for scenario in test_scenarios:
            mock_env = MockEnv(scenario['trades'])
            
            trailing_reward = self._calculate_trailing_stop_rewards(mock_env)
            sltp_reward = self._calculate_dynamic_sltp_rewards(mock_env)
            total_reward = trailing_reward + sltp_reward
            
            print(f"   📊 {scenario['name']}:")
            print(f"      Trailing: {trailing_reward:.4f} | SL/TP: {sltp_reward:.4f} | Total: {total_reward:.4f}")
        
        print("✅ [V3-BRUTAL] Teste de rewards SL/TP + Trailing concluído!")
        print("💡 [INFO] Novos rewards ensinarão modelo a usar trailing stops e SL/TP dinâmico")
    
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

    def _extract_entry_decision(self, action: np.ndarray) -> int:
        """
        🎯 SEVENTEEN: Extrair decisão de entrada do action array
        Returns: 0=HOLD, 1=LONG, 2=SHORT
        """
        try:
            if len(action) == 0:
                return 0  # HOLD

            # Action space 4D: [entry_decision, confidence, pos1_mgmt, pos2_mgmt]
            # Usar mesmos thresholds do cherry.py
            ACTION_THRESHOLD_SHORT = -0.33
            ACTION_THRESHOLD_LONG = 0.33

            raw_decision = float(action[0])

            if raw_decision < ACTION_THRESHOLD_SHORT:
                return 2  # SHORT
            elif raw_decision < ACTION_THRESHOLD_LONG:
                return 0  # HOLD
            else:
                return 1  # LONG

        except Exception as e:
            self.logger.error(f"Erro ao extrair entry decision: {e}")
            return 0  # HOLD por segurança
    
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

    def _sl_moved_against_breakeven(self, trade) -> bool:
        """
        🔍 DETECTA se SL foi movido contra o breakeven
        """
        position_type = trade.get('type', '')
        entry_price = trade.get('entry_price', 0)
        final_sl = trade.get('final_sl', 0)

        if position_type == 'short':
            # Para SHORT: SL acima de entry+5 = contraproducente
            return final_sl > (entry_price + 5)
        elif position_type == 'long':
            # Para LONG: SL abaixo de entry-5 = contraproducente
            return final_sl < (entry_price - 5)

        return False

    def _trailing_moves_were_smart(self, trade) -> bool:
        """
        🧠 AVALIA se os movimentos de trailing foram inteligentes
        """
        position_type = trade.get('type', '')
        pnl = trade.get('pnl', 0)
        price_movement = trade.get('price_direction_during_trail', 0)  # +1 up, -1 down

        if position_type == 'short':
            # SHORT inteligente: preço caindo E SL descendo = ✅
            # SHORT burro: preço subindo E SL subindo = ❌
            if price_movement > 0:  # Preço subiu (contra short)
                return pnl >= 0  # Só inteligente se não perdeu dinheiro
            else:  # Preço desceu (favor do short)
                return True  # Sempre inteligente seguir movimento favorável

        elif position_type == 'long':
            # LONG inteligente: preço subindo E SL subindo = ✅
            if price_movement > 0:  # Preço subiu (favor do long)
                return True  # Sempre inteligente seguir movimento favorável
            else:  # Preço desceu (contra long)
                return pnl >= 0  # Só inteligente se não perdeu dinheiro

        return False

    # ============================================================================
    # 🎯 NOVO SISTEMA DE SL/TP REWARDS - ABORDAGEM HÍBRIDA
    # ============================================================================

    def _calculate_smart_sltp_heuristics(self, env) -> float:
        """
        🎯 HEURÍSTICAS BASEADAS EM TRADING REAL:
        - Risk/Reward ratio ideal: 1:1.5 até 1:2.5
        - SL muito apertado (<7pt) = alta probabilidade de hit
        - TP muito distante (>$80) = baixa probabilidade de hit
        """
        try:
            shaping = 0.0
            positions = getattr(env, 'positions', [])

            if not positions:
                return 0.0

            for position in positions:
                if not isinstance(position, dict):
                    continue

                entry_price = position.get('entry_price', 0)
                sl_price = position.get('sl', 0)
                tp_price = position.get('tp', 0)
                pos_type = position.get('type', '')

                if entry_price == 0 or sl_price == 0 or tp_price == 0:
                    continue

                # Calcular distâncias em pontos
                if pos_type == 'long':
                    sl_distance = abs(entry_price - sl_price)
                    tp_distance = abs(tp_price - entry_price)
                elif pos_type == 'short':
                    sl_distance = abs(sl_price - entry_price)
                    tp_distance = abs(entry_price - tp_price)
                else:
                    continue

                # 🎯 HEURÍSTICA 1: Risk/Reward Ratio
                if sl_distance > 0:
                    rr_ratio = tp_distance / sl_distance

                    # Optimal RR = 1.5 a 2.5
                    if 1.5 <= rr_ratio <= 2.5:
                        # ✅ REWARD: RR ratio no sweet spot
                        shaping += 0.01
                    elif rr_ratio < 1.0:
                        # ❌ PENALTY: Risking mais que reward (burro)
                        penalty = -0.02 * (1.0 - rr_ratio)
                        shaping += penalty
                    elif rr_ratio > 4.0:
                        # ❌ PENALTY: TP muito distante (irrealista)
                        penalty = -0.01 * min((rr_ratio - 4.0) / 2.0, 0.5)
                        shaping += penalty

                # 🎯 HEURÍSTICA 2: SL mínimo para respirar
                if sl_distance < 7:
                    # ❌ PENALTY: SL muito apertado (hit fácil)
                    penalty = -0.015 * (7 - sl_distance) / 7
                    shaping += penalty

                # 🎯 HEURÍSTICA 3: Cap de PnL realista
                lot_size = position.get('lot_size', 0.01)
                potential_pnl = tp_distance * lot_size * 100  # Aproximação USD

                if potential_pnl > 80:
                    # ❌ PENALTY: TP muito ganancioso (hit improvável)
                    penalty = -0.01 * min((potential_pnl - 80) / 20, 0.5)
                    shaping += penalty

            # 🎯 HEURÍSTICA 4: SL ZONE QUALITY (usar feature support_resistance)
            try:
                current_step = getattr(env, 'current_step', 0)
                df = getattr(env, 'df', None)

                if df is not None and 'support_resistance' in df.columns and current_step < len(df):
                    # support_resistance = SL zone quality
                    # ALTO (>0.6) = longe de S/R = zona SEGURA para SL
                    # BAIXO (<0.4) = perto de S/R = zona PERIGOSA para SL
                    sl_zone_quality = df['support_resistance'].iloc[current_step]

                    for position in positions:
                        if not isinstance(position, dict):
                            continue

                        entry_price = position.get('entry_price', 0)
                        sl_price = position.get('sl', 0)
                        pos_type = position.get('type', '')

                        if entry_price == 0 or sl_price == 0:
                            continue

                        # Calcular distância SL
                        if pos_type == 'long':
                            sl_distance = abs(entry_price - sl_price)
                        elif pos_type == 'short':
                            sl_distance = abs(sl_price - entry_price)
                        else:
                            continue

                        # CASO 1: SL zone quality ALTO (zona segura)
                        if sl_zone_quality > 0.6:
                            # SL está longe de S/R = BOM!
                            # Reward se SL está em range realista (10-15pt)
                            if 10 <= sl_distance <= 15:
                                shaping += 0.12  # ÓTIMO: zona segura + SL realista
                            else:
                                shaping += 0.08  # BOM: zona segura

                        # CASO 2: SL zone quality BAIXO (zona perigosa)
                        elif sl_zone_quality < 0.4:
                            # SL está perto de S/R = RUIM!
                            # Penalty maior se SL está no mínimo (mais risco)
                            if sl_distance <= 10.5:
                                shaping -= 0.15  # PÉSSIMO: zona perigosa + SL mínimo
                            else:
                                shaping -= 0.08  # RUIM: zona perigosa
            except:
                pass

            # 🎯 HEURÍSTICA 5: TP MIRADO EM ZONA DE RESISTÊNCIA (usar feature breakout_strength)
            try:
                current_step = getattr(env, 'current_step', 0)
                df = getattr(env, 'df', None)

                if df is not None and 'breakout_strength' in df.columns and current_step < len(df):
                    # breakout_strength = TP target quality
                    # ALTO (>0.6) = resistência PRÓXIMA = alvo BOM para TP
                    # BAIXO (<0.3) = resistência DISTANTE = alvo RUIM para TP
                    tp_target_quality = df['breakout_strength'].iloc[current_step]

                    for position in positions:
                        if not isinstance(position, dict):
                            continue

                        entry_price = position.get('entry_price', 0)
                        tp_price = position.get('tp', 0)
                        pos_type = position.get('type', '')

                        if entry_price == 0 or tp_price == 0:
                            continue

                        # Calcular distância TP
                        if pos_type == 'long':
                            tp_distance = abs(tp_price - entry_price)
                        elif pos_type == 'short':
                            tp_distance = abs(entry_price - tp_price)
                        else:
                            continue

                        # CASO 1: TP target quality ALTO (resistência próxima)
                        if tp_target_quality > 0.6:
                            # Resistência está próxima
                            # Se TP está no range realista (12-18pt) = mira na resistência = ÓTIMO!
                            if 12 <= tp_distance <= 18:
                                shaping += 0.10 * tp_target_quality  # Max +0.10
                            # Se TP fora do range, não aproveita resistência = RUIM
                            else:
                                shaping -= 0.08

                        # CASO 2: TP target quality BAIXO (resistência distante)
                        elif tp_target_quality < 0.3:
                            # Resistência está distante
                            # Se TP está no range realista = conservador = BOM
                            if 12 <= tp_distance <= 18:
                                shaping += 0.05
            except:
                pass

            return shaping

        except Exception as e:
            self.logger.debug(f"Erro em smart_sltp_heuristics: {e}")
            return 0.0

    def _calculate_sltp_improvement_reward(self, env) -> float:
        """
        🎯 RECOMPENSA MELHORIA, NÃO AÇÃO ABSOLUTA
        Compara estado atual vs estado anterior
        """
        try:
            reward = 0.0
            positions = getattr(env, 'positions', [])

            if not positions:
                # Limpar estados de posições fechadas
                self.previous_sltp_state.clear()
                return 0.0

            for position in positions:
                if not isinstance(position, dict):
                    continue

                pos_id = position.get('id', position.get('ticket', str(hash(str(position)))))
                entry_price = position.get('entry_price', 0)
                current_sl = position.get('sl', 0)
                current_tp = position.get('tp', 0)
                pos_type = position.get('type', '')

                if entry_price == 0 or current_sl == 0 or current_tp == 0:
                    continue

                # Calcular RR ratio atual
                if pos_type == 'long':
                    sl_distance = abs(entry_price - current_sl)
                    tp_distance = abs(current_tp - entry_price)
                elif pos_type == 'short':
                    sl_distance = abs(current_sl - entry_price)
                    tp_distance = abs(entry_price - current_tp)
                else:
                    continue

                current_rr = tp_distance / sl_distance if sl_distance > 0 else 0

                # Verificar se temos estado anterior
                if pos_id in self.previous_sltp_state:
                    prev_state = self.previous_sltp_state[pos_id]
                    prev_sl = prev_state.get('sl', 0)
                    prev_tp = prev_state.get('tp', 0)
                    prev_rr = prev_state.get('rr_ratio', 0)

                    # 🎯 RECOMPENSA 1: RR ratio melhorou (mais perto de 2.0)
                    if abs(current_rr - 2.0) < abs(prev_rr - 2.0):
                        reward += 0.005

                    # 🎯 RECOMPENSA 2: SL protegeu lucro (trailing inteligente)
                    current_price = getattr(env, 'current_price', entry_price)

                    if pos_type == 'long':
                        # LONG: SL subiu (proteção)
                        if current_sl > prev_sl and current_sl > entry_price:
                            # SL subiu E está em lucro = inteligente
                            protection_quality = (current_sl - entry_price) / entry_price
                            reward += 0.01 * min(protection_quality * 100, 1.0)
                    elif pos_type == 'short':
                        # SHORT: SL desceu (proteção)
                        if current_sl < prev_sl and current_sl < entry_price:
                            # SL desceu E está em lucro = inteligente
                            protection_quality = (entry_price - current_sl) / entry_price
                            reward += 0.01 * min(protection_quality * 100, 1.0)

                    # 🎯 RECOMPENSA 3: TP ajustado inteligentemente
                    if abs(current_tp - prev_tp) > 0.5:  # TP mudou significativamente
                        # Calcular se ajuste aproximou de target realista
                        lot_size = position.get('lot_size', 0.01)
                        potential_pnl = tp_distance * lot_size * 100

                        if 40 <= potential_pnl <= 80:  # Sweet spot $40-$80
                            reward += 0.005

                # Atualizar estado para próximo step
                self.previous_sltp_state[pos_id] = {
                    'sl': current_sl,
                    'tp': current_tp,
                    'rr_ratio': current_rr
                }

            # Limpar estados de posições que não existem mais
            current_pos_ids = set(
                pos.get('id', pos.get('ticket', str(hash(str(pos)))))
                for pos in positions if isinstance(pos, dict)
            )

            for pos_id in list(self.previous_sltp_state.keys()):
                if pos_id not in current_pos_ids:
                    del self.previous_sltp_state[pos_id]

            return reward

        except Exception as e:
            self.logger.debug(f"Erro em sltp_improvement_reward: {e}")
            return 0.0

    def _calculate_tp_hit_expert_reward(self, env) -> float:
        """
        🎯 REWARD MASSIVO POR TP HIT

        TP HIT = EVENTO VALIOSO que deve ser FORTEMENTE recompensado
        Reward proporcional à distância do TP:
        - TP curto hit (12-18pt): +0.20 (ÓTIMO - realista!)
        - TP médio hit (19-23pt): +0.12
        - TP cap hit (24-25pt): +0.08 (possível mas raro)
        """
        try:
            reward = 0.0

            # Detectar trades fechados NESTE step
            current_trades = getattr(env, 'trades', [])
            current_trades_count = len(current_trades)

            # Se houve novos trades fechados
            if current_trades_count > self.previous_trades_count:
                # Trades novos = trades[previous_count:]
                new_trades = current_trades[self.previous_trades_count:]

                for trade in new_trades:
                    if not isinstance(trade, dict):
                        continue

                    # Verificar se trade fechou por TP HIT
                    close_reason = trade.get('close_reason', '')
                    if close_reason == 'tp_hit':
                        # TP FOI HIT! Recompensar massivamente
                        entry_price = trade.get('entry_price', 0)
                        exit_price = trade.get('exit_price', 0)
                        trade_type = trade.get('type', '')

                        if entry_price > 0 and exit_price > 0:
                            # Calcular distância do TP em pontos
                            if trade_type == 'long':
                                tp_distance = abs(exit_price - entry_price)
                            else:
                                tp_distance = abs(entry_price - exit_price)

                            # RECOMPENSA PROPORCIONAL À DISTÂNCIA
                            if 12 <= tp_distance <= 18:
                                # TP realista hit = ÓTIMO! (range correto 12-18pt)
                                reward += 0.20
                            else:
                                # TP fora do range esperado = pequeno reward
                                reward += 0.05

            # Atualizar contador para próximo step
            self.previous_trades_count = current_trades_count

            return min(reward, 0.5)  # Cap em +0.5

        except Exception as e:
            self.logger.debug(f"Erro em tp_hit_expert_reward: {e}")
            return 0.0

    def _calculate_sl_near_miss_reward(self, env) -> float:
        """
        🎯 REWARD POR SL BEM POSICIONADO (quase hit mas não hit)

        SL BEM POSICIONADO:
        - Preço chegou perto (±2pt) mas NÃO hit
        - Indica que SL está na distância CORRETA

        SL MAL POSICIONADO:
        - Preço nem chegou perto = SL muito distante (desperdício)
        """
        try:
            reward = 0.0
            positions = getattr(env, 'positions', [])
            current_price = getattr(env, 'current_price', 0)

            if current_price == 0:
                return 0.0

            for position in positions:
                if not isinstance(position, dict):
                    continue

                sl_price = position.get('sl', 0)
                pos_type = position.get('type', '')

                if sl_price == 0:
                    continue

                # Calcular distância do preço atual até SL
                distance_to_sl = abs(current_price - sl_price)

                # CASO 1: Preço chegou PERTO do SL (±2pt) mas NÃO hit
                if 2 <= distance_to_sl <= 3:
                    # SL segurou! Quase hit mas não hit = BOM!
                    if pos_type == 'long' and current_price > sl_price:
                        # LONG: preço acima do SL (seguro)
                        reward += 0.10
                    elif pos_type == 'short' and current_price < sl_price:
                        # SHORT: preço abaixo do SL (seguro)
                        reward += 0.10

            return min(reward, 0.3)  # Cap em +0.3

        except Exception as e:
            self.logger.debug(f"Erro em sl_near_miss_reward: {e}")
            return 0.0

    def _calculate_trailing_timing_reward(self, env) -> float:
        """
        🎯 REWARD POR TRAILING NO MOMENTO CERTO

        TIMING BOM:
        - Trailing SL após lucro significativo (+10pt): +0.15 (PROTEGER!)
        - Trailing SL após lucro grande (+15pt): +0.20 (ÓTIMO!)

        TIMING RUIM:
        - Trailing SL sem lucro ou lucro pequeno (<5pt): -0.10 (PREMATURO!)
        - Trailing SL em posição perdedora: -0.15 (BURRICE!)
        """
        try:
            reward = 0.0
            positions = getattr(env, 'positions', [])

            for position in positions:
                if not isinstance(position, dict):
                    continue

                # Verificar se houve trailing SL neste step
                # (comparar SL atual com SL anterior salvo em previous_sltp_state)
                pos_id = position.get('id', position.get('ticket', str(hash(str(position)))))
                current_sl = position.get('sl', 0)

                if pos_id in self.previous_sltp_state:
                    prev_sl = self.previous_sltp_state[pos_id].get('sl', 0)

                    # Detectar se SL mudou (trailing)
                    if abs(current_sl - prev_sl) > 0.5:
                        # SL foi ajustado!
                        entry_price = position.get('entry_price', 0)
                        pos_type = position.get('type', '')
                        current_price = getattr(env, 'current_price', entry_price)

                        if entry_price > 0:
                            # Calcular PnL unrealized
                            if pos_type == 'long':
                                unrealized_pnl_points = current_price - entry_price
                                sl_moved_up = current_sl > prev_sl  # Trailing = SL sobe
                            else:
                                unrealized_pnl_points = entry_price - current_price
                                sl_moved_up = current_sl < prev_sl  # Trailing = SL desce

                            # Verificar se foi TRAILING (SL a favor do trade)
                            if (pos_type == 'long' and sl_moved_up) or (pos_type == 'short' and not sl_moved_up):
                                # É trailing!

                                # CASO 1: Trailing após lucro significativo
                                if unrealized_pnl_points >= 15:
                                    # Lucro grande (+15pt) → trailing = ÓTIMO!
                                    reward += 0.20
                                elif unrealized_pnl_points >= 10:
                                    # Lucro bom (+10pt) → trailing = BOM!
                                    reward += 0.15
                                elif unrealized_pnl_points >= 5:
                                    # Lucro pequeno (+5pt) → trailing = OK
                                    reward += 0.08

                                # CASO 2: Trailing SEM lucro ou COM PREJUÍZO
                                elif unrealized_pnl_points < 0:
                                    # Prejuízo → trailing = BURRICE!
                                    reward -= 0.15
                                else:
                                    # Lucro < 5pt → trailing = PREMATURO!
                                    reward -= 0.10

            return max(min(reward, 0.5), -0.3)  # Cap entre -0.3 e +0.5

        except Exception as e:
            self.logger.debug(f"Erro em trailing_timing_reward: {e}")
            return 0.0

    def _calculate_trend_following_reward(self, env) -> float:
        """
        🎯 TREND FOLLOWING REWARD AMPLIFICADO: Recompensa FORTE por seguir a tendência

        CONCEITO BÁSICO DE TRADING:
        - Mercado subindo → COMPRAR (LONG)
        - Mercado descendo → VENDER (SHORT)
        - Contratendência = PENALTY MASSIVO

        🔥 AMPLIFICADO V2:
        - Reward base aumentado: 0.15 → 0.25 (+67%)
        - Multiplicador por trend_strength: 1.0x a 2.5x
        - Cap aumentado: ±0.3 → ±0.6 (+100%)
        """
        try:
            reward = 0.0

            # Pegar dataframe e step atual
            df = getattr(env, 'df', None)
            current_step = getattr(env, 'current_step', 0)

            if df is None or 'trend_consistency' not in df.columns:
                return 0.0

            if current_step >= len(df):
                return 0.0

            # Pegar trend_consistency E trend_strength
            trend_consistency = df['trend_consistency'].iloc[current_step]

            # 🎯 NOVO: Usar trend_strength_1m se disponível
            if 'trend_strength_1m' in df.columns:
                trend_strength = df['trend_strength_1m'].iloc[current_step]
            else:
                trend_strength = 0.5  # Fallback neutro

            # Detectar DIREÇÃO do trend
            if 'returns_1m' in df.columns and current_step >= 10:
                # Últimos 10 returns para detectar direção
                recent_returns = df['returns_1m'].iloc[max(0, current_step-10):current_step].values
                avg_return = recent_returns.mean() if len(recent_returns) > 0 else 0

                # Pegar posições atuais
                positions = getattr(env, 'positions', [])

                for pos in positions:
                    if not isinstance(pos, dict):
                        continue

                    pos_type = pos.get('type', '')

                    # 🎯 AMPLIFICAR REWARD baseado em trend_strength
                    # trend_strength (0-1): quanto mais forte, maior o multiplicador
                    strength_multiplier = 1.0 + (trend_strength * 1.5)  # 1.0x a 2.5x

                    # CASO 1: TREND UP FORTE (avg_return > 0.001, consistency > 0.6)
                    if avg_return > 0.001 and trend_consistency > 0.6:
                        if pos_type == 'long':
                            # LONG em trend UP = BOM! (amplificado por strength)
                            base_reward = 0.25  # ⬆️ Aumentado de 0.15 para 0.25
                            reward += base_reward * trend_consistency * strength_multiplier
                        elif pos_type == 'short':
                            # SHORT em trend UP = BURRICE! (penalty amplificada)
                            base_penalty = -0.25  # ⬆️ Aumentado de -0.15 para -0.25
                            reward += base_penalty * trend_consistency * strength_multiplier

                    # CASO 2: TREND DOWN FORTE (avg_return < -0.001, consistency > 0.6)
                    elif avg_return < -0.001 and trend_consistency > 0.6:
                        if pos_type == 'short':
                            # SHORT em trend DOWN = BOM!
                            base_reward = 0.25  # ⬆️ Aumentado de 0.15 para 0.25
                            reward += base_reward * trend_consistency * strength_multiplier
                        elif pos_type == 'long':
                            # LONG em trend DOWN = BURRICE!
                            base_penalty = -0.25  # ⬆️ Aumentado de -0.15 para -0.25
                            reward += base_penalty * trend_consistency * strength_multiplier

            return max(min(reward, 0.6), -0.6)  # ⬆️ Cap aumentado de ±0.3 para ±0.6

        except Exception as e:
            self.logger.debug(f"Erro em trend_following_reward: {e}")
            return 0.0

    def _calculate_sltp_gaming_penalty(self, env) -> float:
        """
        🚨 PENALIDADE BRUTAL REFORÇADA: Detectar gaming de SL mínimo + TP máximo

        GAMING PATTERN:
        - SL sempre no mínimo permitido (10-12 pontos)
        - TP sempre no máximo permitido (17-18 pontos)
        - Combinação indica que modelo está GAMANDO reward system

        🔥 REFORÇADO V2:
        - Penalties individuais: -0.05 → -0.12 (+140%)
        - Combo penalty: -0.15 → -0.35 (+133%)
        - Cap total: -2.5 → -3.5 (+40%)
        - NOVO: Bonus para SL/TP no sweet spot
        """
        try:
            penalty = 0.0
            positions = getattr(env, 'positions', [])

            if not positions:
                return 0.0

            for position in positions:
                if not isinstance(position, dict):
                    continue

                entry_price = position.get('entry_price', 0)
                sl_price = position.get('sl', 0)
                tp_price = position.get('tp', 0)
                pos_type = position.get('type', '')
                duration = position.get('duration', 0)

                if entry_price == 0 or sl_price == 0 or tp_price == 0:
                    continue

                # Calcular distâncias em pontos
                if pos_type == 'long':
                    sl_distance = abs(entry_price - sl_price)
                    tp_distance = abs(tp_price - entry_price)
                elif pos_type == 'short':
                    sl_distance = abs(sl_price - entry_price)
                    tp_distance = abs(entry_price - tp_price)
                else:
                    continue

                # 🚨 GAMING DETECTION #1: SL no mínimo - PENALTY AUMENTADA
                if sl_distance <= 10.2:  # SL fixo no mínimo = gaming
                    # ⬆️ AUMENTAR de -0.05 para -0.12 (2.4x mais forte)
                    penalty -= 0.12 * max(1, duration / 10)

                # 🚨 GAMING DETECTION #2: TP no máximo - PENALTY AUMENTADA
                if tp_distance >= 17.8:  # TP fixo no máximo = gaming
                    # ⬆️ AUMENTAR de -0.05 para -0.12 (2.4x mais forte)
                    penalty -= 0.12 * max(1, duration / 10)

                # 🚨 GAMING DETECTION #3: COMBO SL MIN + TP MAX (CRITICAL)
                if sl_distance <= 10.2 and tp_distance >= 17.8:
                    # ⬆️ AUMENTAR de -0.15 para -0.35 (2.3x mais forte)
                    multiplier = min(duration / 5, 5.0)  # Cap em 5x
                    penalty -= 0.35 * multiplier  # Até -1.75 por posição!

                # 🚨 GAMING DETECTION #4: RR ratio extremo
                if sl_distance > 0:
                    rr_ratio = tp_distance / sl_distance

                    # RR > 1.8 com SL mínimo = gaming claro
                    if rr_ratio > 1.8 and sl_distance <= 10.5:
                        penalty -= 0.08 * (rr_ratio - 1.5)

                # 🎯 NOVO: BONUS POR SL/TP NO SWEET SPOT
                # Recompensar ativamente SL 12-14pt e TP 14-16pt
                if 12 <= sl_distance <= 14 and 14 <= tp_distance <= 16:
                    # SWEET SPOT: SL e TP ideais
                    bonus = 0.08 * min(duration / 5, 2.0)  # Max +0.16
                    penalty += bonus  # Adiciona bonus (reduz penalty total)

            # ⬆️ Cap aumentado de -2.5 para -3.5
            return max(penalty, -3.5)

        except Exception as e:
            self.logger.debug(f"Erro em sltp_gaming_penalty: {e}")
            return 0.0

    def _calculate_tp_realism_bonus(self, env) -> float:
        """
        🎯 BONIFICAR TP realista baseado em estrutura de mercado

        TP BOM:
        - Mira em resistência próxima (LONG) ou suporte próximo (SHORT)
        - Distância é múltiplo razoável de ATR (1-2.5 ATR)

        TP RUIM:
        - Ignora resistências próximas
        - Distância irrealista (>2.5 ATR ou no cap de 25 pontos)
        """
        try:
            bonus = 0.0
            positions = getattr(env, 'positions', [])

            if not positions:
                return 0.0

            # Pegar dados de mercado
            current_step = getattr(env, 'current_step', 0)
            df = getattr(env, 'df', None)

            if df is None or 'breakout_strength' not in df.columns or current_step >= len(df):
                return 0.0

            # breakout_strength agora é tp_target_quality
            tp_target_quality = df['breakout_strength'].iloc[current_step]
            current_atr = getattr(env, 'current_atr', 15.0)

            for position in positions:
                if not isinstance(position, dict):
                    continue

                entry_price = position.get('entry_price', 0)
                tp_price = position.get('tp', 0)
                pos_type = position.get('type', '')

                if entry_price == 0 or tp_price == 0:
                    continue

                # Calcular distância do TP em pontos
                if pos_type == 'long':
                    tp_distance = tp_price - entry_price
                else:
                    tp_distance = entry_price - tp_price

                # Calcular distância em múltiplos de ATR
                tp_atr_multiple = tp_distance / current_atr if current_atr > 0 else 0

                # CASO 1: TP target quality ALTO (resistência próxima)
                if tp_target_quality > 0.6:
                    # Resistência está próxima (ex: 1.5 ATR)
                    # Se modelo setou TP próximo (1-2 ATR), REWARD
                    if 1.0 <= tp_atr_multiple <= 2.0:
                        bonus += 0.08 * tp_target_quality
                    # Se modelo setou TP muito longe ignorando resistência, PENALTY
                    elif tp_atr_multiple > 2.5:
                        bonus -= 0.05

                # CASO 2: TP target quality BAIXO (resistência distante)
                elif tp_target_quality < 0.3:
                    # Resistência está longe (>3 ATR)
                    # Se modelo setou TP conservador (<2 ATR), REWARD
                    if tp_atr_multiple < 2.0:
                        bonus += 0.03
                    # Se modelo setou TP no CAP (25 pontos) mirando muito longe, PENALTY
                    elif tp_distance >= 24:
                        bonus -= 0.08

            return max(min(bonus, 0.5), -0.5)

        except Exception as e:
            self.logger.debug(f"Erro em tp_realism_bonus: {e}")
            return 0.0

    def _get_sltp_guidance_weight(self, training_progress: float) -> float:
        """
        🎯 CURRICULUM LEARNING: Guiar mais no início, menos no final

        0-20% treino: Guidance weight = 1.0 (guidance forte)
        20-60% treino: Guidance weight = 0.5 (guidance moderado)
        60-100% treino: Guidance weight = 0.1 (quase sem guidance)
        """
        if training_progress < 0.2:
            return 1.0  # Guidance forte (modelo ainda burro)
        elif training_progress < 0.6:
            return 0.5  # Guidance moderado (modelo aprendendo)
        else:
            return 0.1  # Guidance mínimo (modelo maduro)

    def update_training_progress(self, current_steps: int, total_steps: int = 12000000):
        """
        Atualiza progresso do treinamento para curriculum learning
        Deve ser chamado periodicamente durante o treinamento
        """
        self.training_progress = min(current_steps / total_steps, 1.0)

    def reset(self):
        """Reset para novo episódio"""
        # Reset tracking para novo episódio
        self.last_portfolio_value = self.initial_balance
        self.position_age_steps.clear()
        self.recent_pnl_trend.clear()
        self.previous_sltp_state.clear()  # 🎯 NOVO: Limpar estado de SL/TP

        # Manter apenas stats cumulativas

    def set_smoothing_alpha(self, alpha: float):
        """
        Ajusta o parâmetro de smoothing dinamicamente durante o treinamento

        Args:
            alpha: Taxa de smoothing (0.1=muito suave, 0.5=responsivo, 1.0=sem smoothing)
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("Alpha deve estar entre 0.0 e 1.0")

        old_alpha = self.smoothing_alpha
        self.smoothing_alpha = alpha
        print(f"[V3-BRUTAL] Smoothing alpha alterado: {old_alpha:.2f} → {alpha:.2f}")

        if alpha == 1.0:
            print("[V3-BRUTAL] ⚠️ Smoothing DESABILITADO (alpha=1.0)")
        elif alpha <= 0.1:
            print("[V3-BRUTAL] 📊 Smoothing MUITO SUAVE (alta estabilidade)")
        elif alpha >= 0.5:
            print("[V3-BRUTAL] ⚡ Smoothing RESPONSIVO (baixa estabilidade)")
        else:
            print("[V3-BRUTAL] ⚖️ Smoothing BALANCEADO")

    def get_smoothing_stats(self) -> Dict:
        """Retorna estatísticas do smoothing para debugging"""
        return {
            'smoothing_alpha': self.smoothing_alpha,
            'current_smooth_value': self.smooth_reward,
            'smoothing_initialized': self.smoothing_initialized,
            'step_counter': self.step_counter
        }

    def reset_smoothing(self):
        """Redefine o smoothing (útil para novos episódios ou experimentos)"""
        self.smooth_reward = 0.0
        self.smoothing_initialized = False
        print("[V3-BRUTAL] 🔄 Smoothing reinicializado")


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