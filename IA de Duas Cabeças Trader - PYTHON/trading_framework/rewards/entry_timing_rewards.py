"""
🎯 ENTRY TIMING REWARDS V2 - EIGHTEEN EXPERIMENT
Sistema de recompensas focado em melhorar o TIMING DE ENTRADA

Objetivo: Melhorar Win Rate de 37.7% → 50%+
Método: Multi-Signal Confluence + Behavioral Controls

🔥 MUDANÇAS DO SEVENTEEN → EIGHTEEN:
- ❌ REMOVIDO: Rewards de horário (robô já faz filtro)
- ✅ NOVO: Multi-Signal Confluence Entry (3 camadas)
- ✅ NOVO: Entry Timing After Loss (previne revenge trading)
- ✅ NOVO: Confidence Appropriateness
- ✅ NOVO: Revenge Trading Penalty
- ✅ NOVO: Cut Loss Incentive
- ✅ NOVO: Pattern Recognition (MA Cross, Double Top/Bottom)
- 📊 PESO DOBRADO: 6% → 12% do reward total

Componentes:
1. Entry Timing Quality (50% do Entry Timing)
2. Entry Confluence Reward (30% do Entry Timing)
3. Market Context Reward (20% do Entry Timing)
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
from collections import deque


class MultiSignalConfluenceEntry:
    """
    🎯 SISTEMA DE 3 CAMADAS PARA VALIDAR QUALIDADE DE ENTRADA
    Usa TODOS os intelligent components do Cherry
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Pesos das camadas
        self.layer1_weight = 0.40  # Regime + Volatility
        self.layer2_weight = 0.40  # Momentum + Technical
        self.layer3_weight = 0.20  # Structural

    def calculate_multi_signal_entry_reward(self, env, entry_decision: int, action: np.ndarray) -> Tuple[float, Dict]:
        """
        🎯 CALCULA REWARD BASEADO EM CONFLUÊNCIA DE MÚLTIPLOS SINAIS

        Args:
            env: Environment do cherry
            entry_decision: 1=LONG, 2=SHORT
            action: Array de ações (inclui confidence)

        Returns:
            reward: -1.0 a +1.0
            info: Detalhamento completo
        """
        try:
            # Obter intelligent components
            intelligent_components = getattr(env, '_cached_intelligent_components', None)
            if not intelligent_components:
                return 0.0, {'error': 'No intelligent components'}

            # Layer 1: Regime + Volatility Validation (40%)
            layer1_reward, layer1_info = self._validate_regime_and_volatility(
                env, entry_decision, intelligent_components
            )

            # Layer 2: Momentum + Technical Confluence (40%)
            layer2_reward, layer2_info = self._validate_momentum_and_technical(
                env, entry_decision, intelligent_components, action
            )

            # Layer 3: Structural Confirmation (20%)
            layer3_reward, layer3_info = self._validate_structural_confirmation(
                env, entry_decision
            )

            # Combinar camadas
            total_reward = (
                layer1_reward * self.layer1_weight +
                layer2_reward * self.layer2_weight +
                layer3_reward * self.layer3_weight
            )

            # Info completo
            info = {
                'total_reward': total_reward,
                'layer1_regime_volatility': layer1_reward,
                'layer2_momentum_technical': layer2_reward,
                'layer3_structural': layer3_reward,
                **layer1_info,
                **layer2_info,
                **layer3_info
            }

            return total_reward, info

        except Exception as e:
            self.logger.error(f"Erro em multi-signal entry: {e}")
            return 0.0, {'error': str(e)}

    def _validate_regime_and_volatility(self, env, entry_decision: int,
                                       components: Dict) -> Tuple[float, Dict]:
        """
        🎯 LAYER 1: VALIDAÇÃO DE REGIME E VOLATILIDADE (40%)
        CRÍTICO: Previne entradas em condições impossíveis
        """
        reward = 0.0
        info = {}

        market_regime = components.get('market_regime', {})
        volatility_context = components.get('volatility_context', {})

        regime = market_regime.get('regime', 'unknown')
        regime_strength = market_regime.get('strength', 0.0)
        regime_direction = market_regime.get('direction', 0.0)

        vol_level = volatility_context.get('level', 'normal')
        vol_percentile = volatility_context.get('percentile', 0.5)
        vol_expanding = volatility_context.get('expanding', False)

        # ========================================
        # 1.1 REGIME ALIGNMENT (60% da Layer 1)
        # ========================================

        # LONG entries
        if entry_decision == 1:
            # ✅ IDEAL: LONG em trending_up forte
            if regime == 'trending_up' and regime_strength > 0.5:
                reward += 1.0 * regime_strength
                info['regime_perfect_long'] = True

            # 🔴 CRÍTICO: NUNCA comprar em crash
            elif regime == 'crash':
                reward -= 2.0  # PENALTY MASSIVA
                info['crash_buy_blocked'] = True

            # ⚠️ RUIM: LONG em trending_down
            elif regime == 'trending_down':
                reward -= 1.0 * regime_strength
                info['contra_trend_long'] = True

            # 🟡 NEUTRO: LONG em ranging
            elif regime == 'ranging':
                reward += 0.0
                info['ranging_long'] = True

            # 🟠 QUESTIONÁVEL: LONG em volatile
            elif regime == 'volatile':
                reward -= 0.3
                info['volatile_long'] = True

        # SHORT entries
        elif entry_decision == 2:
            # ✅ IDEAL: SHORT em trending_down forte
            if regime == 'trending_down' and regime_strength > 0.5:
                reward += 1.0 * regime_strength
                info['regime_perfect_short'] = True

            # ⚠️ RUIM: SHORT em trending_up
            elif regime == 'trending_up':
                reward -= 1.0 * regime_strength
                info['contra_trend_short'] = True

            # 🟢 BOM: SHORT em crash
            elif regime == 'crash':
                reward += 0.8
                info['crash_short'] = True

            # 🟡 NEUTRO: SHORT em ranging
            elif regime == 'ranging':
                reward += 0.0
                info['ranging_short'] = True

            # 🟠 QUESTIONÁVEL: SHORT em volatile
            elif regime == 'volatile':
                reward -= 0.3
                info['volatile_short'] = True

        # ========================================
        # 1.2 VOLATILITY VALIDATION (40% da Layer 1)
        # ========================================

        # ✅ IDEAL: Volatilidade normal
        if vol_level == 'normal':
            reward += 0.4
            info['volatility_optimal'] = True

        # ⚠️ ATENÇÃO: Volatilidade extrema alta
        elif vol_level == 'high' and vol_percentile > 0.85:
            reward -= 0.6
            info['volatility_too_high'] = True

        # 🟠 SUBÓTIMO: Volatilidade extrema baixa
        elif vol_level == 'low' and vol_percentile < 0.15:
            reward -= 0.3
            info['volatility_too_low'] = True

        # 🎯 BONUS: Volatilidade expandindo na direção da entrada
        entry_direction = 1 if entry_decision == 1 else -1
        if vol_expanding and (regime_direction * entry_direction > 0):
            reward += 0.3
            info['vol_expansion_aligned'] = True

        info['layer1_raw_score'] = reward

        # Normalizar para -1.0 a +1.0
        normalized_reward = np.tanh(reward)

        return normalized_reward, info

    def _validate_momentum_and_technical(self, env, entry_decision: int,
                                        components: Dict, action: np.ndarray) -> Tuple[float, Dict]:
        """
        🎯 LAYER 2: MOMENTUM + TECHNICAL CONFLUENCE (40%)
        Valida confluência de indicadores técnicos
        """
        reward = 0.0
        info = {}

        momentum_confluence = components.get('momentum_confluence', {})

        momentum_score = momentum_confluence.get('score', 0.0)
        momentum_direction = momentum_confluence.get('direction', 0.0)
        momentum_strength = momentum_confluence.get('strength', 0.0)

        entry_direction = 1 if entry_decision == 1 else -1
        entry_confidence = action[1] if len(action) > 1 else 0.5

        # ========================================
        # 2.1 MOMENTUM CONFLUENCE SCORE (40% da Layer 2)
        # ========================================

        # ✅ ALTA CONFLUÊNCIA + Direção alinhada
        if momentum_score > 0.7 and (momentum_direction * entry_direction > 0):
            reward += 1.0 * momentum_strength
            info['high_confluence_aligned'] = True

        # 🟢 MÉDIA CONFLUÊNCIA + Direção alinhada
        elif momentum_score > 0.5 and (momentum_direction * entry_direction > 0):
            reward += 0.5 * momentum_strength
            info['medium_confluence_aligned'] = True

        # ⚠️ BAIXA CONFLUÊNCIA (sinais mistos)
        elif momentum_score < 0.3:
            reward -= 0.6
            info['low_confluence_warning'] = True

        # 🔴 CONFLUÊNCIA CONTRA A ENTRADA
        elif momentum_direction * entry_direction < -0.3:
            reward -= 0.8
            info['confluence_against'] = True

        # ========================================
        # 2.2 RSI DIVERGENCE DETECTION (30% da Layer 2)
        # ========================================

        divergence_reward, divergence_info = self._detect_rsi_divergence(env, entry_decision)
        reward += divergence_reward * 0.75
        info.update(divergence_info)

        # ========================================
        # 2.3 CONFIDENCE APPROPRIATENESS (30% da Layer 2)
        # ========================================

        # Avaliar se confidence está apropriada para o momentum
        market_quality = momentum_score > 0.6 and abs(momentum_direction) > 0.5

        if market_quality and entry_confidence > 0.7:
            # Alta confiança em mercado de alta qualidade = ÓTIMO
            reward += 0.6
            info['high_confidence_justified'] = True
        elif not market_quality and entry_confidence < 0.4:
            # Baixa confiança em mercado duvidoso = BOM
            reward += 0.4
            info['appropriate_caution'] = True
        elif market_quality and entry_confidence < 0.4:
            # Baixa confiança em mercado bom = PERDEU OPORTUNIDADE
            reward -= 0.3
            info['missed_opportunity'] = True
        elif not market_quality and entry_confidence > 0.7:
            # Alta confiança em mercado ruim = PERIGOSO
            reward -= 0.8
            info['overconfidence_danger'] = True

        info['layer2_raw_score'] = reward

        # Normalizar para -1.0 a +1.0
        normalized_reward = np.tanh(reward)

        return normalized_reward, info

    def _detect_rsi_divergence(self, env, entry_decision: int) -> Tuple[float, Dict]:
        """
        🎯 DETECTAR DIVERGÊNCIAS DE RSI
        Divergência = sinal técnico muito forte de reversão
        """
        reward = 0.0
        info = {}

        try:
            df = getattr(env, 'df', None)
            current_step = getattr(env, 'current_step', 0)

            if df is None or 'rsi_14_1m' not in df.columns or current_step < 20:
                return 0.0, {}

            # Obter RSI e preços recentes (últimas 20 barras)
            rsi_recent = df['rsi_14_1m'].iloc[current_step-20:current_step+1].values
            close_recent = df['close_1m'].iloc[current_step-20:current_step+1].values

            if len(rsi_recent) < 20 or len(close_recent) < 20:
                return 0.0, {}

            # BULLISH DIVERGENCE (para LONG)
            if entry_decision == 1:
                price_old_low = np.min(close_recent[5:12])
                price_recent_low = np.min(close_recent[12:])
                rsi_old_low = np.min(rsi_recent[5:12])
                rsi_recent_low = np.min(rsi_recent[12:])

                # Divergência bullish: preço baixa, RSI sobe
                if price_recent_low < price_old_low * 0.998:
                    if rsi_recent_low > rsi_old_low + 2:
                        divergence_strength = (rsi_recent_low - rsi_old_low) / 50.0
                        reward = 1.0 * min(divergence_strength, 1.0)
                        info['bullish_divergence'] = True
                        info['divergence_strength'] = divergence_strength

            # BEARISH DIVERGENCE (para SHORT)
            elif entry_decision == 2:
                price_old_high = np.max(close_recent[5:12])
                price_recent_high = np.max(close_recent[12:])
                rsi_old_high = np.max(rsi_recent[5:12])
                rsi_recent_high = np.max(rsi_recent[12:])

                # Divergência bearish: preço sobe, RSI desce
                if price_recent_high > price_old_high * 1.002:
                    if rsi_recent_high < rsi_old_high - 2:
                        divergence_strength = (rsi_old_high - rsi_recent_high) / 50.0
                        reward = 1.0 * min(divergence_strength, 1.0)
                        info['bearish_divergence'] = True
                        info['divergence_strength'] = divergence_strength

            return reward, info

        except Exception as e:
            self.logger.error(f"Erro em RSI divergence: {e}")
            return 0.0, {}

    def _validate_structural_confirmation(self, env, entry_decision: int) -> Tuple[float, Dict]:
        """
        🎯 LAYER 3: STRUCTURAL CONFIRMATION (20%)
        Valida estrutura de mercado usando features do dataframe
        """
        reward = 0.0
        info = {}

        try:
            df = getattr(env, 'df', None)
            current_step = getattr(env, 'current_step', 0)

            if df is None or current_step >= len(df):
                return 0.0, {}

            # 3.1 BREAKOUT STRENGTH (40% da Layer 3)
            if 'breakout_strength' in df.columns:
                breakout = df['breakout_strength'].iloc[current_step]
                if breakout > 0.65:
                    reward += 0.8
                    info['sr_proximity_good'] = True
                elif breakout < 0.35:
                    reward -= 0.4
                    info['sr_proximity_bad'] = True

            # 3.2 SUPPORT/RESISTANCE QUALITY (30% da Layer 3)
            if 'support_resistance' in df.columns:
                sr_quality = df['support_resistance'].iloc[current_step]
                if sr_quality > 0.6:
                    reward += 0.6
                    info['sl_zone_safe'] = True
                elif sr_quality < 0.4:
                    reward -= 0.6
                    info['sl_zone_dangerous'] = True

            # 3.3 PRICE POSITION (15% da Layer 3)
            if 'price_position' in df.columns:
                price_pos = df['price_position'].iloc[current_step]

                if entry_decision == 1:  # LONG
                    if price_pos < 0.35:
                        reward += 0.3
                        info['price_at_support'] = True
                    elif price_pos > 0.75:
                        reward -= 0.3
                        info['buying_high'] = True

                elif entry_decision == 2:  # SHORT
                    if price_pos > 0.65:
                        reward += 0.3
                        info['price_at_resistance'] = True
                    elif price_pos < 0.25:
                        reward -= 0.3
                        info['selling_low'] = True

            # 3.4 VOLUME MOMENTUM (15% da Layer 3)
            if 'volume_momentum' in df.columns:
                vol_momentum = df['volume_momentum'].iloc[current_step]
                if vol_momentum > 0.6:
                    reward += 0.3
                    info['volume_surge'] = True
                elif vol_momentum < 0.3:
                    reward -= 0.2
                    info['volume_weak'] = True

            info['layer3_raw_score'] = reward

            # Normalizar para -1.0 a +1.0
            normalized_reward = np.tanh(reward)

            return normalized_reward, info

        except Exception as e:
            self.logger.error(f"Erro em structural validation: {e}")
            return 0.0, {}


class EntryTimingRewards:
    """
    Sistema de recompensas V2 para timing de entrada (EIGHTEEN)
    Integra-se ao V3 Brutal Reward System
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Pesos dos componentes (relativo ao shaping total)
        self.timing_quality_weight = 0.50  # 50% do Entry Timing
        self.confluence_weight = 0.30       # 30% do Entry Timing
        self.market_context_weight = 0.20   # 20% do Entry Timing

        # 🎯 NOVO: Multi-Signal Confluence System
        self.multi_signal_system = MultiSignalConfluenceEntry()

        # 🎯 NOVO: Tracking para behavioral controls
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.last_trades = deque(maxlen=10)

        # Cache para performance
        self.step_counter = 0

    def calculate_entry_timing_rewards(self, env, entry_decision: int, action: np.ndarray) -> Tuple[float, Dict]:
        """
        Calcula recompensas de timing de entrada V2 (EIGHTEEN)

        Args:
            env: Environment do trading
            entry_decision: 0=HOLD, 1=LONG, 2=SHORT
            action: Array de ações do modelo

        Returns:
            total_reward: Recompensa total
            info: Dicionário com detalhes
        """
        self.step_counter += 1

        # Só calcular se houver entrada (BUY ou SELL)
        if entry_decision not in [1, 2]:
            return 0.0, {'entry_timing_active': False}

        try:
            # Atualizar tracking de trades
            self._update_trade_tracking(env)

            # Componente 1: Entry Timing Quality (50%)
            timing_quality, timing_info = self._calculate_entry_timing_quality(env, entry_decision, action)

            # Componente 2: Entry Confluence (30%)
            confluence_reward, confluence_info = self._calculate_entry_confluence_reward(env, entry_decision, action)

            # Componente 3: Market Context (20%)
            market_context, context_info = self._calculate_market_context_reward(env, entry_decision)

            # Total (pesos já aplicados dentro de cada função)
            total_reward = timing_quality + confluence_reward + market_context

            # Info completo
            info = {
                'entry_timing_active': True,
                'total_entry_timing_reward': total_reward,
                'timing_quality_reward': timing_quality,
                'confluence_reward': confluence_reward,
                'market_context_reward': market_context,
                **timing_info,
                **confluence_info,
                **context_info
            }

            return total_reward, info

        except Exception as e:
            self.logger.error(f"Erro no cálculo entry timing rewards: {e}")
            return 0.0, {'error': str(e)}

    def _update_trade_tracking(self, env):
        """Atualiza tracking de wins/losses consecutivas"""
        try:
            trades = getattr(env, 'trades', [])
            if not trades:
                return

            # Verificar se há novo trade
            if len(trades) > len(self.last_trades):
                last_trade = trades[-1]
                last_pnl = last_trade.get('pnl_usd', 0.0)

                # Atualizar consecutivas
                if last_pnl > 0:
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0

                # Adicionar ao histórico
                self.last_trades.append(last_trade)

        except Exception as e:
            self.logger.error(f"Erro em trade tracking: {e}")

    def _calculate_entry_timing_quality(self, env, entry_decision: int, action: np.ndarray) -> Tuple[float, Dict]:
        """
        COMPONENTE 1: Entry Timing Quality (50% do Entry Timing)
        """
        try:
            reward = 0.0
            info = {}

            intelligent_components = getattr(env, '_cached_intelligent_components', None)
            if not intelligent_components:
                return 0.0, {'error': 'No intelligent components'}

            market_regime = intelligent_components.get('market_regime', {})
            momentum_confluence = intelligent_components.get('momentum_confluence', {})
            volatility_context = intelligent_components.get('volatility_context', {})

            # 1.1 Market Context Alignment (30%)
            market_alignment_reward = self._calculate_market_alignment(
                entry_decision, market_regime, momentum_confluence
            )
            reward += market_alignment_reward * 0.30
            info['market_alignment_reward'] = market_alignment_reward

            # 1.2 Volatility Timing (20%)
            volatility_timing_reward = self._calculate_volatility_timing(
                entry_decision, volatility_context, momentum_confluence
            )
            reward += volatility_timing_reward * 0.20
            info['volatility_timing_reward'] = volatility_timing_reward

            # 1.3 Momentum Confluence (20%)
            momentum_reward = self._calculate_momentum_timing(
                env, entry_decision, momentum_confluence, market_regime
            )
            reward += momentum_reward * 0.20
            info['momentum_timing_reward'] = momentum_reward

            # 1.4 🎯 NOVO: Revenge Trading Penalty (15%)
            revenge_penalty, revenge_info = self._calculate_revenge_trading_penalty(env, entry_decision)
            reward += revenge_penalty * 0.15
            info.update(revenge_info)

            # 1.5 🎯 NOVO: Cut Loss Incentive (15%)
            cut_loss_reward, cut_loss_info = self._calculate_cut_loss_incentive(env, action)
            reward += cut_loss_reward * 0.15
            info.update(cut_loss_info)

            # Aplicar peso do componente (50% do Entry Timing)
            final_reward = reward * self.timing_quality_weight

            return final_reward, info

        except Exception as e:
            self.logger.error(f"Erro em timing quality: {e}")
            return 0.0, {'error': str(e)}

    def _calculate_market_alignment(self, entry_decision: int, market_regime: Dict,
                                   momentum_confluence: Dict) -> float:
        """Market Context Alignment"""
        reward = 0.0

        regime = market_regime.get('regime', 'unknown')
        regime_strength = market_regime.get('strength', 0.0)
        momentum_direction = momentum_confluence.get('direction', 0.0)
        momentum_strength = momentum_confluence.get('strength', 0.0)

        # LONG entries
        if entry_decision == 1:
            if regime == 'trending_up' and momentum_direction > 0.3:
                reward += 0.3 * momentum_strength
            if regime == 'trending_down':
                reward -= 0.5 * regime_strength
            if regime == 'crash':
                reward -= 1.0
            if regime == 'ranging':
                reward -= 0.1

        # SHORT entries
        elif entry_decision == 2:
            if regime == 'trending_down' and momentum_direction < -0.3:
                reward += 0.3 * momentum_strength
            if regime == 'trending_up':
                reward -= 0.5 * regime_strength
            if regime == 'ranging':
                reward -= 0.1

        return reward

    def _calculate_volatility_timing(self, entry_decision: int, volatility_context: Dict,
                                    momentum_confluence: Dict) -> float:
        """Volatility Timing"""
        reward = 0.0

        vol_level = volatility_context.get('level', 'normal')
        vol_percentile = volatility_context.get('percentile', 0.5)
        vol_expanding = volatility_context.get('expanding', False)
        momentum_direction = momentum_confluence.get('direction', 0.0)

        entry_direction = 1 if entry_decision == 1 else -1

        if vol_level == 'normal':
            reward += 0.2

        if vol_level == 'high' and vol_percentile > 0.9:
            reward -= 0.3

        if vol_level == 'low' and vol_percentile < 0.1:
            reward -= 0.2

        if vol_expanding and (momentum_direction * entry_direction > 0):
            reward += 0.15

        return reward

    def _calculate_momentum_timing(self, env, entry_decision: int,
                                  momentum_confluence: Dict, market_regime: Dict) -> float:
        """Momentum Confluence"""
        reward = 0.0

        momentum_score = momentum_confluence.get('score', 0.0)
        momentum_strength = momentum_confluence.get('strength', 0.0)
        regime_direction = market_regime.get('direction', 0.0)

        try:
            current_step = env.current_step
            rsi = env.df['rsi_14_1m'].iloc[current_step] if 'rsi_14_1m' in env.df.columns else 50.0
        except:
            rsi = 50.0

        if momentum_score > 0.7:
            reward += 0.4 * momentum_strength

        if momentum_score < 0.3:
            reward -= 0.3

        if entry_decision == 1 and rsi < 35 and regime_direction > 0:
            reward += 0.25

        if entry_decision == 2 and rsi > 65 and regime_direction < 0:
            reward += 0.25

        return reward

    def _calculate_revenge_trading_penalty(self, env, entry_decision: int) -> Tuple[float, Dict]:
        """
        🎯 NOVO: Penalty por revenge trading
        """
        reward = 0.0
        info = {}

        if self.consecutive_losses >= 1:
            # Penalty escalante: -0.3, -0.6, -0.9...
            penalty = -0.3 * self.consecutive_losses
            reward = penalty
            info['revenge_trading_penalty'] = penalty
            info['consecutive_losses'] = self.consecutive_losses

        return reward, info

    def _calculate_cut_loss_incentive(self, env, action: np.ndarray) -> Tuple[float, Dict]:
        """
        🎯 NOVO: Incentivo para corte rápido de perdas
        """
        reward = 0.0
        info = {}

        positions = getattr(env, 'positions', [])
        if not positions:
            return 0.0, {}

        entry_decision = action[0] if len(action) > 0 else 0

        for pos in positions:
            unrealized_pnl = pos.get('unrealized_pnl', 0.0)
            position_age = pos.get('age', 0)

            # Posição em perda >0.5% do balance
            if unrealized_pnl < -5.0:  # Aproximadamente -0.5% de $1000
                # Detectar se está tentando fechar
                if entry_decision < 0.33:
                    if position_age < 30:
                        reward += 0.5  # BONUS MASSIVO
                        info['quick_cut_loss'] = True
                    elif position_age < 60:
                        reward += 0.25
                        info['cut_loss'] = True
                elif position_age > 60:
                    reward -= 0.3
                    info['holding_loser'] = True

        return reward, info

    def _calculate_entry_confluence_reward(self, env, entry_decision: int, action: np.ndarray) -> Tuple[float, Dict]:
        """
        COMPONENTE 2: Entry Confluence Reward (30% do Entry Timing)
        """
        try:
            reward = 0.0
            info = {}

            # 2.1 🎯 NOVO: Multi-Signal Confluence (70%)
            multi_signal_reward, multi_signal_info = self.multi_signal_system.calculate_multi_signal_entry_reward(
                env, entry_decision, action
            )
            reward += multi_signal_reward * 0.70
            info.update(multi_signal_info)

            # 2.2 🎯 NOVO: Entry Timing After Loss (15%)
            timing_after_loss_reward, timing_info = self._calculate_entry_timing_after_loss(env, entry_decision)
            reward += timing_after_loss_reward * 0.15
            info.update(timing_info)

            # 2.3 🎯 NOVO: Pattern Recognition (15%)
            pattern_reward, pattern_info = self._calculate_pattern_recognition(env, entry_decision)
            reward += pattern_reward * 0.15
            info.update(pattern_info)

            # Aplicar peso do componente (30% do Entry Timing)
            final_reward = reward * self.confluence_weight

            return final_reward, info

        except Exception as e:
            self.logger.error(f"Erro em confluence reward: {e}")
            return 0.0, {'error': str(e)}

    def _calculate_entry_timing_after_loss(self, env, entry_decision: int) -> Tuple[float, Dict]:
        """
        🎯 NOVO: Penalty por entrar muito rápido após fechar posição
        """
        reward = 0.0
        info = {}

        try:
            trades = getattr(env, 'trades', [])
            positions = getattr(env, 'positions', [])

            if not positions and trades:
                last_trade = trades[-1]
                last_pnl = last_trade.get('pnl_usd', 0.0)
                last_close_step = last_trade.get('close_step', 0)
                current_step = getattr(env, 'current_step', 0)

                steps_since_close = current_step - last_close_step

                if entry_decision in [1, 2]:
                    # Entrada IMEDIATA após fechar
                    if steps_since_close < 5:
                        penalty = -0.8
                        reward = penalty
                        info['always_in_market_penalty'] = penalty
                        info['steps_since_close'] = steps_since_close

                    # Entrada rápida após PERDA
                    elif last_pnl < 0 and steps_since_close < 10:
                        penalty = -0.5
                        reward = penalty
                        info['quick_reentry_after_loss'] = penalty

                    # Esperou tempo adequado
                    elif 10 <= steps_since_close <= 30:
                        bonus = 0.2
                        reward = bonus
                        info['patient_reentry'] = bonus

            return reward, info

        except Exception as e:
            self.logger.error(f"Erro em entry timing after loss: {e}")
            return 0.0, {}

    def _calculate_pattern_recognition(self, env, entry_decision: int) -> Tuple[float, Dict]:
        """
        🎯 NOVO: Reward por entrar em padrões técnicos
        """
        reward = 0.0
        info = {}

        try:
            df = getattr(env, 'df', None)
            if df is None or len(df) < 60:
                return 0.0, {}

            close = df['close'].values if 'close' in df.columns else None
            high = df['high'].values if 'high' in df.columns else None
            low = df['low'].values if 'low' in df.columns else None

            if close is None:
                return 0.0, {}

            # MA CROSS (20 vs 50)
            if len(close) >= 51:
                sma20_now = np.mean(close[-20:])
                sma50_now = np.mean(close[-50:])
                sma20_prev = np.mean(close[-21:-1])
                sma50_prev = np.mean(close[-51:-1])

                # LONG: golden cross
                if entry_decision == 1 and sma20_prev <= sma50_prev and sma20_now > sma50_now:
                    reward += 0.4
                    info['ma_cross_long'] = True

                # SHORT: death cross
                elif entry_decision == 2 and sma20_prev >= sma50_prev and sma20_now < sma50_now:
                    reward += 0.4
                    info['ma_cross_short'] = True

            # DOUBLE BOTTOM/TOP
            if high is not None and low is not None and len(high) >= 25:
                last_low = np.min(low[-5:])
                prev_low = np.min(low[-20:-5])
                last_high = np.max(high[-5:])
                prev_high = np.max(high[-20:-5])

                price_ref = close[-1]
                tolerance = 0.0015 * price_ref

                # LONG: double bottom
                if entry_decision == 1 and abs(last_low - prev_low) <= tolerance and close[-1] > last_low:
                    reward += 0.3
                    info['double_bottom'] = True

                # SHORT: double top
                elif entry_decision == 2 and abs(last_high - prev_high) <= tolerance and close[-1] < last_high:
                    reward += 0.3
                    info['double_top'] = True

            return reward, info

        except Exception as e:
            self.logger.error(f"Erro em pattern recognition: {e}")
            return 0.0, {}

    def _calculate_market_context_reward(self, env, entry_decision: int) -> Tuple[float, Dict]:
        """
        COMPONENTE 3: Market Context Reward (20% do Entry Timing)
        NOTA: Rewards de horário foram REMOVIDAS (robô já faz filtro)
        """
        try:
            reward = 0.0
            info = {}

            # 3.1 Position Context (100% - único componente agora)
            position_context_reward = self._calculate_position_context(env, entry_decision)
            reward += position_context_reward * 1.0
            info['position_context_reward'] = position_context_reward

            # Aplicar peso do componente (20% do Entry Timing)
            final_reward = reward * self.market_context_weight

            return final_reward, info

        except Exception as e:
            self.logger.error(f"Erro em market context: {e}")
            return 0.0, {'error': str(e)}

    def _calculate_position_context(self, env, entry_decision: int) -> float:
        """Intraday Position Context"""
        reward = 0.0

        positions = getattr(env, 'positions', [])
        entry_type = 'long' if entry_decision == 1 else 'short'

        # Primeira entrada do episódio
        if len(positions) == 0:
            reward += 0.2

        # Segunda entrada para hedge
        if len(positions) == 1:
            existing_type = positions[0].get('type', 'long')
            if existing_type != entry_type:
                reward += 0.15

        return reward


# Exportar classes principais
__all__ = ['EntryTimingRewards', 'MultiSignalConfluenceEntry']
