"""
🎯 SELECTIVE TRADING REWARD SYSTEM V4-SELECTIVE
Sistema de reward otimizado que ENSINA SELETIVIDADE E PACIÊNCIA
Baseado no V4 Inno com componentes anti-overtrading

🔧 PRINCIPAIS MODIFICAÇÕES:
- PENALIZAÇÃO por overtrading (>50% tempo posicionado)
- REWARD por paciência em condições ruins
- BONUS por qualidade de entrada
- PENALIZAÇÃO específica para "always in market"
- Sistema de cooldown adaptativo integrado

📊 DISTRIBUIÇÃO DO REWARD:
- PnL = 60% (reduzido de 70% para dar espaço aos novos componentes)
- Shaping inteligente = 15%
- Seletividade = 15% (NOVO - anti-overtrading)
- Qualidade de entrada = 10% (NOVO - ensina timing)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from collections import deque

class SelectiveTradingReward:
    """
    Sistema de reward focado em SELETIVIDADE e QUALIDADE
    Ensina o modelo a NÃO tradear quando não há oportunidades
    """
    
    def __init__(self, initial_balance: float = 1000.0):
        self.initial_balance = initial_balance
        self.logger = logging.getLogger(__name__)
        
        # Configurações V4-SELECTIVE
        self.pain_multiplier = 1.5
        self.risk_penalty_threshold = 0.15
        self.max_reward = 2.0
        
        # 🎯 NOVOS PARÂMETROS ANTI-OVERTRADING
        self.max_position_time_ratio = 0.5  # Máximo 50% do tempo posicionado
        self.optimal_position_time_ratio = 0.25  # Ótimo: 25% do tempo
        self.min_position_time_ratio = 0.05  # Mínimo: 5% do tempo (evitar inatividade total)
        
        # Thresholds de qualidade de entrada
        self.min_volatility_for_entry = 0.0003  # Volatilidade mínima para entrada ser considerada boa
        self.min_trend_strength = 0.001  # Força de tendência mínima
        
        # Tracking original do V4
        self.last_portfolio_value = initial_balance
        self.position_performance = {}
        self.recent_pnl_trend = deque(maxlen=10)
        
        # NOVA: Buffer para normalização adaptativa
        self.reward_buffer = deque(maxlen=1000)
        self.normalization_active = False
        
        # Activity tracking
        self.last_significant_action_step = 0
        
        # 🎯 NOVO TRACKING ANTI-OVERTRADING
        self.steps_in_position = 0
        self.total_steps = 0
        self.trade_quality_history = deque(maxlen=20)  # Histórico de qualidade dos trades
        self.patience_counter = 0  # Contador de paciência (steps sem posição em condições ruins)
        self.last_trade_result = None  # 'win', 'loss', ou None
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        
        # Cache de performance
        self.step_counter = 0
        self.cached_metrics = {}
        
        # Métricas para debugging
        self.total_rewards_given = 0
        self.positive_rewards = 0
        self.negative_rewards = 0
        self.overtrading_penalties_applied = 0
        self.patience_rewards_given = 0
        self.quality_bonuses_given = 0
        
    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict, bool]:
        """
        Calcula reward V4-SELECTIVE:
        60% PnL + 15% Shaping + 15% Seletividade + 10% Qualidade
        
        Action structure:
        [0] entry_decision: [0,2] → 0=HOLD, 1=LONG, 2=SHORT
        [1] entry_confidence: [0,1] → Confidence level
        [2] pos1_mgmt: [-1,1] → Position 1 management
        [3] pos2_mgmt: [-1,1] → Position 2 management
        """
        
        self.step_counter += 1
        self.total_steps += 1
        
        # 🎯 TRACKING DE POSIÇÃO (para calcular tempo posicionado)
        has_position = len(getattr(env, 'positions', [])) > 0
        if has_position:
            self.steps_in_position += 1
        
        # 🔧 DETECTAR SE ESTÁ EM COOLDOWN FORÇADO
        is_in_cooldown = getattr(env, 'cooldown_counter', 0) > 0
        
        # 1. PNL REWARD (60% do reward)
        pnl_reward, pnl_info = self._calculate_pure_pnl_reward(env)
        
        # 2. SHAPING INTELIGENTE (15% do reward)
        if self.step_counter % 5 == 0:
            shaping_reward, shaping_info = self._calculate_intelligent_shaping(env, action)
            self.cached_metrics['shaping'] = (shaping_reward, shaping_info)
        else:
            shaping_reward, shaping_info = self.cached_metrics.get('shaping', (0.0, {}))
        
        # 3. 🎯 SELETIVIDADE REWARD (15% do reward) - NOVO!
        selectivity_reward, selectivity_info = self._calculate_selectivity_reward(env, action, has_position, is_in_cooldown)
        
        # 4. 🎯 QUALIDADE DE ENTRADA (10% do reward) - NOVO!
        quality_reward, quality_info = self._calculate_entry_quality_reward(env, action, old_state)
        
        # 5. 🎯 MANAGEMENT QUALITY (bonus adicional) - NOVO!
        mgmt_reward, mgmt_info = self._calculate_management_quality_reward(env, action, has_position)
        
        # 5.1 🚨 ACTIVITY BONUS - RESTAURADO (essencial para operação)
        activity_reward, activity_info = self._calculate_activity_bonus(env, action)
        
        # 5.2 🎯 TRAILING STOP QUALITY - NOVO! (ensina management head)
        trailing_reward, trailing_info = self._calculate_trailing_stop_reward(env, action)
        
        # 6. REWARD FINAL V4-SELECTIVE (com management e activity bonus)
        pure_pnl_component = pnl_reward * 0.60
        shaping_component = shaping_reward * 0.15
        selectivity_component = selectivity_reward * 0.15
        quality_component = quality_reward * 0.10
        
        raw_total = pure_pnl_component + shaping_component + selectivity_component + quality_component + mgmt_reward + activity_reward + trailing_reward
        
        # 6.1 NORMALIZAÇÃO ADAPTATIVA
        total_reward = self._adaptive_normalize(raw_total)
        
        # 6. Early termination apenas em casos extremos
        done = False  # Nunca terminar por reward
        
        # 7. Clipping final conservador
        total_reward = np.clip(total_reward, -1.5, 1.5)
        
        # Safeguard numérico
        if not np.isfinite(total_reward) or abs(total_reward) > 100:
            total_reward = 0.0
        
        # 8. Update tracking
        if self.step_counter % 5 == 0:
            self._update_tracking(env)
        
        # 9. Update stats
        if self.step_counter % 10 == 0:
            self._update_stats(total_reward)
        
        # 10. Info detalhado
        info = {
            'pnl_reward': pnl_reward,
            'shaping_reward': shaping_reward,
            'selectivity_reward': selectivity_reward,
            'quality_reward': quality_reward,
            'mgmt_reward': mgmt_reward,
            'activity_reward': activity_reward,
            'trailing_reward': trailing_reward,
            'total_reward': total_reward,
            'pure_pnl_component': pure_pnl_component,
            'shaping_component': shaping_component,
            'selectivity_component': selectivity_component,
            'quality_component': quality_component,
            'position_time_ratio': self.steps_in_position / max(1, self.total_steps),
            'is_in_cooldown': is_in_cooldown,
            'v4_selective_mode': True,
            **pnl_info,
            **shaping_info,
            **selectivity_info,
            **quality_info,
            **mgmt_info,
            **activity_info,
            **trailing_info
        }
        
        return total_reward, info, done
    
    def _calculate_selectivity_reward(self, env, action: np.ndarray, has_position: bool, is_in_cooldown: bool) -> Tuple[float, Dict]:
        """
        🎯 NOVO: Reward por seletividade e anti-overtrading
        """
        try:
            reward = 0.0
            info = {}
            
            # Calcular ratio de tempo posicionado
            position_time_ratio = self.steps_in_position / max(1, self.total_steps)
            info['position_time_ratio'] = position_time_ratio
            
            # 🔧 NÃO PENALIZAR DURANTE COOLDOWN FORÇADO
            if not is_in_cooldown:
                # 1. PENALIZAÇÃO POR OVERTRADING (só quando não está em cooldown)
                if position_time_ratio > self.max_position_time_ratio:
                    # Penalização progressiva por estar muito tempo posicionado
                    excess = position_time_ratio - self.max_position_time_ratio
                    penalty = -1.0 * excess * 2.0  # Penalização forte
                    reward += penalty
                    info['overtrading_penalty'] = penalty
                    self.overtrading_penalties_applied += 1
                    
                # 2. REWARD POR RATIO ÓTIMO
                elif self.optimal_position_time_ratio * 0.8 <= position_time_ratio <= self.optimal_position_time_ratio * 1.2:
                    # Bonus por estar no range ótimo (20-30% do tempo)
                    bonus = 0.3
                    reward += bonus
                    info['optimal_ratio_bonus'] = bonus
                    
                # 3. PENALIZAÇÃO POR INATIVIDADE EXTREMA (só se não está em cooldown)
                elif position_time_ratio < self.min_position_time_ratio and self.total_steps > 100:
                    # Penalização leve por nunca tradear
                    penalty = -0.1
                    reward += penalty
                    info['inactivity_penalty'] = penalty
            else:
                info['cooldown_active'] = True  # Sinalizar que cooldown está ativo
            
            # 4. REWARD POR PACIÊNCIA EM CONDIÇÕES RUINS
            if not has_position:
                # Verificar se as condições de mercado são ruins
                volatility = self._get_market_volatility(env)
                trend_strength = self._get_trend_strength(env)
                
                if volatility < self.min_volatility_for_entry or trend_strength < self.min_trend_strength:
                    # Condições ruins - recompensar por NÃO tradear
                    self.patience_counter += 1
                    if self.patience_counter > 10:  # Após 10 steps de paciência
                        patience_reward = 0.1  # Pequeno reward constante
                        reward += patience_reward
                        info['patience_reward'] = patience_reward
                        self.patience_rewards_given += 1
                else:
                    # Condições boas mas não está tradeando - reset counter
                    self.patience_counter = 0
            else:
                self.patience_counter = 0
            
            # 5. BONUS POR SAIR DE POSIÇÕES PERDEDORAS RAPIDAMENTE
            if has_position:
                positions = getattr(env, 'positions', [])
                for pos in positions:
                    unrealized_pnl = pos.get('unrealized_pnl', 0.0)
                    position_age = pos.get('age', 0)
                    
                    # Se está perdendo e sai rápido, pequeno bonus
                    if unrealized_pnl < -0.01 * self.initial_balance and position_age < 20:
                        if action[0] < 0.33:  # Ação de fechar (HOLD/CLOSE)
                            cut_loss_bonus = 0.05
                            reward += cut_loss_bonus
                            info['cut_loss_bonus'] = cut_loss_bonus
            
            # 6. PENALIZAÇÃO POR "ALWAYS IN MARKET" BEHAVIOR (só se não está em cooldown)
            # Detectar se está abrindo nova posição logo após fechar
            if not is_in_cooldown and not has_position and self.last_trade_result == 'recent_close':
                if action[0] > 0.33:  # Tentando abrir nova posição
                    always_in_penalty = -0.2
                    reward += always_in_penalty
                    info['always_in_market_penalty'] = always_in_penalty
            
            info['selectivity_total'] = reward
            return reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no selectivity reward: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_entry_quality_reward(self, env, action: np.ndarray, old_state: Dict) -> Tuple[float, Dict]:
        """
        🎯 NOVO: Reward baseado na QUALIDADE da entrada
        Usa entry_confidence (action[1]) para avaliar qualidade
        """
        try:
            reward = 0.0
            info = {}
            
            # Extrair entry_decision e entry_confidence do action
            entry_decision = action[0] if len(action) > 0 else 0
            entry_confidence = action[1] if len(action) > 1 else 0.5
            
            # Detectar se abriu nova posição
            old_positions = old_state.get('positions', []) if old_state else []
            current_positions = getattr(env, 'positions', [])
            
            opened_new_position = len(current_positions) > len(old_positions)
            
            if opened_new_position and entry_decision > 0:  # Ação de entrada (LONG ou SHORT)
                # Avaliar qualidade da entrada
                volatility = self._get_market_volatility(env)
                trend_strength = self._get_trend_strength(env)
                spread = self._get_spread(env)
                
                quality_score = 0.0
                market_good = False
                
                # 1. Volatilidade adequada (nem muito baixa nem muito alta)
                if self.min_volatility_for_entry <= volatility <= self.min_volatility_for_entry * 10:
                    quality_score += 0.2
                    info['good_volatility'] = True
                    market_good = True
                elif volatility > self.min_volatility_for_entry * 10:
                    quality_score -= 0.1  # Volatilidade muito alta é ruim
                    info['excessive_volatility'] = True
                else:
                    quality_score -= 0.2  # Volatilidade muito baixa é pior
                    info['low_volatility'] = True
                
                # 2. Tendência clara
                if trend_strength > self.min_trend_strength:
                    quality_score += 0.2
                    info['clear_trend'] = True
                    market_good = market_good and True
                else:
                    quality_score -= 0.1
                    info['weak_trend'] = True
                    market_good = False
                
                # 3. Spread aceitável
                if spread < 0.0001:  # Spread baixo
                    quality_score += 0.1
                    info['good_spread'] = True
                else:
                    quality_score -= 0.1
                    info['bad_spread'] = True
                
                # 🎯 4. CONFIDENCE APPROPRIATENESS - NOVO!
                # Recompensar confidence apropriada ao mercado
                if market_good and entry_confidence > 0.7:
                    # Alta confiança em mercado bom = ÓTIMO
                    confidence_bonus = 0.3
                    quality_score += confidence_bonus
                    info['high_confidence_good_market'] = True
                elif not market_good and entry_confidence < 0.4:
                    # Baixa confiança em mercado ruim = BOM (reconheceu incerteza)
                    caution_bonus = 0.2
                    quality_score += caution_bonus
                    info['appropriate_caution'] = True
                elif market_good and entry_confidence < 0.4:
                    # Baixa confiança em mercado bom = PERDEU OPORTUNIDADE
                    missed_penalty = -0.2
                    quality_score += missed_penalty
                    info['missed_opportunity'] = True
                elif not market_good and entry_confidence > 0.7:
                    # Alta confiança em mercado ruim = PERIGOSO
                    overconfidence_penalty = -0.3
                    quality_score += overconfidence_penalty
                    info['overconfidence_penalty'] = True
                
                # 5. Padrões de entrada (MA cross, topo/fundo duplo) – SHAPING LEVE
                pattern_bonus = 0.0
                pattern_info = {}
                try:
                    # Obter séries recentes de close/high/low (fallback para *_5m)
                    df = getattr(env, 'df', None)
                    if df is not None and len(df) > 60:
                        close = None
                        high = None
                        low = None
                        for name in ['close', 'close_5m']:
                            if name in df.columns:
                                close = df[name].values
                                break
                        for name in ['high', 'high_5m']:
                            if name in df.columns:
                                high = df[name].values
                                break
                        for name in ['low', 'low_5m']:
                            if name in df.columns:
                                low = df[name].values
                                break
                        if close is not None:
                            import numpy as _np
                            # MA cross (20 vs 50)
                            if len(close) >= 51:
                                def _sma(arr, n):
                                    return _np.mean(arr[-n:]) if len(arr) >= n else _np.mean(arr)
                                sma20_now = _sma(close, 20)
                                sma50_now = _sma(close, 50)
                                sma20_prev = _sma(close[:-1], 20)
                                sma50_prev = _sma(close[:-1], 50)
                                # Long: cruzamento para cima recente
                                if entry_decision == 1 and sma20_prev <= sma50_prev and sma20_now > sma50_now:
                                    pattern_bonus += 0.08
                                    pattern_info['ma_cross_long'] = True
                                # Short: cruzamento para baixo recente
                                if entry_decision == 2 and sma20_prev >= sma50_prev and sma20_now < sma50_now:
                                    pattern_bonus += 0.08
                                    pattern_info['ma_cross_short'] = True
                            # Topo/ Fundo duplo simples (janela 20)
                            if high is not None and low is not None and len(high) >= 25:
                                win = 20
                                c_slice = close[-win:]
                                h_slice = high[-win:]
                                l_slice = low[-win:]
                                idx_last = len(c_slice) - 1
                                # Último low alto e um low anterior similar
                                last_low = _np.min(l_slice[-5:])
                                prev_low = _np.min(l_slice[:-5]) if len(l_slice) > 5 else _np.min(l_slice)
                                last_high = _np.max(h_slice[-5:])
                                prev_high = _np.max(h_slice[:-5]) if len(h_slice) > 5 else _np.max(h_slice)
                                price_ref = close[-1]
                                tol = max(1e-6, 0.0015 * price_ref)  # ~0.15%
                                # Fundo duplo para long
                                if entry_decision == 1 and abs(last_low - prev_low) <= tol and close[-1] > last_low:
                                    pattern_bonus += 0.06
                                    pattern_info['double_bottom'] = True
                                # Topo duplo para short
                                if entry_decision == 2 and abs(last_high - prev_high) <= tol and close[-1] < last_high:
                                    pattern_bonus += 0.06
                                    pattern_info['double_top'] = True
                            # Removidos sinais adicionais (slope, momentum, breakout, squeeze, mean-revert, candle)
                except Exception:
                    pass

                quality_score += pattern_bonus
                info.update(pattern_info)

                # 6. Timing baseado em perdas consecutivas
                if self.consecutive_losses > 2:
                    # Penalizar entrada após múltiplas perdas (revenge trading)
                    quality_score -= 0.3
                    info['revenge_trading_penalty'] = True
                
                info['entry_confidence'] = entry_confidence
                info['market_quality'] = 'good' if market_good else 'bad'
                
                # Converter score em reward
                reward = quality_score * 0.5  # Escalar para range apropriado
                
                # Adicionar ao histórico
                self.trade_quality_history.append(quality_score)
                
                if quality_score > 0:
                    self.quality_bonuses_given += 1
                
                info['quality_score'] = quality_score
                info['opened_position'] = True
            else:
                info['opened_position'] = False
            
            info['quality_total'] = reward
            return reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no quality reward: {e}")
            return 0.0, {'error': str(e)}
    
    def _get_market_volatility(self, env) -> float:
        """Calcula volatilidade atual do mercado"""
        try:
            # Pegar dados recentes do environment
            if hasattr(env, 'df') and len(env.df) > 20:
                recent_prices = env.df['close'].iloc[-20:].values
                returns = np.diff(recent_prices) / recent_prices[:-1]
                volatility = np.std(returns)
                return volatility
            return 0.001  # Default
        except:
            return 0.001
    
    def _get_trend_strength(self, env) -> float:
        """Calcula força da tendência atual"""
        try:
            if hasattr(env, 'df') and len(env.df) > 50:
                prices = env.df['close'].iloc[-50:].values
                # Calcular tendência simples (slope da regressão linear)
                x = np.arange(len(prices))
                z = np.polyfit(x, prices, 1)
                slope = z[0]
                # Normalizar pelo preço médio
                trend_strength = abs(slope) / np.mean(prices)
                return trend_strength
            return 0.0
        except:
            return 0.0
    
    def _calculate_management_quality_reward(self, env, action: np.ndarray, has_position: bool) -> Tuple[float, Dict]:
        """
        🎯 NOVO: Reward por qualidade de management das posições
        Avalia pos1_mgmt (action[2]) e pos2_mgmt (action[3])
        """
        try:
            reward = 0.0
            info = {}
            
            if not has_position:
                info['no_positions'] = True
                return 0.0, info
            
            # Extrair management actions
            pos1_mgmt = action[2] if len(action) > 2 else 0.0
            pos2_mgmt = action[3] if len(action) > 3 else 0.0
            
            positions = getattr(env, 'positions', [])
            
            for idx, pos in enumerate(positions[:2]):  # Máximo 2 posições
                # Usar management action correspondente
                mgmt_value = pos1_mgmt if idx == 0 else pos2_mgmt
                
                # Calcular PnL da posição
                unrealized_pnl = pos.get('unrealized_pnl', 0.0)
                position_age = pos.get('age', 0)
                pnl_percent = unrealized_pnl / self.initial_balance if self.initial_balance > 0 else 0
                
                # 1. GESTÃO DE POSIÇÃO PERDEDORA
                if pnl_percent < -0.01:  # Perdendo >1%
                    if mgmt_value < -0.5:  # Foco em SL (proteção)
                        # BOM: Apertando stop em posição perdedora
                        protection_bonus = 0.05
                        reward += protection_bonus
                        info[f'pos{idx}_protection'] = True
                    elif mgmt_value > 0.5:  # Foco em TP
                        # RUIM: Aumentando target em posição perdedora
                        greed_penalty = -0.05
                        reward += greed_penalty
                        info[f'pos{idx}_greed'] = True
                
                # 2. GESTÃO DE POSIÇÃO VENCEDORA
                elif pnl_percent > 0.02:  # Ganhando >2%
                    if mgmt_value > 0.3:  # Foco em TP
                        # BOM: Deixando lucro correr
                        runner_bonus = 0.03
                        reward += runner_bonus
                        info[f'pos{idx}_runner'] = True
                    elif mgmt_value < -0.7:  # Muito foco em SL
                        # QUESTIONÁVEL: Muito conservador com lucro
                        conservative_penalty = -0.02
                        reward += conservative_penalty
                        info[f'pos{idx}_too_conservative'] = True
                
                # 3. GESTÃO ADAPTATIVA POR IDADE
                if position_age > 100:  # Posição antiga
                    if abs(mgmt_value) > 0.5:  # Ação decisiva
                        # BOM: Tomando decisão em posição estagnada
                        decisive_bonus = 0.02
                        reward += decisive_bonus
                        info[f'pos{idx}_decisive'] = True
                
                info[f'pos{idx}_mgmt'] = mgmt_value
                info[f'pos{idx}_pnl_pct'] = pnl_percent * 100
            
            info['mgmt_total'] = reward
            return reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no management reward: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_trailing_stop_reward(self, env, action: np.ndarray) -> Tuple[float, Dict]:
        """
        🎯 NOVO: Reward por qualidade de uso do trailing stop
        Ensina o modelo a usar trailing stop para proteger lucros
        """
        try:
            reward = 0.0
            info = {'trailing_analysis': {}}
            
            # Verificar se houve trades recentes
            trades = getattr(env, 'trades', [])
            if not trades:
                info['no_trades'] = True
                return 0.0, info
            
            last_trade = trades[-1]
            
            # 1. 🎯 REWARD POR ATIVAÇÃO DE TRAILING STOP
            if last_trade.get('trailing_activated', False):
                activation_bonus = 0.15  # Bonus moderado por ativar trailing
                reward += activation_bonus
                info['trailing_analysis']['activation_bonus'] = activation_bonus
                
                # Bonus adicional se foi ativado no timing certo
                if last_trade.get('trailing_timing', False):
                    timing_bonus = 0.10
                    reward += timing_bonus
                    info['trailing_analysis']['timing_bonus'] = timing_bonus
            
            # 2. 🎯 REWARD POR PROTEÇÃO DE LUCRO VIA TRAILING
            if last_trade.get('trailing_protected', False):
                protection_bonus = 0.20  # Bonus maior por proteger lucros
                reward += protection_bonus
                info['trailing_analysis']['protection_bonus'] = protection_bonus
                
                # Bonus extra se o trade teve exit_reason = trailing_stop
                if last_trade.get('exit_reason', '') == 'trailing_stop':
                    execution_bonus = 0.15
                    reward += execution_bonus
                    info['trailing_analysis']['execution_bonus'] = execution_bonus
            
            # 3. 🎯 REWARD POR MOVIMENTOS INTELIGENTES DO TRAILING
            trailing_moves = last_trade.get('trailing_moves', 0)
            if trailing_moves > 0:
                # Reward moderado por mover o trailing (máximo 3 moves)
                move_bonus = min(trailing_moves * 0.05, 0.15)
                reward += move_bonus
                info['trailing_analysis']['move_bonus'] = move_bonus
                info['trailing_analysis']['trailing_moves'] = trailing_moves
            
            # 4. 🎯 PENALIDADE POR OPORTUNIDADE PERDIDA
            if last_trade.get('missed_trailing_opportunity', False):
                missed_penalty = -0.10  # Penalidade moderada
                reward += missed_penalty
                info['trailing_analysis']['missed_penalty'] = missed_penalty
            
            # 5. 🎯 REWARD POR AJUSTES DE TP
            if last_trade.get('tp_adjusted', False):
                # Verificar se o ajuste foi inteligente
                pnl_usd = last_trade.get('pnl_usd', 0.0)
                if pnl_usd > 0:  # TP adjustment que resultou em lucro
                    tp_bonus = 0.08
                    reward += tp_bonus
                    info['trailing_analysis']['tp_bonus'] = tp_bonus
                elif pnl_usd < 0:  # TP adjustment que resultou em perda (sinal ruim)
                    tp_penalty = -0.05
                    reward += tp_penalty
                    info['trailing_analysis']['tp_penalty'] = tp_penalty
            
            # 6. 🎯 ANÁLISE DO MANAGEMENT ACTION ATUAL
            # Verificar se modelo está tentando gerenciar posições existentes
            positions = getattr(env, 'positions', [])
            if positions and len(action) >= 4:
                # Analisar management actions [2] e [3]
                mgmt_actions = action[2:4]
                active_mgmt = any(abs(a) > 0.5 for a in mgmt_actions)
                
                if active_mgmt:
                    # Small bonus for active management when in position
                    for i, pos in enumerate(positions[:2]):
                        unrealized_pnl = pos.get('unrealized_pnl', 0.0)
                        mgmt_action = mgmt_actions[i] if i < len(mgmt_actions) else 0.0
                        
                        # Se posição em lucro e modelo está gerenciando ativamente
                        if unrealized_pnl > 0 and abs(mgmt_action) > 0.8:
                            active_mgmt_bonus = 0.03
                            reward += active_mgmt_bonus
                            info['trailing_analysis'][f'active_mgmt_pos{i}'] = active_mgmt_bonus
            
            info['trailing_analysis']['total_trailing_reward'] = reward
            return reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no trailing stop reward: {e}")
            return 0.0, {'error': str(e)}
    
    def _get_spread(self, env) -> float:
        """Calcula spread atual"""
        try:
            if hasattr(env, 'df') and 'spread' in env.df.columns:
                return env.df['spread'].iloc[-1]
            return 0.0001  # Default
        except:
            return 0.0001
    
    def _calculate_pure_pnl_reward(self, env) -> Tuple[float, Dict]:
        """PnL reward do V4 original"""
        try:
            realized_pnl = getattr(env, 'total_realized_pnl', 0.0)
            unrealized_pnl = getattr(env, 'total_unrealized_pnl', 0.0)
            
            # V4: Valorizar unrealized PnL
            total_pnl = realized_pnl + (unrealized_pnl * 0.8)
            pnl_percent = total_pnl / self.initial_balance
            
            # Reward base amplificado
            pnl_percent_clipped = np.clip(pnl_percent, -0.15, 0.15)
            base_reward = pnl_percent_clipped * 3.0  # CORRIGIDO: Reduzido de 10.0 para 3.0 (estabilidade)
            
            # Pain multiplication para perdas
            if pnl_percent_clipped < -0.03:
                pain_factor = 1.0 + (self.pain_multiplier - 1.0) * np.tanh(abs(pnl_percent_clipped) * 20)
                base_reward *= pain_factor
            elif pnl_percent_clipped > 0.02:
                bonus_factor = 1.0 + 0.2 * np.tanh(pnl_percent_clipped * 50)
                base_reward *= bonus_factor
            
            # 🎯 TRACKING DE RESULTADO DO ÚLTIMO TRADE
            trades = getattr(env, 'trades', [])
            if trades and len(trades) > len(self.recent_pnl_trend):
                last_trade = trades[-1]
                last_pnl = last_trade.get('pnl_usd', 0.0)
                
                # Atualizar tracking de wins/losses
                if last_pnl > 0:
                    self.last_trade_result = 'win'
                    self.consecutive_wins += 1
                    self.consecutive_losses = 0
                else:
                    self.last_trade_result = 'loss'
                    self.consecutive_losses += 1
                    self.consecutive_wins = 0
                
                self.recent_pnl_trend.append(last_pnl)
            elif not getattr(env, 'positions', []):
                # Acabou de fechar posição
                self.last_trade_result = 'recent_close'
            
            info = {
                'realized_pnl': realized_pnl,
                'unrealized_pnl': unrealized_pnl,
                'total_pnl': total_pnl,
                'pnl_percent': pnl_percent * 100,
                'consecutive_losses': self.consecutive_losses,
                'consecutive_wins': self.consecutive_wins
            }
            
            return base_reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no PnL reward: {e}")
            return 0.0, {'error': str(e)}
    
    def _adaptive_normalize(self, reward: float) -> float:
        """
        🎯 NORMALIZAÇÃO ADAPTATIVA - Estabiliza reward baseado em histórico
        Previne explosão de reward mantendo distribuição estável
        """
        try:
            # Adicionar ao buffer
            self.reward_buffer.append(reward)
            
            # Normalização só ativa após coletar dados suficientes
            if len(self.reward_buffer) < 50:
                # Clipping conservador durante warmup
                return np.clip(reward, -2.0, 2.0)
            
            # Ativar normalização adaptativa
            if not self.normalization_active and len(self.reward_buffer) >= 100:
                self.normalization_active = True
                self.logger.info("🎯 V4 SELECTIVE: Normalização adaptativa ATIVADA")
            
            if self.normalization_active:
                # Calcular rolling statistics (últimos 500 rewards)
                recent_rewards = list(self.reward_buffer)[-500:]
                mean_reward = np.mean(recent_rewards)
                std_reward = np.std(recent_rewards) + 1e-6  # Evitar divisão por zero
                
                # Z-score normalização com clipping 3-sigma
                normalized = (reward - mean_reward) / (2.5 * std_reward)  # 2.5-sigma
                return np.clip(normalized, -1.0, 1.0)
            else:
                # Clipping conservador enquanto coleta dados
                return np.clip(reward, -2.0, 2.0)
                
        except Exception as e:
            self.logger.error(f"Erro na normalização adaptativa: {e}")
            return np.clip(reward, -1.0, 1.0)
    
    def _calculate_activity_bonus(self, env, action: np.ndarray) -> Tuple[float, Dict]:
        """
        🚨 ACTIVITY BONUS - RESTAURADO
        Incentiva o modelo a tomar ações para quebrar inércia de HOLD infinito
        """
        try:
            reward = 0.0
            info = {}
            
            # Calcular magnitude da ação
            action_magnitude = float(np.abs(action).max()) if len(action) > 0 else 0.0
            entry_decision = action[0] if len(action) > 0 else 0
            entry_confidence = action[1] if len(action) > 1 else 0.5
            
            # 1. BONUS BASE POR QUALQUER AÇÃO DECISIVA
            if action_magnitude > 0.5:
                decisive_bonus = 0.02  # Pequeno mas essencial
                reward += decisive_bonus
                info['decisive_action_bonus'] = decisive_bonus
            elif action_magnitude > 0.2:
                moderate_bonus = 0.01
                reward += moderate_bonus  
                info['moderate_action_bonus'] = moderate_bonus
            
            # 2. BONUS POR ENTRADA (quebra inércia de HOLD)
            if entry_decision > 0:  # LONG ou SHORT
                entry_bonus = 0.03 * entry_confidence  # Proporcional à confiança
                reward += entry_bonus
                info['entry_action_bonus'] = entry_bonus
            
            # 3. BONUS POR MANAGEMENT ATIVO
            if len(action) > 2:
                mgmt_actions = action[2:4] if len(action) >= 4 else action[2:]
                mgmt_magnitude = float(np.abs(mgmt_actions).max())
                
                if mgmt_magnitude > 0.3:
                    mgmt_bonus = 0.015
                    reward += mgmt_bonus
                    info['management_action_bonus'] = mgmt_bonus
            
            # 4. ANTI-INÉRCIA: Bonus por quebrar período de inatividade
            if hasattr(self, 'last_significant_action_step'):
                steps_since_action = self.total_steps - self.last_significant_action_step
                if steps_since_action > 20 and action_magnitude > 0.3:
                    inertia_break_bonus = min(0.05, steps_since_action * 0.001)
                    reward += inertia_break_bonus
                    info['inertia_break_bonus'] = inertia_break_bonus
            
            # Atualizar tracking se ação significativa
            if action_magnitude > 0.3:
                self.last_significant_action_step = self.total_steps
            
            info['activity_total'] = reward
            info['action_magnitude'] = action_magnitude
            
            return reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no activity bonus: {e}")
            return 0.0, {'error': str(e)}
    
    def _calculate_intelligent_shaping(self, env, action: np.ndarray) -> Tuple[float, Dict]:
        """Shaping do V4 original (simplificado)"""
        try:
            shaping_reward = 0.0
            info = {}
            
            # Portfolio progress
            current_portfolio = getattr(env, 'portfolio_value', self.initial_balance)
            progress = (current_portfolio - self.last_portfolio_value) / self.initial_balance
            
            if abs(progress) > 0.001:
                progress_reward = progress * 0.1 * 20.0
                shaping_reward += progress_reward
                info['progress_reward'] = progress_reward
            
            # Momentum dos últimos trades
            if len(self.recent_pnl_trend) >= 3:
                recent_avg = np.mean(list(self.recent_pnl_trend)[-3:])
                momentum = recent_avg / self.initial_balance
                
                if momentum > 0.01:
                    momentum_reward = 0.05 * momentum * 25.0
                    shaping_reward += momentum_reward
                elif momentum < -0.01:
                    momentum_reward = 0.05 * momentum * 10.0
                    shaping_reward += momentum_reward
                else:
                    momentum_reward = 0.0
                
                info['momentum_reward'] = momentum_reward
            
            return shaping_reward, info
            
        except Exception as e:
            self.logger.error(f"Erro no shaping: {e}")
            return 0.0, {'error': str(e)}
    
    def _update_tracking(self, env):
        """Update tracking variables"""
        try:
            self.last_portfolio_value = getattr(env, 'portfolio_value', self.initial_balance)
            
            # Limpar posições antigas
            current_positions = getattr(env, 'positions', [])
            current_pos_ids = set(pos.get('id', str(hash(str(pos)))) for pos in current_positions)
            
            for pos_id in list(self.position_performance.keys()):
                if pos_id not in current_pos_ids:
                    del self.position_performance[pos_id]
                    
        except:
            pass
    
    def _update_stats(self, reward: float):
        """Atualizar estatísticas"""
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
            'zero_ratio': (total - self.positive_rewards - self.negative_rewards) / total,
            'position_time_ratio': self.steps_in_position / max(1, self.total_steps),
            'overtrading_penalties': self.overtrading_penalties_applied,
            'patience_rewards': self.patience_rewards_given,
            'quality_bonuses': self.quality_bonuses_given,
            'version': 'V4_SELECTIVE'
        }
    
    def get_suggested_cooldown(self) -> int:
        """
        🎯 RETORNA COOLDOWN ADAPTATIVO SUGERIDO
        Para ser usado pelo environment
        """
        base_cooldown = 7
        
        # Ajustar baseado no último resultado
        if self.last_trade_result == 'loss':
            if self.consecutive_losses >= 3:
                return 20  # Cooldown longo após múltiplas perdas
            else:
                return 12  # Cooldown médio após perda
        elif self.last_trade_result == 'win':
            if self.consecutive_wins >= 3:
                return 10  # Cooldown médio após múltiplos wins (evitar overconfidence)
            else:
                return 5  # Cooldown curto após win
        
        # Ajustar baseado no overtrading
        position_ratio = self.steps_in_position / max(1, self.total_steps)
        if position_ratio > 0.5:
            return 15  # Forçar cooldown maior se está overtrading
        
        return base_cooldown
    
    def reset(self):
        """Reset para novo episódio"""
        # Reset tracking
        self.last_portfolio_value = self.initial_balance
        self.position_performance.clear()
        self.recent_pnl_trend.clear()
        
        # Reset anti-overtrading tracking (mantém entre episódios para aprendizado)
        # Não resetar steps_in_position e total_steps para manter histórico global
        
        # Reset trade tracking
        self.patience_counter = 0
        self.last_trade_result = None
        # Manter consecutive losses/wins para próximo episódio (memória)


# Factory function
def create_selective_daytrade_reward_system(initial_balance: float = 1000.0):
    """Factory function para o sistema V4-SELECTIVE"""
    return SelectiveTradingReward(initial_balance)
