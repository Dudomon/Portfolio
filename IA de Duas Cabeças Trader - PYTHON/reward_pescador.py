import os
import numpy as np


class PescadorRewardSystem:
    """
    Reward system for scalp "pescador":
    - Rewards quick, small wins and penalizes holding too long.
    - Encourages entries aligned with fast MA cross and price slope.
    - Penalizes large adverse excursion (MAE) during a trade.

    Interface compatível com silus.TradingEnv:
      reward, info, done_flag = calculate_reward_and_info(env, action, old_state)
    """

    def __init__(self, initial_balance=500.0,
                 quick_close_steps=10,
                 time_penalty_per_step=0.005,  # 0.02 → 0.005 (4x menor)
                 mae_penalty_scale=10.0,
                 entry_align_bonus=0.12,
                 entry_misaligned_penalty=0.1,
                 ema_span=9,
                 activity_bonus=0.15,  # 0.05 → 0.15 (3x maior)
                 inactivity_threshold=50,
                 inactivity_penalty=0.005):  # 0.01 → 0.005 (2x menor)
        self.initial_balance = initial_balance
        self.quick_close_steps = quick_close_steps
        self.time_penalty_per_step = time_penalty_per_step
        self.mae_penalty_scale = mae_penalty_scale
        self.entry_align_bonus = entry_align_bonus
        self.entry_misaligned_penalty = entry_misaligned_penalty
        self.ema_span = ema_span
        self.activity_bonus = activity_bonus
        self.inactivity_threshold = inactivity_threshold
        self.inactivity_penalty = inactivity_penalty

        # Trackers
        self._last_seen_trades_len = 0
        self._last_seen_open_count = 0
        self.last_trade_step = -999

    # Utilidades internas
    def _get_series(self, env, name):
        base_tf = getattr(env, 'base_tf', '5m')
        col = f"{name}_{base_tf}"
        if hasattr(env, 'df') and col in env.df.columns:
            return env.df[col]
        return None

    def _tanh_scale(self, x, scale):
        try:
            return float(np.tanh(x / max(1e-6, scale)))
        except Exception:
            return 0.0

    def _is_fast_cross_aligned(self, env, side):
        try:
            close = self._get_series(env, 'close')
            if close is None:
                return False
            # Calcular EMA rápida (span pequeno). Cálculo apenas até current_step para custo mínimo
            t = int(env.current_step)
            if t < 2:
                return False
            segment = close.iloc[:t+1]
            ema = segment.ewm(span=self.ema_span, adjust=False).mean()
            c_prev, c_curr = float(close.iloc[t-1]), float(close.iloc[t])
            e_prev, e_curr = float(ema.iloc[-2]), float(ema.iloc[-1])
            slope = e_curr - e_prev
            if side == 'long':
                crossed_up = (c_prev < e_prev) and (c_curr > e_curr)
                return crossed_up and slope > 0
            else:
                crossed_down = (c_prev > e_prev) and (c_curr < e_curr)
                return crossed_down and slope < 0
        except Exception:
            return False

    def _mae_penalty_for_trade(self, env, trade):
        try:
            side = trade.get('type', 'long')
            entry_step = int(trade.get('entry_step', env.current_step))
            exit_step = int(trade.get('exit_step', env.current_step))
            entry_price = float(trade.get('entry_price', 0.0))
            lot_size = float(trade.get('lot_size', 0.02))
            if exit_step <= entry_step + 1:
                return 0.0
            high = self._get_series(env, 'high')
            low = self._get_series(env, 'low')
            if high is None or low is None:
                return 0.0
            window_high = float(np.max(high.iloc[entry_step:exit_step+1]))
            window_low = float(np.min(low.iloc[entry_step:exit_step+1]))
            if side == 'long':
                adverse = max(0.0, entry_price - window_low)
            else:
                adverse = max(0.0, window_high - entry_price)
            # Converter em USD aproximado (mesma escala do env: * 100)
            mae_usd = adverse * lot_size * 100.0
            # Penalidade suavizada
            return -0.5 * self._tanh_scale(mae_usd, self.mae_penalty_scale)
        except Exception:
            return 0.0

    def get_suggested_cooldown(self):
        # Heurística simples; pode ser sobrescrita pelo env
        return 5

    def calculate_reward_and_info(self, env, action, old_state):
        reward = 0.0
        info = {
            'pescador': True,
            'components': {}
        }

        # Penalidade por tempo segurando posição (curto e pontual)
        try:
            open_count = len(getattr(env, 'positions', []))
            if open_count > 0:
                time_pen = -self.time_penalty_per_step * min(3, open_count)
                reward += time_pen
                info['components']['time_penalty'] = time_pen
        except Exception:
            pass

        # Recompensa/Penalidade de alinhamento no momento de uma nova entrada
        try:
            current_open = len(getattr(env, 'positions', []))
            if current_open > self._last_seen_open_count:
                # Processar apenas novas posições abertas neste step
                new_positions = current_open - self._last_seen_open_count
                for i in range(new_positions):
                    # Última posição deve ser a mais recente
                    pos = env.positions[-(i+1)] if env.positions else None
                    side = pos.get('type', 'long') if pos else ('long' if (action[0] if isinstance(action, (list, np.ndarray)) else 0) == 1 else 'short')
                    aligned = self._is_fast_cross_aligned(env, side)
                    comp_key = f'entry_align_{i}'
                    if aligned:
                        reward += self.entry_align_bonus
                        info['components'][comp_key] = self.entry_align_bonus
                    else:
                        reward -= self.entry_misaligned_penalty
                        info['components'][comp_key] = -self.entry_misaligned_penalty
        except Exception:
            pass

        # Recompensa por trades fechados neste passo + bônus por rapidez + penalidade MAE
        try:
            trades = getattr(env, 'trades', [])
            if isinstance(trades, list) and len(trades) > self._last_seen_trades_len:
                new_trades = trades[self._last_seen_trades_len:]
                for idx, t in enumerate(new_trades):
                    pnl_usd = float(t.get('pnl_usd', 0.0))
                    duration = int(t.get('duration', 0))
                    # Escalar PnL para manter reward estável
                    pnl_comp = self._tanh_scale(pnl_usd, 10.0)  # ~[-1,1]
                    reward += pnl_comp
                    info['components'][f'pnl_{idx}'] = pnl_comp
                    # Bônus por fechar rápido
                    if duration <= self.quick_close_steps:
                        base_bonus = 0.3
                        # Bônus extra quando o gate está raro
                        rare_gate_bonus = 0.0
                        try:
                            if hasattr(env, 'get_pescador_pass_rate'):
                                pr = env.get_pescador_pass_rate()
                                if pr < 0.08:
                                    rare_gate_bonus = 0.05
                        except Exception:
                            pass
                        reward += (base_bonus + rare_gate_bonus)
                        info['components'][f'quick_bonus_{idx}'] = base_bonus + rare_gate_bonus
                    # Bônus extra se fechar por TP dentro da janela curta
                    exit_reason = t.get('exit_reason', '')
                    if isinstance(exit_reason, str) and 'TP' in exit_reason.upper() and duration <= self.quick_close_steps:
                        reward += 0.1
                        info['components'][f'quick_tp_bonus_{idx}'] = 0.1
                    # Penalidade de MAE
                    mae_pen = self._mae_penalty_for_trade(env, t)
                    if mae_pen != 0.0:
                        reward += mae_pen
                        info['components'][f'mae_penalty_{idx}'] = mae_pen
                    # Penalidade extra por segurar demais
                    if duration > self.quick_close_steps:
                        over = duration - self.quick_close_steps
                        extra_pen = -0.01 * over
                        reward += extra_pen
                        info['components'][f'overstay_penalty_{idx}'] = extra_pen
                # Update trackers
                self._last_seen_trades_len = len(trades)
                self.last_trade_step = int(getattr(env, 'current_step', 0))
        except Exception:
            pass

        # Sistema de atividade/inatividade APRIMORADO
        try:
            current_step = int(getattr(env, 'current_step', 0))
            steps_since_last_trade = current_step - self.last_trade_step
            
            # Bônus MELHORADO por atividade recente (últimos 100 steps vs 20)
            if steps_since_last_trade <= 100:  # 20 → 100 steps (janela expandida)
                # Bônus decrescente: máximo nos primeiros 20 steps, diminui gradualmente
                activity_multiplier = max(0.3, 1.0 - (steps_since_last_trade / 100.0))
                activity_reward = self.activity_bonus * activity_multiplier
                reward += activity_reward
                info['components']['activity_bonus'] = activity_reward
                info['components']['activity_multiplier'] = activity_multiplier
            
            # 🎣 NOVO: Bônus adicional por frequência de trades na sessão
            try:
                trades_count = len(getattr(env, 'trades', []))
                # Últimos 500 steps como "sessão" de trading
                recent_trades = [t for t in env.trades if (current_step - t.get('exit_step', 0)) <= 500]
                if len(recent_trades) >= 3:  # 3+ trades na sessão = ativo
                    frequency_bonus = 0.05 * min(len(recent_trades), 10)  # Cap em 10 trades
                    reward += frequency_bonus
                    info['components']['frequency_bonus'] = frequency_bonus
                    info['components']['recent_trades_count'] = len(recent_trades)
            except Exception:
                pass
            
            # Penalidade crescente por inatividade prolongada (SUAVIZADA)
            if steps_since_last_trade > self.inactivity_threshold:
                excess_inactive = min(steps_since_last_trade - self.inactivity_threshold, 200)  # Cap em 200 steps
                inactivity_pen = -self.inactivity_penalty * (excess_inactive / 50)  # Crescimento gradual
                reward += inactivity_pen
                info['components']['inactivity_penalty'] = inactivity_pen
                info['components']['steps_inactive'] = steps_since_last_trade
                
        except Exception:
            pass

        # Atualizar contador de posições abertas
        try:
            self._last_seen_open_count = len(getattr(env, 'positions', []))
        except Exception:
            pass

        # Nunca encerramos episódio por reward neste sistema
        return reward, info, False
