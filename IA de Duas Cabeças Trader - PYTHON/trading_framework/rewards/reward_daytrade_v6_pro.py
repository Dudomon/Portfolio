"""
Reward V6 Pro – DayTrade Gold (Silus)

Objetivo: Fornecer um sinal matemático forte e claro para o PPO maximizar, com
PNL dominante e controles de risco profissionais. Evita intenções verbais e
foca em termos diretamente correlacionados ao resultado de trading.

Componentes (por step):
1) Base PnL (dominante):
   r_pnl = w_pnl * tanh((portfolio_value - last_portfolio_value) / scale_pnl)
   • last_portfolio_value é atualizado a cada step.

2) Fechamentos (discreto no step em que ocorre):
   r_close = w_close * tanh(sum_pnl_closed / scale_close)
   • Função suave (tanh) reduz saturação e estabiliza gradientes.

3) Risco (por posição aberta):
   • MAE penalty: penaliza excursão adversa (em pontos ou proporção) desde a entrada.
     r_mae = - w_mae * tanh(MAE_points / mae_scale)
   • Holding penalty: penaliza segurar por muito tempo sem resolver.
     r_time = - w_time * min(1.0, age / time_scale)

4) Gestão de tamanho (opcional, fraco):
   • Evita saturar sempre no máximo: r_size = - w_size if lot_size ~ max_lot.

Combinação:
   r_total = r_pnl + r_close + sum_pos(open_pos: r_mae + r_time + r_size)

Sinais opcionais de qualidade devem ser leves para não sobrepor PnL.

Observações:
• PPO vê esse sinal como maximização pura, alinhado ao objetivo financeiro.
• Todas as constantes configuráveis no V6_CONFIG.
"""

from __future__ import annotations
from typing import Dict, Tuple, Any, Optional
import numpy as np


V6_CONFIG = {
    # Pesos
    'w_pnl': 1.0,          # Dominante: impacto do delta de equity por step
    'w_close': 0.6,        # Reforço na barra de fechamento
    'w_mae': 0.3,          # Penalização por excursão adversa
    'w_time': 0.15,        # Penalização por tempo segurando
    'w_size': 0.05,        # Penalização suave por lote máximo constante
    'w_activity': 0.3,     # 🚀 AUMENTADO: 0.08 -> 0.3 (combate inatividade)

    # Escalas (normalizações)
    'scale_pnl': 5.0,      # USD ou pontos efetivos (ajustar conforme env)
    'scale_close': 20.0,   # USD agregado de fechamentos por step
    'mae_scale': 10.0,     # pontos de ouro (~$ por 0.01 lot)
    'time_scale': 50.0,    # steps até saturar penalidade (≈ 4h em M5)

    # Bônus de Atividade
    'activity_decisive_threshold': 0.15,     # Threshold para ação decisiva
    'activity_moderate_threshold': 0.05,     # Threshold para ação moderada
    'activity_decisive_bonus': 0.5,          # 🚀 AUMENTADO: 0.02 -> 0.5  
    'activity_moderate_bonus': 0.3,          # 🚀 AUMENTADO: 0.01 -> 0.3
    'activity_inaction_penalty': -0.2,       # 🚀 AUMENTADO: -0.005 -> -0.2
    'activity_inaction_threshold': 0.02,     # Threshold para considerar inação
    'inactivity_steps_threshold': 30,        # Steps para considerar inatividade prolongada
    'inactivity_max_multiplier': 3.0,        # Multiplicador máximo por inatividade

    # Limites/robustez
    'max_abs_reward': 2.5, # clamp final por step
}


class RewardV6Pro:
    def __init__(self, initial_balance: float = 500.0, config: Optional[Dict[str, Any]] = None):
        self.initial_balance = float(initial_balance)
        self.cfg = {**V6_CONFIG, **(config or {})}
        self.last_portfolio_value = self.initial_balance
        self._last_positions_snapshot = []

    # Utilidades
    def _tanh(self, x: float, scale: float) -> float:
        if scale <= 0:
            return 0.0
        return float(np.tanh(x / scale))

    def _get_df_cols(self, env, base: str) -> Optional[np.ndarray]:
        try:
            df = getattr(env, 'df', None)
            if df is None:
                return None
            for name in [f"{base}_5m", base]:
                if name in df.columns:
                    return df[name].values
            return None
        except Exception:
            return None

    def _adverse_excursion_points(self, env, pos: Dict[str, Any]) -> float:
        try:
            entry_price = float(pos.get('entry_price', 0.0))
            entry_step = int(pos.get('entry_step', getattr(env, 'current_step', 0)))
            ptype = pos.get('type', 'long')
            high = self._get_df_cols(env, 'high')
            low = self._get_df_cols(env, 'low')
            if high is None or low is None:
                return 0.0
            cur_step = int(getattr(env, 'current_step', len(low)-1))
            start = max(0, min(entry_step, len(low)-1))
            end = max(start, min(cur_step, len(low)-1))
            if ptype == 'long':
                window_min = float(np.min(low[start:end+1]))
                mae = max(0.0, entry_price - window_min)
            else:
                window_max = float(np.max(high[start:end+1]))
                mae = max(0.0, window_max - entry_price)
            return mae
        except Exception:
            return 0.0

    def _age_steps(self, env, pos: Dict[str, Any]) -> int:
        try:
            entry_step = int(pos.get('entry_step', getattr(env, 'current_step', 0)))
            cur = int(getattr(env, 'current_step', 0))
            return max(0, cur - entry_step)
        except Exception:
            return 0

    def _sum_closed_pnl(self, env, old_state: Dict[str, Any]) -> float:
        try:
            # Comparar trades/positions do old_state com env atual
            old_positions = old_state.get('positions', []) if old_state else []
            cur_positions = getattr(env, 'positions', []) or []
            old_count = len(old_positions)
            cur_count = len(cur_positions)
            # Heurística: se reduziu o número, houve fechamento.
            if cur_count >= old_count:
                return 0.0
            # Buscar PnL do env para fechamentos recentes
            trades = getattr(env, 'trades', []) or []
            if not trades:
                return 0.0
            # Somar últimos |old_count - cur_count| fechamentos
            closed_needed = old_count - cur_count
            recent = trades[-closed_needed:]
            return float(sum(t.get('pnl_usd', 0.0) for t in recent))
        except Exception:
            return 0.0

    def calculate_reward_and_info(self, env, action: np.ndarray, old_state: Dict[str, Any]) -> Tuple[float, Dict[str, Any], bool]:
        info: Dict[str, Any] = {}
        try:
            # 1) Base PnL (delta equity)
            portfolio = float(getattr(env, 'portfolio_value', self.last_portfolio_value))
            delta_equity = portfolio - self.last_portfolio_value
            r_pnl = self.cfg['w_pnl'] * self._tanh(delta_equity, self.cfg['scale_pnl'])
            info['base_pnl'] = r_pnl
            info['delta_equity'] = delta_equity

            # 2) Fechamentos no step
            sum_closed = self._sum_closed_pnl(env, old_state)
            r_close = self.cfg['w_close'] * self._tanh(sum_closed, self.cfg['scale_close']) if sum_closed != 0 else 0.0
            info['closed_pnl_sum'] = sum_closed
            info['close_component'] = r_close

            # 3) Risco por posição
            r_risk_total = 0.0
            positions = getattr(env, 'positions', []) or []
            max_lot = float(getattr(env, 'max_lot_size', 0.03))
            for pos in positions[:2]:  # máx 2 posições
                mae_pts = self._adverse_excursion_points(env, pos)
                r_mae = - self.cfg['w_mae'] * self._tanh(mae_pts, self.cfg['mae_scale'])
                age = self._age_steps(env, pos)
                r_time = - self.cfg['w_time'] * min(1.0, age / self.cfg['time_scale'])
                lot = float(pos.get('lot_size', 0.0))
                r_size = - self.cfg['w_size'] if max_lot > 0 and lot >= max_lot - 1e-6 else 0.0
                r_risk_total += (r_mae + r_time + r_size)
            info['risk_component'] = r_risk_total

            # 4) Bônus de Atividade (independente do PnL)
            action_magnitude = float(np.abs(action).max()) if len(action) > 0 else 0.0
            r_activity = 0.0

            # Incentivo básico por decisão
            if action_magnitude > self.cfg['activity_decisive_threshold']:
                r_activity += self.cfg['activity_decisive_bonus']
            elif action_magnitude > self.cfg['activity_moderate_threshold']:
                r_activity += self.cfg['activity_moderate_bonus']

            # Multiplicador por inatividade prolongada
            steps_inactive = getattr(env, 'steps_since_last_trade', 0)
            if steps_inactive > self.cfg['inactivity_steps_threshold']:
                inactivity_multiplier = min(
                    self.cfg['inactivity_max_multiplier'], 
                    1.0 + steps_inactive / 50
                )
                r_activity *= inactivity_multiplier

            # Penalidade suave por inação completa
            if action_magnitude < self.cfg['activity_inaction_threshold']:
                r_activity += self.cfg['activity_inaction_penalty']

            # Aplicar peso configurável
            r_activity *= self.cfg['w_activity']
            info['activity_component'] = r_activity
            info['action_magnitude'] = action_magnitude
            info['steps_inactive'] = steps_inactive

            # Total
            total = r_pnl + r_close + r_risk_total + r_activity
            total = float(np.clip(total, -self.cfg['max_abs_reward'], self.cfg['max_abs_reward']))

            # Atualiza estado
            self.last_portfolio_value = portfolio

            # Done nunca por reward
            return total, info, False

        except Exception as e:
            # Fallback seguro
            return 0.0, {'error': str(e)}, False


def create_v6_pro_reward_system(initial_balance: float = 500.0, config: Optional[Dict[str, Any]] = None) -> RewardV6Pro:
    return RewardV6Pro(initial_balance=initial_balance, config=config)

