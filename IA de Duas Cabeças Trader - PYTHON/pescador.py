"""
Pescador - Variante do pipeline de treinamento (baseado em silus.py) para scalp "pescador".
Altera apenas:
- Tag/dirs de experimento
- Ambiente: usa PescadorRewardSystem (rewards curtas e pontuais)
- Otimizações de performance aplicadas

Mantém o restante (convergência, política, dataset) do silus.
"""
import os
import sys

# Importar base (silus)
import silus as base

from reward_pescador import PescadorRewardSystem


# 1) Ajustar TAGs e diretórios para separar artefatos
base.EXPERIMENT_TAG = "PESCADOR"
base.DIFF_MODEL_DIR = f"Otimizacao/treino_principal/models/{base.EXPERIMENT_TAG}"
base.DIFF_CHECKPOINT_DIR = f"Otimizacao/treino_principal/checkpoints/{base.EXPERIMENT_TAG}"
base.DIFF_ENVSTATE_DIR = f"trading_framework/training/checkpoints/{base.EXPERIMENT_TAG}"

os.makedirs(base.DIFF_MODEL_DIR, exist_ok=True)
os.makedirs(base.DIFF_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(base.DIFF_ENVSTATE_DIR, exist_ok=True)


# 2) Ambiente especializado: apenas troca o reward system e pequenos ajustes de SL/TP
class PescadorEnv(base.TradingEnv):
    def __init__(self, df, window_size=20, is_training=True, initial_balance=None, trading_params=None):
        super().__init__(df, window_size=window_size, is_training=is_training,
                         initial_balance=initial_balance, trading_params=trading_params)

        # Substituir reward system por pescador
        self.reward_system = PescadorRewardSystem(initial_balance=self.initial_balance)

        # 🚀 TESTE: DESABILITAR COOLDOWNS COMPLETAMENTE
        self.cooldown_after_trade = 0
        self.cooldown_base = 0
        self.slot_cooldowns = {i: 0 for i in range(getattr(self, 'max_positions', 2))}
        print("🚀 [PESCADOR] Cooldowns COMPLETAMENTE DESABILITADOS para teste")
        
        # 🎯 ACTIVITY SYSTEM FIXO - Timeout constante 4h (sem progressão)
        if is_training:
            from trading_framework.enhancements.activity_enhancement import create_activity_enhancement_system
            self.activity_system = create_activity_enhancement_system(
                position_timeout=48,  # 4h fixo para pescador
                progressive_timeout=False,  # DESABILITAR progressão automática
                training_steps_total=12000000
            )
            print("🎣 [PESCADOR] Activity System FIXO - Timeout SEMPRE 4h (scalping constante)")
        else:
            self.activity_system = None

        # Ajustes leves para scalp curto (mantém ranges realistas)
        self.sl_range_min = 0.3
        self.sl_range_max = 0.7
        self.tp_range_min = 0.5
        self.tp_range_max = 1.0  # ranges curtos para scalps rápidos
        self.sl_tp_step = 0.1

        # Episódios mais longos para mais oportunidades em baixa volatilidade  
        self.MAX_STEPS = 3000  # 2000 -> 3000 steps

        # GATE PESCADOR - filtros de entrada (com adaptação + cache)
        self._pescador_ema = None
        self._pescador_prev_ema = None
        self._gate_cache = {}  # 🚀 OTIMIZAÇÃO: Cache para gate decisions
        self._gate_cache_step = -1
        # Base thresholds balanceados para pescador
        self.gate_base_min_slope = 0.01   # 0.02 -> 0.01 (50% mais permissivo)
        self.gate_base_min_atr = 0.2      # 0.3 -> 0.2 (33% mais permissivo)  
        self.gate_base_session_start = 5  # 7 -> 5 (2h mais cedo)
        self.gate_base_session_end = 20   # 18 -> 20 (2h mais tarde)
        # Dinâmicos
        self.gate_min_slope = self.gate_base_min_slope
        self.gate_min_atr = self.gate_base_min_atr
        self.gate_max_spread = 2.0   # 1.0 -> 2.0 (2x mais permissivo)
        self.gate_session_start = self.gate_base_session_start
        self.gate_session_end = self.gate_base_session_end
        # Estatísticas para adaptação
        self._gate_attempts_window = 1000
        self._gate_recent_attempts = []  # lista de bool (passou ou não)
        self._gate_last_adapt_step = 0

    def _pescador_update_ema(self, price: float, span: int = 9):
        alpha = 2.0 / (span + 1)
        if self._pescador_ema is None:
            self._pescador_ema = price
            self._pescador_prev_ema = price
        else:
            self._pescador_prev_ema = self._pescador_ema
            self._pescador_ema = alpha * price + (1 - alpha) * self._pescador_ema

    def _pescador_pass_gate(self) -> bool:
        """🚀 OTIMIZAÇÃO: Gate com cache para evitar recálculos no mesmo step"""
        # Cache para evitar múltiplos cálculos no mesmo step
        if self._gate_cache_step == self.current_step and 'result' in self._gate_cache:
            return self._gate_cache['result']
            
        try:
            price = float(self.df[f'close_{self.base_tf}'].iloc[self.current_step])
            self._pescador_update_ema(price, span=9)
            slope = (self._pescador_ema - self._pescador_prev_ema) if (self._pescador_ema is not None and self._pescador_prev_ema is not None) else 0.0

            # ATR
            atr_val = None
            if 'atr_14' in self.df.columns:
                atr_val = self.df['atr_14'].iloc[self.current_step]
            # Spread
            spread_val = None
            if 'spread' in self.df.columns:
                spread_val = self.df['spread'].iloc[self.current_step]
            # Hour
            try:
                ts = self.df.index[self.current_step]
                hour = int(getattr(ts, 'hour', None) or 0)
            except Exception:
                hour = 0

            if abs(slope) < self.gate_min_slope:
                return False
            if atr_val is not None and atr_val < self.gate_min_atr:
                return False
            if spread_val is not None and spread_val > self.gate_max_spread:
                return False
            if not (self.gate_session_start <= hour < self.gate_session_end):
                result = False
            else:
                result = True
                
            # 🚀 OTIMIZAÇÃO: Cachear resultado para o step atual
            self._gate_cache_step = self.current_step
            self._gate_cache['result'] = result
            return result
            
        except Exception:
            # Cache também resultados de erro
            self._gate_cache_step = self.current_step
            self._gate_cache['result'] = False
            return False

    def _pescador_register_gate_attempt(self, passed: bool):
        try:
            self._gate_recent_attempts.append(bool(passed))
            if len(self._gate_recent_attempts) > self._gate_attempts_window:
                self._gate_recent_attempts.pop(0)
        except Exception:
            pass

    def get_pescador_pass_rate(self) -> float:
        try:
            if not self._gate_recent_attempts:
                return 0.0
            return sum(1 for x in self._gate_recent_attempts if x) / len(self._gate_recent_attempts)
        except Exception:
            return 0.0

    def _pescador_adapt_gate(self):
        try:
            # Adaptar a cada 100 tentativas
            if len(self._gate_recent_attempts) < 100:
                return
            if self.current_step - self._gate_last_adapt_step < 100:
                return
            self._gate_last_adapt_step = self.current_step
            rate = self.get_pescador_pass_rate()
            # Relaxar quando muito baixo
            if rate < 0.08:
                # relax 25% até mínimos
                self.gate_min_slope = max(0.01, self.gate_min_slope * 0.75)
                self.gate_min_atr = max(0.2, self.gate_min_atr * 0.85)
                # expandir sessão em 1h cada lado dentro dos limites
                self.gate_session_start = max(0, self.gate_session_start - 1)
                self.gate_session_end = min(23, self.gate_session_end + 1)
            elif rate > 0.15:
                # Restaurar gradualmente para base
                self.gate_min_slope = min(self.gate_base_min_slope, self.gate_min_slope * 1.05)
                self.gate_min_atr = min(self.gate_base_min_atr, self.gate_min_atr * 1.05)
                # contrair sessão em 1h rumo à base
                if self.gate_session_start < self.gate_base_session_start:
                    self.gate_session_start += 1
                if self.gate_session_end > self.gate_base_session_end:
                    self.gate_session_end -= 1
        except Exception:
            pass

    def step(self, action):
        try:
            import numpy as np
            # Forçar HOLD se não passar no gate de entrada
            if isinstance(action, (list, tuple)):
                action = np.array(action, dtype=np.float32)
            if action is not None and len(action) >= 1:
                entry_decision = float(action[0])
                # 🚀 GATE COMPLETAMENTE DESABILITADO - Pescador deve operar livremente
                if False:  # entry_decision > 0 and not self._pescador_pass_gate():
                    # 🚀 DEBUG: Log bloqueios do gate
                    if self.current_step % 1000 == 0:
                        print(f"[PESCADOR-GATE] BLOQUEADO step={self.current_step} | slope={self._pescador_ema - self._pescador_prev_ema if self._pescador_ema and self._pescador_prev_ema else 0:.4f}")
                    
                    # Registrar tentativa e adaptar
                    self._pescador_register_gate_attempt(False)
                    self._pescador_adapt_gate()
                    action = action.copy()
                    action[0] = 0.0  # HOLD
                    if len(action) > 2:
                        action[2] = 0.0  # zerar mgmt pos1
                    if len(action) > 3:
                        action[3] = 0.0  # zerar mgmt pos2
                # 🎣 GATE COMPLETAMENTE REMOVIDO - Sem interferência adaptativa

            # 🚀 OTIMIZAÇÃO PERFORMANCE: Logs menos frequentes (como silus.py)
            try:
                if self.current_step % 10000 == 0 and self.current_step > 0:  # 5000 -> 10000
                    pr = self.get_pescador_pass_rate() * 100.0
                    print(
                        f"[PESCADOR-GATE] step={self.current_step} | pass_rate_win={pr:.1f}% | "
                        f"slope>={self.gate_min_slope:.3f} atr>={self.gate_min_atr:.3f} | "
                        f"sess={self.gate_session_start}-{self.gate_session_end} | attempts_win={len(self._gate_recent_attempts)}"
                    )
            except Exception:
                pass
            return super().step(action)
        except Exception:
            return super().step(action)
    
    def _close_position(self, position, current_step_or_reason=None):
        """Override para desabilitar aplicação de cooldowns após fechamento"""
        result = super()._close_position(position, current_step_or_reason)
        
        # 🚀 TESTE: Zerar todos os cooldowns após qualquer fechamento
        self.cooldown_after_trade = 0
        for slot in self.slot_cooldowns:
            self.slot_cooldowns[slot] = 0
            
        return result
    
    # REMOVIDO: Override de _calculate_reward_and_info que estava quebrando o processamento
    # O silus.py faz processamento essencial (SL/TP, ordens) antes do reward


# 3) Função de criação de ambiente (monkey patch na base)
def make_wrapped_env_pescador(df, window_size, is_training, initial_portfolio=None, current_steps=0):
    if initial_portfolio is None:
        initial_portfolio = base.TRADING_CONFIG["portfolio_inicial"]

    trading_params = base.get_gold_trading_params_for_phase(current_steps)
    current_phase = base.get_current_phase_config(current_steps)
    if current_steps > 0:
        print(f"🏆 GOLD PHASE: {current_phase['name']} ({current_steps:,} steps)")
        print(f"   Focus: {current_phase['focus']}")

    env = PescadorEnv(df, window_size=window_size, is_training=is_training,
                      initial_balance=initial_portfolio, trading_params=trading_params)
    env.seed(base.SEED)
    env.action_space.seed(base.SEED)
    env.observation_space.seed(base.SEED)
    return env


# Patching: garantir que os componentes do pipeline usem nosso env/dirs
# Importante: não sobrescrever base.TradingEnv para evitar conflitos de super().__init__
base.make_wrapped_env = make_wrapped_env_pescador

# 🚀 DOBRAR LEARNING RATE PARA PESCADOR (mais agressivo)
base.BEST_PARAMS["learning_rate"] = 6.0e-05  # 3e-05 -> 6e-05 (2x)
base.BEST_PARAMS["critic_learning_rate"] = 4.0e-05  # 2e-05 -> 4e-05 (2x)
print(f"🚀 [PESCADOR] Learning rates DOBRADOS - Actor: {base.BEST_PARAMS['learning_rate']:.1e}, Critic: {base.BEST_PARAMS['critic_learning_rate']:.1e}")


def main():
    # Reusar o main do silus após patching
    base.main()


if __name__ == "__main__":
    main()
