# PLANO REFINADO: UNIFIED REWARD COM COMPONENTES ESPECIALIZADOS

## FILOSOFIA CENTRAL

Manter o backbone unificado recebendo UM ÚNICO REWARD FINAL, mas esse reward é composto por componentes que dão feedback específico sobre diferentes aspectos da decisão, sem quebrar a arquitetura V7.

---

## 🎯 ESTRUTURA DO SISTEMA

```python
class UnifiedRewardWithComponents:
    """
    Reward único alimenta o backbone, mas com componentes específicos
    que dão feedback diferenciado sobre timing vs gestão
    """

    def __init__(self):
        # PESOS CONSERVADORES - começar pequeno
        self.base_weight = 0.8        # Reward tradicional (dominante)
        self.timing_weight = 0.1      # Componente de timing
        self.management_weight = 0.1  # Componente de gestão

        # HISTÓRICO para análise
        self.component_history = {
            'base': [], 'timing': [], 'management': [], 'total': []
        }
```

## 📊 COMPONENTE 1: TIMING QUALITY

```python
def calculate_timing_component(self, action, info, env):
    """
    Componente que avalia qualidade do TIMING da entrada
    Não substitui profit/loss, apenas adiciona feedback sobre momento
    """
    timing_bonus = 0.0

    if info.get('new_position_opened', False):
        entry_quality = action[1]  # Entry quality do modelo
        current_step = env.current_step

        # MÉTODO 1: Volatilidade recente (proxy para movimento futuro)
        if current_step >= 20:
            recent_volatility = self._calculate_recent_volatility(env, current_step, window=20)
            volatility_threshold = 0.005  # Ajustável

            if recent_volatility > volatility_threshold:
                # Mercado ativo = bom timing
                timing_bonus = +0.2 * entry_quality
            else:
                # Mercado parado = timing questionável
                timing_bonus = -0.1 * (1.0 - entry_quality)

        # MÉTODO 2: Momentum confluence (múltiplos timeframes)
        momentum_score = self._evaluate_momentum_confluence(env, current_step)
        if momentum_score > 0.6:  # Momentum forte
            timing_bonus += 0.1 * entry_quality
        elif momentum_score < 0.3:  # Momentum fraco
            timing_bonus -= 0.1 * (1.0 - entry_quality)

    return timing_bonus

def _calculate_recent_volatility(self, env, step, window=20):
    """Calcular volatilidade das últimas N barras"""
    if step < window:
        return 0.0

    recent_prices = env.df['close_5m'].iloc[step-window:step]
    returns = recent_prices.pct_change().dropna()
    return returns.std() if len(returns) > 0 else 0.0

def _evaluate_momentum_confluence(self, env, step):
    """Avaliar confluência de momentum em diferentes timeframes"""
    if step < 50:
        return 0.5  # Neutro

    # Momentum 1: Short term (5 bars)
    short_momentum = self._calculate_momentum(env.df['close_5m'], step, 5)

    # Momentum 2: Medium term (20 bars)
    medium_momentum = self._calculate_momentum(env.df['close_5m'], step, 20)

    # Confluence: quando ambos apontam na mesma direção
    if (short_momentum > 0 and medium_momentum > 0) or (short_momentum < 0 and medium_momentum < 0):
        return 0.8  # Alta confluência
    else:
        return 0.2  # Baixa confluência
```

## ⚙️ COMPONENTE 2: MANAGEMENT QUALITY

```python
def calculate_management_component(self, action, info, env):
    """
    Componente que avalia qualidade da GESTÃO de posições
    Avalia SL/TP placement, duration, trailing usage
    """
    management_bonus = 0.0

    # 1. AVALIAÇÃO DE NOVA POSIÇÃO (SL/TP placement)
    if info.get('new_position_opened', False):
        management_bonus += self._evaluate_sl_tp_placement(action, info, env)

    # 2. AVALIAÇÃO DE FECHAMENTO (efficiency)
    if info.get('position_closed', False):
        management_bonus += self._evaluate_closure_efficiency(action, info, env)

    # 3. AVALIAÇÃO DE GESTÃO ATIVA (trailing, adjustments)
    management_bonus += self._evaluate_active_management(action, info, env)

    return management_bonus

def _evaluate_sl_tp_placement(self, action, info, env):
    """Avaliar qualidade inicial de SL/TP"""
    if 'new_position' not in info:
        return 0.0

    position = info['new_position']
    entry_price = position.get('entry_price', 0)
    sl_price = position.get('sl_price', 0)
    tp_price = position.get('tp_price', 0)

    if entry_price == 0 or sl_price == 0 or tp_price == 0:
        return 0.0

    # Calcular distâncias
    sl_distance = abs(sl_price - entry_price)
    tp_distance = abs(tp_price - entry_price)
    risk_reward_ratio = tp_distance / sl_distance if sl_distance > 0 else 1.0

    # REWARD baseado em risk/reward ratio
    if 1.5 <= risk_reward_ratio <= 3.0:  # Ratio saudável
        return +0.1
    elif risk_reward_ratio < 1.0:  # Ratio ruim
        return -0.1
    else:
        return 0.0  # Ratio questionável mas aceitável

def _evaluate_closure_efficiency(self, action, info, env):
    """Avaliar eficiência do fechamento"""
    if 'closed_trade' not in info:
        return 0.0

    trade = info['closed_trade']
    duration = trade.get('duration', 0)
    exit_reason = trade.get('exit_reason', '')
    pnl = trade.get('pnl_usd', 0)

    bonus = 0.0

    # 1. Penalizar trades muito curtos (overtrading)
    if duration < 30:  # Menos de 150min
        bonus -= 0.15 * (1.0 - duration/30)  # Penalidade progressiva

    # 2. Reward por uso eficiente de trailing
    if trade.get('trailing_activated', False) and pnl > 0:
        bonus += 0.1

    # 3. Reward por saídas inteligentes (não apenas SL hit)
    if exit_reason == 'take_profit':
        bonus += 0.05
    elif exit_reason == 'trailing_stop':
        bonus += 0.08
    elif exit_reason == 'time_exit' and pnl > 0:  # Exit inteligente por tempo
        bonus += 0.03

    return bonus

def _evaluate_active_management(self, action, info, env):
    """Avaliar gestão ativa durante a posição"""
    bonus = 0.0

    # Verificar ajustes de SL/TP nas actions atuais
    if len(action) >= 11:  # V7 action space
        sl_adjusts = action[5:8]   # SL adjusts
        tp_adjusts = action[8:11]  # TP adjusts

        # Reward por uso moderado de adjustments (não zero, não excessivo)
        sl_activity = np.mean(np.abs(sl_adjusts))
        tp_activity = np.mean(np.abs(tp_adjusts))

        if 0.1 <= sl_activity <= 1.0:  # Atividade moderada
            bonus += 0.02
        if 0.1 <= tp_activity <= 1.0:
            bonus += 0.02

    return bonus
```

## 🔧 INTEGRAÇÃO NO SISTEMA EXISTENTE

```python
# MODIFICAR em daytrader.py - step()
def step(self, action):
    # ... código existente até calcular reward base ...

    # REWARD BASE (tradicional)
    base_reward = self.reward_calculator.calculate_step_reward(
        self.current_step, info, self.episode_metrics, action
    )

    # COMPONENTES ESPECIALIZADOS
    timing_component = self.unified_reward_system.calculate_timing_component(
        action, info, self
    )
    management_component = self.unified_reward_system.calculate_management_component(
        action, info, self
    )

    # REWARD FINAL UNIFICADO
    final_reward = (
        self.unified_reward_system.base_weight * base_reward +
        self.unified_reward_system.timing_weight * timing_component +
        self.unified_reward_system.management_weight * management_component
    )

    # LOGGING para análise
    if self.current_step % 1000 == 0:
        self._log_reward_components(base_reward, timing_component, management_component, final_reward)

    return observation, final_reward, done, info
```

## 📊 SISTEMA DE MONITORAMENTO

```python
class ComponentRewardMonitor:
    """Monitor para analisar eficácia dos componentes"""

    def __init__(self, window_size=1000):
        self.window_size = window_size
        self.components_history = {
            'base': deque(maxlen=window_size),
            'timing': deque(maxlen=window_size),
            'management': deque(maxlen=window_size),
            'total': deque(maxlen=window_size)
        }

    def log_step(self, base, timing, management, total):
        self.components_history['base'].append(base)
        self.components_history['timing'].append(timing)
        self.components_history['management'].append(management)
        self.components_history['total'].append(total)

    def analyze_components(self):
        """Análise dos componentes e seus impactos"""
        if len(self.components_history['total']) < 100:
            return

        base_mean = np.mean(self.components_history['base'])
        timing_mean = np.mean(self.components_history['timing'])
        mgmt_mean = np.mean(self.components_history['management'])

        # Correlações
        base_arr = np.array(self.components_history['base'])
        timing_arr = np.array(self.components_history['timing'])
        mgmt_arr = np.array(self.components_history['management'])

        timing_base_corr = np.corrcoef(timing_arr, base_arr)[0,1]
        mgmt_base_corr = np.corrcoef(mgmt_arr, base_arr)[0,1]

        print(f"\n📊 REWARD COMPONENTS ANALYSIS:")
        print(f"  Base Reward: {base_mean:.4f}")
        print(f"  Timing Component: {timing_mean:.4f}")
        print(f"  Management Component: {mgmt_mean:.4f}")
        print(f"  Timing-Base Correlation: {timing_base_corr:.3f}")
        print(f"  Management-Base Correlation: {mgmt_base_corr:.3f}")

        # Alertas
        if abs(timing_mean) > 0.05:
            print(f"  ⚠️ Timing component very active: {timing_mean:.4f}")
        if abs(mgmt_mean) > 0.05:
            print(f"  ⚠️ Management component very active: {mgmt_mean:.4f}")
```

## 🚀 IMPLEMENTAÇÃO INCREMENTAL

### FASE 1: SETUP (2h)

1. Criar UnifiedRewardWithComponents
2. Integrar no daytrader com pesos ZERO (só logging)
3. Validar que não quebra nada

### FASE 2: TIMING (4h)

1. Implementar calculate_timing_component
2. Ativar com peso 0.05 (muito baixo)
3. Treinar por 50k steps e analisar

### FASE 3: MANAGEMENT (4h)

1. Implementar calculate_management_component
2. Ativar com peso 0.05
3. Treinar por 50k steps e analisar

### FASE 4: OTIMIZAÇÃO (6h)

1. Ajustar pesos baseado nos resultados
2. Treinar modelo completo
3. Comparar com baseline

## ⚡ CONFIGURAÇÃO DE SEGURANÇA

```python
# FLAGS no início do daytrader.py
USE_COMPONENT_REWARDS = True
COMPONENT_REWARD_WEIGHTS = {
    'base': 0.8,      # Manter reward tradicional dominante
    'timing': 0.1,    # Começar conservador
    'management': 0.1 # Começar conservador
}

# FALLBACK automático
if convergence_issues_detected():
    COMPONENT_REWARD_WEIGHTS = {'base': 1.0, 'timing': 0.0, 'management': 0.0}
    print("⚠️ Fallback: Component rewards disabled due to convergence issues")
```

Este sistema mantém a arquitetura V7 intacta, mas refina o sinal de reward para dar feedback mais específico sobre diferentes aspectos da decisão. O backbone unificado continua vendo um único reward, mas esse reward agora carrega informação mais rica sobre qualidade de timing e gestão.