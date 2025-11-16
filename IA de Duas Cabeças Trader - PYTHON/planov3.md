# 🎯 PLANO V3 - CORREÇÃO DO OVERTRADING COMPULSIVO

## 📋 CONTEXTO DO PROBLEMA
O modelo atualmente apresenta comportamento de "sempre estar posicionado" - sai de um trade por SL/TP e imediatamente entra em outro, sem discriminação de qualidade. Paradoxalmente:
- **Treino (2x volatilidade)**: Opera POUCO
- **Produção (volatilidade normal)**: Opera DEMAIS

## 🔧 IMPLEMENTAÇÕES NECESSÁRIAS

### 1️⃣ **VOLATILIDADE VARIÁVEL NO TREINO**

#### Localização: `daytrader.py` - Classe `TradingEnv`

#### Implementação:
```python
# ADICIONAR na __init__ da TradingEnv:
self.volatility_schedule = [0.5, 0.5, 1.0, 2.0, 0.5, 1.0, 3.0, 0.5]
self.volatility_idx = 0
self.episodes_per_volatility = 10  # Mudar volatilidade a cada 10 episódios

# MODIFICAR no reset():
def reset(self):
    # Rotacionar volatilidade baseado em episódios
    if self.episode_count % self.episodes_per_volatility == 0:
        self.current_volatility = self.volatility_schedule[self.volatility_idx % len(self.volatility_schedule)]
        self.volatility_idx += 1
        print(f"🔄 Volatilidade ajustada para: {self.current_volatility}x")
    
    # Aplicar volatilidade ao dataset
    self._apply_volatility_multiplier(self.current_volatility)
```

#### Modificações no Dataset:
```python
def _apply_volatility_multiplier(self, multiplier):
    """Aplicar multiplicador de volatilidade aos dados"""
    # Preservar dados originais
    if not hasattr(self, 'original_df'):
        self.original_df = self.df.copy()
    
    # Aplicar volatilidade
    volatility_cols = ['high_5m', 'low_5m', 'close_5m', 'open_5m']
    for col in volatility_cols:
        if col in self.df.columns:
            base_price = self.original_df[col].mean()
            deviation = (self.original_df[col] - base_price)
            self.df[col] = base_price + (deviation * multiplier)
```

### 2️⃣ **REWARD POR DURAÇÃO DE POSIÇÃO**

#### Localização: `trading_framework/rewards/reward_daytrade_v2.py`

#### Implementação:
```python
# ADICIONAR no calculate_step_reward():

def calculate_position_lifetime_penalty(self, info):
    """Penalizar trades muito curtos para evitar overtrading"""
    penalty = 0.0
    
    # Verificar trades fechados neste step
    if 'closed_trade' in info and info['closed_trade']:
        trade = info['closed_trade']
        duration = trade.get('duration', 0)
        
        # Penalizar trades menores que 50 steps (250min em 5m bars)
        MIN_DURATION = 50
        if duration < MIN_DURATION:
            # Penalidade progressiva: 0% reward em duration=0, 100% em duration=50
            penalty_factor = 1.0 - (duration / MIN_DURATION)
            penalty = -0.1 * penalty_factor  # Máximo -0.1 para trades instantâneos
            
            if self.verbose and duration < 10:  # Log apenas trades muito curtos
                print(f"⚠️ Trade muito curto: {duration} steps, penalidade: {penalty:.3f}")
    
    return penalty

# INTEGRAR na função principal:
def calculate_step_reward(self, ...):
    # ... código existente ...
    
    # Adicionar penalidade por duração
    lifetime_penalty = self.calculate_position_lifetime_penalty(info)
    total_reward += lifetime_penalty
    
    # ... resto do código ...
```

### 3️⃣ **DUAL REWARD SYSTEM - SEPARATION OF CONCERNS**

#### Localização: Criar novo arquivo `trading_framework/rewards/dual_reward_system.py`

#### Implementação Completa:
```python
import numpy as np
from typing import Dict, Tuple, Any

class DualRewardSystem:
    """
    Sistema de reward duplo para TwoHeadV7Intuition
    Separa rewards para Entry Head e Management Head
    """
    
    def __init__(self, 
                 future_window: int = 100,
                 movement_threshold: float = 10.0,  # 10 pontos de movimento
                 verbose: bool = False):
        self.future_window = future_window
        self.movement_threshold = movement_threshold
        self.verbose = verbose
        self.entry_rewards_history = []
        self.management_rewards_history = []
        
    def calculate(self, action: np.ndarray, info: Dict[str, Any], env) -> Tuple[float, float]:
        """
        Calcula rewards separados para cada head
        
        Returns:
            entry_reward: Reward para Entry Head (timing de entrada)
            management_reward: Reward para Management Head (gestão de posição)
        """
        entry_reward = 0.0
        management_reward = 0.0
        
        # 1. ENTRY HEAD REWARD - Baseado em timing
        if info.get('new_position_opened', False):
            entry_reward = self._calculate_entry_timing_reward(action, info, env)
            
        # 2. MANAGEMENT HEAD REWARD - Baseado em gestão
        if info.get('position_closed', False):
            management_reward = self._calculate_management_reward(action, info, env)
            
        # Log para debug
        if self.verbose and (entry_reward != 0 or management_reward != 0):
            print(f"📊 Dual Reward: Entry={entry_reward:.3f}, Management={management_reward:.3f}")
            
        return entry_reward, management_reward
    
    def _calculate_entry_timing_reward(self, action: np.ndarray, info: Dict, env) -> float:
        """
        Reward para Entry Head baseado APENAS em timing
        Não considera profit/loss, apenas se havia movimento futuro
        """
        current_step = env.current_step
        entry_quality = action[1]  # Entry quality do modelo
        
        # Olhar movimento futuro (não podemos ver o futuro real no treino, usar proxy)
        if current_step + self.future_window < len(env.df):
            current_price = env.df['close_5m'].iloc[current_step]
            future_prices = env.df['close_5m'].iloc[current_step:current_step + self.future_window]
            
            # Calcular movimento máximo nos próximos N steps
            max_high = future_prices.max()
            max_low = future_prices.min()
            max_movement = max(abs(max_high - current_price), abs(current_price - max_low))
            
            # Reward baseado em movimento futuro
            if max_movement >= self.movement_threshold:
                # Havia movimento significativo - boa entrada
                entry_reward = 0.5 * entry_quality  # Reward proporcional à confidence
            else:
                # Mercado parado - má entrada
                entry_reward = -0.5 * (1.0 - entry_quality)  # Punir mais se estava confiante
        else:
            # Final do dataset, reward neutro
            entry_reward = 0.0
            
        return entry_reward
    
    def _calculate_management_reward(self, action: np.ndarray, info: Dict, env) -> float:
        """
        Reward para Management Head baseado em eficiência de gestão
        Considera SL/TP placement, trailing stop usage, etc.
        """
        if 'closed_trade' not in info:
            return 0.0
            
        trade = info['closed_trade']
        management_reward = 0.0
        
        # 1. Eficiência do SL
        if trade.get('exit_reason') == 'stop_loss':
            # Verificar se SL estava bem posicionado
            sl_distance = abs(trade.get('sl_price', 0) - trade.get('entry_price', 0))
            if sl_distance < 5:  # SL muito apertado
                management_reward -= 0.2
            elif sl_distance > 20:  # SL muito largo
                management_reward -= 0.1
            else:
                management_reward += 0.1  # SL adequado
                
        # 2. Eficiência do TP
        elif trade.get('exit_reason') == 'take_profit':
            # Verificar se capturou movimento adequado
            tp_distance = abs(trade.get('tp_price', 0) - trade.get('entry_price', 0))
            if tp_distance < 5:  # TP muito conservador
                management_reward -= 0.1
            elif tp_distance > 30:  # TP irrealista
                management_reward -= 0.2
            else:
                management_reward += 0.2  # TP adequado
                
        # 3. Uso de trailing stop
        if trade.get('trailing_activated', False):
            management_reward += 0.15  # Bonus por usar trailing
            
        # 4. Penalizar gestão de trades muito curtos
        duration = trade.get('duration', 0)
        if duration < 20:  # Menos de 100min
            management_reward -= 0.3
            
        return management_reward
```

### 4️⃣ **INTEGRAÇÃO NO DAYTRADER**

#### Localização: `daytrader.py`

#### Modificações Necessárias:

```python
# 1. IMPORTAR o sistema dual
from trading_framework.rewards.dual_reward_system import DualRewardSystem

# 2. ADICIONAR na __init__ da TradingEnv:
self.dual_reward_system = DualRewardSystem(
    future_window=100,
    movement_threshold=10.0,
    verbose=False
)
self.use_dual_rewards = True  # Flag para ativar/desativar

# 3. MODIFICAR o step() para incluir dual rewards:
def step(self, action):
    # ... código existente ...
    
    # Calcular reward tradicional
    base_reward = self._calculate_reward(...)
    
    # Adicionar dual rewards se ativado
    if self.use_dual_rewards:
        entry_reward, management_reward = self.dual_reward_system.calculate(
            action, info, self
        )
        
        # Combinar rewards com pesos
        ENTRY_WEIGHT = 0.3
        MANAGEMENT_WEIGHT = 0.3
        BASE_WEIGHT = 0.4
        
        total_reward = (BASE_WEIGHT * base_reward + 
                       ENTRY_WEIGHT * entry_reward + 
                       MANAGEMENT_WEIGHT * management_reward)
        
        # Adicionar ao info para logging
        info['entry_reward'] = entry_reward
        info['management_reward'] = management_reward
        info['base_reward'] = base_reward
    else:
        total_reward = base_reward
    
    return obs, total_reward, done, info
```

### 5️⃣ **FLAGS DE CONTROLE E TESTES**

#### Adicionar no início do `daytrader.py`:

```python
# FLAGS DE CONTROLE V3
USE_VARIABLE_VOLATILITY = True  # Ativar volatilidade variável
USE_POSITION_LIFETIME_PENALTY = True  # Ativar penalidade por duração
USE_DUAL_REWARD_SYSTEM = True  # Ativar sistema dual de rewards
DUAL_REWARD_VERBOSE = False  # Logs detalhados do dual reward

# Configurações ajustáveis
VOLATILITY_SCHEDULE = [0.5, 0.5, 1.0, 2.0, 0.5, 1.0, 3.0, 0.5]
MIN_POSITION_DURATION = 50  # Steps mínimos para não ter penalidade
MOVEMENT_THRESHOLD = 10.0  # Pontos de movimento para considerar "mercado ativo"
```

## 📊 MÉTRICAS DE VALIDAÇÃO

### Adicionar logging específico para monitorar eficácia:

```python
class OverTradingMonitor:
    """Monitor para detectar e reportar overtrading"""
    
    def __init__(self):
        self.trades_per_episode = []
        self.avg_duration_per_episode = []
        self.consecutive_trades = []  # Trades sem gap temporal
        
    def log_episode(self, trades):
        self.trades_per_episode.append(len(trades))
        
        if trades:
            avg_duration = np.mean([t['duration'] for t in trades])
            self.avg_duration_per_episode.append(avg_duration)
            
            # Detectar trades consecutivos
            consecutive = 0
            for i in range(1, len(trades)):
                gap = trades[i]['entry_step'] - trades[i-1]['exit_step']
                if gap < 5:  # Menos de 5 steps entre trades
                    consecutive += 1
            self.consecutive_trades.append(consecutive)
    
    def report(self):
        print("\n📊 OVERTRADING REPORT:")
        print(f"  Trades/episódio: {np.mean(self.trades_per_episode):.1f}")
        print(f"  Duração média: {np.mean(self.avg_duration_per_episode):.1f} steps")
        print(f"  Trades consecutivos: {np.mean(self.consecutive_trades):.1f}/ep")
        
        # Alertas
        if np.mean(self.trades_per_episode) > 20:
            print("  ⚠️ ALERTA: Overtrading detectado!")
        if np.mean(self.avg_duration_per_episode) < 30:
            print("  ⚠️ ALERTA: Trades muito curtos!")
```

## 🚀 ORDEM DE IMPLEMENTAÇÃO

1. **PRIMEIRO**: Implementar volatilidade variável (mais simples)
2. **SEGUNDO**: Adicionar penalidade por duração
3. **TERCEIRO**: Implementar dual reward system
4. **QUARTO**: Adicionar monitoring e ajustar hiperparâmetros

## ⚠️ AVISOS IMPORTANTES

1. **TESTAR CADA MUDANÇA ISOLADAMENTE** antes de combinar
2. **COMEÇAR COM PESOS CONSERVADORES** (0.1) e aumentar gradualmente
3. **MONITORAR CONVERGÊNCIA** - dual rewards podem desestabilizar inicialmente
4. **BACKUP DO MODELO ATUAL** antes de qualquer mudança

## 🎯 RESULTADO ESPERADO

- **Redução de 70%** em trades consecutivos
- **Aumento de 150%** na duração média dos trades
- **Entry Quality médio > 0.6** (atual ~0.4)
- **Win rate estável ou melhor**

---

*Documento preparado para implementação pelo Sonnet. Cada seção é independente e pode ser implementada/testada separadamente.*