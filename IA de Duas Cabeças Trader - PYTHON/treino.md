# 🎯 SPEC DE TREINAMENTO - MODELO TRADER EXCEPCIONAL V7 INTUITION

## 📊 OBJETIVO PRINCIPAL
Criar um agente de trading de **alta performance** para **GOLD (GC_YAHOO)** capaz de:
- ✅ **Win Rate > 55%** consistente
- ✅ **Profit Factor > 1.5** sustentável  
- ✅ **Sharpe Ratio > 1.2** (risk-adjusted)
- ✅ **Max Drawdown < 15%** (gestão de risco)
- ✅ **Retorno médio > 20%** ao ano

## 🏗️ ARQUITETURA BASE: V7 INTUITION

### Core Components
- **Backbone Unificado**: Visão compartilhada do mercado
- **Actor LSTM**: Decisões temporais com memória
- **Critic MLP**: Avaliação rápida de value
- **Gradient Mixing**: Cross-pollination entre heads
- **Action Space 11D**: Controle completo de trading

### Action Space Breakdown (11 dimensões)
```python
[0] entry_decision     # 0=HOLD, 1=LONG, 2=SHORT
[1] entry_quality      # 0.0-1.0 (confiança na entrada)
[2] temporal_signal    # -1.0 a +1.0 (timing bias)
[3] risk_appetite      # 0.0-1.0 (agressividade)
[4] market_regime_bias # -1.0 a +1.0 (regime detection)
[5-7] sl_adjusts       # Stop loss dinâmico (3 níveis)
[8-10] tp_adjusts      # Take profit dinâmico (3 níveis)
```

## 🥇 ESPECIFICAÇÕES PARA GOLD (GC)

### Características do Ativo
- **Volatilidade Média Diária**: 0.8-1.5%
- **Range Médio**: $15-30 por dia
- **Horário Principal**: London/NY overlap (8h-12h EST)
- **Correlações**: USD inversa, Risk-off asset
- **Sazonalidade**: Alta em incerteza geopolítica

### Ranges Otimizados SL/TP para Gold
```python
GOLD_TRADING_PARAMS = {
    # Stop Loss: Mais apertado que outros ativos
    'stop_loss_base': 5.0,      # $5 base (0.25% em $2000)
    'stop_loss_range': (3.0, 12.0),  # $3-12 flexível
    'stop_loss_levels': [
        {'multiplier': 0.6, 'name': 'tight'},    # $3-7.2
        {'multiplier': 1.0, 'name': 'normal'},   # $5-12
        {'multiplier': 1.5, 'name': 'wide'}      # $7.5-18
    ],
    
    # Take Profit: Targets realistas para daytrading
    'take_profit_base': 10.0,    # $10 base (0.5% em $2000)
    'take_profit_range': (5.0, 25.0),  # $5-25 flexível
    'take_profit_levels': [
        {'multiplier': 0.5, 'name': 'quick'},    # $5-12.5
        {'multiplier': 1.0, 'name': 'normal'},   # $10-25
        {'multiplier': 2.0, 'name': 'runner'}    # $20-50
    ],
    
    # Risk Management
    'risk_reward_min': 1.5,      # Mínimo 1.5:1
    'position_size_max': 0.02,   # Max 2% do portfolio
    'daily_loss_limit': 0.03,    # Max 3% perda diária
    'trailing_activation': 8.0,   # Ativar trailing em $8 profit
    'trailing_distance': 4.0      # Trailing stop $4 do pico
}
```

## 📈 FASES DE TREINAMENTO PROGRESSIVO

### Phase 1: Foundation (0-2M steps)
**Objetivo**: Aprender mecânica básica de trading
- Dataset: Condições normais de mercado
- Foco: Entry/Exit timing, position sizing básico
- Reward: Simples (PnL + win rate)
- Success Criteria: Win rate > 45%, Drawdown < 20%

### Phase 2: Risk Management (2M-4M steps)
**Objetivo**: Dominar gestão de risco
- Dataset: Mix 50% normal + 50% volátil
- Foco: Stop loss dinâmico, position sizing adaptativo
- Reward: PnL + risk metrics (Sharpe, drawdown penalty)
- Success Criteria: Profit Factor > 1.0, Max DD < 15%

### Phase 3: Market Regimes (4M-6M steps)
**Objetivo**: Adaptar a diferentes condições
- Dataset: Trending (30%), Ranging (40%), Volatile (30%)
- Foco: Regime detection, strategy switching
- Reward: Regime-specific rewards
- Success Criteria: Performance consistente em todos regimes

### Phase 4: Advanced Patterns (6M-8M steps)
**Objetivo**: Reconhecer patterns complexos
- Dataset: Patterns específicos de Gold (breakouts, reversals)
- Foco: Multi-timeframe analysis, confluence trading
- Reward: Pattern completion bonus
- Success Criteria: Win rate > 50% em patterns

### Phase 5: Optimization (8M-10M steps)
**Objetivo**: Fine-tuning e maximização
- Dataset: Full historical + recent data
- Foco: Otimização de entries, maximização de RR
- Reward: Sharpe-weighted returns
- Success Criteria: Sharpe > 1.0, PF > 1.3

### Phase 6: Mastery (10M-12M steps)
**Objetivo**: Performance excepcional consistente
- Dataset: Live-like conditions com slippage/spread
- Foco: Consistência, adaptabilidade, robustez
- Reward: Full complexity (PnL + risk + consistency)
- Success Criteria: Todos KPIs atingidos

## 🎯 REWARD SYSTEM PROGRESSIVO

### Base Reward Components
```python
reward = (
    0.40 * pnl_component +          # Lucro direto
    0.20 * risk_adjusted_component + # Sharpe/Sortino
    0.15 * consistency_component +   # Win rate stability
    0.15 * risk_management_component + # DD control
    0.10 * execution_quality_component # Entry/Exit precision
)
```

### Adaptive Weights por Phase
- **Phase 1-2**: PnL 60%, Risk 40%
- **Phase 3-4**: PnL 40%, Risk 30%, Patterns 30%
- **Phase 5-6**: Balanced 20% each component

## 🔧 HYPERPARAMETERS OTIMIZADOS

### PPO Configuration
```python
PPO_CONFIG = {
    'learning_rate': 3e-5,           # Stable learning
    'n_steps': 2048,                  # Good trajectory length
    'batch_size': 64,                 # Optimal for GPU
    'n_epochs': 10,                   # Sufficient updates
    'gamma': 0.99,                    # Long-term thinking
    'gae_lambda': 0.95,              # Advantage estimation
    'clip_range': 0.2,               # Standard clipping
    'clip_range_vf': None,           # No value clipping
    'ent_coef': 0.01,                # Exploration bonus
    'vf_coef': 0.5,                  # Value function weight
    'max_grad_norm': 0.5,            # Gradient clipping
    'target_kl': 0.02                # KL divergence limit
}
```

### V7 Intuition Specific
```python
V7_CONFIG = {
    'v7_shared_lstm_hidden': 512,
    'v7_features_dim': 256,
    'backbone_shared_dim': 256,
    'regime_embed_dim': 32,
    'gradient_mixing_strength': 0.3,
    'enable_interference_monitoring': True,
    'adaptive_sharing': True
}
```

## 📊 MÉTRICAS DE SUCESSO

### KPIs Primários
1. **Win Rate**: > 55% em 1000+ trades
2. **Profit Factor**: > 1.5 sustentável
3. **Sharpe Ratio**: > 1.2 anualizado
4. **Max Drawdown**: < 15% do capital
5. **Recovery Time**: < 50 trades após DD

### KPIs Secundários
- Average Win/Loss Ratio: > 1.8
- Consistency Score: > 0.7 (estabilidade)
- Execution Quality: > 80% (timing)
- Adaptation Speed: < 100 trades para novo regime
- Risk-Reward Achievement: > 70% dos targets

## 🚀 ESTRATÉGIAS ESPECIALIZADAS PARA GOLD

### 1. London Open Breakout
- Horário: 3:00-4:00 AM EST
- Setup: Range dos primeiros 30min
- Entry: Breakout com volume
- SL: Oposto do range
- TP: 2x range inicial

### 2. NY Session Momentum
- Horário: 8:30-10:30 AM EST
- Setup: Continuação de trend London
- Entry: Pullback to VWAP
- SL: Below VWAP
- TP: Previous high/low

### 3. Risk-Off Reversals
- Trigger: VIX spike, USD weakness
- Setup: Oversold/Overbought extremes
- Entry: Reversal candle patterns
- SL: Beyond extreme
- TP: 50% retracement

### 4. Asian Session Range
- Horário: 7:00 PM - 2:00 AM EST
- Setup: Tight range trading
- Entry: Range boundaries
- SL: Outside range
- TP: Opposite boundary

## 🛡️ RISK MANAGEMENT FRAMEWORK

### Position Sizing
```python
position_size = min(
    base_size * (1 + confidence_score),
    max_position_size,
    available_capital * 0.02
)
```

### Dynamic Stop Loss
```python
stop_loss = base_sl * (
    1.0 + 
    volatility_multiplier * 0.3 +
    regime_adjustment * 0.2 -
    confidence_bonus * 0.1
)
```

### Trailing Stop Activation
```python
if profit >= activation_threshold:
    trailing_stop = max(
        entry_price + (profit * 0.5),
        current_price - trailing_distance
    )
```

## 📅 CRONOGRAMA DE IMPLEMENTAÇÃO

### Semana 1-2: Setup e Baseline
- [ ] Configurar ambiente com parâmetros Gold
- [ ] Estabelecer baseline com modelo atual
- [ ] Implementar métricas de tracking
- [ ] Criar datasets por phase

### Semana 3-6: Training Phases 1-3
- [ ] Phase 1: Foundation training (2M steps)
- [ ] Phase 2: Risk management (2M steps)
- [ ] Phase 3: Market regimes (2M steps)
- [ ] Checkpoints e avaliações intermediárias

### Semana 7-10: Training Phases 4-6
- [ ] Phase 4: Advanced patterns (2M steps)
- [ ] Phase 5: Optimization (2M steps)
- [ ] Phase 6: Mastery (2M steps)
- [ ] Fine-tuning final

### Semana 11-12: Validation e Deploy
- [ ] Backtesting extensivo
- [ ] Paper trading validation
- [ ] Stress testing
- [ ] Deploy preparation

## 🎓 TÉCNICAS AVANÇADAS

### 1. Curriculum Learning Adaptativo
- Ajuste automático de dificuldade
- Replay de cenários difíceis
- Progressive data augmentation

### 2. Meta-Learning Components
- Rapid adaptation a novos padrões
- Few-shot learning para eventos raros
- Transfer learning de outros metais

### 3. Ensemble Strategies
- Multiple timeframe consensus
- Confidence-weighted decisions
- Voting mechanism para entries

### 4. Advanced Reward Shaping
- Curiosity-driven exploration
- Hindsight experience replay
- Inverse reinforcement learning

## 📈 MONITORAMENTO E AJUSTES

### Real-time Metrics
- Dashboard com KPIs em tempo real
- Alertas para degradação de performance
- Logs detalhados de todas decisões

### Adjustment Triggers
- Performance drop > 10%: Review immediato
- Drawdown > 10%: Reduzir risk appetite
- Win rate < 45%: Retrain última phase
- Novo regime detectado: Adaptive learning

### Continuous Improvement
- A/B testing de strategies
- Hyperparameter optimization ongoing
- Feature engineering iterativo
- Dataset expansion mensal

## 🏆 RESULTADO ESPERADO

Um modelo de trading para Gold que seja:
- **Lucrativo**: 20-30% retorno anual consistente
- **Robusto**: Performance estável em diferentes condições
- **Adaptável**: Rápida adaptação a mudanças de mercado
- **Confiável**: Drawdowns controlados e recuperação rápida
- **Escalável**: Capaz de gerenciar portfolios maiores

---

**NOTA**: Este spec deve ser revisado e ajustado baseado nos resultados de cada fase. O sucesso depende de iteração contínua e refinamento baseado em dados reais de performance.

**FILOSOFIA CORE**: "Disciplina supera inteligência. Consistência supera brillhantismo. Risk management supera profit maximization."