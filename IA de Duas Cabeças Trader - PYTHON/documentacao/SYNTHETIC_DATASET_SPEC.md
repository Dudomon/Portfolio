# 🔥 SYNTHETIC DATASET SPECIFICATION
## Dataset Sintético Inteligente para Trading de Ouro

### 📊 PROBLEMA IDENTIFICADO
- Dataset atual: 99.8% baixa volatilidade
- V7 Intuition "over-trained" em micro-movimentos
- Max positions sempre atingido por falta de seletividade real

### 🎯 OBJETIVOS DO DATASET SINTÉTICO

#### 1. DISTRIBUIÇÃO DE VOLATILIDADE REALISTA
```
- 45% Consolidação (baixa volatilidade) - 0.002-0.008%
- 35% Tendências (média volatilidade) - 0.008-0.025% 
- 15% Breakouts (alta volatilidade) - 0.025-0.080%
- 5% Eventos extremos (muito alta) - 0.080-0.200%
```

#### 2. PADRÕES DE MERCADO AUTÊNTICOS
```
- Gaps de abertura (2-5% dos dias)
- Reversões em suporte/resistência  
- Breakouts com retestes
- Fakeouts (30% dos breakouts)
- Consolidações triangulares/retangulares
- Trends com pullbacks realistas
```

#### 3. CICLOS TEMPORAIS INTELIGENTES
```
- Horários de maior volume: 8h-12h, 13h30-17h (GMT)
- Baixa atividade: 17h-22h, 2h-6h
- Eventos de notícias: spikes aleatórios
- Fins de semana: gaps de abertura
```

### 🏗️ ARQUITETURA DO GERADOR

#### MÓDULO 1: BASE PRICE ENGINE
```python
class BasePriceEngine:
    def __init__(self):
        self.base_trend = 0.0001  # Trend diário médio
        self.mean_reversion_strength = 0.3
        self.momentum_persistence = 0.7
    
    def generate_base_movement(self, current_price, regime):
        # Gera movimento base considerando regime atual
```

#### MÓDULO 2: Volatility Regime Controller  
```python
class VolatilityRegimeController:
    def __init__(self):
        self.regimes = {
            'consolidation': {'prob': 0.45, 'vol_range': (0.002, 0.008)},
            'trending': {'prob': 0.35, 'vol_range': (0.008, 0.025)},
            'breakout': {'prob': 0.15, 'vol_range': (0.025, 0.080)},
            'extreme': {'prob': 0.05, 'vol_range': (0.080, 0.200)}
        }
```

#### MÓDULO 3: Pattern Injection System
```python
class PatternInjector:
    def inject_support_resistance(self, price_series):
        # Injeta níveis de S/R baseados em Fibonacci
    
    def inject_breakout_pattern(self, price_series):
        # Cria breakouts realistas com volume
        
    def inject_fakeout_pattern(self, price_series):
        # Simula breakouts falsos (bear/bull traps)
```

#### MÓDULO 4: Market Microstructure
```python
class MarketMicrostructure:
    def add_bid_ask_spread(self, prices):
        # Adiciona spread realista
        
    def add_intraday_seasonality(self, prices, timestamps):
        # Padrões de volume/volatilidade por horário
        
    def add_weekend_gaps(self, prices, timestamps):
        # Gaps de fim de semana realistas
```

### 📈 GERAÇÃO EM CAMADAS

#### LAYER 1: Macroeconomic Trends
- Trends de longo prazo (semanas/meses)
- Ciclos sazonais do ouro
- Correlação com USD/inflação

#### LAYER 2: Daily Market Regimes  
- Determina regime do dia (consolidação/trend/breakout)
- Duração típica de cada regime
- Transições suaves entre regimes

#### LAYER 3: Intraday Patterns
- Padrões de abertura/fechamento
- Lunch time consolidation
- Power hour movements

#### LAYER 4: Micro Movements
- Noise realista 
- Order flow simulation
- HFT-style micro reversals

### 🎯 FEATURES ESPECIAIS

#### 1. ADAPTIVE DIFFICULTY
```python
# Dataset progressivo para curriculum learning
Easy Mode: Padrões óbvios, volatilidade alta
Medium Mode: Mix realista
Hard Mode: Muito noise, fakeouts frequentes
Expert Mode: Condições extremas
```

#### 2. VALIDATION MODES
```python
# Diferentes tipos de validação
Stress Test: Só condições adversas
Calm Market: Só baixa volatilidade  
Volatile Market: Só alta volatilidade
Mixed Conditions: Distribuição realista
```

#### 3. ECONOMIC EVENTS SIMULATION
```python
# Simula eventos fundamentais
NFP Release: Spike + reversão
Fed Meetings: Volatilidade pré/pós
Inflation Data: Trends direcionais
Geopolitical: Gaps + uncertainty
```

### 📊 MÉTRICAS DE QUALIDADE

#### Statistical Validation
- Kurtosis similar ao ouro real (3.2-4.8)
- Skewness próximo de zero (-0.2 a +0.2)
- Autocorrelação realista
- Heteroscedasticidade apropriada

#### Trading Validation  
- Sharpe ratio de estratégias simples (0.3-0.8)
- Drawdown máximo realista (15-25%)
- Win rate de mean reversion (45-55%)
- Win rate de trend following (35-45%)

#### Visual Validation
- Charts indistinguíveis de dados reais
- Padrões reconhecíveis por traders humanos
- Volume/price relationship convincente

### 🛠️ IMPLEMENTAÇÃO EM FASES

#### FASE 1: MVP Generator (1-2 dias)
- Engine básico com 4 regimes de volatilidade
- Padrões simples (trends, consolidações)
- 1M de barras 5min para teste inicial

#### FASE 2: Pattern Enhancement (2-3 dias)  
- Sistema de S/R dinâmico
- Breakouts e fakeouts realistas
- Intraday seasonality

#### FASE 3: Advanced Features (3-4 dias)
- Economic events simulation
- Multi-timeframe coherence
- Adaptive difficulty system

#### FASE 4: Validation & Tuning (2 dias)
- Statistical validation
- A/B testing vs dados reais
- Fine-tuning de parâmetros

### 📁 ESTRUTURA DE ARQUIVOS

```
synthetic_dataset_generator/
├── core/
│   ├── base_engine.py          # Motor principal
│   ├── volatility_controller.py # Controle de regimes
│   └── pattern_injector.py     # Injeção de padrões
├── features/
│   ├── support_resistance.py   # Níveis S/R
│   ├── breakout_system.py     # Sistema de breakouts
│   └── microstructure.py     # Microestrutura
├── validation/
│   ├── statistical_tests.py   # Testes estatísticos
│   ├── visual_validation.py   # Validação visual
│   └── trading_validation.py  # Backtest validation
├── configs/
│   ├── gold_5m_config.yaml    # Config para ouro 5min
│   ├── forex_config.yaml      # Config para forex
│   └── crypto_config.yaml     # Config para crypto
└── generators/
    ├── curriculum_generator.py # Geração progressiva
    ├── stress_test_generator.py # Cenários extremos
    └── mixed_generator.py      # Dataset balanceado
```

### 🎯 PARÂMETROS CONFIGURÁVEIS

```yaml
# gold_5m_realistic.yaml
market_config:
  base_price: 2000.0
  daily_drift: 0.0001
  annual_volatility: 0.18
  
regimes:
  consolidation:
    probability: 0.45
    min_duration: 50  # barras
    max_duration: 300
    volatility_range: [0.002, 0.008]
    
  trending:
    probability: 0.35
    min_duration: 100
    max_duration: 800
    volatility_range: [0.008, 0.025]
    
patterns:
  support_resistance:
    strength_levels: [0.3, 0.5, 0.7, 0.9]
    retest_probability: 0.7
    break_probability: 0.3
    
  breakouts:
    false_breakout_rate: 0.3
    volume_surge_multiplier: 2.5
    retest_probability: 0.8
```

### 🚀 RESULTADO ESPERADO

**DATASET FINAL:**
- 2M+ barras de 5min (≈7 anos de dados)
- Distribuição de volatilidade realista
- Padrões autênticos e desafiadores
- Validação estatística rigorosa
- Multiple difficulty levels

**IMPACTO NO V7:**
- ✅ Aprende quando NÃO tradear
- ✅ Identifica oportunidades reais
- ✅ Max positions raramente atingido
- ✅ Performance estável em diferentes condições
- ✅ Generalização superior

**TEMPO ESTIMADO:** 8-10 dias para implementação completa
**ROI ESPERADO:** 300-500% melhoria na performance do modelo