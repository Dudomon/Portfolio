# 🎯 OPÇÃO 2 ELABORADA: MULTI-SIGNAL CONFLUENCE ENTRY

## 📊 INTELLIGENT COMPONENTS DISPONÍVEIS NO CHERRY

### **1. Market Regime** (cherry.py linha 5837-5903)
```python
{
    'regime': 'trending_up' | 'trending_down' | 'ranging' | 'volatile' | 'crash' | 'unknown',
    'strength': float (0.0-2.0),  # Força da tendência
    'direction': float (1.0 | -1.0 | 0.0)  # Direção
}
```

### **2. Volatility Context** (cherry.py linha 5905-5953)
```python
{
    'level': 'high' | 'normal' | 'low',
    'percentile': float (0.0-1.0),  # Posição no range histórico
    'expanding': bool  # Volatilidade expandindo?
}
```

### **3. Momentum Confluence** (cherry.py linha 5955-6033)
```python
{
    'score': float (0.0-1.0),  # Confluência de indicadores
    'direction': float (-1.0 a 1.0),  # Direção do momentum
    'strength': float (0.0-1.0)  # Força do momentum
}
```

### **4. Features do DataFrame**
```python
# Já calculadas no df (linha 5392-5402):
- volume_momentum: Momentum de volume vs média
- price_position: Posição do preço no range 20-bar
- breakout_strength: Distância para S/R (TP target zones)
- trend_consistency: Consistência da tendência
- support_resistance: SL zone quality
- volatility_regime: Volatilidade 20 vs 50
- market_structure: Volatility spike detection
```

---

## 🎯 OPÇÃO 2: MULTI-SIGNAL CONFLUENCE ENTRY (ELABORADA)

**Conceito**: Sistema de validação em 3 camadas que usa TODOS os intelligent components para determinar qualidade de entrada

### 📊 ARQUITETURA DO SISTEMA

```
ENTRY QUALITY = Layer 1 (40%) + Layer 2 (40%) + Layer 3 (20%)

Layer 1: REGIME + VOLATILITY VALIDATION (40%)
├─ Market Regime Alignment
├─ Volatility Appropriateness
└─ Crash Detection

Layer 2: MOMENTUM + TECHNICAL CONFLUENCE (40%)
├─ Momentum Confluence Score
├─ RSI Divergence Detection
├─ MACD Alignment
└─ Trend Consistency

Layer 3: STRUCTURAL CONFIRMATION (20%)
├─ Breakout Strength (S/R proximity)
├─ Support/Resistance Quality
├─ Price Position in Range
└─ Volume Momentum
```

---

## 🔥 IMPLEMENTAÇÃO COMPLETA

```python
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

            # 🟡 NEUTRO: LONG em ranging (pode funcionar)
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

            # 🟢 BOM: SHORT em crash (pegar o movimento)
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
            # Stops prematuros muito prováveis
            reward -= 0.6
            info['volatility_too_high'] = True

        # 🟠 SUBÓTIMO: Volatilidade extrema baixa
        elif vol_level == 'low' and vol_percentile < 0.15:
            # Targets demorados
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
        reward += divergence_reward * 0.75  # Peso significativo
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
            # Baixa confiança em mercado duvidoso = BOM (reconheceu incerteza)
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

            # ========================================
            # BULLISH DIVERGENCE (para LONG)
            # ========================================
            if entry_decision == 1:
                # Procurar: Preço faz lower low, mas RSI faz higher low

                # Dividir em 2 janelas: antiga (bars 0-10) e recente (bars 10-20)
                price_old_low = np.min(close_recent[5:12])
                price_recent_low = np.min(close_recent[12:])

                rsi_old_low = np.min(rsi_recent[5:12])
                rsi_recent_low = np.min(rsi_recent[12:])

                # Divergência bullish: preço baixa, RSI sobe
                if price_recent_low < price_old_low * 0.998:  # Preço fez lower low (-0.2%)
                    if rsi_recent_low > rsi_old_low + 2:  # RSI fez higher low (+2 pontos)
                        # BULLISH DIVERGENCE confirmada!
                        divergence_strength = (rsi_recent_low - rsi_old_low) / 50.0
                        reward = 1.0 * min(divergence_strength, 1.0)
                        info['bullish_divergence'] = True
                        info['divergence_strength'] = divergence_strength
                        info['price_old_low'] = price_old_low
                        info['price_recent_low'] = price_recent_low
                        info['rsi_old_low'] = rsi_old_low
                        info['rsi_recent_low'] = rsi_recent_low

            # ========================================
            # BEARISH DIVERGENCE (para SHORT)
            # ========================================
            elif entry_decision == 2:
                # Procurar: Preço faz higher high, mas RSI faz lower high

                price_old_high = np.max(close_recent[5:12])
                price_recent_high = np.max(close_recent[12:])

                rsi_old_high = np.max(rsi_recent[5:12])
                rsi_recent_high = np.max(rsi_recent[12:])

                # Divergência bearish: preço sobe, RSI desce
                if price_recent_high > price_old_high * 1.002:  # Preço fez higher high (+0.2%)
                    if rsi_recent_high < rsi_old_high - 2:  # RSI fez lower high (-2 pontos)
                        # BEARISH DIVERGENCE confirmada!
                        divergence_strength = (rsi_old_high - rsi_recent_high) / 50.0
                        reward = 1.0 * min(divergence_strength, 1.0)
                        info['bearish_divergence'] = True
                        info['divergence_strength'] = divergence_strength
                        info['price_old_high'] = price_old_high
                        info['price_recent_high'] = price_recent_high
                        info['rsi_old_high'] = rsi_old_high
                        info['rsi_recent_high'] = rsi_recent_high

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

            # ========================================
            # 3.1 BREAKOUT STRENGTH (40% da Layer 3)
            # ========================================
            # breakout_strength = TP target zones (distância para S/R)
            if 'breakout_strength' in df.columns:
                breakout = df['breakout_strength'].iloc[current_step]

                # ALTO: Resistência/suporte próximo (bom para TP)
                if breakout > 0.65:
                    reward += 0.8
                    info['sr_proximity_good'] = True
                # BAIXO: Longe de S/R (TP distante)
                elif breakout < 0.35:
                    reward -= 0.4
                    info['sr_proximity_bad'] = True

            # ========================================
            # 3.2 SUPPORT/RESISTANCE QUALITY (30% da Layer 3)
            # ========================================
            # support_resistance = SL zone quality
            if 'support_resistance' in df.columns:
                sr_quality = df['support_resistance'].iloc[current_step]

                # ALTO: Zona segura para SL (longe de S/R)
                if sr_quality > 0.6:
                    reward += 0.6
                    info['sl_zone_safe'] = True
                # BAIXO: Zona perigosa para SL (próximo de S/R)
                elif sr_quality < 0.4:
                    reward -= 0.6
                    info['sl_zone_dangerous'] = True

            # ========================================
            # 3.3 PRICE POSITION (15% da Layer 3)
            # ========================================
            # price_position = Posição no range 20-bar
            if 'price_position' in df.columns:
                price_pos = df['price_position'].iloc[current_step]

                # LONG: Melhor comprar quando preço está em baixo do range
                if entry_decision == 1:
                    if price_pos < 0.35:  # Preço no terço inferior
                        reward += 0.3
                        info['price_at_support'] = True
                    elif price_pos > 0.75:  # Preço no topo (comprar caro)
                        reward -= 0.3
                        info['buying_high'] = True

                # SHORT: Melhor vender quando preço está no topo do range
                elif entry_decision == 2:
                    if price_pos > 0.65:  # Preço no terço superior
                        reward += 0.3
                        info['price_at_resistance'] = True
                    elif price_pos < 0.25:  # Preço no fundo (vender barato)
                        reward -= 0.3
                        info['selling_low'] = True

            # ========================================
            # 3.4 VOLUME MOMENTUM (15% da Layer 3)
            # ========================================
            # volume_momentum = Volume vs média
            if 'volume_momentum' in df.columns:
                vol_momentum = df['volume_momentum'].iloc[current_step]

                # Volume expandindo = movimento forte
                if vol_momentum > 0.6:
                    reward += 0.3
                    info['volume_surge'] = True
                # Volume muito baixo = movimento fraco
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
```

---

## 📊 EXEMPLO DE OUTPUT

```python
# Exemplo de entrada PERFEITA (LONG):
{
    'total_reward': 0.85,  # Excelente!
    'layer1_regime_volatility': 0.9,   # trending_up forte + vol normal
    'layer2_momentum_technical': 0.8,  # Alta confluência + divergência bullish
    'layer3_structural': 0.85,         # S/R favorável + preço em suporte

    'regime_perfect_long': True,
    'volatility_optimal': True,
    'high_confluence_aligned': True,
    'bullish_divergence': True,
    'divergence_strength': 0.12,
    'high_confidence_justified': True,
    'sr_proximity_good': True,
    'sl_zone_safe': True,
    'price_at_support': True,
    'volume_surge': True
}

# Exemplo de entrada PÉSSIMA (LONG em crash):
{
    'total_reward': -0.92,  # Péssimo!
    'layer1_regime_volatility': -0.95,  # CRASH detectado!
    'layer2_momentum_technical': -0.7,  # Momentum contra + baixa confluência
    'layer3_structural': 0.2,           # Estrutura OK mas irrelevante

    'crash_buy_blocked': True,          # BLOQUEIO CRÍTICO
    'confluence_against': True,
    'overconfidence_danger': True,
    'buying_high': True
}
```

---

## 🎯 VANTAGENS DESTE SISTEMA

1. ✅ **Usa TODOS os intelligent components** do cherry (não desperdiça informação)
2. ✅ **Sistema de 3 camadas** (crítico → técnico → estrutural)
3. ✅ **Detecta divergências** (sinal técnico forte)
4. ✅ **Valida confidence** (apropriada ao contexto)
5. ✅ **Bloqueio de crash** (NUNCA compra em queda forte)
6. ✅ **Weights balanceados** (40/40/20)
7. ✅ **Normalização** (tanh para -1.0 a +1.0)
8. ✅ **Info detalhado** (debugging completo)

---

## 💡 INTEGRAÇÃO NO ENTRY TIMING

```python
# entry_timing_rewards.py

class EntryTimingRewards:
    def __init__(self):
        # ... código existente ...

        # 🎯 NOVO: Sistema de Multi-Signal Confluence
        self.multi_signal_system = MultiSignalConfluenceEntry()

    def _calculate_entry_confluence_reward(self, env, entry_decision: int) -> Tuple[float, Dict]:
        """
        COMPONENTE 2: Entry Confluence Reward (30% do Entry Timing)
        """
        reward = 0.0
        info = {}

        # 2.1 Multi-Signal Confluence (70% deste componente = 21% do Entry Timing)
        multi_signal_reward, multi_signal_info = self.multi_signal_system.calculate_multi_signal_entry_reward(
            env, entry_decision, action
        )
        reward += multi_signal_reward * 0.70
        info.update(multi_signal_info)

        # 2.2 Multi-Indicator Confirmation (30% deste componente = 9% do Entry Timing)
        confirmation_reward, confirm_info = self._calculate_multi_indicator_confirmation(
            env, entry_decision, ...
        )
        reward += confirmation_reward * 0.30
        info.update(confirm_info)

        # Aplicar peso do componente (30% do Entry Timing)
        final_reward = reward * self.confluence_weight

        return final_reward, info
```

---

## 🏆 COMPARAÇÃO: OPÇÃO 1 vs OPÇÃO 2

| Aspecto | Opção 1 (Entry Timing After Loss) | Opção 2 (Multi-Signal Confluence) |
|---------|-----------------------------------|-----------------------------------|
| **Foco** | Timing comportamental | Qualidade técnica |
| **Complexity** | Simples (usa apenas trades history) | Complexa (usa todos intelligent components) |
| **Impacto** | Previne revenge trading | Melhora timing de entrada |
| **Peso** | -0.8 penalty / +0.2 bonus | -1.0 a +1.0 (range completo) |
| **Implementação** | 50 linhas | 400 linhas |
| **Manutenção** | Fácil | Média |
| **Debugging** | Simples | Detalhado (3 layers) |

---

## 💡 RECOMENDAÇÃO FINAL

**USE AMBAS!**

- **Opção 1** (Entry Timing After Loss): 15% do Entry Confluence
- **Opção 2** (Multi-Signal Confluence): 70% do Entry Confluence
- Multi-Indicator Confirmation: 15% do Entry Confluence

Isso combina:
- ✅ Prevenção de revenge trading (comportamental)
- ✅ Validação técnica robusta (estrutural)
- ✅ Uso máximo dos intelligent components
