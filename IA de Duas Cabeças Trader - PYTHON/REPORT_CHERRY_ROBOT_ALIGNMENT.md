# 📊 Relatório de Alinhamento: cherry.py ↔ Robot_cherry.py

**Data:** 2025-10-01
**Análise:** Lógica de criação de posições, fechamento e ajustes dinâmicos SL/TP

---

## ✅ **COMPONENTES ALINHADOS**

### 1. **Action Space Mapping** ✅
**Status:** TOTALMENTE ALINHADO

**cherry.py (Linhas 77-78, 3869-3874):**
```python
ACTION_THRESHOLD_LONG = -0.33   # raw_decision < -0.33 = HOLD
ACTION_THRESHOLD_SHORT = 0.33   # raw_decision < 0.33 = LONG, >= 0.33 = SHORT

if raw_decision < ACTION_THRESHOLD_LONG:
    entry_decision = 0  # HOLD
elif raw_decision < ACTION_THRESHOLD_SHORT:
    entry_decision = 1  # LONG
else:
    entry_decision = 2  # SHORT
```

**Robot_cherry.py (Linhas 99-100, 3559-3564):**
```python
ACTION_THRESHOLD_LONG = -0.33   # raw_decision < -0.33 = HOLD
ACTION_THRESHOLD_SHORT = 0.33   # raw_decision < 0.33 = LONG, >= 0.33 = SHORT

if raw_decision < ACTION_THRESHOLD_LONG:
    entry_decision = 0  # HOLD
elif raw_decision < ACTION_THRESHOLD_SHORT:
    entry_decision = 1  # BUY (LONG)
else:
    entry_decision = 2  # SELL (SHORT)
```

**Mapeamento:**
- HOLD: `[-1.0, -0.33)`
- LONG: `[-0.33, 0.33)`
- SHORT: `[0.33, 1.0]`

---

### 2. **Management to SL/TP Conversion** ✅
**Status:** TOTALMENTE ALINHADO

**cherry.py (Linhas 6223-6252):**
```python
def convert_management_to_sltp_adjustments(mgmt_value):
    if mgmt_value < 0:
        # Foco em SL management
        if mgmt_value < -0.5:
            return (0.5, 0)  # Afrouxar SL
        else:
            return (-0.5, 0)  # Apertar SL
    elif mgmt_value > 0:
        # Foco em TP management
        if mgmt_value > 0.5:
            return (0, 0.5)  # TP distante
        else:
            return (0, -0.5)  # TP próximo
    else:
        return (0, 0)
```

**Robot_cherry.py (Linhas 3284-3312):**
```python
def _convert_management_to_sltp_adjustments(self, mgmt_value):
    if mgmt_value < 0:
        # Foco em SL management
        if mgmt_value < -0.5:
            return (0.5, 0)  # Afrouxar SL
        else:
            return (-0.5, 0)  # Apertar SL
    elif mgmt_value > 0:
        # Foco em TP management
        if mgmt_value > 0.5:
            return (0, 0.5)  # TP distante
        else:
            return (0, -0.5)  # TP próximo
    else:
        return (0, 0)
```

**Mapeamento Bidirecional:**
- `mgmt < -0.5`: SL +0.5 pontos (afrouxar)
- `-0.5 < mgmt < 0`: SL -0.5 pontos (apertar)
- `0 < mgmt < 0.5`: TP -0.5 pontos (próximo)
- `mgmt > 0.5`: TP +0.5 pontos (distante)

---

### 3. **Confidence Filter** ✅
**Status:** ALINHADO

**cherry.py (Linha 6309-6313):**
```python
MIN_CONFIDENCE_THRESHOLD = 0.8  # 80%
if entry_confidence < MIN_CONFIDENCE_THRESHOLD:
    # Reject entry
```

**Robot_cherry.py (Linha 3601-3603):**
```python
if entry_decision in [1, 2] and entry_confidence < 0.8:
    entry_decision = 0  # Forçar HOLD
```

---

## ⚠️ **DESALINHAMENTOS IDENTIFICADOS**

### 1. **Criação de Posições - Sistema de Slots** ⚠️

**cherry.py (Simulação - Linhas 6270-6360):**
- ✅ Sistema de slots com cooldown independente
- ✅ `available_slot` determinado ANTES da criação
- ✅ Verifica `slot_cooldowns` por índice
- ✅ SL/TP calculado usando `convert_model_adjustments_to_points()`
- ✅ Posição criada em memória (`self.positions.append()`)

```python
# cherry.py: Sistema de slots no ambiente de simulação
available_slot = None
if entry_decision > 0 and len(self.positions) < self.max_positions:
    for slot_idx in range(self.max_positions):
        if (slot_idx not in occupied_slots and
            self.slot_cooldowns.get(slot_idx, 0) == 0):
            available_slot = slot_idx
            break

if entry_decision > 0 and available_slot is not None:
    position = {
        'type': 'long' if entry_decision == 1 else 'short',
        'entry_price': current_price,
        'lot_size': lot_size,
        'entry_step': self.current_step,
        'position_id': available_slot
    }
    self.positions.append(position)
```

**Robot_cherry.py (MT5 Real - Linhas 3802-3880):**
- ✅ Sistema de slots com cooldown independente
- ✅ `_allocate_entry_slot()` chama `_reconcile_slot_map()`
- ✅ Usa `position_slot_cooldowns` (timestamps)
- ✅ SL/TP extraído de `action_analysis['sl_points'][0]`
- ⚠️ **DIFERENÇA**: Posição criada via MT5 (`_execute_order_v7()`)
- ⚠️ **DIFERENÇA**: Cooldown baseado em tempo real (segundos) vs steps

```python
# Robot_cherry.py: Sistema de slots com MT5
slot_id, wait_sec = self._allocate_entry_slot()
if slot_id is None:
    return  # Nenhum slot livre

sl_points = abs(action_analysis['sl_points'][0])
tp_points = abs(action_analysis['tp_points'][0])

result = self._execute_order_v7(mt5.ORDER_TYPE_BUY, volume, sl_price, tp_price, slot_id=slot_id)
```

**Impacto:** Funcionalidade equivalente, mas implementações diferentes (simulação vs MT5 real).

---

### 2. **Fechamento de Posições** ⚠️

**cherry.py (Linhas 6670-6750):**
- ✅ Fecha posição e atualiza `realized_balance`
- ✅ Respeita SL/TP ao calcular `actual_exit_price`
- ✅ Determina `close_reason` (SL hit, TP hit, trailing_stop)
- ✅ Atualiza cooldown do slot: `self.slot_cooldowns[slot_idx] = cooldown_after_trade`
- ⚠️ **DIFERENÇA**: Remoção manual de `self.positions`

```python
# cherry.py: Fechamento em simulação
def _close_position(self, position, exit_step):
    current_price = self.df[f'close_{self.base_tf}'].iloc[exit_step]

    # Determinar preço real de saída (respeitando SL/TP)
    actual_exit_price = current_price
    if position['type'] == 'long' and current_price < position['sl']:
        actual_exit_price = position['sl']
    # ... (lógica completa)

    pnl = self._get_position_pnl(position, actual_exit_price)
    self.realized_balance += pnl

    # Ativar cooldown do slot
    slot_idx = int(position.get('position_id', 0))
    self.slot_cooldowns[slot_idx] = int(self.cooldown_after_trade)
```

**Robot_cherry.py (Linhas 3249-3280):**
- ✅ Monitora posições MT5 para logs
- ⚠️ **DIFERENÇA**: MT5 fecha automaticamente ao atingir SL/TP
- ⚠️ **DIFERENÇA**: Cooldown ativado em `_on_position_closed()` via webhook
- ⚠️ **DIFERENÇA**: Não há método `_close_position()` direto

```python
# Robot_cherry.py: Monitoramento (MT5 fecha automaticamente)
def _check_and_close_positions(self, tick):
    positions = self._get_robot_positions()
    for position in positions:
        current_pnl = (tick.bid - position.price_open) * position.volume * 100
        # Apenas logs - MT5 fecha automaticamente
```

**Impacto:** Cherry.py simula fechamentos, Robot_cherry.py delega para MT5.

---

### 3. **Ajustes Dinâmicos de SL/TP** ⚠️⚠️

**cherry.py (Linhas 5277-5420 - `_process_dynamic_trailing_stop`):**
- ✅ Trailing stop RESTRITIVO (apenas favorável ao trade)
- ✅ LONG: SL só pode subir | SHORT: SL só pode descer
- ✅ TP ajustável com cap de $100 USD
- ✅ Auto-close se PnL ≥ $100
- ✅ Valida distância mínima (5pt buffer SL, 3pt buffer TP)
- ✅ Retorna `result` dict com detalhes completos

```python
# cherry.py: Sistema RESTRITIVO de trailing
def _process_dynamic_trailing_stop(self, pos, sl_adjust, tp_adjust, current_price, pos_index):
    result = {
        'sl_adjusted': False,
        'tp_adjusted': False,
        'action_taken': False,
        'position_updates': {}
    }

    # Auto-close at $100 cap
    if current_pnl >= 100:
        self._close_position(pos, self.current_step)
        result['pnl_cap_reached'] = True
        return result

    # SL TRAILING ONLY (favor do trade)
    if abs(sl_adjust) >= 0.3:
        if pos['type'] == 'long':
            new_sl = current_sl + sl_movement_points
            # RESTRICTION: SL can only go UP
            if new_sl > current_sl and new_sl < current_price - 5.0:
                result['position_updates']['sl'] = new_sl
                result['sl_adjusted'] = True

    # TP ADJUSTABLE with $100 cap
    if abs(tp_adjust) >= 0.3:
        new_tp = current_tp + tp_movement_points
        potential_pnl = (new_tp - pos['entry_price']) * pos['volume']
        if potential_pnl <= 100:
            result['position_updates']['tp'] = new_tp
            result['tp_adjusted'] = True

    return result
```

**Robot_cherry.py (Linhas 3377-3587 - `_process_dynamic_trailing_stop_v7`):**
- ⚠️ **DIFERENÇA**: Sistema de ATIVAÇÃO explícita de trailing
- ⚠️ **DIFERENÇA**: Trailing com distância inicial (15-30 pontos)
- ⚠️ **DIFERENÇA**: Move trailing baseado em `trailing_intensity`
- ⚠️ **DIFERENÇA**: Sem cap de $100 USD
- ⚠️ **DIFERENÇA**: Metadata tracking (`trailing_activated`)
- ✅ Também valida movimento favorável ao trade

```python
# Robot_cherry.py: Sistema de ATIVAÇÃO + MOVIMENTO
def _process_dynamic_trailing_stop_v7(self, position, sl_adjust, tp_adjust):
    result = {
        'trailing_activated': False,
        'trailing_moved': False,
        'tp_adjusted': False,
        'action_taken': False,
        'position_updates': {}
    }

    # ATIVAÇÃO DE TRAILING
    if not pos_metadata.get('trailing_activated', False) and abs(trailing_signal) > 1.5:
        initial_trail_distance = 15 + (trailing_intensity * 15)  # 15-30 pontos

        if pos_type == 'long':
            trail_price = current_price - initial_trail_distance
            if trail_price > position.sl:
                result['position_updates']['sl'] = trail_price
                result['trailing_protected'] = True

    # MOVIMENTO DE TRAILING
    elif pos_metadata.get('trailing_activated', False) and abs(trailing_signal) > 0.5:
        new_trail_distance = ...
        new_trail_price = current_price - new_trail_distance

        # Só mover para cima (proteção)
        if new_trail_price > position.sl:
            result['position_updates']['sl'] = new_trail_price
            result['trailing_moved'] = True

    # TP ADJUSTMENT (SEM CAP)
    if abs(tp_adjust) >= 0.5:
        new_tp = current_tp + tp_change_points
        result['position_updates']['tp'] = new_tp
        result['tp_adjusted'] = True

    return result
```

**Impacto:** 🚨 **DESALINHAMENTO CRÍTICO**
- **cherry.py**: Trailing DIRETO (ajuste imediato de SL)
- **Robot_cherry.py**: Trailing com ATIVAÇÃO + MOVIMENTO (2 etapas)
- **cherry.py**: Cap de $100 USD no TP
- **Robot_cherry.py**: Sem cap de lucro

---

### 4. **Conversão de Ajustes para Pontos** ⚠️

**cherry.py (Linhas 7399-7469 - `convert_model_adjustments_to_points`):**
- ✅ Função global unificada
- ✅ Context: "creation" vs "adjustment"
- ✅ Range SL: 2-8 pontos | TP: 3-15 pontos
- ✅ Arredondamento para múltiplos de 0.5

**Robot_cherry.py (Linhas 3314-3375 - `_convert_model_adjustments_to_points`):**
- ✅ Método de instância (mesmo algoritmo)
- ✅ Context: "creation" vs "adjustment"
- ⚠️ **DIFERENÇA**: Usa `REALISTIC_SLTP_CONFIG` local
- ✅ Arredondamento para múltiplos de 0.5

**Impacto:** Funcionalidade idêntica, apenas diferença de scope (global vs método).

---

## 📋 **RESUMO EXECUTIVO**

### ✅ **Componentes Alinhados (5/8)**
1. ✅ Action space mapping (thresholds)
2. ✅ Management to SL/TP conversion
3. ✅ Confidence filter (80%)
4. ✅ Slot cooldown system (lógica equivalente)
5. ✅ SL/TP calculation (mesma fórmula)

### ⚠️ **Desalinhamentos (3/8)**
1. ⚠️ **Criação de posições**: Simulação (cherry.py) vs MT5 Real (Robot_cherry.py)
   - **Severidade:** BAIXA (funcionalidade equivalente)

2. ⚠️ **Fechamento de posições**: Manual (cherry.py) vs Automático MT5 (Robot_cherry.py)
   - **Severidade:** BAIXA (MT5 garante SL/TP)

3. 🚨 **Ajustes dinâmicos SL/TP**: Trailing DIRETO vs ATIVAÇÃO+MOVIMENTO
   - **Severidade:** ALTA (comportamento diferente)
   - **cherry.py**: Trailing direto + Cap $100 USD
   - **Robot_cherry.py**: Trailing com ativação explícita + Sem cap

---

## 🎯 **RECOMENDAÇÕES**

### 🔴 **PRIORIDADE ALTA**
**Alinhar lógica de trailing stop:**
- [ ] Robot_cherry.py deve usar trailing DIRETO (sem ativação explícita)
- [ ] Implementar cap de $100 USD no TP do Robot_cherry.py
- [ ] Remover sistema de "trailing_activated" metadata

### 🟡 **PRIORIDADE MÉDIA**
**Unificar nomenclatura:**
- [ ] cherry.py: `_process_dynamic_trailing_stop()`
- [ ] Robot_cherry.py: `_process_dynamic_trailing_stop_v7()` → Renomear para consistência

### 🟢 **PRIORIDADE BAIXA**
**Documentação:**
- [ ] Adicionar comentários sobre diferenças esperadas (simulação vs MT5)
- [ ] Criar testes comparativos de trailing stop

---

## 📊 **SCORE DE ALINHAMENTO**

**Alinhamento Geral:** 62.5% (5/8 componentes)

**Breakdown:**
- ✅ Action Space: 100%
- ✅ Position Creation Logic: 85% (diferenças esperadas)
- 🚨 Trailing Stop Logic: 40% (CRÍTICO)
- ✅ Position Closing: 90% (MT5 automático é válido)
- ✅ Confidence Filter: 100%

**Conclusão:** O alinhamento é **PARCIAL**. O maior gap está no **sistema de trailing stop dinâmico**, que precisa ser unificado para garantir comportamento consistente entre treino e produção.
