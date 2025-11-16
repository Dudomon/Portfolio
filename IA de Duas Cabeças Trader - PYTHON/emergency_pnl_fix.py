#!/usr/bin/env python3
"""
🚨 EMERGENCY PNL FIX - Aplicar correção imediata no daytrader.py
"""

def create_emergency_fix():
    """Criar patch de emergência para _get_position_pnl"""
    
    return '''
    def _get_position_pnl(self, pos, current_price):
        """PnL com EMERGENCY PROTECTION contra bugs"""
        price_diff = 0
        if pos['type'] == 'long':
            price_diff = current_price - pos['entry_price']
        else:
            price_diff = pos['entry_price'] - current_price
        
        # PnL base
        base_pnl = price_diff * pos['lot_size'] * 100
        
        # 🚨 EMERGENCY PROTECTION: Limitar PnL a máximo fisicamente possível
        max_sl_points = 15  # SL máximo configurado + margem
        max_loss_allowed = max_sl_points * pos['lot_size'] * 100
        max_gain_allowed = max_sl_points * 3 * pos['lot_size'] * 100  # TP até 3x SL
        
        # Clipping de segurança
        if base_pnl < -max_loss_allowed:
            print(f"🚨 EMERGENCY CLIP: PnL ${base_pnl:.2f} → ${-max_loss_allowed:.2f}")
            return -max_loss_allowed
        elif base_pnl > max_gain_allowed:
            print(f"🚨 EMERGENCY CLIP: PnL ${base_pnl:.2f} → ${max_gain_allowed:.2f}")
            return max_gain_allowed
        
        return base_pnl
    '''

def create_sl_validation_fix():
    """Criar validação de SL ao criar posições"""
    
    return '''
    # ADICIONAR APÓS position['tp'] = current_price - tp_price_diff
    
    # 🚨 EMERGENCY VALIDATION: Garantir SL/TP válidos
    if 'sl' not in position or position['sl'] <= 0:
        print(f"🚨 POSITION SEM SL! Adicionando SL emergencial")
        if position['type'] == 'long':
            position['sl'] = current_price - 8.0  # 8 pontos emergency SL
        else:
            position['sl'] = current_price + 8.0
    
    if 'tp' not in position or position['tp'] <= 0:
        print(f"🚨 POSITION SEM TP! Adicionando TP emergencial") 
        if position['type'] == 'long':
            position['tp'] = current_price + 15.0  # 15 pontos emergency TP
        else:
            position['tp'] = current_price - 15.0
    
    # Log para debug
    sl_points = abs(position['entry_price'] - position['sl'])
    tp_points = abs(position['entry_price'] - position['tp'])
    print(f"📊 Nova posição: {position['type']} SL={sl_points:.1f}pts TP={tp_points:.1f}pts")
    '''

if __name__ == "__main__":
    print("🚨 EMERGENCY FIXES CRIADOS")
    print("\n1. PnL PROTECTION:")
    print(create_emergency_fix())
    print("\n2. SL VALIDATION:")  
    print(create_sl_validation_fix())
    print("\n🎯 APLICAR MANUALMENTE NO daytrader.py!")