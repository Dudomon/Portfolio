#!/usr/bin/env python3
"""
🛡️ EMERGENCY REWARD PROTECTION - Proteger reward system contra PnL bugs
"""

def create_reward_protection():
    """Criar wrapper de proteção para rewards"""
    
    return '''
def protected_reward_wrapper(original_reward_func):
    """Wrapper para proteger reward system contra PnL bugs"""
    
    def wrapper(*args, **kwargs):
        try:
            # Calcular reward normal
            reward = original_reward_func(*args, **kwargs)
            
            # 🚨 EMERGENCY PROTECTION: Detectar rewards impossíveis
            if abs(reward) > 50:  # Reward muito alto = possível bug PnL
                print(f"🚨 REWARD PROTECTION: {reward:.2f} → clipped")
                reward = np.clip(reward, -15.0, 15.0)
            
            # Detectar NaN/Inf
            if not np.isfinite(reward):
                print(f"🚨 REWARD NaN/Inf DETECTED: → 0.0")
                reward = 0.0
                
            return reward
            
        except Exception as e:
            print(f"🚨 REWARD CALCULATION ERROR: {e} → 0.0")
            return 0.0
    
    return wrapper

# APLICAR EM TODOS OS REWARD SYSTEMS:
# calculate_reward = protected_reward_wrapper(calculate_reward)
    '''

def create_pnl_sanity_check():
    """Sistema para detectar PnL impossíveis"""
    
    return '''
def sanity_check_trade_pnl(trade_info, max_sl_points=15):
    """Verificar se PnL do trade é fisicamente possível"""
    
    pnl = trade_info.get('pnl_usd', 0)
    lot_size = trade_info.get('lot_size', 0.01)
    
    # Calcular perda máxima fisicamente possível
    max_loss = max_sl_points * lot_size * 100
    max_gain = max_sl_points * 3 * lot_size * 100  # TP geralmente 3x SL
    
    if pnl < -max_loss:
        print(f"🚨 IMPOSSIBLE LOSS: ${pnl:.2f} > max ${max_loss:.2f}")
        print(f"   Trade: {trade_info}")
        return False, -max_loss  # Retornar PnL corrigido
    elif pnl > max_gain:
        print(f"🚨 IMPOSSIBLE GAIN: ${pnl:.2f} > max ${max_gain:.2f}")
        return False, max_gain
    
    return True, pnl  # PnL é válido
    '''

if __name__ == "__main__":
    print("🛡️ REWARD PROTECTION SYSTEMS:")
    print("\n1. REWARD WRAPPER:")
    print(create_reward_protection())
    print("\n2. PnL SANITY CHECK:")
    print(create_pnl_sanity_check())
    print("\n🎯 INTEGRAR NOS REWARD SYSTEMS!")