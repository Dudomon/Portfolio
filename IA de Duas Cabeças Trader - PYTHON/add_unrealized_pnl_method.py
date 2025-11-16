#!/usr/bin/env python3
"""
Adicionar método _calculate_unrealized_pnl que está faltando
"""
import sys
import os
sys.path.append("D:/Projeto")

# Ler o arquivo atual
with open("D:/Projeto/trading_framework/rewards/reward_daytrade_v2.py", "r", encoding="utf-8") as f:
    content = f.read()

# Método que estava faltando
unrealized_pnl_method = '''
    def _calculate_unrealized_pnl(self, env) -> float:
        """Calcular PnL não realizado das posições abertas"""
        try:
            if not hasattr(env, 'positions') or not env.positions:
                return 0.0
            
            total_unrealized = 0.0
            current_price = getattr(env, 'current_price', 0)
            
            if current_price == 0:
                # Tentar obter preço atual do dataset
                if hasattr(env, 'df') and hasattr(env, 'current_step'):
                    try:
                        current_price = env.df['close_5m'].iloc[env.current_step - 1]
                    except (IndexError, KeyError):
                        return 0.0
                else:
                    return 0.0
            
            for position in env.positions:
                if isinstance(position, dict):
                    entry_price = position.get('entry_price', 0)
                    size = position.get('lot_size', position.get('size', 0))
                    side = position.get('type', position.get('side', 'long'))
                    
                    if entry_price > 0 and size > 0:
                        if side.lower() in ['long', 'buy']:
                            unrealized = (current_price - entry_price) * size
                        else:  # short
                            unrealized = (entry_price - current_price) * size
                        
                        total_unrealized += unrealized
            
            return total_unrealized
            
        except Exception as e:
            return 0.0
'''

# Encontrar onde inserir o método (antes do método _get_current_state)
insert_pos = content.find("    def _get_current_state(self, env)")
if insert_pos == -1:
    print("❌ Não encontrou posição para inserir método")
    exit(1)

# Inserir o método
new_content = content[:insert_pos] + unrealized_pnl_method + "\n" + content[insert_pos:]

# Salvar arquivo
with open("D:/Projeto/trading_framework/rewards/reward_daytrade_v2.py", "w", encoding="utf-8") as f:
    f.write(new_content)

print("✅ Método _calculate_unrealized_pnl adicionado com sucesso")