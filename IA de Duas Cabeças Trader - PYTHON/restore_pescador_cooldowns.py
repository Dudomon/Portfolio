#!/usr/bin/env python3
"""
Script para restaurar cooldowns do pescador.py após teste
"""

def restore_cooldowns():
    pescador_file = "D:\\Projeto\\pescador.py"
    
    # Código para restaurar (removendo as linhas de teste)
    remove_lines = [
        "        # 🚀 TESTE: DESABILITAR COOLDOWNS COMPLETAMENTE",
        "        self.cooldown_after_trade = 0",
        "        self.cooldown_base = 0", 
        "        self.slot_cooldowns = {i: 0 for i in range(getattr(self, 'max_positions', 2))}",
        "        print(\"🚀 [PESCADOR] Cooldowns COMPLETAMENTE DESABILITADOS para teste\")",
        "",  # linha em branco extra
        "    def _close_position(self, position, current_step_or_reason=None):",
        "        \"\"\"Override para desabilitar aplicação de cooldowns após fechamento\"\"\"",
        "        result = super()._close_position(position, current_step_or_reason)",
        "        ",
        "        # 🚀 TESTE: Zerar todos os cooldowns após qualquer fechamento",
        "        self.cooldown_after_trade = 0",
        "        for slot in self.slot_cooldowns:",
        "            self.slot_cooldowns[slot] = 0",
        "            ",
        "        return result",
        "    # REMOVIDO: Override de _calculate_reward_and_info que estava quebrando o processamento",
        "    # O silus.py faz processamento essencial (SL/TP, ordens) antes do reward",
        "# 🚀 DOBRAR LEARNING RATE PARA PESCADOR (mais agressivo)",
        "base.BEST_PARAMS[\"learning_rate\"] = 6.0e-05  # 3e-05 -> 6e-05 (2x)",
        "base.BEST_PARAMS[\"critic_learning_rate\"] = 4.0e-05  # 2e-05 -> 4e-05 (2x)",
        "print(f\"🚀 [PESCADOR] Learning rates DOBRADOS - Actor: {base.BEST_PARAMS['learning_rate']:.1e}, Critic: {base.BEST_PARAMS['critic_learning_rate']:.1e}\")"
    ]
    
    try:
        with open(pescador_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Remover linhas específicas do teste
        filtered_lines = []
        skip_until_return = False
        
        for line in lines:
            line_stripped = line.strip()
            
            # Pular bloco _close_position completo
            if "def _close_position(self, position, close_reason=" in line:
                skip_until_return = True
                continue
            elif skip_until_return and line_stripped == "return result":
                skip_until_return = False
                continue
            elif skip_until_return:
                continue
            
            # Pular linhas específicas de cooldown
            should_skip = False
            for remove_line in remove_lines:
                if remove_line.strip() and remove_line.strip() in line_stripped:
                    should_skip = True
                    break
            
            if not should_skip:
                filtered_lines.append(line)
        
        # Escrever arquivo restaurado
        with open(pescador_file, 'w', encoding='utf-8') as f:
            f.writelines(filtered_lines)
        
        print("✅ Cooldowns e LR do pescador.py restaurados com sucesso!")
        print("   - Removidas linhas de teste de cooldown=0")
        print("   - Removido override de _close_position")
        print("   - Learning rate volta ao padrão do silus.py")
        print("   - Sistema volta ao padrão herdado do silus.py")
        
    except Exception as e:
        print(f"❌ Erro ao restaurar: {e}")

if __name__ == "__main__":
    restore_cooldowns()