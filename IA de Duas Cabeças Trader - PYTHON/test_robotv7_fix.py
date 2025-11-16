#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE: Verificar se RobotV7 abre corretamente
"""

import subprocess
import sys
import time
import os

def test_robotv7():
    """Testar se RobotV7 abre sem travar"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ RobotV7 não encontrado: {robot_path}")
        return False
    
    print("🧪 Testando RobotV7...")
    print("⏳ Aguarde 10 segundos para verificar se a janela abre...")
    
    try:
        # Lançar RobotV7
        process = subprocess.Popen([sys.executable, robot_path])
        
        # Aguardar 10 segundos
        time.sleep(10)
        
        # Verificar se ainda está rodando
        if process.poll() is None:
            print("✅ RobotV7 está rodando!")
            print("✅ Se você consegue ver a janela, a correção funcionou!")
            
            response = input("\nA janela do RobotV7 está visível? (s/N): ")
            if response.lower() == 's':
                print("🎉 SUCESSO! Correção funcionou!")
                return True
            else:
                print("❌ Janela ainda não está visível")
                print("💡 Tente clicar no ícone na barra de tarefas")
                
                # Tentar finalizar o processo
                try:
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                except:
                    pass
                return False
        else:
            print("❌ RobotV7 fechou inesperadamente")
            return False
            
    except Exception as e:
        print(f"❌ Erro ao testar: {e}")
        return False

if __name__ == "__main__":
    print("🧪 TESTE DE CORREÇÃO - RobotV7")
    print("=" * 40)
    
    success = test_robotv7()
    
    if not success:
        print("\n💡 DICAS ADICIONAIS:")
        print("1. Verifique se há múltiplas instâncias rodando")
        print("2. Tente reiniciar o computador")
        print("3. Execute como administrador")
        print("4. Verifique antivírus/firewall")
    
    input("\nPressione Enter para sair...")
