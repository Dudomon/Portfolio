#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🧪 TESTE ESPECÍFICO: Minimizar/Restaurar Janela
Testa se a janela pode ser minimizada e restaurada corretamente
"""

import subprocess
import sys
import time
import os

def test_minimize_restore():
    """Teste específico de minimização e restauração"""
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ RobotV7 não encontrado: {robot_path}")
        return False
    
    print("🧪 TESTE: Minimizar/Restaurar Janela")
    print("=" * 40)
    print("⏳ Iniciando RobotV7...")
    
    try:
        # Lançar RobotV7
        process = subprocess.Popen([sys.executable, robot_path])
        
        # Aguardar inicialização
        time.sleep(8)
        
        # Verificar se ainda está rodando
        if process.poll() is None:
            print("✅ RobotV7 iniciado com sucesso!")
            print()
            print("📋 INSTRUÇÕES PARA TESTE:")
            print("1. Verifique se a janela do RobotV7 está visível")
            print("2. Minimize a janela (clique no botão minimizar)")
            print("3. Clique no ícone na barra de tarefas para restaurar")
            print("4. Repita o processo 2-3 vezes")
            print()
            
            # Aguardar teste manual
            input("Pressione Enter quando terminar o teste de minimização...")
            
            # Perguntar resultado
            print()
            result = input("A janela restaurou corretamente da barra de tarefas? (s/N): ")
            
            # Finalizar processo
            try:
                process.terminate()
                time.sleep(2)
                if process.poll() is None:
                    process.kill()
                print("✅ Processo finalizado")
            except:
                print("⚠️ Processo pode ainda estar rodando")
            
            if result.lower() == 's':
                print("🎉 SUCESSO! Problema de minimização corrigido!")
                return True
            else:
                print("❌ Problema ainda persiste")
                return False
        else:
            print("❌ RobotV7 fechou inesperadamente")
            return False
            
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        return False

if __name__ == "__main__":
    success = test_minimize_restore()
    
    if not success:
        print("\n💡 SOLUÇÕES ALTERNATIVAS:")
        print("1. Use Alt+Tab para alternar entre janelas")
        print("2. Reinicie o computador")
        print("3. Execute como administrador")
        print("4. Verifique se há conflitos com outros programas")
    
    input("\nPressione Enter para sair...")
