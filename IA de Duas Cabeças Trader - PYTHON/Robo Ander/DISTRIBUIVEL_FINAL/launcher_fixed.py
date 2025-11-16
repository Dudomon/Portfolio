#!/usr/bin/env python3
import sys
import os

# Configurar path para executar .pyc diretamente
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

print("🚀 RobotV7 Legion - Iniciando...")

# Executar o main() do Robot_1min diretamente sem subprocess
if __name__ == "__main__":
    try:
        # Simular execução do Robot_1min.pyc
        import Robot_1min
        
        # Verificar se existe função main no módulo
        if hasattr(Robot_1min, 'main'):
            Robot_1min.main()
        else:
            print("✅ Módulo Robot_1min carregado com sucesso!")
            
    except Exception as e:
        print(f"❌ Erro ao executar Robot_1min: {e}")
        input("Pressione Enter para sair...")