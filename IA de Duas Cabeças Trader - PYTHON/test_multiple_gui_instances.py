#!/usr/bin/env python3
"""
🧪 TEST MULTIPLE GUI INSTANCES - Testar múltiplas instâncias da GUI corrigida
"""

import subprocess
import time
import sys
import os

def test_multiple_instances():
    """Testar múltiplas instâncias do RobotV7 com GUI corrigida"""
    
    print("🧪 TESTANDO MÚLTIPLAS INSTÂNCIAS DA GUI CORRIGIDA")
    print("=" * 60)
    
    robot_path = "Modelo PPO Trader/RobotV7.py"
    
    if not os.path.exists(robot_path):
        print(f"❌ Arquivo não encontrado: {robot_path}")
        return
    
    print("🚀 Iniciando 3 instâncias do RobotV7...")
    
    processes = []
    
    try:
        # Iniciar 3 instâncias
        for i in range(3):
            print(f"   Iniciando instância {i+1}...")
            
            # Usar pythonw para evitar múltiplas janelas de console
            process = subprocess.Popen([
                sys.executable, robot_path
            ], 
            creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32" else 0
            )
            
            processes.append(process)
            time.sleep(2)  # Delay entre instâncias
        
        print(f"✅ {len(processes)} instâncias iniciadas!")
        print("\n📋 INSTRUÇÕES PARA TESTE:")
        print("1. Verifique se todas as 3 janelas aparecem")
        print("2. Minimize algumas janelas")
        print("3. Aguarde 10-15 segundos")
        print("4. Clique nas janelas minimizadas na barra de tarefas")
        print("5. Verifique se elas restauram corretamente")
        print("6. Monitore se alguma janela trava ou fica não responsiva")
        
        print(f"\n⏰ Aguardando 30 segundos para teste...")
        time.sleep(30)
        
        print("\n📊 VERIFICANDO STATUS DOS PROCESSOS:")
        for i, process in enumerate(processes):
            if process.poll() is None:
                print(f"   Instância {i+1}: ✅ Rodando (PID: {process.pid})")
            else:
                print(f"   Instância {i+1}: ❌ Finalizada (código: {process.returncode})")
        
        print("\n🔍 TESTE CONCLUÍDO!")
        print("Se as janelas não travaram e respondem normalmente,")
        print("as correções foram bem-sucedidas!")
        
        # Aguardar input do usuário
        input("\nPressione Enter para finalizar todas as instâncias...")
        
    except KeyboardInterrupt:
        print("\n⚠️ Teste interrompido pelo usuário")
    
    finally:
        # Finalizar todos os processos
        print("\n🛑 Finalizando instâncias...")
        for i, process in enumerate(processes):
            try:
                if process.poll() is None:
                    process.terminate()
                    print(f"   Instância {i+1}: Finalizada")
            except:
                pass
        
        # Aguardar finalização
        time.sleep(2)
        
        # Force kill se necessário
        for process in processes:
            try:
                if process.poll() is None:
                    process.kill()
            except:
                pass

def create_test_summary():
    """Criar resumo das correções aplicadas"""
    
    print("\n📋 RESUMO DAS CORREÇÕES APLICADAS:")
    print("=" * 60)
    
    corrections = [
        "✅ Thread-safe logging com queue",
        "✅ Cleanup de callbacks ao fechar janela", 
        "✅ Gerenciamento inteligente de visibilidade com debouncing",
        "✅ Verificação de responsividade da GUI",
        "✅ Intervalos adaptativos para updates",
        "✅ Restauração suave de janelas sem roubo agressivo de foco",
        "✅ Remoção de conflitos entre múltiplas instâncias",
        "✅ Proper cleanup de threads e recursos"
    ]
    
    for correction in corrections:
        print(f"   {correction}")
    
    print(f"\n🎯 BENEFÍCIOS ESPERADOS:")
    print("   • Janelas não ficam mais travadas/minimizadas")
    print("   • Melhor responsividade da GUI")
    print("   • Múltiplas instâncias funcionam sem conflito")
    print("   • Menos uso de recursos (CPU/memória)")
    print("   • Cleanup adequado ao fechar")

if __name__ == "__main__":
    create_test_summary()
    
    response = input("\nDeseja testar múltiplas instâncias agora? (s/n): ")
    if response.lower() in ['s', 'sim', 'y', 'yes']:
        test_multiple_instances()
    else:
        print("Teste cancelado. Execute o script novamente quando quiser testar.")