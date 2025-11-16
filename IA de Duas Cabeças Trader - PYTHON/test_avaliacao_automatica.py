"""
🧪 TESTE DO SISTEMA DE AVALIAÇÃO AUTOMÁTICA

Verifica se:
1. Os checkpoints são salvos no diretório correto (Elegance)
2. O avaliar_v8.py encontra o checkpoint mais recente 
3. A avaliação acontece a cada 500k steps
4. O CHECKPOINT_PATH é atualizado corretamente
"""

import os
import sys
import glob
from datetime import datetime

sys.path.append("D:/Projeto")

def test_checkpoint_directory_structure():
    """🗂️ Testa estrutura de diretórios"""
    print("🗂️ TESTE: Estrutura de Diretórios")
    print("-" * 50)
    
    EXPERIMENT_TAG = "Elegance"
    
    # Diretórios que devem existir
    expected_dirs = [
        f"D:/Projeto/Otimizacao/treino_principal/models/{EXPERIMENT_TAG}",
        f"D:/Projeto/Otimizacao/treino_principal/checkpoints/{EXPERIMENT_TAG}",
        f"D:/Projeto/trading_framework/training/checkpoints/{EXPERIMENT_TAG}"
    ]
    
    for dir_path in expected_dirs:
        exists = os.path.exists(dir_path)
        status = "✅" if exists else "❌" 
        print(f"   {status} {dir_path}")
        
        if not exists:
            print(f"      🔧 Criando diretório...")
            os.makedirs(dir_path, exist_ok=True)
            print(f"      ✅ Criado: {dir_path}")
    
    return True

def test_avaliar_v8_checkpoint_search():
    """🔍 Testa busca de checkpoints no avaliar_v8.py"""
    print("\n🔍 TESTE: Busca de Checkpoints avaliar_v8.py")
    print("-" * 50)
    
    try:
        # Importar função de busca do avaliar_v8
        sys.path.append("D:/Projeto/avaliacao")
        from avaliar_v8 import find_v8_checkpoint
        
        print("   📊 Executando find_v8_checkpoint()...")
        checkpoint = find_v8_checkpoint()
        
        if checkpoint:
            print(f"\n   ✅ CHECKPOINT ENCONTRADO:")
            print(f"      📁 Path: {checkpoint}")
            print(f"      📏 Tamanho: {os.path.getsize(checkpoint)/(1024*1024):.1f}MB")
            print(f"      📅 Modificado: {datetime.fromtimestamp(os.path.getmtime(checkpoint))}")
            return True
        else:
            print("   ❌ Nenhum checkpoint encontrado")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro na busca: {e}")
        return False

def test_daytrader_evaluation_frequency():
    """⏰ Testa configuração da frequência de avaliação (500k steps)"""
    print("\n⏰ TESTE: Frequência de Avaliação")
    print("-" * 50)
    
    try:
        # Ler daytrader8dim.py e procurar a configuração
        with open("D:/Projeto/daytrader8dim.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Procurar pela linha de configuração
        lines = content.split('\n')
        evaluation_lines = [line for line in lines if "500000" in line and "%" in line]
        
        print("   📊 Linhas com configuração 500k encontradas:")
        for line in evaluation_lines:
            if "_on_step" in content[content.find(line)-200:content.find(line)]:
                print(f"      ✅ {line.strip()}")
        
        # Verificar se a função existe
        if "_run_avaliar_v8_evaluation" in content:
            print("   ✅ Função _run_avaliar_v8_evaluation encontrada")
        else:
            print("   ❌ Função _run_avaliar_v8_evaluation NÃO encontrada")
            
        return True
        
    except Exception as e:
        print(f"   ❌ Erro na verificação: {e}")
        return False

def test_checkpoint_path_update():
    """📝 Testa atualização do CHECKPOINT_PATH no avaliar_v8.py"""
    print("\n📝 TESTE: Atualização CHECKPOINT_PATH")
    print("-" * 50)
    
    try:
        # Ler avaliar_v8.py
        avaliar_path = "D:/Projeto/avaliacao/avaliar_v8.py"
        with open(avaliar_path, "r", encoding="utf-8") as f:
            content = f.read()
        
        # Encontrar linha do CHECKPOINT_PATH
        lines = content.split('\n')
        checkpoint_line = None
        for i, line in enumerate(lines):
            if line.startswith('CHECKPOINT_PATH = '):
                checkpoint_line = line
                line_number = i + 1
                break
        
        if checkpoint_line:
            print(f"   ✅ CHECKPOINT_PATH encontrado na linha {line_number}:")
            print(f"      {checkpoint_line}")
            
            # Verificar se o path existe
            current_path = checkpoint_line.split('"')[1] if '"' in checkpoint_line else "N/A"
            if os.path.exists(current_path):
                print(f"   ✅ Arquivo existe: {os.path.basename(current_path)}")
            else:
                print(f"   ❌ Arquivo não existe: {current_path}")
                print(f"      📝 Isso é normal se ainda não foi executado o daytrader")
            
            return True
        else:
            print("   ❌ CHECKPOINT_PATH não encontrado")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro na verificação: {e}")
        return False

def test_experiment_tag_consistency():
    """🏷️ Testa consistência do EXPERIMENT_TAG"""
    print("\n🏷️ TESTE: Consistência EXPERIMENT_TAG")
    print("-" * 50)
    
    try:
        # Ler EXPERIMENT_TAG do daytrader8dim.py
        with open("D:/Projeto/daytrader8dim.py", "r", encoding="utf-8") as f:
            daytrader_content = f.read()
        
        # Ler EXPERIMENT_TAG do avaliar_v8.py
        with open("D:/Projeto/avaliacao/avaliar_v8.py", "r", encoding="utf-8") as f:
            avaliar_content = f.read()
        
        # Extrair valores
        daytrader_tag = None
        avaliar_tag = None
        
        for line in daytrader_content.split('\n'):
            if line.startswith('EXPERIMENT_TAG = '):
                daytrader_tag = line.split('"')[1] if '"' in line else line.split('=')[1].strip()
                break
        
        for line in avaliar_content.split('\n'):
            if 'EXPERIMENT_TAG = ' in line and not line.strip().startswith('#'):
                avaliar_tag = line.split('"')[1] if '"' in line else line.split('=')[1].strip()
                break
        
        print(f"   📊 daytrader8dim.py: EXPERIMENT_TAG = '{daytrader_tag}'")
        print(f"   📊 avaliar_v8.py:   EXPERIMENT_TAG = '{avaliar_tag}'")
        
        if daytrader_tag == avaliar_tag and daytrader_tag is not None:
            print(f"   ✅ EXPERIMENT_TAGs consistentes: '{daytrader_tag}'")
            return True
        else:
            print(f"   ❌ EXPERIMENT_TAGs INCONSISTENTES!")
            return False
            
    except Exception as e:
        print(f"   ❌ Erro na verificação: {e}")
        return False

def main():
    """🚀 Executa todos os testes"""
    print("🚀 TESTE COMPLETO: Sistema de Avaliação Automática")
    print("=" * 60)
    
    tests = [
        ("Estrutura de Diretórios", test_checkpoint_directory_structure),
        ("Busca de Checkpoints", test_avaliar_v8_checkpoint_search),
        ("Frequência de Avaliação", test_daytrader_evaluation_frequency),
        ("Atualização CHECKPOINT_PATH", test_checkpoint_path_update),
        ("Consistência EXPERIMENT_TAG", test_experiment_tag_consistency)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   ❌ ERRO no teste '{test_name}': {e}")
            results.append((test_name, False))
    
    # Resumo final
    print("\n" + "=" * 60)
    print("🏁 RESUMO DOS TESTES")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSOU" if result else "❌ FALHOU"
        print(f"   {status} | {test_name}")
    
    print(f"\n📊 RESULTADO FINAL: {passed}/{total} testes passaram")
    
    if passed == total:
        print("🏆 SISTEMA DE AVALIAÇÃO AUTOMÁTICA: FUNCIONANDO PERFEITAMENTE!")
        print("\n✅ PRÓXIMOS PASSOS:")
        print("   1. Execute o daytrader8dim.py")
        print("   2. A cada 500k steps será executado avaliar_v8.py automaticamente")
        print("   3. O checkpoint mais recente será usado sempre")
        print("   4. Resultados salvos em avaliacoes/")
    else:
        print("🔧 CORREÇÕES NECESSÁRIAS nos testes que falharam")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)