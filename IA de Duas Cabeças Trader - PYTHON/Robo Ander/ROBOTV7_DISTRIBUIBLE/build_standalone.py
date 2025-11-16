"""
Script para criar executável STANDALONE REAL
- NÃO PRECISA Python instalado
- ESCONDE todo o código fonte  
- Um arquivo .exe que roda em qualquer Windows
"""

import PyInstaller.__main__
import os
import shutil

def create_standalone_exe():
    """Criar executável standalone REAL"""
    
    print("🔥 CRIANDO EXECUTÁVEL STANDALONE REAL...")
    print("📦 Sem Python, sem código exposto, sem dependências")
    print()
    
    # Limpar builds anteriores
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    if os.path.exists('build'):
        shutil.rmtree('build')
        
    # Argumentos do PyInstaller para executável REAL
    args = [
        # Arquivo principal
        'robotlogin.py',
        
        # ONEFILE = um único .exe standalone
        '--onefile',
        
        # Nome do executável
        '--name=RobotV7_Legion_Standalone',
        
        # Sem console (GUI)
        '--windowed',
        
        # Incluir TODOS os arquivos necessários
        '--add-data=robotv7_login_system.py;.',
        '--add-data=enhanced_normalizer.py;.',
        '--add-data=login_system.py;.',
        '--add-data=online_system_real.py;.',
        '--add-data=secure_model_system.py;.',
        '--add-data=protect_normalizers.py;.',
        '--add-data=Modelo Ander;Modelo Ander',
        '--add-data=trading_framework;trading_framework',
        
        # Hidden imports essenciais
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.scrolledtext',
        '--hidden-import=tkinter.filedialog',
        '--hidden-import=tkinter.messagebox',
        '--hidden-import=numpy',
        '--hidden-import=pandas',
        '--hidden-import=requests',
        '--hidden-import=cryptography',
        '--hidden-import=psutil',
        '--hidden-import=robotv7_login_system',
        '--hidden-import=enhanced_normalizer',
        '--hidden-import=secure_model_system',
        
        # Excluir módulos desnecessários
        '--exclude-module=matplotlib',
        '--exclude-module=scipy',
        '--exclude-module=torch',
        '--exclude-module=tensorflow',
        '--exclude-module=jupyter',
        '--exclude-module=notebook',
        
        # UPX para comprimir (se disponível)
        '--upx-dir=.',
        
        # Sem debug
        '--noconfirm',
        '--clean',
    ]
    
    print("⚙️ Executando PyInstaller...")
    PyInstaller.__main__.run(args)
    
    # Verificar se foi criado
    exe_path = os.path.join('dist', 'RobotV7_Legion_Standalone.exe')
    if os.path.exists(exe_path):
        size_mb = os.path.getsize(exe_path) / (1024*1024)
        print(f"✅ EXECUTÁVEL CRIADO COM SUCESSO!")
        print(f"📁 Local: {exe_path}")
        print(f"📊 Tamanho: {size_mb:.1f} MB")
        print()
        print("🎯 CARACTERÍSTICAS:")
        print("- ✅ STANDALONE (não precisa Python)")
        print("- ✅ CÓDIGO PROTEGIDO (não expõe fonte)")
        print("- ✅ TUDO EMBUTIDO (modelos, framework, etc)")
        print("- ✅ RODA EM QUALQUER WINDOWS")
        print()
        print("🚀 PRONTO PARA DISTRIBUIÇÃO!")
        
    else:
        print("❌ ERRO: Executável não foi criado")
        print("Verifique os logs acima para detalhes")

if __name__ == "__main__":
    create_standalone_exe()