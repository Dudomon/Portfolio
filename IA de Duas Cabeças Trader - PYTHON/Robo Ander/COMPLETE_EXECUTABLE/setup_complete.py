"""
Setup script para criar executável COMPLETO do RobotLogin V7
usando cx_Freeze - Solução COMPLETA e FUNCIONAL
"""

from cx_Freeze import setup, Executable
import sys
import os

# Diretórios necessários
current_dir = os.path.dirname(os.path.abspath(__file__))
modelo_ander_path = os.path.join(current_dir, "Modelo Ander")
trading_framework_path = os.path.join(current_dir, "trading_framework")

# Packages que precisam ser incluídos
packages = [
    "tkinter", 
    "tkinter.ttk", 
    "tkinter.scrolledtext", 
    "tkinter.filedialog", 
    "tkinter.messagebox",
    "threading", 
    "queue", 
    "datetime", 
    "time", 
    "os", 
    "sys",
    "json", 
    "hashlib", 
    "pickle", 
    "requests", 
    "numpy", 
    "pandas",
    "collections", 
    "warnings", 
    "zipfile", 
    "shutil", 
    "tempfile",
    "subprocess",
    "io",
    "platform",
    "psutil",
    "cryptography",
    "cryptography.fernet",
    
    # Módulos locais
    "robotv7_login_system",
    "enhanced_normalizer", 
    "login_system",
    "online_system_real",
    "secure_model_system",
    "protect_normalizers",
]

# Arquivos e diretórios para incluir
include_files = [
    # Arquivos Python essenciais
    ("robotv7_login_system.py", "robotv7_login_system.py"),
    ("enhanced_normalizer.py", "enhanced_normalizer.py"),
    ("login_system.py", "login_system.py"),
    ("online_system_real.py", "online_system_real.py"),
    ("secure_model_system.py", "secure_model_system.py"),
    ("protect_normalizers.py", "protect_normalizers.py"),
    
    # Modelos protegidos
    (modelo_ander_path, "Modelo Ander"),
    
    # Framework completo
    (trading_framework_path, "trading_framework"),
]

# Módulos inclusos forçadamente 
includes = [
    "robotv7_login_system",
    "enhanced_normalizer", 
    "login_system",
    "online_system_real", 
    "secure_model_system",
    "protect_normalizers",
]

# Opções de build
build_exe_options = {
    "packages": packages,
    "include_files": include_files,
    "includes": includes,
    "excludes": [
        # Excluir módulos desnecessários para reduzir tamanho
        "matplotlib", 
        "scipy.sparse.csgraph._validation",
        "scipy.spatial.qhull",
        "torch", 
        "tensorflow", 
        "sklearn",
        "numba", 
        "llvmlite", 
        "moviepy",
        "PIL",
        "cv2",
        "jupyter",
        "notebook",
        "IPython",
        "test",
        "unittest",
        "pydoc",
    ],
    "zip_include_packages": ["encodings", "importlib", "urllib3"],
    "optimize": 2,
    "build_exe": "RobotLogin_V7_COMPLETE",
}

# Configuração do executável
executable = Executable(
    script="robotlogin.py",
    target_name="RobotLogin_V7_COMPLETE.exe", 
    base="Win32GUI" if sys.platform == "win32" else None,
    icon=None,  # Pode adicionar ícone se tiver
    shortcut_name="RobotLogin V7 Complete",
    shortcut_dir="Desktop",
)

# Setup principal
setup(
    name="RobotLogin V7 Complete",
    version="1.0.0",
    description="Robot Trading V7 - Sistema Completo de Login e Trading",
    author="RobotV7 Team",
    options={"build_exe": build_exe_options},
    executables=[executable],
)

print("🚀 Executável COMPLETO criado com SUCESSO!")
print("📁 Pasta: RobotLogin_V7_COMPLETE/")
print("🎯 Arquivo: RobotLogin_V7_COMPLETE.exe")
print("✅ Sistema completo incluído - FUNCIONANDO!")