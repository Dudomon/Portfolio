"""
Setup script para criar executável do RobotLogin V7
usando cx_Freeze - alternativa ao PyInstaller
"""

from cx_Freeze import setup, Executable
import sys
import os

# Dependências básicas
packages = [
    "tkinter", "threading", "queue", "datetime", "time", "os", "sys",
    "json", "hashlib", "pickle", "requests", "numpy", "pandas",
    "collections", "warnings", "zipfile", "shutil", "tempfile"
]

# Incluir arquivos necessários
include_files = [
    ("robotv7_login_system.py", "robotv7_login_system.py"),
    ("enhanced_normalizer.py", "enhanced_normalizer.py"),
    ("login_system.py", "login_system.py"),
    ("online_system_real.py", "online_system_real.py"),
    ("secure_model_system.py", "secure_model_system.py"),
    ("protect_normalizers.py", "protect_normalizers.py"),
]

# Opções de build
build_exe_options = {
    "packages": packages,
    "include_files": include_files,
    "excludes": [
        "matplotlib", "scipy", "torch", "tensorflow", 
        "sklearn", "numba", "llvmlite", "moviepy"
    ],
    "optimize": 2,
}

# Configurações do executável
executable = Executable(
    script="robotlogin_lite.py",
    target_name="RobotLogin_V7.exe",
    base="Win32GUI" if sys.platform == "win32" else None,
    icon=None  # Pode adicionar ícone se tiver
)

setup(
    name="RobotLogin V7",
    version="1.0",
    description="Robot Trading V7 Login System",
    author="Claude Code",
    options={"build_exe": build_exe_options},
    executables=[executable]
)