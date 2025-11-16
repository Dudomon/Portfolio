# -*- mode: python ; coding: utf-8 -*-

import sys
import os

# Adicionar caminhos necessários
sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

block_cipher = None

a = Analysis(
    ['robotlogin.py'],
    pathex=[
        'D:\\Projeto\\Robo Ander',
        'D:\\Projeto',
        'D:\\Projeto\\trading_framework'
    ],
    binaries=[],
    datas=[
        # Incluir arquivos de dados necessários
        ('secure_model_system.py', '.'),
        ('protect_normalizers.py', '.'),
        ('enhanced_normalizer.py', '.'),
        ('robotv7_login_system.py', '.'),
        ('login_system.py', '.'),
        ('online_system_real.py', '.'),
    ],
    hiddenimports=[
        # Trading framework
        'trading_framework',
        'trading_framework.policies',
        'trading_framework.policies.two_head_v11_sigmoid',
        'trading_framework.extractors',
        'trading_framework.extractors.transformer_extractor',
        
        # Security modules
        'secure_model_system',
        'protect_normalizers',
        
        # ML libraries
        'stable_baselines3',
        'sb3_contrib',
        'torch',
        'tensorflow',
        'sklearn',
        'numpy',
        'pandas',
        
        # GUI
        'tkinter',
        'tkinter.ttk',
        'tkinter.scrolledtext',
        'tkinter.filedialog',
        'tkinter.messagebox',
        
        # Crypto
        'cryptography',
        'cryptography.fernet',
        
        # MT5
        'MetaTrader5',
        
        # Others
        'requests',
        'pickle',
        'json',
        'hashlib',
        'threading',
        'queue',
        'datetime',
        'time',
        'os',
        'sys',
        'warnings'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='RobotLogin_V7',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Manter console para logs
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None  # Pode adicionar ícone se tiver
)