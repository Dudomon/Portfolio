# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# Paths
current_dir = os.path.abspath('.')
trading_framework_path = os.path.join(current_dir, 'trading_framework')
modelo_ander_path = os.path.join(current_dir, 'Modelo Ander')

# Hidden imports necessários
hiddenimports = [
    # Módulos do sistema
    'robotv7_login_system',
    'enhanced_normalizer', 
    'login_system',
    'online_system_real',
    'secure_model_system',
    'protect_normalizers',
    
    # Framework de trading
    'trading_framework.rewards.reward_system_simple',
    'trading_framework.environments.trading_env',
    'trading_framework.policies.two_head_v11_sigmoid',
    'trading_framework.extractors.transformer_v9_daytrading',
    
    # Bibliotecas essenciais
    'numpy',
    'pandas',
    'tkinter',
    'tkinter.ttk',
    'tkinter.scrolledtext',
    'tkinter.filedialog',
    'tkinter.messagebox',
    'threading',
    'queue',
    'datetime',
    'time',
    'json',
    'hashlib',
    'pickle',
    'requests',
    'collections',
    'warnings',
    'zipfile',
    'shutil',
    'tempfile',
    'subprocess',
    'os',
    'sys',
    'io',
    
    # MetaTrader5 (opcional)
    'MetaTrader5',
    
    # Stable-Baselines3 (se disponível)
    'stable_baselines3',
    'stable_baselines3.common.vec_env',
    'stable_baselines3.common.callbacks',
    
    # Cryptography para modelos seguros
    'cryptography',
    'cryptography.fernet',
    'psutil',
]

# Arquivos de dados
datas = [
    # Pasta de modelos com todos os arquivos .secure
    (modelo_ander_path, 'Modelo Ander'),
    
    # Framework de trading completo
    (trading_framework_path, 'trading_framework'),
]

# Analisar o arquivo principal
a = Analysis(
    ['robotlogin.py'],
    pathex=[current_dir],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Excluir bibliotecas pesadas que não são necessárias
        'matplotlib',
        'scipy',
        'torch', 
        'tensorflow',
        'sklearn',
        'numba',
        'llvmlite',
        'moviepy',
        'PIL',
        'cv2',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remover binários desnecessários para reduzir tamanho
a.binaries = TOC([x for x in a.binaries if not x[0].lower().startswith('api-ms-win')])
a.binaries = TOC([x for x in a.binaries if not x[0].lower().startswith('ucrtbase')])

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='RobotLogin_V7_Complete',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,  # Console habilitado para debug
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Pode adicionar ícone aqui se tiver
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='RobotLogin_V7_Complete'
)