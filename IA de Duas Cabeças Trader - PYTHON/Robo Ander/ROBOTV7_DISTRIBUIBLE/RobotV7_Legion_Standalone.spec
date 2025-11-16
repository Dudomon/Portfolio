# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['robotlogin.py'],
    pathex=[],
    binaries=[],
    datas=[('robotv7_login_system.py', '.'), ('enhanced_normalizer.py', '.'), ('login_system.py', '.'), ('online_system_real.py', '.'), ('secure_model_system.py', '.'), ('protect_normalizers.py', '.'), ('Modelo Ander', 'Modelo Ander'), ('trading_framework', 'trading_framework')],
    hiddenimports=['tkinter', 'tkinter.ttk', 'tkinter.scrolledtext', 'tkinter.filedialog', 'tkinter.messagebox', 'numpy', 'pandas', 'requests', 'cryptography', 'psutil', 'robotv7_login_system', 'enhanced_normalizer', 'secure_model_system'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['matplotlib', 'scipy', 'torch', 'tensorflow', 'jupyter', 'notebook'],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RobotV7_Legion_Standalone',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
