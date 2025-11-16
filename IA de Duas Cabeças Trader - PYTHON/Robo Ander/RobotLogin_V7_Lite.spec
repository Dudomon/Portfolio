# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['robotlogin_lite.py'],
    pathex=[],
    binaries=[],
    datas=[('robotv7_login_system.py', '.'), ('enhanced_normalizer.py', '.'), ('login_system.py', '.'), ('online_system_real.py', '.'), ('secure_model_system.py', '.'), ('protect_normalizers.py', '.')],
    hiddenimports=['tkinter', 'tkinter.ttk', 'tkinter.filedialog', 'tkinter.messagebox'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='RobotLogin_V7_Lite',
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
