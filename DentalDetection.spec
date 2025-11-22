# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('tooth_float32.tflite', '.')],
    hiddenimports=[
        'tflite_runtime',
        'tflite_runtime.interpreter',
        'cv2',
        'numpy',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tensorflow',  # Exclude full TensorFlow
        'matplotlib',
        'tkinter',
        'IPython',
        'notebook',
        'pandas',
        'scipy',
        'PIL',
        'Pillow',
        'pytest',
        'setuptools',
        '_pytest',
        'cryptography',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate binaries to reduce size
a.binaries = [x for x in a.binaries if not x[0].startswith('api-ms-win')]
a.binaries = [x for x in a.binaries if not x[0].startswith('vcruntime')]

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='DentalDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress with UPX
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,  # Compress with UPX
    upx_exclude=[],
    name='DentalDetection',
)
