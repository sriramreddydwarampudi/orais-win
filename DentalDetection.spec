# -*- mode: python ; coding: utf-8 -*-
import sys
from PyInstaller.utils.hooks import collect_all, collect_submodules

block_cipher = None

# Collect all TensorFlow modules
tf_hidden_imports = collect_submodules('tensorflow')
cv2_hidden_imports = collect_submodules('cv2')

# Collect all data files
datas = [('tooth_float32.tflite', '.')]
binaries = []

# Collect TensorFlow binaries and data
tmp_ret = collect_all('tensorflow')
datas += tmp_ret[0]; binaries += tmp_ret[1]

# Collect OpenCV data
tmp_ret = collect_all('cv2')
datas += tmp_ret[0]; binaries += tmp_ret[1]

a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=[
        'tensorflow',
        'tensorflow.python',
        'tensorflow.python.util',
        'tensorflow.python.framework',
        'tensorflow.lite',
        'tensorflow.lite.python',
        'tensorflow.lite.python.interpreter',
        'tensorflow.lite.python.lite',
        'cv2',
        'numpy',
        'numpy.core',
        'numpy.core._methods',
        'numpy.core._dtype_ctypes',
        'PySide6',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
    ] + tf_hidden_imports + cv2_hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
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
        'tensorboard',
        'tensorflow_estimator',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

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
    upx=False,  # Disable UPX for TensorFlow compatibility
    console=True,  # Enable console to see errors during testing
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
    upx=False,
    upx_exclude=[],
    name='DentalDetection',
)
