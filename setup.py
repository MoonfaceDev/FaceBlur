from cx_Freeze import setup, Executable

# Dependencies are automatically detected, but it might need
# fine tuning.
build_options = {'packages': [], 'excludes': []}

import sys
base = 'Win32GUI' if sys.platform=='win32' else None

executables = [
    Executable('C:\\Users\\elaic\\PycharmProjects\\FaceBlur\\blur.py', base=base, target_name = 'FaceBlur.exe')
]

setup(name='FaceBlur',
      version = '1.0',
      description = 'Blur faces in videos',
      options = {'build_exe': build_options},
      executables = executables)
