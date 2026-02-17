# setup_fluid.py
#
# Run with:
# python C:/Users/Rory_/Documents/VSCode/ResearchPersonal/setup_fluid.py build_ext --compiler=mingw32 --inplace *> full_output.txt
#
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np
import os

mingw_path = None
for path in os.environ['PATH'].split(';'):
    if 'msys64' in path.lower() and 'ucrt64' in path.lower():
        mingw_path = path
        break

if mingw_path and os.path.exists(os.path.join(mingw_path, 'gcc.exe')):
    print(f"Found GCC at: {mingw_path}")
    
    os.environ['CC'] = 'gcc'
    os.environ['CXX'] = 'g++'
    os.environ['DISTUTILS_USE_SDK'] = '1'
    
    os.environ['PATH'] = mingw_path + ';' + os.environ['PATH']

extensions = [
    Extension(
        "fluid_sim_hist",
        ["C:/Users/Rory_/Documents/VSCode/ResearchPersonal/fluid_sim_hist.pyx"],
        include_dirs=[np.get_include()],
extra_compile_args=[
            '-fopenmp',
            '-O3',
            #'-fopt-info-vec-missed',
            '-fopt-info-vec-optimized',
            '-fopt-info-omp',
            '-ffast-math',
            '-march=native',
            '-mtune=native',
            '-mavx2',
            '-msse4.2',
            '-funroll-loops',
            '-ftree-vectorize',
            '-fopenmp-simd',
            '-fno-signed-zeros',
            '-fno-trapping-math',
            '-fno-math-errno'
        ],
        extra_link_args=[
        '-fopenmp',
        '-static',           # Static link GCC runtime
        '-static-libgcc',    # Static link libgcc
        '-static-libstdc++', # Static link libstdc++
],
    )
]

setup(
    name="fluid_sim_hist",
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            'language_level': "3",
            'boundscheck': False,
            'wraparound': False,
            'initializedcheck': False,
            'cdivision': True,
            'nonecheck': False,
            'overflowcheck': False,
            'embedsignature': False,
            'cdivision_warnings': False,
            'binding': False
        }
    )
)