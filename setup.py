from __future__ import absolute_import, print_function
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from numpy.distutils.misc_util import get_numpy_include_dirs
from os import path
from glob import glob

class build_ext_openmp(build_ext):
    # https://www.openmp.org/resources/openmp-compilers-tools/
    # python setup.py build_ext --help-compiler
    openmp_compile_args = {
        'msvc':  ['/openmp'],
        'intel': ['-qopenmp'],
        '*':     ['-fopenmp']
    }
    openmp_link_args = openmp_compile_args # ?

    def build_extension(self, ext):
        compiler = self.compiler.compiler_type.lower()
        if compiler.startswith('intel'):
            compiler = 'intel'
        if compiler not in self.openmp_compile_args:
            compiler = '*'

        _extra_compile_args = list(ext.extra_compile_args)
        _extra_link_args    = list(ext.extra_link_args)
        try:
            ext.extra_compile_args += self.openmp_compile_args[compiler]
            ext.extra_link_args    += self.openmp_link_args[compiler]
            super(build_ext_openmp, self).build_extension(ext)
        except:
            print('compiling with OpenMP support failed, re-trying without')
            ext.extra_compile_args = _extra_compile_args
            ext.extra_link_args    = _extra_link_args
            super(build_ext_openmp, self).build_extension(ext)

            
#------------------------------------------------------------------------------------


# cf. https://github.com/mkleehammer/pyodbc/issues/82#issuecomment-231561240
_dir = path.dirname(__file__)

with open(path.join(_dir,'stardist','version.py'), encoding="utf-8") as f:
    exec(f.read())

with open(path.join(_dir,'README.md'), encoding="utf-8") as f:
    long_description = f.read()

qhull_root = path.join(_dir, 'stardist', 'lib', 'qhull_src', 'src')
qhull_src = sorted(glob(path.join(qhull_root, '*', '*.c*')))[::-1]
common_src = ['stardist/lib/utils.cpp']

setup(
    name='stardist',
    version=__version__,
    description='StarDist',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/mpicbg-csbd/stardist',
    author='Uwe Schmidt, Martin Weigert',
    author_email='uschmidt@mpi-cbg.de, mweigert@mpi-cbg.de',
    license='BSD 3-Clause License',
    packages=find_packages(),
    python_requires='>=3.5',

    cmdclass={'build_ext': build_ext_openmp},
    
    ext_modules=[
        Extension(
            'stardist.lib.stardist2d',
            sources=['stardist/lib/stardist2d.cpp','stardist/lib/clipper.cpp'] + common_src,
            extra_compile_args = ['-std=c++11'],
            include_dirs=get_numpy_include_dirs(),
        ),
        Extension(
            'stardist.lib.stardist3d',
            sources=['stardist/lib/stardist3d.cpp', 'stardist/lib/stardist3d_impl.cpp'] + common_src + qhull_src,
            extra_compile_args = ['-std=c++11'],
            include_dirs=get_numpy_include_dirs() + [qhull_root],
        ),
    ],

    package_data={'stardist': ['kernels/*.cl']},

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],

    install_requires=[
        'csbdeep>=0.4.0',
        'scikit-image',
        'numba',
    ],
)
