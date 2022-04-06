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

        # thanks to @jaimergp (https://github.com/conda-forge/staged-recipes/pull/17766)
        # issue: qhull has a mix of c and c++ source files
        #        gcc warns about passing -std=c++11 for c files, but clang errors out
        compile_original = self.compiler._compile
        def compile_patched(obj, src, ext, cc_args, extra_postargs, pp_opts):
            # remove c++ specific (extra) options for c files
            if src.lower().endswith('.c'):
                extra_postargs = [arg for arg in extra_postargs if not arg.lower().startswith('-std')]
            return compile_original(obj, src, ext, cc_args, extra_postargs, pp_opts)
        # monkey patch the _compile method
        self.compiler._compile = compile_patched

        _extra_compile_args = list(ext.extra_compile_args)
        _extra_link_args    = list(ext.extra_link_args)
        try:
            assert False
            ext.extra_compile_args += self.openmp_compile_args[compiler]
            ext.extra_link_args    += self.openmp_link_args[compiler]
            super(build_ext_openmp, self).build_extension(ext)
        except:
            print('compiling with OpenMP support failed, re-trying without')
            ext.extra_compile_args = _extra_compile_args
            ext.extra_link_args    = _extra_link_args
            super(build_ext_openmp, self).build_extension(ext)
        finally:
            # restore original _compile, just in case
            self.compiler._compile = compile_original


#------------------------------------------------------------------------------------


# cf. https://github.com/mkleehammer/pyodbc/issues/82#issuecomment-231561240
_dir = path.dirname(__file__)

with open(path.join(_dir,'stardist','version.py'), encoding="utf-8") as f:
    exec(f.read())

with open(path.join(_dir,'README.md'), encoding="utf-8") as f:
    long_description = f.read()


external_root = path.join(_dir, 'stardist', 'lib', 'external')

qhull_root = path.join(external_root, 'qhull_src', 'src')
qhull_src = sorted(glob(path.join(qhull_root, '*', '*.c*')))[::-1]

nanoflann_root = path.join(external_root, 'nanoflann')

clipper_root = path.join(external_root, 'clipper')
clipper_src = sorted(glob(path.join(clipper_root, '*.cpp*')))[::-1]


setup(
    name='stardist',
    version=__version__,
    description='StarDist - Object Detection with Star-convex Shapes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/stardist/stardist',
    author='Uwe Schmidt, Martin Weigert',
    author_email='research@uweschmidt.org, martin.weigert@epfl.ch',
    license='BSD-3-Clause',
    packages=find_packages(),
    python_requires='>=3.6',

    cmdclass={'build_ext': build_ext_openmp},

    ext_modules=[
        Extension(
            'stardist.lib.stardist2d',
            sources = ['stardist/lib/stardist2d.cpp', 'stardist/lib/utils.cpp'] + clipper_src,
            extra_compile_args = ['-std=c++11'],
            include_dirs = get_numpy_include_dirs() + [clipper_root, nanoflann_root],
        ),
        Extension(
            'stardist.lib.stardist3d',
            sources = ['stardist/lib/stardist3d.cpp', 'stardist/lib/stardist3d_impl.cpp', 'stardist/lib/utils.cpp'] + qhull_src,
            extra_compile_args = ['-std=c++11'],
            include_dirs = get_numpy_include_dirs() + [qhull_root, nanoflann_root],
        ),
    ],

    package_data={'stardist': [ 'kernels/*.cl', 'data/images/*' ]},

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',

        'Operating System :: OS Independent',

        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],

    install_requires=[
        'csbdeep>=0.6.3',
        'scikit-image',
        'numba',
        'imageio',
    ],

    extras_require={
        "tf1":  ["csbdeep[tf1]>=0.6.3"],
        "test": ["pytest"],
        "bioimageio": ["bioimageio.core>=0.5.0","importlib-metadata"],
    },

)
