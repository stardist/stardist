from __future__ import absolute_import, print_function
from setuptools import setup, find_packages, Extension
from numpy.distutils.misc_util import get_numpy_include_dirs
from os import path


def openmp_available():
    import os, subprocess, shutil
    from distutils.sysconfig import get_config_var
    from tempfile import mkdtemp
    try:
        curdir = os.getcwd()
        tmpdir = mkdtemp()
        os.chdir(tmpdir)
        compiler = os.environ.get('CXX',get_config_var('CXX')).split()[0]
        filename = 'test.cpp'
        with open(filename,'w') as f:
            f.write("#include <omp.h>\nint main() { return omp_get_num_threads(); }")
        with open(os.devnull,'w') as void:
            return 0 == subprocess.call([compiler, '-fopenmp', filename], stdout=void, stderr=void)
    except:
        return False
    finally:
        os.chdir(curdir)
        shutil.rmtree(tmpdir)


_dir = path.abspath(path.dirname(__file__))

with open(path.join(_dir,'stardist','version.py')) as f:
    exec(f.read())

with open(path.join(_dir,'README.md')) as f:
    long_description = f.read()

extra_args  = [] # ['-std=c++11']
extra_args += ['-fopenmp'] if openmp_available() else []


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

    ext_modules=[
        Extension(
            'stardist.lib.stardist',
            sources=['stardist/lib/stardist.cpp','stardist/lib/clipper.cpp'],
            include_dirs=get_numpy_include_dirs(),
            extra_compile_args=extra_args,
            extra_link_args=extra_args,
        )
    ],

    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'License :: OSI Approved :: BSD License',

        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],

    install_requires=[
        "csbdeep",
        "scikit-image",
    ],

)