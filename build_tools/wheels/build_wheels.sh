#!/bin/bash
set -e
set -x

# Reproducible builds
export SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct 2>/dev/null || echo "0")
export PYTHONHASHSEED=0

if [[ $(uname) == "Darwin" ]]; then
    # Use conda-forge llvm-openmp for macOS OpenMP support
    # This version has lower LC_VERSION_MIN_MACOSX than Homebrew's
    # Following scikit-learn's approach:
    # https://github.com/scikit-learn/scikit-learn/blob/main/build_tools/wheels/build_wheels.sh

    if [[ "$CIBW_BUILD" == *-macosx_arm64 ]]; then
        export MACOSX_DEPLOYMENT_TARGET=12.0
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-arm64/llvm-openmp-11.1.0-hf3c4609_1.tar.bz2"
    else
        export MACOSX_DEPLOYMENT_TARGET=10.9
        OPENMP_URL="https://anaconda.org/conda-forge/llvm-openmp/11.1.0/download/osx-64/llvm-openmp-11.1.0-hda6cdc1_1.tar.bz2"
    fi

    # Create conda env with llvm-openmp
    conda create -n build -y $OPENMP_URL

    # Find the conda prefix (miniforge vs miniconda)
    if [[ -d "$HOME/miniforge3/envs/build" ]]; then
        PREFIX="$HOME/miniforge3/envs/build"
    elif [[ -d "$HOME/miniconda3/envs/build" ]]; then
        PREFIX="$HOME/miniconda3/envs/build"
    else
        echo "ERROR: Could not find conda environment"
        exit 1
    fi

    # Verify libomp exists
    if [[ ! -f "$PREFIX/lib/libomp.dylib" ]]; then
        echo "ERROR: libomp.dylib not found in $PREFIX/lib"
        ls -la "$PREFIX/lib/" || true
        exit 1
    fi

    # Use Apple clang with OpenMP
    export CC=/usr/bin/clang
    export CXX=/usr/bin/clang++
    export CPPFLAGS="-Xpreprocessor -fopenmp"
    export CFLAGS="-I$PREFIX/include"
    export CXXFLAGS="-I$PREFIX/include"
    export LDFLAGS="-Wl,-rpath,$PREFIX/lib -L$PREFIX/lib -lomp"

    echo "=== Build configuration ==="
    echo "MACOSX_DEPLOYMENT_TARGET=$MACOSX_DEPLOYMENT_TARGET"
    echo "CC=$CC"
    echo "CXX=$CXX"
    echo "CPPFLAGS=$CPPFLAGS"
    echo "CFLAGS=$CFLAGS"
    echo "CXXFLAGS=$CXXFLAGS"
    echo "LDFLAGS=$LDFLAGS"
    echo "PREFIX=$PREFIX"
    otool -l "$PREFIX/lib/libomp.dylib" | grep -A3 LC_VERSION_MIN || true
    echo "==========================="
fi

python -m pip install cibuildwheel
python -m cibuildwheel --output-dir wheelhouse
