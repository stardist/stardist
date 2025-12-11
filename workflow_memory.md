# Workflow Fix Attempts

## Attempt 1: Solution A - MACOSX_DEPLOYMENT_TARGET=10.14
- **Commit**: 001a7a7
- **Result**: FAILED
- **Error**: `libunwind.1.0.dylib has a minimum target of 14.0`
- **Lesson**: Homebrew's llvm@14 bundles libunwind with 14.0 minimum. 10.14 is not enough.

## Attempt 2: Solution C - conda-forge compilers
- **Commit**: da1cd0e (fixed syntax in 5fbfcdb)
- **Result**: FAILED
- **Error**: `symbol not found in flat namespace '___kmpc_dispatch_deinit'`
- **Lesson**: conda-forge `compilers` metapackage has newer clang with OpenMP symbols not in older libomp.

## Attempt 3: System clang + conda-forge OpenMP
- **Commit**: 6a3a862
- **Result**: FAILED
- **Error**: Same OpenMP symbol error
- **Lesson**: System clang doesn't have proper OpenMP support without extra flags.

## Attempt 4: Solution B - MACOSX_DEPLOYMENT_TARGET=14.0 + Homebrew llvm + conda-forge libomp
- **Commit**: ffdd79b
- **Result**: FAILED
- **Error**: Segfault in `non_maximum_suppression_inds` at runtime
- **Lesson**: Mixing conda-forge libomp (11.1.0) with Homebrew llvm@14 causes runtime crash.

## Attempt 5: Homebrew llvm@14 for both compiler and OpenMP
- **Commit**: 5230781
- **Result**: FAILED
- **Error**: Same segfault in NMS code
- **Lesson**: Homebrew llvm@14's libomp.dylib still causes segfault at runtime.

## Attempt 6: GCC-13 with MACOSX_DEPLOYMENT_TARGET=14.0 - SUCCESS!
- **Commit**: c2405f2
- **Result**: SUCCESS!
- **Lesson**: GCC-13 with 14.0 target builds working wheels that pass all tests.
- Uses same compiler as tests.yml (GCC-13)
- Homebrew GCC libs require 14.0 minimum target on macOS 15 runner

## Solution Summary
**Working configuration:**
- Compiler: GCC-13 (Homebrew)
- MACOSX_DEPLOYMENT_TARGET: 14.0 for x86_64, 12.0 for arm64
- No need for conda-forge OpenMP or Homebrew llvm

**Why clang failed:**
- Homebrew llvm@14 links against libunwind with 14.0 minimum
- Mixing conda-forge libomp with Homebrew clang caused runtime segfaults
- System clang lacks native OpenMP support

**Trade-off:**
- Wheel requires macOS 14.0+ (Sonoma)
- Users on older macOS must build from source

## Next Steps
- Full matrix build running (11 jobs: cp38-cp313 × x86_64+arm64)
- If all pass, merge to `wheels` branch for final verification
- Then create PR to main
