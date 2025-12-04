# NumPy 2.0 Compatibility Guide

This document provides instructions for building and testing StarDist with NumPy 2.0 support.

## Overview

StarDist's C/C++ extensions (stardist2d and stardist3d) have been updated to support both NumPy 1.x (≥1.20) and NumPy 2.x. The extensions use the NumPy 2.0 API with backward compatibility shims.

## Requirements

- Python ≥3.6
- NumPy ≥1.20 (supports both 1.x and 2.x)
- C/C++ compiler with C++11 support
- Build dependencies: `setuptools`, `wheel`

## Building from Source

### Step 1: Install Build Dependencies

```bash
pip install setuptools wheel numpy
```

### Step 2: Clean Previous Builds (Optional)

If you previously built StarDist with NumPy 1.x, clean old build artifacts:

```bash
# Remove old build files
rm -rf build/ dist/ *.egg-info
# Remove compiled extensions
find . -name "*.so" -delete
find . -name "*.pyc" -delete
```

### Step 3: Build and Install

#### Option A: Editable Install (for development)

```bash
pip install -e . --no-build-isolation
```

The `--no-build-isolation` flag ensures the build uses your installed NumPy version.

#### Option B: Standard Install

```bash
pip install .
```

#### Option C: Build Wheel

```bash
python setup.py bdist_wheel
pip install dist/stardist-*.whl
```

## Verifying the Installation

### Quick Verification

```python
import numpy as np
print(f"NumPy version: {np.__version__}")

# Test imports
import stardist
from stardist.lib import stardist2d, stardist3d
print("✓ StarDist C extensions loaded successfully")
```

### Running Compatibility Tests

The repository includes a comprehensive test suite for NumPy compatibility:

```bash
# Run NumPy compatibility tests
pytest tests/test_numpy_compat.py -v

# Run all tests (requires additional dependencies)
pytest tests/ -v
```

### Manual Function Tests

#### Test 2D Star Distance

```python
import numpy as np
from stardist.lib.stardist2d import c_star_dist

# Create test image
img = np.zeros((64, 64), dtype=np.uint16)
img[20:40, 20:40] = 1  # Simple square object

# Calculate star distances
result = c_star_dist(img, n_rays=32, grid_y=1, grid_x=1)
print(f"2D result shape: {result.shape}")  # Should be (64, 64, 32)
```

#### Test 3D Star Distance

```python
import numpy as np
from stardist.lib.stardist3d import c_star_dist3d

# Create test volume
img = np.zeros((32, 32, 32), dtype=np.uint16)
img[10:20, 10:20, 10:20] = 1  # Simple cube object

# Create normalized direction vectors
n_rays = 96
pdz = np.random.randn(n_rays).astype(np.float32)
pdy = np.random.randn(n_rays).astype(np.float32)
pdx = np.random.randn(n_rays).astype(np.float32)
norm = np.sqrt(pdx**2 + pdy**2 + pdz**2)
pdx /= norm; pdy /= norm; pdz /= norm

# Calculate star distances
result = c_star_dist3d(img, pdz, pdy, pdx, n_rays, grid_z=1, grid_y=1, grid_x=1)
print(f"3D result shape: {result.shape}")  # Should be (32, 32, 32, 96)
```

## Testing with Different NumPy Versions

### Test with NumPy 1.x

```bash
# Install NumPy 1.26 (last 1.x version)
pip uninstall -y stardist numpy
pip install 'numpy<2'
pip install -e . --no-build-isolation

# Verify
python -c "import numpy; print(f'NumPy {numpy.__version__}'); import stardist; print('Success')"

# Run tests
pytest tests/test_numpy_compat.py -v
```

### Test with NumPy 2.x

```bash
# Install NumPy 2.x
pip uninstall -y stardist numpy
pip install 'numpy>=2.0'
pip install -e . --no-build-isolation

# Verify
python -c "import numpy; print(f'NumPy {numpy.__version__}'); import stardist; print('Success')"

# Run tests
pytest tests/test_numpy_compat.py -v
```

## Troubleshooting

### Import Error: "numpy.core.multiarray failed to import"

This error occurs when trying to use extensions compiled with NumPy 1.x in a NumPy 2.x environment. Solution:

```bash
# Rebuild the extensions with your current NumPy version
pip uninstall -y stardist
pip install -e . --no-build-isolation
```

### Compilation Errors

If you encounter compilation errors:

1. Ensure you have a C++11 compatible compiler:
   - Linux: GCC ≥4.8 or Clang ≥3.3
   - macOS: Xcode command line tools
   - Windows: MSVC ≥14.0 or MinGW

2. Check NumPy installation:
   ```bash
   python -c "import numpy; print(numpy.get_include())"
   ```

3. Try building with verbose output:
   ```bash
   pip install -e . --no-build-isolation -v
   ```

### Test Failures

If tests fail:

1. Verify NumPy version compatibility:
   ```python
   import numpy
   print(numpy.__version__)  # Should be ≥1.20
   ```

2. Check that extensions are properly compiled:
   ```bash
   python -c "from stardist.lib import stardist2d, stardist3d"
   ```

3. Run tests with verbose output:
   ```bash
   pytest tests/test_numpy_compat.py -vv
   ```

## Technical Details

### Compilation Flags

The extensions are compiled with these NumPy 2.0 compatibility flags:

```python
-DNPY_NO_DEPRECATED_API=NPY_2_0_API_VERSION
-DNPY_TARGET_VERSION=NPY_2_0_API_VERSION
```

These flags:
- Use the NumPy 2.0 API version
- Disable deprecated API usage
- Enable backward compatibility with NumPy 1.x (≥1.20)

### Why Recompilation May Be Needed

- Extensions compiled with NumPy 1.x header files **will not work** with NumPy 2.x runtime
- Extensions compiled with NumPy 2.x header files **will work** with both NumPy 1.x and 2.x runtimes (when using NPY_2_0_API_VERSION)
- Always recompile when switching between major NumPy versions for best compatibility

## Additional Resources

- [NumPy 2.0 Migration Guide](https://numpy.org/devdocs/numpy_2_0_migration_guide.html)
- [StarDist Documentation](https://github.com/stardist/stardist)
- [Test Suite](tests/test_numpy_compat.py) - Reference implementation for testing
