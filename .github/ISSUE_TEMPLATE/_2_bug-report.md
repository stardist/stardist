---
name: "\U0001F41B Bug Report"
about: Submit a bug report to help us improve StarDist
title: ''
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To reproduce**
Steps to reproduce the behavior, ideally by providing a code snippet, Python script, or Jupyter notebook.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Data and screenshots**
If applicable, add data (e.g. images) or screenshots to help explain your problem.

**Environment (please complete the following information):**
 - StarDist version [e.g. 0.6.2]
 - CSBDeep version [e.g. 0.6.1]
 - TensorFlow version [e.g. 2.3.1]
 - OS: [Windows, Linux, Mac]
 - GPU memory (if applicable): [e.g. 8 GB]

You may run this code and paste the output:
```python
import importlib, platform

print(f'os: {platform.platform()}')
for m in ('stardist','csbdeep','tensorflow'):
    try:
        print(f'{m}: {importlib.import_module(m).__version__}')
    except ModuleNotFoundError:
        print(f'{m}: not installed')

import tensorflow as tf
try:
    print(f'tensorflow GPU: {tf.test.is_gpu_available()}')
except:
    print(f'tensorflow GPU: {tf.config.list_physical_devices("GPU")}')
```