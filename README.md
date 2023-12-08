# MyPy Type Checking for NumPy/Jax/PyTorch Einsum Operations

`mypy_einsum` is a [Mypy](https://mypy.readthedocs.io/) plugin for type checking [`np.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html), [`jax.numpy.einsum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html), and [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html) operations.

The Einstein summation convention can be used to compute many multi-dimensional, linear algebraic array operations. `einsum` provides a succinct way of representing these.
However, since `einsum` equations are passed as a string, it is very easy to overlook typos or other bugs as linters are unable to help. `mypy_einsum` is a [Mypy](https://mypy.readthedocs.io/) plugin that that is able to statically verify the correctness of `einsum` equations with needing to execute the code.

## Installation

`mypy_einsum` can be installed with [`pip`](https://pip.pypa.io/):

```shell
pip install mypy-einsum
```

## Setup

To enable the plugin, add it to you projects [Mypy configuration file](https://mypy.readthedocs.io/en/stable/config_file.html).
Usually `mypy.ini`:

```ini
[mypy]
plugins = mypy_einsum
```

or `pyproject.toml`:

```toml
[tool.mypy]
plugins = ["mypy_einsum"]
```

## Example

Can you spot the error in the running code?

```python
import numpy as np

a = np.arange(9).reshape(3, 3)

np.einsum("ik,kj->ij", a)
```

Well, you don't need to `mypy_einsum` can to it for you:

```shell
❯ mypy example.py --pretty
example.py:5: error: Number of einsum subscripts must be equal to the
number of operands.  [einsum]
    np.einsum("ik,kj->ij", a)
              ^~~~~~~~~~~
Found 1 error in 1 file (checked 1 source file)
```

And it's is pretty simple fix after reading the error message:

```python
np.einsum("ik,kj->ij", a, a)
```

```bash
❯ mypy example.py
Success: no issues found in 1 source file
```

## Supported Operations

- [`np.einsum`](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html)
- [`np.einsum_path`](https://numpy.org/doc/stable/reference/generated/numpy.einsum_path.html)
- [`jax.numpy.einsum`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html)
- [`jax.numpy.einsum_path`](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum_path.html)
- [`torch.einsum`](https://pytorch.org/docs/stable/generated/torch.einsum.html)
