# fmt: off
from array_module import einsum as einsum
import numpy as np

A: np.ndarray

einsum("")  # E: Number of einsum subscripts must be equal to the number of operands
einsum("", 0, 0)  # E: Number of einsum subscripts must be equal to the number of operands
einsum(",", 0, [0], [0])  # E: Number of einsum subscripts must be equal to the number of operands
einsum(",", [0])  # E: Number of einsum subscripts must be equal to the number of operands

einsum("i..", [0, 0])  # E: Invalid Ellipses
einsum(".i...", [0, 0])  # E: Invalid Ellipses
einsum("j->..j", [0, 0])  # E: Invalid Ellipses
einsum("j->.j...", [0, 0])  # E: Invalid Ellipses

einsum("i%...", [0, 0])  # E: Character % is not a valid symbol
einsum("...j$", [0, 0])  # E: Character % is not a valid symbol
einsum("i->&", [0, 0])  # E: Character & is not a valid symbol

einsum("i->ij", [0, 0])  # E: Output character j did not appear in the input

einsum("ij->jij", [[0, 0], [0, 0]])  # E: Output character j appeared multiple times.
