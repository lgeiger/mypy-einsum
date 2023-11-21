# fmt: off
import numpy as np

A: np.ndarray

np.einsum("")  # E: Number of einsum subscripts must be equal to the number of operands
np.einsum("", 0, 0)  # E: Number of einsum subscripts must be equal to the number of operands
np.einsum(",", 0, [0], [0])  # E: Number of einsum subscripts must be equal to the number of operands
np.einsum(",", [0])  # E: Number of einsum subscripts must be equal to the number of operands

np.einsum("i..", [0, 0])  # E: Invalid Ellipses
np.einsum(".i...", [0, 0])  # E: Invalid Ellipses
np.einsum("j->..j", [0, 0])  # E: Invalid Ellipses
np.einsum("j->.j...", [0, 0])  # E: Invalid Ellipses

np.einsum("i%...", [0, 0])  # E: Character % is not a valid symbol
np.einsum("...j$", [0, 0])  # E: Character % is not a valid symbol
np.einsum("i->&", [0, 0])  # E: Character & is not a valid symbol

np.einsum("i->ij", [0, 0])  # E: Output character j did not appear in the input

np.einsum("ij->jij", [[0, 0], [0, 0]])  # E: Output character j appeared multiple times.
