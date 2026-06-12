# Tensor_network_NH_tb

Demo code for the preprint: XXX.

This repository shows how to encode very large non-Hermitian tight-binding
Hamiltonians in a binary tensor-network representation and how to evaluate
real-space spectral functions with a kernel polynomial method.

## Contents

- `source_ed.jl` and `small_ed.ipynb`: exact-diagonalization checks for small
  systems.
- `2D_lattice.jl`, `NHtk.jl`, and `extra_util.jl`: tensor-network utilities.
- `cube_tensor_script.jl`: script version of the large-system tensor-network calculation.
- `cubic_tensor_demo.ipynb`: notebook version of `test.jl`, organized into
  step-by-step cells for the paper demo.

For a quick smoke test, reduce `L` and `kpm_order` in `cube_tensor_script.jl` or
`cubic_tensor_demo.ipynb` before running the full calculation.
