# Local validation of the revised examples

Checked on 2026-09-29 with Julia 1.10.3 on Windows, using the repository-root
Project/Manifest and one BLAS thread.

## Numerical checks

`julia --project=. test/runtests.jl`: **269 checks passed**.

- 67 coordinate-mapping and input checks.
- 198 matrix/recurrence checks on a `4^3` lattice: periodic and tanh fields,
  amplitudes 0, 1.4 and 5, two complex energies and two probes. These compare
  the full MPO matrix with an independent sparse construction, verify the
  Hermitized blocks and derivative operator, check the finite-size scale
  bound, compare raw TN/sparse odd moments, and verify smaller-Nc reweighting
  against an independently propagated shorter recurrence.
- Four checks of the ED biorthogonal residues, including comparison with
  the diagonal resolvent for a non-normal complex matrix.

The moment tests use chi=64 and cutoff=1e-12 to test the implementation with
small truncation error. The zero-amplitude tanh case is represented exactly
as a zero MPO because QTCI requires a nonzero initial pivot.

## Executed example

`julia --project=. cube_tensor_script.jl` completed with the default
`16^3`, lambda=2, E=0.3+0.4i, site0=0, Nc=100, a=12, chi=100 and cutoff=1e-8.
It wrote the spectrum, 100 scalar moments with bond diagnostics, run metadata
and timing. The physical complex density differed from the corresponding
sparse calculation by approximately **5.85e-7** in absolute value and
**2.88e-5** relative to the sparse value. This is a check at one point, not a
full spectral error bound.

All 18 Julia source/example/test files were parsed with Julia 1.10.3, checking
for parse-error nodes. The optional Python plotting script also generated a
PNG from the executed example's CSV. The other examples share the tested
model/recurrence API; their full grids and large-system settings were not
rerun as part of this local validation.

The saved original calculation environment is byte-identical to the supplied
Project/Manifest. Both notebooks contain no stored cell outputs. Generated
validation data and images are excluded from the source files by `.gitignore`.
