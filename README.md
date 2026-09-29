# Tensor-network NHKPM examples

[![arXiv](https://img.shields.io/badge/arXiv-2606.16424-B31B1B)](https://arxiv.org/abs/2606.16424)

Example implementations accompanying the manuscript and its revised response:
QTCI construction of a three-dimensional non-Hermitian tight-binding model,
MPO Hermitization, and site-resolved NHKPM with a streaming MPS recurrence.
All examples run as ordinary Julia scripts. The repository contains source
and software environments; calculated data and cluster job scripts are not
included.

## Getting started

Use Julia **1.10.3**. From this repository directory:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. cube_tensor_script.jl
```

The default example evaluates one energy and one corner probe on a `16^3`
lattice with 100 odd moments. It saves complex densities, scalar moments,
MPS bond dimensions, timing, parameters, package versions and source SHA-256
hashes in a new `outputs/single_point_.../` directory. Initial startup includes
package precompilation and JIT compilation. Outputs are ignored by Git.

The scripts never activate or install a Julia environment internally. If
you already have the matching packages in your usual project, invoke them
with your normal Julia environment instead. `cube_tensor_demo.ipynb` is a
step-by-step notebook calling the same revised implementation; its first
cell activates the root project.

## Source and examples

| File | Purpose |
| --- | --- |
| `cube_tensor_script.jl` | One spectral point, with editable parameters at the top of `main` |
| `src/RevisedTN.jl`, `src/model.jl` | Revised public API, model construction and Hermitization |
| `NHtk.jl` | Revised `KPM_ldos_bysite_streaming` plus the original helper routines |
| `src/SparseKPMBenchmark.jl` | Sparse Hamiltonian and vector recurrence used as the numerical reference |
| `src/EDReference.jl` | Small-system biorthogonal site-resolved ED residues and Gaussian broadening |
| `examples/spectrum_grid.jl` | Small complex-energy grid; comments give the response benchmark grid |
| `examples/domain_wall.jl` | QTCI tanh potential, two probes, represented-potential checks |
| `examples/convergence.jl` | Vary chi, Nc and cutoff; reuse scalar moments for smaller Nc |
| `examples/spatial_corner.jl` | Independently evaluate 64 sites in the outer `4 x 4 x 4` corner cube |
| `examples/scaling.jl` | Warmed TN/sparse per-point timings and small-system ED timing |
| `examples/compare_sparse_ed.jl` | Matched TN/sparse probes and a site-resolved ED reference |
| `examples/plot_spectrum.py` | Optional PNG plotting for `spectrum.csv` and `spatial.csv` |
| `test/runtests.jl` | Small-system matrix, Hermitization, moment and ED checks |

See `VALIDATION.md` for the checks actually run on this revision.

For example:

```sh
julia --project=. examples/domain_wall.jl
julia --project=. examples/convergence.jl
julia --project=. examples/compare_sparse_ed.jl
julia --project=. test/runtests.jl
```

The examples use small grids/orders by default to keep them practical.
Change the configuration block in each script for production calculations.
They run serially and save completed points; they do not implement job arrays,
automatic restart or result merging.

For optional plots, install Python's `numpy` and `matplotlib` in your Python
environment, then pass an actual output filename:

```sh
python examples/plot_spectrum.py outputs/grid_XXXXX/spectrum.csv
python examples/plot_spectrum.py outputs/spatial_corner_XXXXX/spatial.csv
```

Spectral plots use Im(E) horizontally and Re(E) vertically. Both probes share
the same physical-amplitude color scale, starting at zero. The plot uses
`abs(rho)` without changing the saved complex values. A complete rectangular
grid is plotted as cells; an incomplete grid is shown as individual points.

## Coordinates, model and QTCI construction

`L` is the **number of binary bits per coordinate**, not a linear size:

| L | Linear size | Physical sites |
| --- | --- | --- |
| 4 | 16 | 4096 |
| 10 | 1024 | 2^30 |
| 14 | 16384 | 2^42 |

Physical sites and coordinates are zero based:
`site0 = x + side*y + side^2*z`, with `side=2^L`. The tensor chain has the
**z/y/x block ordering**, most significant bit first in each block; x is
the fastest physical coordinate. Hermitization adds a leading auxiliary bit.
The Hamiltonian has open boundaries and the same staggered real hopping as
the original script, with `t1=1` and `t2=1.4`.

The periodic onsite field is built from two QTCI factors as in the original
calculation. With `s(q)=+1` for `q mod 4` in `{0,3}` and `-1` otherwise:

```text
V(x,y,z) = i * lambda * s(x)*s(y)*s(z)
                         * s(x div 4)*s(y div 4)*s(z div 4).
```

To replace it by a smooth domain wall:

```julia
config = ModelConfig(L=4, lambda=1.4, tci_tolerance=1e-8)
field = tanh_field(16; gamma=1.4, x0=6.0, R=3.0)
model = build_model(config; loss_field=field)
```

The callback receives the zero-based flattened index and returns a **real**
field. It follows QTCI -> scalar MPS -> diagonal MPO, and is multiplied by
`i` once. It replaces the periodic field. The response probes are `(4,7,7)`
and `(11,7,7)`. `domain_wall.jl` checks the represented MPO against the callback
along both probe slices; these samples do not constitute a global error bound.

## Revised numerical implementation

For `A = omega*I - H`, the Hermitized MPO represents `[0 A; A' 0]`.
Its adjoint block uses `swapprime(dag(A), 0=>1)` to conjugate **and** exchange
MPO input/output indices. The recurrence rescales this MPO **once** by `a`:

```text
Hbar = Hherm / a
D = sigma_minus tensor I
t0 = |R_l>,  t1 = Hbar |R_l>
d0 = 0,      d1 = |L_l>
t(n+1) = 2 Hbar t(n) - t(n-1)
d(n+1) = 2 D t(n) + 2 Hbar d(n) - d(n-1)
mu(m) = <L_l | d(2m-1)>
```

`D` differentiates with respect to the **rescaled** conjugate energy; there
is no extra `1/a` in this operator. Only the two most recent ordinary and
derivative MPSs, plus update temporaries, are kept. Odd scalar overlaps are
taken before the kernel sum. No final MPS-sum compression is performed.

- `n_odd` / `Nc` counts odd contributions. The last polynomial order is
  `2*Nc-1`; `Nc=500` means order 999.
- Every recurrence MPO-MPS application and MPS addition explicitly uses
  `alg="densitymatrix"`, `maxdim=chi`, and `cutoff=1e-8` by default.
  The cutoff controls discarded density-matrix weight subject to the bond cap.
- `chi` limits **propagated MPS** bonds. `hamiltonian_bond` and
  `hermitized_bond` report MPO bonds; these are different quantities.
- QTCI's independent construction tolerance is `tci_tolerance=1e-8`.
  The Hamiltonian-building MPO arithmetic retains the production code's
  settings; the recurrence's chi/cutoff are not applied to Hamiltonian construction.
- Kernel indexing and the scalar prefactor preserve the convention used for
  the revised calculations. Both TN and sparse examples use this convention.
- `scaled_density` is the direct kernel sum; `physical_density =
  scaled_density/a^2` is the result in the original complex-energy plane.
  Numerical comparisons use the complex values, and plots can display their
  absolute value. Signed small-weight oscillations are not clipped.
- Reweighting at a different `Nc` recalculates its kernel from the same raw
  moments. Changes to energy, site, chi, cutoff, scale or Hamiltonian require
  another recurrence.

`a=12` is the fixed bound used for the reported parameter windows. The code
does not run DMRG to select it. For a changed model/window, check that the
singular values of `omega*I-H` are strictly below `a`. The small-system test
checks this through the Hermitized eigenvalues. Increasing Nc does not repair
an insufficient bound.

### Settings from the revised calculations

| Calculation | L | loss | Nc | chi | Energy/probe |
| --- | --- | --- | --- | --- | --- |
| 16^3 TN/sparse benchmark | 4 | lambda=2 | 200 | 100 | site0=0; Re 0..3.5 (101), Im -2.1..2.1 (121) |
| Convergence | 10 | lambda=1.4 or 5 | 100,200,300,400,500 | 20,40,60,80,100 | Selected energy windows/probes |
| Trillion spectrum | 14 | lambda=1.4 | 500 | 100 | site0=0; Re 0..2 (101), Im -1.1..1.1 (111) |
| Trillion spatial sample | 14 | lambda=1.4 | 500 | 100 | E=0.26i; x,y,z=0..3 |
| Tanh domain wall | 4 | gamma=1.4, x0=6, R=3 | 500 | 100 | Two probes; Re -4.5..4.5 (451), Im -1.6..1.6 (161) |

Unless varied in a convergence example, use `a=12`, compression cutoff
`1e-8` and QTCI tolerance `1e-8`. These example scripts expose the relevant
parameters; running their small defaults does not reproduce every paper figure.

## Timing and memory

The scaling example separates model/Hermitization construction from the TN
recurrence and warms the algorithm before measurements. TN/sparse solver
times are for **one energy and one probe**; ED time is for all eigenpairs.
ED spectra use biorthogonal local residues and a separately chosen Gaussian
broadening. The ED kernel is not identical to a finite-order KPM kernel.

`@timed` bytes are cumulative Julia allocations, **not peak RSS**. Measure
peak process memory externally when needed. At `L=14`, a single ComplexF64
physical-space vector occupies **64 TiB**, or **128 TiB** in the doubled
Hermitized space. These are analytic single-vector storage estimates; they
are not sparse-process measurements or runtime comparisons.

## Environments and provenance

| Component | Recorded version |
| --- | --- |
| Julia | 1.10.3 |
| ITensors.jl | 0.9.10 |
| ITensorMPS.jl | 0.3.20 |
| Quantics.jl | 0.4.6 |
| QuanticsTCI.jl | 0.7.2 |
| TensorCrossInterpolation.jl | 0.9.17 |
| TCIITensorConversion.jl | 0.2.1 |

The root `Project.toml` pins these direct packages and `Manifest.toml` records
the resolved example environment. `environments/paper/` contains the unchanged
saved Project/Manifest from the calculation environment, including its original
transitive versions and additional packages. It is separate from the lean
example environment.

The original repository revision **3b59549** identifies the stored-MPS
implementation, not this revised streaming implementation. The revised core
is extracted from the saved convergence/domain-wall/sparse sources used in
the response calculations, with reusable example interfaces and explicit
compression-algorithm keywords. Each example records source hashes and active
versions. A published revision/tag must refer to the commit containing these
files; no DOI or release identifier is assigned by these scripts.

`2D_lattice.jl`, `extra_util.jl`, `source_ed.jl` and `small_ed.ipynb` remain
as historical utilities. `source_ed.jl`/`small_ed.ipynb` use Plots and
`extra_util.jl` additionally requires FFTW, which is not a dependency of the
revised examples (install it in a separate legacy environment if needed).
For the revised calculations use the entry points above. The old MPS-sum
functions remain available for inspection; their original defaults/final
compression differ from the new scalar-overlap path.
