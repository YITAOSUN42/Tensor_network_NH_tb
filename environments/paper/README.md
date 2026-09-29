# Recorded calculation environment

`Project.toml` and `Manifest.toml` are unchanged copies of the saved Julia
1.10.3 environment accompanying the revised calculations. They include
packages used elsewhere in that working environment (for example CUDA,
HDF5 and Plots); those packages are not required by the new CPU examples.
The manifest contains registry package identifiers, versions and tree hashes,
with no local source-path dependencies.

The repository root provides a smaller runnable project and a newly resolved
manifest. It pins the six direct tensor-network packages to the recorded
versions. Its transitive dependency resolution is separate from this saved
environment. Use this directory's environment when the original full package
resolution is needed; use the root environment for the standalone examples.

These environment files document dependencies. They do not by themselves
identify the source revision or reproduce the original cluster resources.
