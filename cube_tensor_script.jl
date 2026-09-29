# Run from this repository: julia --project=. cube_tensor_script.jl
include(joinpath(@__DIR__, "src", "RevisedTN.jl"))
include(joinpath(@__DIR__, "examples", "example_io.jl"))
using .RevisedTN
using Random, LinearAlgebra

function main()
    Random.seed!(1234)
    BLAS.set_num_threads(1)
    # L is BITS PER COORDINATE, not the side length.
    # L=4 -> 16^3; L=10 -> 1024^3; L=14 -> 16384^3.
    config = ModelConfig(L=4, lambda=2.0, t1=1.0, t2=1.4, tci_tolerance=1e-8)
    omega, site0 = 0.3 + 0.4im, 0
    Nc, chi, scale, cutoff = 100, 100, 12.0, 1e-8
    output = new_output("single_point")
    write_run_info(output; config, omega, site0, Nc, chi, scale, cutoff, seed=1234)

    build = @timed build_model(config)
    model = build.value
    # Save scalar diagnostics, never the MPSs themselves.
    rows = NamedTuple[]
    observer(m, mu, t, d) = push!(rows, (moment=m, order=2*m-1,
        moment_re=real(mu), moment_im=imag(mu), bond_t=maxlinkdim(t), bond_d=maxlinkdim(d)))
    run = @timed local_spectrum(model, omega; site0, n_odd=Nc, scale,
        maxdim=chi, cutoff, return_moments=true, observer, progress_every=100)
    result = run.value
    write_rows(joinpath(output, "moments.csv"), rows)
    write_rows(joinpath(output, "spectrum.csv"), [spectral_row(omega, site0, result)])
    write_rows(joinpath(output, "timing.csv"), [(build_seconds=build.time,
        first_call_seconds=run.time, allocated_bytes=run.bytes)])
    println("D_H=$(result.hamiltonian_bond), D_herm=$(result.hermitized_bond), chi=$chi")
    println("rho=$(result.physical_density), |rho|=$(abs(result.physical_density))")
    println("Saved to $output (first-call timing includes JIT; allocations are not peak RSS)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
