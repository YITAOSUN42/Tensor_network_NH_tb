include(joinpath(@__DIR__, "..", "src", "RevisedTN.jl"))
include("example_io.jl")
using .RevisedTN
using ITensors, ITensorMPS, Random, LinearAlgebra

function main()
    Random.seed!(1234)
    BLAS.set_num_threads(1)
    config = ModelConfig(L=4, lambda=1.4)
    gamma, x0, R = 1.4, 6.0, 3.0
    probes = [(4,7,7), (11,7,7)]
    # Two energies for a short example. The full response grid is
    # [complex(re,im) for re in -4.5:0.02:4.5 for im in -1.6:0.02:1.6].
    # Set Nc=500 for the response settings; 100 is a cheaper demonstration.
    energies = [2.5-0.8im, 2.5+0.8im]
    Nc, chi, scale, cutoff = 100, 100, 12.0, 1e-8
    side = 2^config.L
    field = tanh_field(side; gamma, x0, R)
    output = new_output("domain_wall")
    write_run_info(output; config, gamma, x0, R, probes, energies, Nc, chi, scale, cutoff, seed=1234)
    model = build_model(config; loss_field=field)

    # Validate the represented potential along both probe slices, using the
    # same binary site mapping as the KPM probe (not just the callback values).
    checks = NamedTuple[]
    for (_,y,z) in probes, x in 0:side-1
        i0 = site_index(x,y,z,side)
        ket = MPS(ComplexF64, model.sites, RevisedTN.to_binary_vector(i0,3*config.L))
        exact = 1im*field(i0)
        actual = inner(ket', model.loss_mpo, ket)
        push!(checks, (x=x,y=y,z=z,site0=i0,exact_im=imag(exact),
            mpo_re=real(actual),mpo_im=imag(actual),abs_error=abs(actual-exact)))
    end
    maximum(r.abs_error for r in checks) <= 100*config.tci_tolerance*max(1,abs(gamma)) ||
        error("Potential MPO sample check failed")
    write_rows(joinpath(output, "potential_checks.csv"), checks)

    rows = NamedTuple[]
    for xyz in probes, omega in energies
        i0 = site_index(xyz..., side)
        r = local_spectrum(model, omega; site0=i0, n_odd=Nc, scale, maxdim=chi, cutoff)
        push!(rows, spectral_row(omega, i0, r))
        write_rows(joinpath(output, "spectrum.csv"), rows)
    end
    println("Saved $output")
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
