include(joinpath(@__DIR__, "..", "src", "RevisedTN.jl"))
include("example_io.jl")
using .RevisedTN
using Random, LinearAlgebra

function main()
    Random.seed!(1234)
    BLAS.set_num_threads(1)
    # For the response: L=10, chi=[20,40,60,80,100],
    # Nc=[100,200,300,400,500]. Increase only the parameters below.
    config = ModelConfig(L=2, lambda=1.4)
    chis, orders, cutoffs = [20,40], [20,40,60], [1e-8]
    # To vary cutoff independently: chis=[100], orders=[500],
    # cutoffs=[1e-6,1e-8,1e-10]. The Hamiltonian QTCI tolerance stays fixed.
    energies, site0, scale = [0.26im, 0.4im], 0, 12.0
    output = new_output("convergence")
    write_run_info(output; config, chis, orders, cutoffs, energies, site0, scale, seed=1234)
    model = build_model(config)
    spectra, moments = NamedTuple[], NamedTuple[]
    for chi in chis, cutoff in cutoffs, (energy_id,omega) in enumerate(energies)
        observer(m, mu, t, d) = push!(moments, (energy_id=energy_id,
            re=real(omega),im=imag(omega),site0=site0,chi=chi,cutoff=cutoff,
            moment=m,order=2*m-1,moment_re=real(mu),moment_im=imag(mu),
            bond_t=maxlinkdim(t),bond_d=maxlinkdim(d)))
        r = local_spectrum(model, omega; site0, n_odd=maximum(orders), scale,
            maxdim=chi, cutoff, return_moments=true, observer)
        for Nc in orders
            rho = density_from_moments(r.moments, Nc; scale)
            push!(spectra, merge((chi=chi,cutoff=cutoff,Nc=Nc), spectral_row(omega,site0,rho)))
        end
        write_rows(joinpath(output,"moments.csv"), moments)
        write_rows(joinpath(output,"spectrum.csv"), spectra)
    end
    println("Saved $output; each energy/chi/cutoff needs only one recurrence")
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
