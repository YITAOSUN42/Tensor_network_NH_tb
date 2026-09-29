include(joinpath(@__DIR__, "..", "src", "RevisedTN.jl"))
include(joinpath(@__DIR__, "..", "src", "SparseKPMBenchmark.jl"))
include(joinpath(@__DIR__, "..", "src", "EDReference.jl"))
include("example_io.jl")
using .RevisedTN, .SparseKPMBenchmark, .EDReference
using Random, LinearAlgebra, SparseArrays

function main()
    Random.seed!(1234)
    BLAS.set_num_threads(1)
    config = ModelConfig(L=2, lambda=2.0)
    potential = :periodic  # :tanh selects the identical callback for TN and sparse/ED.
    site0, Nc, chi, scale, cutoff, eta = 0, 30, 100, 12.0, 1e-8, 0.15
    energies = [0.0+0.2im, 0.5+0.3im, 1.0+0.6im]
    output = new_output("compare")
    gamma, x0, R = 1.4, 6.0, 3.0
    # For the response domain wall choose L=4, potential=:tanh, and site0
    # from site_index(4,7,7,16) or site_index(11,7,7,16). Dense ED is costlier.
    potential in (:periodic,:tanh) || error("unknown potential")
    write_run_info(output; config, potential, gamma, x0, R, site0,
        Nc, chi, scale, cutoff, eta, energies, seed=1234)
    s = 2^config.L
    field = potential == :tanh ? tanh_field(s; gamma, x0, R) : nothing
    model = build_model(config;loss_field=field)
    H = build_sparse_hoti(s,s,s; lambda=config.lambda,tx1=config.t1,tx2=config.t2,loss_field=field)
    modes = local_ed_modes(H,[site0])
    println("ED eigenvector condition: $(modes.condition), residual: $(modes.relative_residual)")
    rows = NamedTuple[]
    for omega in energies
        tn = local_spectrum(model,omega;site0,n_odd=Nc,scale,maxdim=chi,cutoff)
        sp = nhkpm_local_sparse(H,omega;site0,n_odd=Nc,scale)
        ed = only(broaden_local_modes(modes,omega;eta))
        a,b = tn.physical_density, sp.physical_density
        push!(rows,(re=real(omega),im=imag(omega),site0=site0,
            tn_re=real(a),tn_im=imag(a),sparse_re=real(b),sparse_im=imag(b),
            abs_complex_error=abs(a-b),ed_re=real(ed),ed_im=imag(ed)))
    end
    write_rows(joinpath(output,"comparison.csv"),rows)
    println("Saved $output; ED uses Gaussian broadening, KPM uses its finite-order kernel")
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
