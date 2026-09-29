include(joinpath(@__DIR__, "..", "src", "RevisedTN.jl"))
include("example_io.jl")
using .RevisedTN
using Random, LinearAlgebra

function main()
    Random.seed!(1234)
    BLAS.set_num_threads(1)
    # Response trillion calculation: change L=14 and Nc=500.
    config = ModelConfig(L=4, lambda=1.4)
    omega, Nc, chi, scale, cutoff = 0.26im, 100, 100, 12.0, 1e-8
    output = new_output("spatial_corner")
    write_run_info(output; config, omega, Nc, chi, scale, cutoff, region="x,y,z=0:3", seed=1234)
    model = build_model(config)
    # Hermitization depends on energy, so reuse it for all 64 sites.
    Hherm, sites, D = hermitize_model(model, omega)
    rows = NamedTuple[]
    for z in 0:3, y in 0:3, x in 0:3
        i0 = site_index(x,y,z,model.side)
        scaled = KPM_ldos_bysite_streaming(Hherm,Nc,i0,scale,sites,D;maxdim=chi,cutoff)
        r = (scaled_density=scaled, physical_density=scaled/scale^2)
        push!(rows, merge((x=x,y=y,z=z), spectral_row(omega,i0,r)))
        write_rows(joinpath(output,"spatial.csv"), rows)
        println("$(length(rows))/64 sites")
    end
    println("Saved $output; all 64 sites evaluated independently")
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
