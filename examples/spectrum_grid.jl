include(joinpath(@__DIR__, "..", "src", "RevisedTN.jl"))
include("example_io.jl")
using .RevisedTN
using Random, LinearAlgebra

function main()
    Random.seed!(1234)
    BLAS.set_num_threads(1)
    # Small demonstration grid. For the 16^3 response benchmark use L=4,
    # lambda=2, Nc=200, re=range(0,3.5; length=101),
    # im=range(-2.1,2.1; length=121), and chi=100.
    config = ModelConfig(L=2, lambda=2.0)
    re_grid = range(0.0, 3.5; length=5)
    im_grid = range(-2.1, 2.1; length=7)
    Nc, chi, scale, cutoff, site0 = 30, 100, 12.0, 1e-8, 0
    output = new_output("grid")
    write_run_info(output; config, re_grid, im_grid, Nc, chi, scale, cutoff, site0, seed=1234)
    model = build_model(config)
    rows = NamedTuple[]
    for re in re_grid, im in im_grid
        omega = complex(re, im)
        r = local_spectrum(model, omega; site0, n_odd=Nc, scale, maxdim=chi, cutoff)
        push!(rows, spectral_row(omega, site0, r))
        # Flush completed energies so an interrupted example still has its results.
        write_rows(joinpath(output, "spectrum.csv"), rows)
        println("$(length(rows))/$(length(re_grid)*length(im_grid)) energies")
    end
    println("Saved $output; plot with examples/plot_spectrum.py")
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
