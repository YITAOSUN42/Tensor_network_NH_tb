include(joinpath(@__DIR__, "..", "src", "RevisedTN.jl"))
include(joinpath(@__DIR__, "..", "src", "SparseKPMBenchmark.jl"))
include("example_io.jl")
using .RevisedTN, .SparseKPMBenchmark
using Random, LinearAlgebra

function main()
    Random.seed!(1234)
    BLAS.set_num_threads(1)
    bits = [2,3,4]  # Extend for TN only; sparse/dense allocations are guarded below.
    lambda, omega, site0 = 1.4, 0.26im, 0
    Nc, chi, scale, cutoff, tci_tolerance = 50, 100, 12.0, 1e-8, 1e-8
    sparse_max_sites, ed_max_sites = 4096, 512
    output = new_output("scaling")
    write_run_info(output;bits,lambda,omega,site0,Nc,chi,scale,cutoff,tci_tolerance,
        sparse_max_sites,ed_max_sites,seed=1234)
    # Warm the same algorithm/type paths, outside the measured repetitions.
    warm = build_model(ModelConfig(L=2,lambda=lambda,tci_tolerance=tci_tolerance))
    local_spectrum(warm,omega;n_odd=2,scale,maxdim=chi,cutoff)
    Hw = build_sparse_hoti(4,4,4;lambda)
    nhkpm_local_sparse(Hw,omega;n_odd=2,scale)
    eigen(Matrix(Hw))
    rows = NamedTuple[]
    for L in bits
        build = @timed build_model(ModelConfig(L=L,lambda=lambda,tci_tolerance=tci_tolerance))
        model = build.value
        hbuild = @timed hermitize_model(model,omega)
        Hherm, sites, D = hbuild.value
        tn = @timed KPM_ldos_bysite_streaming(Hherm,Nc,site0,scale,sites,D;maxdim=chi,cutoff)
        push!(rows,(L=L,sites=model.total_sites,method="TN",
            build_seconds=build.time+hbuild.time,solve_seconds=tn.time,
            allocated_bytes=tn.bytes,D_H=maxlinkdim(model.hamiltonian)))
        if model.total_sites <= sparse_max_sites
            spbuild = @timed build_sparse_hoti(model.side,model.side,model.side;lambda)
            H = spbuild.value
            sp = @timed nhkpm_local_sparse(H,omega;site0,n_odd=Nc,scale)
            push!(rows,(L=L,sites=model.total_sites,method="sparse",
                build_seconds=spbuild.time,solve_seconds=sp.time,allocated_bytes=sp.bytes,D_H=0))
            if model.total_sites <= ed_max_sites
                dense = @timed Matrix(H)
                ed = @timed eigen(dense.value)
                push!(rows,(L=L,sites=model.total_sites,method="ED_eigenpairs",
                    build_seconds=spbuild.time+dense.time,solve_seconds=ed.time,
                    allocated_bytes=ed.bytes,D_H=0))
            end
        end
        write_rows(joinpath(output,"scaling.csv"),rows)
    end
    println("Saved $output")
    println("TN/sparse timings are per energy/site; ED returns all eigenpairs.")
    println("Allocated bytes are cumulative Julia allocations, NOT peak process memory.")
end
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
