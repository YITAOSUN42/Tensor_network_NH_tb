using Test, Random, LinearAlgebra, SparseArrays
include(joinpath(@__DIR__,"..","src","RevisedTN.jl"))
include(joinpath(@__DIR__,"..","src","SparseKPMBenchmark.jl"))
include(joinpath(@__DIR__,"..","src","EDReference.jl"))
using .RevisedTN, .SparseKPMBenchmark, .EDReference
using ITensors, ITensorMPS
BLAS.set_num_threads(1)
Random.seed!(1234)
# These tests intentionally contract tiny full matrices (up to 14 indices).
ITensors.set_warn_order(20)

function dense_mpo(H,sites)
    s = reverse(sites)
    n = 2^length(s)
    return reshape(Array(reduce(*,H),prime.(s)...,s...),n,n)
end

@testset "Coordinates and input checks" begin
    for i in 0:63
        @test site_index(coordinates(i,4)...,4) == i
    end
    @test_throws ArgumentError build_model(ModelConfig(L=0))
    @test_throws ArgumentError tanh_field(4;R=0)
    @test_throws ArgumentError density_from_moments([1+0im],2)
end

@testset "MPO construction, Hermitization, moments and sparse reference" begin
    for lambda in (0.0,1.4,5.0), domainwall in (false,true)
        config = ModelConfig(L=2,lambda=lambda,tci_tolerance=1e-10)
        field = tanh_field(4;gamma=lambda,x0=1.2,R=0.8)
        model = domainwall ? build_model(config;loss_field=field) : build_model(config)
        H = build_sparse_hoti(4,4,4;lambda)
        if domainwall
            H = H-spdiagm(0=>diag(H))+spdiagm(0=>[1im*field(i) for i in 0:63])
        end
        dense = dense_mpo(model.hamiltonian,model.sites)
        @test norm(dense-Matrix(H))/norm(H) < 1e-8
        for omega in (0.3+0.4im,0.0+0.26im)
            herm,sites,D = hermitize_model(model,omega)
            B = dense_mpo(herm,sites)
            A = Matrix(omega*I-H)
            expected = [zeros(ComplexF64,64,64) A; A' zeros(ComplexF64,64,64)]
            @test norm(B-B')/norm(B) < 1e-10
            @test norm(B-expected)/norm(expected) < 1e-8
            @test norm(dense_mpo(D,sites)-[zeros(64,128); Matrix{Float64}(I,64,64) zeros(64,64)]) < 1e-10
            @test maximum(abs,eigvals(Hermitian(B)))/12 < 1
            for site0 in (0,21)
                bonds = Tuple{Int,Int}[]
                observer(m,mu,t,d) = push!(bonds,(maxlinkdim(t),maxlinkdim(d)))
                tn = local_spectrum(model,omega;site0,n_odd=12,scale=12,
                    maxdim=64,cutoff=1e-12,return_moments=true,observer)
                sp = nhkpm_local_sparse(H,omega;site0,n_odd=12,scale=12,return_moments=true)
                @test tn.moments ≈ sp.moments rtol=1e-6 atol=1e-7
                @test tn.physical_density ≈ sp.physical_density rtol=1e-6 atol=1e-8
                @test length(bonds) == 12
                @test maximum(maximum,bonds) <= 64
                @test density_from_moments(tn.moments,12;scale=12).scaled_density ≈ tn.scaled_density
                # Independent short recurrence catches a wrong Nc-dependent kernel.
                short = local_spectrum(model,omega;site0,n_odd=4,scale=12,maxdim=64,cutoff=1e-12)
                @test density_from_moments(tn.moments,4;scale=12).scaled_density ≈ short.scaled_density
            end
        end
    end
end

@testset "ED biorthogonal residues" begin
    # Non-normal complex matrix with a known, well-conditioned eigenbasis.
    R = ComplexF64[1 0.2im 0.1; 0.1 1 0.3; 0.2 -0.1im 1]
    E = ComplexF64[-1+0.2im,0.3-0.1im,1.2+0.4im]
    H = R*Diagonal(E)/R
    modes = local_ed_modes(H,[0,2])
    @test vec(sum(modes.residues;dims=1)) ≈ ones(2)
    @test modes.relative_residual < 1e-12
    z = 2+2im
    @test vec(transpose(1 ./ (z .- modes.energies))*modes.residues) ≈ diag(inv(z*I-H))[[1,3]]
    @test all(isfinite,broaden_local_modes(modes,0.2im;eta=0.1))
end
