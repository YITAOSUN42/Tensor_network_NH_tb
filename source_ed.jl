using LinearAlgebra
using Plots

# ------------------------------------------------------------
# 1) 2D PT-SSH Hamiltonian (A,B sites per cell; checkerboard ±iγ)
#    Intracell A↔B: r
#    Intercell B(x,y)↔A(x+1,y): tx;  B(x,y)↔A(x,y+1): ty
# ------------------------------------------------------------
function build_PT_2D_SSH(Nx::Int=20, Ny::Int=20;
                         r::Real=5.0, tx::Real=1.0, ty::Real=1.0,
                         γ::Real=0.5, periodic::Bool=true,
                         rprime::Real=0.0,   # non-reciprocal only in intracell bond
                         t3::Real=0.0)       # reciprocal diagonal next-nearest

    ncell = Nx * Ny
    dim   = 2 * ncell
    H = zeros(ComplexF64, dim, dim)

    idx(x, y, s) = 2 * ((x-1) + Nx*(y-1)) + s  # x=1..Nx, y=1..Ny, s=1(A) or 2(B)

    for y in 1:Ny, x in 1:Nx
        ia = idx(x, y, 1)  # A(x,y)
        ib = idx(x, y, 2)  # B(x,y)

        # On-site gain/loss (PT)
        H[ia, ia] +=  1im * γ
        H[ib, ib] += -1im * γ

        # Intracell A <-> B: non-reciprocal r ± r′
        H[ia, ib] += -(r + rprime)  # B -> A
        H[ib, ia] += -(r - rprime)  # A -> B

        # neighbors (with wrap if periodic)
        xn = (x < Nx) ? x+1 : 1
        yp = (y < Ny) ? y+1 : 1
        ym = (y > 1)  ? y-1 : Ny

        # Intercell along +x: reciprocal tx
        if periodic || x < Nx
            ia_r = idx(xn, y, 1)
            H[ia_r, ib] += -tx   # B(x,y) -> A(x+1,y)
            H[ib, ia_r] += -tx   # A(x+1,y) -> B(x,y)
        end

        # Intercell along +y: reciprocal ty
        if periodic || y < Ny
            ia_u = idx(x, yp, 1)
            H[ia_u, ib] += -ty   # B(x,y) -> A(x,y+1)
            H[ib, ia_u] += -ty   # A(x,y+1) -> B(x,y)
        end

        # Diagonal next-nearest: reciprocal t3
        if t3 != 0
            # NE: (x+1, y+1)
            if periodic || (x < Nx && y < Ny)
                ia_ne = idx(xn, yp, 1)
                H[ia_ne, ib] += -t3
                H[ib, ia_ne] += -t3
            end
            # SE: (x+1, y-1)
            #if periodic || (x < Nx && y > 1)
            #    ia_se = idx(xn, ym, 1)
            #    H[ia_se, ib] += -t3
            #    H[ib, ia_se] += -t3
            #end
        end
    end

    return H
end

using SparseArrays

"""
    idx(x, y, N)

1-based linear index for site (x,y) with x,y = 1..N.
"""
@inline function idx(x::Int, y::Int, N::Int)
    return (x-1)*N + y
end

"""
    build_hamiltonian(N; tx=1.0, ty=1.0, λ=1.0, β=1/3, ϕ=0.0, sparse=true)

Construct the OBC Hamiltonian for an N×N square lattice with on-site potential
V_{x,y} = i * 2λ * cos(2πβ x + ϕ) * cos(2πβ y + ϕ).

- (x,y) are 1..N in code; the cos() uses (x-1),(y-1) to match the usual lattice indexing j=0..N-1.
- Returns an `AbstractMatrix{ComplexF64}` (sparse by default).
"""
function build_hamiltonian(N::Int;
    tx1=1.0, tx2=0.8, ty1=1.0, ty2=0.8,
    λ=1.0, β=1/3, ϕ=0.0, sparse::Bool=true)

    dim = N * N
    H = sparse ? spzeros(ComplexF64, dim, dim) : zeros(ComplexF64, dim, dim)

    for x in 1:N, y in 1:N
        j = idx(x, y, N)

        # On-site potential
        Vx = cos(2π*β*(x-1) + ϕ)
        Vy = cos(2π*β*(y-1) + ϕ)
        H[j, j] = 1im * 2 * λ * Vx * Vy

        if x + 1 ≤ N
            k = idx(x + 1, y, N)
            tx_eff = isodd(x) ? tx1 : tx2

            if iseven(y) 
                tx_eff *= -1
            end

            H[j, k] = tx_eff 
            H[k, j] = tx_eff 
        end

        # hopping in y direction  
        if y + 1 ≤ N
            k = idx(x, y + 1, N)
            ty_eff = isodd(y) ? ty1 : ty2

            H[j, k] = ty_eff
            H[k, j] = ty_eff
        end
    end

    return H
end
function build_hamiltonian_flux_1_4(N::Int;
    tx1=1.0, tx2=0.8, ty1=1.0, ty2=0.8,
    λ=1.0, β1=1/3, β2=1/3, ϕ=0.0, γ=0.0, sparse::Bool=true)

    dim = N * N
    H = sparse ? spzeros(ComplexF64, dim, dim) : zeros(ComplexF64, dim, dim)
 
    α = 1/2

    for x in 1:N, y in 1:N
        j = idx(x, y, N)
 
        Vx = cos(2π*β1*(x - 1) + ϕ)
        Vy = cos(2π*β2*(y - 1) + ϕ)
        H[j, j] = 1im * 2 * λ * Vx * Vy
 
        if x + 1 ≤ N
            k = idx(x + 1, y, N)
            tx_eff = mod(x,2) == 0 ? tx1 : tx2
            #tx_eff = isodd(x + y) ? tx1 : tx2

     
            H[j, k] = tx_eff - γ
            H[k, j] = tx_eff + γ
        end
 
        if y + 1 ≤ N
            k = idx(x, y + 1, N)
            ty_eff = mod(y,2) == 0 ? ty1 : ty2
            #ty_eff = isodd(x + y) ? ty1 : ty2
        
            phase = exp(2π * 1im * α * (x - 1))
    
            H[j, k] = (ty_eff - γ) * phase
            H[k, j] = (ty_eff + γ) * conj(phase)  
        end
        V0 = 0. 
   
        V_2x2 = V0 * (cos(π/4 * (x - 2.5)   )^2 + cos(π/4 * (y - 2.5))^2)  

        H[j, j] +=  V_2x2
        
    end

    return H
end

function build_hamiltonian_skin(N::Int;
    tx1=1.0, tx2=0.8, ty1=1.0, ty2=0.8,
    λ=1.0, β1=1/3, β2=1/3,ϕ=0.0, γ=0.0, sparse::Bool=true)

    dim = N * N
    H = sparse ? spzeros(ComplexF64, dim, dim) : zeros(ComplexF64, dim, dim)

    for x in 1:N, y in 1:N
        j = idx(x, y, N)

        # On-site potential
        #Vx = cos(2π*β1*(x  - 1) + ϕ)
        #Vy = cos(2π*β2*(y -1) + ϕ)
 
 
        H[j, j] = -1im   * λ * -(-1)^(mod(y,2))

        # ----- hopping in x direction -----
        if x + 1 ≤ N
            k = idx(x + 1, y, N)
            tx_eff = isodd(x) ? tx1 : tx2
 
       
            if isodd(x)
                if isodd(y)
                    H[j, k] = tx_eff - γ
                    H[k, j] = tx_eff + γ
                else
                    H[j, k] =   (tx_eff - γ)
                    H[k, j] =  (tx_eff + γ)
                end
            else
                if isodd(y)
                    H[j, k] = tx_eff  # - γ
                    H[k, j] = tx_eff  # + γ
                else
                    H[j, k] =  tx_eff  # - γ
                    H[k, j] =   tx_eff  # + γ
                end
            end
        end

        # ----- hopping in y direction -----
        if y + 1 ≤ N
            k = idx(x, y + 1, N)
            ty_eff = isodd(y) ? ty1 : ty2

        
            if isodd(y)
                H[j, k] = ty_eff -  γ
                H[k, j] = ty_eff  +  γ
            else
                H[j, k] = ty_eff  # -  γ
                H[k, j] = ty_eff   # + γ
            end
        end
    end

    return H
end

"""
    reshape_state(ψ, N) -> N×N matrix

Reshape a length N^2 eigenvector into a 2D grid ψ[x,y] with x,y=1..N.
"""
function reshape_state(ψ::AbstractVector, N::Int)
    @assert length(ψ) == N*N
    return reshape(ψ, N, N)  # columns correspond to y increasing
end

function plot_potential(N; λ=1.0, β=1/3, ϕ=0.0)
    pot = [2*λ*cos(2π*β*(x-1) + ϕ) * cos(2π*β*(y-1) + ϕ)
           for x in 1:N, y in 1:N]

    plt = heatmap(
        1:N, 1:N, pot',
        xlabel="x", ylabel="y",
        title="Imaginary onsite potential (Im V)",
        color=:viridis, aspect_ratio=1
    )
    return plt
end



function spectrum_plot(H)
    ev = eigvals(H)
    p = scatter(real.(ev), imag.(ev), ms=1,
            xlabel="Re(E)", ylabel="Im(E)",
            title="spectrum" )
    return p
end

# Example (matches your Python line):
# spectrum_plot_SSH(30, 30; r=2.9, tx=1.0, ty=1.0, γ=0.8, periodic=false)





# ------------------------------------------------------------
# 2) Hermitization of (ωI - H):
#    Build M = [ 0                (ωI - H)
#                (ω* I - H†)      0       ]
#    where H† is the conjugate transpose of H.
# ------------------------------------------------------------
function hermitize(ω::Number, H::AbstractMatrix)
    N = size(H, 1)
    @assert size(H,2) == N "H must be square"
    T = promote_type(eltype(H), ComplexF64)
    Z  = zeros(T, N, N)
    Iₙ = Matrix{T}(I, N, N)
    A  = complex(ω) * Iₙ .- H
    B  = conj(complex(ω)) * Iₙ .- adjoint(H)  # adjoint(H) = H†
    # Assemble 2N×2N block matrix
    M = [Z  A;
         B  Z]
    return M
end

# --- Tiny sanity check (uncomment to try) ---
# Nx, Ny = 6, 6
# H = build_PT_2D_SSH(Nx, Ny; r=4.0, tx=1.0, ty=1.0, γ=0.8, periodic=true)
# M = hermitize(0.0, H)
# K, J = block_shift_mats(size(H,1))
# println(size(H), "  ->  ", size(M), " ; K,J: ", size(K), " / ", size(J))

function Jackson_kernel(N::Int)    # Jackson kernel of order 0 to N-1, gn[n]=g_(n-1)
  gn=zeros(N)
  q=pi/(N+1)
  for n=0:N-1
    gn[n+1]=((N-n+1)*cos(n*q)+sin(n*q)*cot(q))/(N+1)
  end
  return gn
end

# Helper: build K = [0 0; I 0] sized to a given square matrix M (which must be 2N×2N)
function block_K_like(M::AbstractMatrix)
    L1, L2 = size(M)
    @assert L1 == L2 "M must be square"
    @assert iseven(L1) "M must be of size 2N×2N"
    N = L1 ÷ 2
    T = eltype(M)
    Z  = zeros(T, N, N)
    Iₙ = Matrix{T}(I, N, N)
    K = [Z  Z;
         Iₙ Z]
    return K
end

function block_J_like(M::AbstractMatrix)
    L1, L2 = size(M)
    @assert L1 == L2 "M must be square"
    @assert iseven(L1) "M must be of size 2N×2N"
    N = L1 ÷ 2
    T = eltype(M)
    Z  = zeros(T, N, N)
    Iₙ = Matrix{T}(I, N, N)
    J = [Z Iₙ;
         Z Z]
    return J
end

"""
    F_series_odd(M, N_pol; K=nothing)

Compute and return the list `[F₁(M), F₃(M), …, F_{2N_pol-1}(M)]` where
the sequence `F_n(M)` is defined by:
    F₀ = 0,
    F₁ = K  (K = [0 0; I 0]),
    F_{n+1} = 2*K*T_n(M) + 2*M*F_n - F_{n-1},   for n ≥ 1,

and `T_n(M)` are Chebyshev polynomials of the first kind with
    T₀ = I,  T₁ = M,  T_{n} = 2*M*T_{n-1} - T_{n-2}.

Arguments:
- `M`      : a 2N×2N square matrix.
- `N_pol`  : positive integer; number of odd-index outputs to return.
- `K`      : optional precomputed block matrix `[0 0; I 0]` (must match M’s size).

The implementation stores only the two most recent `T_n` and `F_n` to save memory.
"""
function F_series_odd(M::AbstractMatrix, N_pol::Integer; K::Union{Nothing,AbstractMatrix}=nothing)
    @assert N_pol ≥ 1 "N_pol must be ≥ 1"
    L1, L2 = size(M)
    @assert L1 == L2 "M must be square"
    @assert iseven(L1) "M must be of size 2N×2N"

    # Build K if not supplied
    Kmat = isnothing(K) ? block_K_like(M) : K
    @assert size(Kmat) == size(M) "K must have the same size as M"

    T = eltype(M)
    # Chebyshev seeds: T0 = I, T1 = M
    T_prev = Matrix{T}(I, L1, L1)  # T₀
    T_cur  = Matrix(M)             # T₁

    # F seeds: F0 = 0, F1 = K
    F_prev = zeros(T, L1, L1)      # F₀
    F_cur  = Matrix(Kmat)          # F₁

    # Collect odd-indexed F's
    results = Matrix{T}[]
    push!(results, F_cur)          # F₁

    # Current indices
    m = 1  # we have F₁; need up to F_{2N_pol-1}
    while length(results) < N_pol
        println(m)
        # Use recurrence with current n = m (since it uses T_n to get F_{n+1})
        # F_{m+1} = 2*K*T_m + 2*M*F_m - F_{m-1}
        F_next = 2 * (Kmat * T_cur) + 2 * (M * F_cur) - F_prev

        # Advance F
        F_prev, F_cur = F_cur, F_next
        m += 1

        # Advance Chebyshev T: T_{n+1} = 2*M*T_n - T_{n-1}
        T_next = 2 * (M * T_cur) - T_prev
        T_prev, T_cur = T_cur, T_next

        # Keep only odd-indexed F's
        if isodd(m)
            push!(results, F_cur)
        end
    end

    return results
end

# -------------------------
# # Example usage (uncomment to test):
# N = 4
# M = randn(ComplexF64, 2N, 2N)  # any 2N×2N matrix
# Fs = F_series_odd(M, 3)        # returns [F₁, F₃, F₅]
# @show length(Fs), size.(Fs)

"""
    jackson_weighted_sum(M, N_pol; gn=Jackson_kernel(2N_pol), K=block_K_like(M))

Compute  Σ_{n=1}^{N_pol} (2/π^2)*(-1)^(n+1)*gn[2n-1] * K * F_{2n-1}(M),
where F₀=0, F₁=K, and
    F_{n+1} = 2*K*T_n(M) + 2*M*F_n - F_{n-1},
with Chebyshev T of the first kind (standard sign):
    T₀=I, T₁=M, T_{n+1}=2*M*T_n - T_{n-1}.

Arguments:
- M::AbstractMatrix   (size 2N×2N)
- N_pol::Int          (number of odd-index terms)
- gn::AbstractVector  (Jackson kernel; length ≥ 2*N_pol)
- K::AbstractMatrix   (defaults to [0 0; I 0], size same as M)
"""
function jackson_weighted_sum(M::AbstractMatrix, N_pol::Int;
                              gn::AbstractVector = Jackson_kernel(2N_pol),
                              K::AbstractMatrix  = block_K_like(M),
                              J::AbstractMatrix  = block_J_like(M))
    @assert N_pol ≥ 1
    L1, L2 = size(M)
    @assert L1 == L2 "M must be square"
    @assert size(K) == (L1, L1) "K must be the same size as M"
    @assert size(J) == (L1, L1) "J must be the same size as M"
    @assert length(gn) ≥ 2*N_pol "gn must have length at least 2*N_pol"

    # Choose a common numeric type for accumulation
    CT = promote_type(eltype(M), eltype(K), eltype(gn), Complex)

    # Seeds: Chebyshev T and F
    T_prev = Matrix{CT}(I, L1, L1)       # T₀
    T_cur  = Matrix{CT}(M)               # T₁
    F_prev = zeros(CT, L1, L1)           # F₀
    F_cur  = Matrix{CT}(K)               # F₁

    acc = zeros(CT, L1, L1)

    # Helper for sign (-1)^(n+1) without pow
    sign_n(n) = isodd(n) ? one(CT) : -one(CT)

    # n = 1 term uses current F_cur = F₁
    coeff = (CT(2) / (π^2)) * sign_n(1) * CT(gn[2*1 - 1])
    acc .+= coeff .* (J * F_cur)

    # We will step m = 1,2,3,... updating F_m and T_m; collect odd m’s.
    m = 1
    collected = 1
    
    while collected < N_pol
        println(m)
        # F_{m+1} and T_{m+1}
        F_next = CT(2) .* (K * T_cur) .+ CT(2) .* (M * F_cur) .- F_prev
        T_next = CT(2) .* (M * T_cur) .- T_prev

        F_prev, F_cur = F_cur, F_next
        T_prev, T_cur = T_cur, T_next
        m += 1

        if isodd(m)
            n = (m + 1) ÷ 2  # maps odd m=1,3,5,... -> n=1,2,3,...
            coeff = (CT(2) / (π^2)) * sign_n(n) * CT(gn[2*n - 1])
            acc .+= coeff .* (J * F_cur)
            collected += 1
        end
    end

    return acc
end

"""
    DOS_map(H; xlims=(-4.0,4.0), ylims=(-1.0,1.0), nx=201, ny=101,
            N_pol=100, scale=10.0, stat=:real)

Compute a 2D grid of DOS values over ω = x + i*y.
For each ω, build M(ω) = hermitize(ω, H) / scale, then compute
    S(ω) = jackson_weighted_sum(M(ω), N_pol),
and store a scalar from S(ω):

- stat=:real  -> Re(tr(S))      (default)
- stat=:imag  -> Im(tr(S))
- stat=:abs   -> abs(tr(S))
- stat=:norm  -> opnorm(S)

Returns (xgrid, ygrid, Z) where Z is ny×nx (rows ↔ y, cols ↔ x).
Also returns a heatmap plot as `p` for convenience.
"""
function DOS_map(H;
                 xlims=(-4.0,4.0), ylims=(-1.0,1.0),
                 nx::Int=201, ny::Int=101,
                 N_pol::Int=100, scale::Real=10.0,
                 stat::Symbol=:real)

    @assert nx ≥ 2 && ny ≥ 2
    @assert N_pol ≥ 1
    xgrid = range(xlims[1], xlims[2], length=nx)
    ygrid = range(ylims[1], ylims[2], length=ny)

    # Precompute gn and a size-compatible K once (saves time)
    gn = Jackson_kernel(2*N_pol)
    M0 = hermitize(0.0 + 0.0im, H) / scale
    K  = block_K_like(M0)
    J = block_J_like(M0)

    # How to reduce S(ω) to a scalar
    dos_scalar = let stat=stat
        s -> begin
            val = tr(s)
            stat === :real ? real(val) :
            stat === :imag ? imag(val) :
            stat === :abs  ? abs(val)  :
            stat === :norm ? opnorm(s) :
            error("stat must be one of :real, :imag, :abs, :norm")
        end
    end

    Z = zeros(Float64, ny, nx)  # rows = y, cols = x
    for (jy, y) in enumerate(ygrid)
        for (ix, x) in enumerate(xgrid)
            ω = complex(x, y)
            M = hermitize(ω, H) / scale
            S = jackson_weighted_sum(M, N_pol; gn=gn, K=K, J=J)
            Z[jy, ix] = dos_scalar(S)
        end
    end

    # Plot
    p = heatmap(xgrid, ygrid, Z,
                xlabel="Re(ω)", ylabel="Im(ω)",
                colorbar_title = (stat==:real ? "Re tr(DOS)" :
                                  stat==:imag ? "Im tr(DOS)" :
                                  stat==:abs  ? "|tr(DOS)|"   :
                                  "‖DOS‖"),
                aspect_ratio=:equal,
                title="DOS map (N_pol=$N_pol, scale=$scale)")

    return xgrid, ygrid, Z, p
end


# ---- Example usage ----
# Assume H is already built (e.g. from build_PT_2D_SSH), and your other functions exist.
# ω grid: x ∈ [-4,4], y ∈ [-1,1], N_pol = 100, scale = 10 (as in your snippet)
# xg, yg, Z, p = DOS_map(H; xlims=(-4,4), ylims=(-1,1), nx=201, ny=101,
#                        N_pol=100, scale=10.0, stat=:real)
# display(p)
# savefig(p, "dos_map.png")


############################## For DOS examination
"""
    dos(H, E_min, E_max, delta; eta=delta, per_site=true)

Compute the density of states ρ(E) on a grid E = E_min:delta:E_max for the (possibly
non-Hermitian) Hamiltonian `H` using Lorentzian broadening:

    ρ(E) = (1/(π N)) * Σ_j Im[ 1 / (E + iη - λ_j) ]

where λ_j are eigenvalues of H. For Hermitian H this reduces to the usual sum of
Lorentzians with width η.

Arguments
- `H`        :: AbstractMatrix (real or complex)
- `E_min`    :: Real  — start of energy range
- `E_max`    :: Real  — end of energy range
- `delta`    :: Real  — energy grid step (resolution)

Keywords
- `eta`      :: Real  — Lorentzian broadening (default = `delta`)
- `per_site` :: Bool  — if true, normalize by matrix size N (default true)

Returns
- `E::Vector{Float64}`, `rho::Vector{Float64}`

Notes
- For spectra with nonzero Im(λ), choose η ≥ typical |Im(λ)| to get a smooth, positive DOS.
"""
function dos(H::AbstractMatrix, E_min::Real, E_max::Real, delta::Real; eta::Real=delta, per_site::Bool=true)
    @assert delta > 0 "delta must be positive"
    @assert E_max >= E_min "E_max must be ≥ E_min"
    η = float(eta) > 0 ? float(eta) : eps(Float64)

    # Eigenvalues (works for Hermitian & non-Hermitian)
    λ = eigvals(H)
    a = real.(λ)
    b = imag.(λ)

    E = collect(range(float(E_min), float(E_max); step=float(delta)))
    ρ = zeros(Float64, length(E))

    # Accumulate Im[1/(E + iη - λ_j)] = (η - b) / ((E - a)^2 + (η - b)^2)
    @inbounds for j in eachindex(a)
        #w = η - b[j]
        denom = @. (E - a[j])^2 + η^2
        @. ρ += η / denom
    end

    N = size(H,1)
    norm = (per_site ? N : 1)
    @. ρ = ρ / (π * norm)

    return E, ρ
end

function ldos(H::AbstractMatrix, E=0.0; η::Real=1e-2)
    dim = size(H,1)
    z = E + im*η
    G = inv(z*I - H)   # dense inverse, okay for moderate size
    ρ = zeros(Float64, dim)
    for j in 1:dim
        ρ[j] = -imag(G[j,j]) / π
    end
    return ρ
end

"""
    visualize_ldos(H, N; E=0.0, η=1e-2)

Compute and plot LDOS on an N×N lattice.
"""
function visualize_ldos(H, N; E=0.0, η=1e-2)
    ρ = ldos(H, E; η=η)
    ρ_grid = reshape(ρ, N, N)
    heatmap(
        1:N, 1:N, ρ_grid',
        xlabel="x", ylabel="y",
        title="LDOS at E = $E",
        color=:inferno, aspect_ratio=1
    )
    return ρ_grid
end
