module SparseKPMBenchmark

using LinearAlgebra
using SparseArrays
using Printf

export build_sparse_hoti,
       lattice_index,
       lattice_coordinates,
       nhkpm_local_sparse,
       nhkpm_grid_sparse,
       write_grid_csv

"""
    lattice_index(x, y, z, Lx, Ly, Lz)

One-based matrix index for the real-space convention used by
`cube_tensor_script.jl`: `x` is the fastest-running coordinate.
"""
@inline function lattice_index(x::Int, y::Int, z::Int,
                               Lx::Int, Ly::Int, Lz::Int)
    1 <= x <= Lx || throw(BoundsError(1:Lx, x))
    1 <= y <= Ly || throw(BoundsError(1:Ly, y))
    1 <= z <= Lz || throw(BoundsError(1:Lz, z))
    return ((z - 1) * Ly + (y - 1)) * Lx + x
end

"""
    lattice_coordinates(site0, Lx, Ly, Lz)

Convert the zero-based `site_index` used by the tensor-network script to
one-based `(x, y, z)` coordinates.
"""
@inline function lattice_coordinates(site0::Int,
                                     Lx::Int, Ly::Int, Lz::Int)
    nsites = Lx * Ly * Lz
    0 <= site0 < nsites || throw(BoundsError(0:(nsites - 1), site0))
    x = mod(site0, Lx) + 1
    y = mod(div(site0, Lx), Ly) + 1
    z = div(site0, Lx * Ly) + 1
    return x, y, z
end

# sqrt(2) * cos(pi*m/2 + pi/4), evaluated exactly for integer m.
@inline s4(m::Int) = (mod(m, 4) == 0 || mod(m, 4) == 3) ? 1.0 : -1.0

"""
    build_sparse_hoti(Lx, Ly, Lz; kwargs...)

Construct the sparse non-Hermitian 3D BBH-type Hamiltonian represented by
`cube_tensor_script.jl`. Open boundary conditions are used in all directions.

The hopping signs are

    s_x = 1,  s_y = (-1)^x,  s_z = (-1)^(x+y),

for one-based coordinates. The onsite term is

    i * lambda * prod_mu s4(r_mu-1) * s4(fld(r_mu-1, 4)).

This is the product of `onsite_envelope` and `onsite_unit_cell_value` in
the tensor-network construction.

With `loss_field=f`, replace this onsite term by `i*f(site0)` using a real
callback on zero-based flattened indices. This matches `build_model` in
`RevisedTN`, including the tanh domain-wall example.
"""
function build_sparse_hoti(Lx::Int, Ly::Int, Lz::Int;
                           tx1::Real=1.0, tx2::Real=1.4,
                           ty1::Real=tx1, ty2::Real=tx2,
                           tz1::Real=tx1, tz2::Real=tx2,
                           lambda::Real=5.0,
                           loss_field=nothing)
    Lx > 0 && Ly > 0 && Lz > 0 ||
        throw(ArgumentError("lattice dimensions must be positive"))

    nsites = Lx * Ly * Lz
    # One diagonal entry plus two entries for each of at most three bonds.
    capacity = 7 * nsites
    rows = Int[]
    cols = Int[]
    vals = ComplexF64[]
    sizehint!(rows, capacity)
    sizehint!(cols, capacity)
    sizehint!(vals, capacity)

    @inline function add_entry!(i::Int, j::Int, value::Number)
        push!(rows, i)
        push!(cols, j)
        push!(vals, ComplexF64(value))
        return nothing
    end

    @inline function add_hopping!(i::Int, j::Int, value::Real)
        add_entry!(i, j, value)
        add_entry!(j, i, value)
        return nothing
    end

    for z in 1:Lz, y in 1:Ly, x in 1:Lx
        i = lattice_index(x, y, z, Lx, Ly, Lz)

        nx, ny, nz = x - 1, y - 1, z - 1
        bx, by, bz = fld(nx, 4), fld(ny, 4), fld(nz, 4)
        modulation = (s4(nx) * s4(ny) * s4(nz) *
                      s4(bx) * s4(by) * s4(bz))
        # Optional callback uses the same ZERO-BASED index as RevisedTN.
        # It replaces the periodic field and must return a finite real value.
        field = loss_field === nothing ? lambda * modulation : loss_field(i - 1)
        field isa Real && isfinite(field) ||
            throw(ArgumentError("loss_field must return a finite real value"))
        add_entry!(i, i, 1im * field)

        if x < Lx
            j = lattice_index(x + 1, y, z, Lx, Ly, Lz)
            tx = isodd(x) ? tx1 : tx2
            add_hopping!(i, j, tx)
        end

        if y < Ly
            j = lattice_index(x, y + 1, z, Lx, Ly, Lz)
            ty = isodd(y) ? ty1 : ty2
            sy = isodd(x) ? -1.0 : 1.0
            add_hopping!(i, j, sy * ty)
        end

        if z < Lz
            j = lattice_index(x, y, z + 1, Lx, Ly, Lz)
            tz = isodd(z) ? tz1 : tz2
            sz = iseven(x + y) ? 1.0 : -1.0
            add_hopping!(i, j, sz * tz)
        end
    end

    return sparse(rows, cols, vals, nsites, nsites)
end

"""
Apply the rescaled Hermitized matrix

    Hbar = [0  A; A'  0] / scale,   A = omega*I - H,

without explicitly assembling the doubled sparse matrix.
"""
function mul_hbar!(out::Vector{ComplexF64},
                   A::SparseMatrixCSC{ComplexF64,Int},
                   vector::Vector{ComplexF64}, inv_scale::Float64)
    nsites = size(A, 1)
    length(vector) == 2 * nsites || throw(DimensionMismatch("invalid vector"))
    length(out) == 2 * nsites || throw(DimensionMismatch("invalid output"))

    upper_out = @view out[1:nsites]
    lower_out = @view out[(nsites + 1):(2 * nsites)]
    upper_in = @view vector[1:nsites]
    lower_in = @view vector[(nsites + 1):(2 * nsites)]

    mul!(upper_out, A, lower_in)
    mul!(lower_out, adjoint(A), upper_in)
    upper_out .*= inv_scale
    lower_out .*= inv_scale
    return out
end

"""
Return the unnormalized Jackson numerators used in `NHtk.jl`.
The common division by `total_orders + 1` is applied in the final NHKPM
prefactor, exactly as in the tensor-network routine.
"""
function jackson_numerators(total_orders::Int)
    total_orders > 0 || throw(ArgumentError("total_orders must be positive"))
    q = pi / (total_orders + 1)
    return [
        (total_orders - n + 1) * cos(q * n) + sin(q * n) / tan(q)
        for n in 0:(total_orders - 1)
    ]
end

"""
    nhkpm_local_sparse(H, omega; site0=0, n_odd=500, scale=12.0,
                       return_moments=false)

Evaluate the site-resolved NHKPM spectral function with sparse matrix-vector
products. `site0` is zero based to match `KPM_Tn_NH_bysite`.

The recurrence is identical to the tensor-network implementation:

    t_0 = |R>,                 t_1 = Hbar |R>
    d_0 = 0,                   d_1 = D_zeta |R> = |L>
    t_(n+1) = 2 Hbar t_n - t_(n-1)
    d_(n+1) = 2 D_zeta t_n + 2 Hbar d_n - d_(n-1).

There are `n_odd` retained odd derivative moments, through order
`2*n_odd - 1`. `scaled_density` is the quantity directly comparable with
`get_energy_from_T_MPS`. `physical_density = scaled_density / scale^2`
converts the density per unit area back to the original complex-energy plane.
"""
function nhkpm_local_sparse(H::SparseMatrixCSC,
                            omega::Number;
                            site0::Int=0,
                            n_odd::Int=500,
                            scale::Real=12.0,
                            return_moments::Bool=false)
    size(H, 1) == size(H, 2) || throw(DimensionMismatch("H must be square"))
    n_odd > 0 || throw(ArgumentError("n_odd must be positive"))
    isfinite(scale) && scale > 0 ||
        throw(ArgumentError("scale must be finite and positive"))
    isfinite(omega) || throw(ArgumentError("omega must be finite"))

    nsites = size(H, 1)
    0 <= site0 < nsites || throw(BoundsError(0:(nsites - 1), site0))
    site = site0 + 1
    total_orders = 2 * n_odd
    max_polynomial_order = total_orders - 1
    inv_scale = 1.0 / Float64(scale)

    Hc = convert(SparseMatrixCSC{ComplexF64,Int}, H)
    A = spdiagm(0 => fill(ComplexF64(omega), nsites)) - Hc
    kernel = jackson_numerators(total_orders)

    t_prev = zeros(ComplexF64, 2 * nsites)
    t_curr = similar(t_prev)
    t_next = similar(t_prev)
    # These are initial states, not scratch buffers: d_0 = 0 and
    # d_1 = |L_l>. similar(...) would leave all other entries uninitialized.
    d_prev = zeros(ComplexF64, 2 * nsites)
    d_curr = zeros(ComplexF64, 2 * nsites)
    d_next = similar(t_prev)

    # |R_l> occupies the upper auxiliary sector. D_zeta |R_l> = |L_l>.
    t_prev[site] = 1.0
    mul_hbar!(t_curr, A, t_prev, inv_scale)
    d_curr[nsites + site] = 1.0

    moments = return_moments ? Vector{ComplexF64}(undef, n_odd) : nothing
    first_moment = d_curr[nsites + site]
    return_moments && (moments[1] = first_moment)
    weighted_sum = kernel[1] * first_moment

    for next_order in 2:max_polynomial_order
        # The right-hand sides must use t_n and d_n before rotating buffers.
        mul_hbar!(t_next, A, t_curr, inv_scale)
        @. t_next = 2 * t_next - t_prev

        mul_hbar!(d_next, A, d_curr, inv_scale)
        @. d_next = 2 * d_next - d_prev
        @views d_next[(nsites + 1):(2 * nsites)] .+= 2 .* t_curr[1:nsites]

        t_prev, t_curr, t_next = t_curr, t_next, t_prev
        d_prev, d_curr, d_next = d_curr, d_next, d_prev

        if isodd(next_order)
            moment_number = (next_order + 1) ÷ 2
            moment = d_curr[nsites + site]
            return_moments && (moments[moment_number] = moment)

            # Julia array entry `next_order` represents kernel order
            # `next_order - 1`, matching NHtk.jl's jackson_kernel[l-1].
            sign = isodd(moment_number) ? 1.0 : -1.0
            weighted_sum += sign * kernel[next_order] * moment
        end
    end

    scaled_density = (2 / (pi^2 * (total_orders + 1))) * weighted_sum
    physical_density = scaled_density / Float64(scale)^2
    isfinite(scaled_density) && isfinite(physical_density) ||
        error("Non-finite NHKPM density at omega=$omega, site0=$site0, " *
              "n_odd=$n_odd, scale=$scale; check the recurrence and ensure " *
              "the Hermitized spectrum lies strictly inside (-1, 1).")
    result = (
        scaled_density=scaled_density,
        physical_density=physical_density,
        site0=site0,
        n_odd=n_odd,
        max_polynomial_order=max_polynomial_order,
        scale=Float64(scale),
    )
    return return_moments ? merge(result, (moments=moments,)) : result
end

"""
    nhkpm_grid_sparse(H, real_grid, imag_grid; kwargs...)

Evaluate `scaled_density` on a complex-energy grid. Rows correspond to
`imag_grid` and columns to `real_grid`, following `get_spectrum` in `NHtk.jl`.
"""
function nhkpm_grid_sparse(H::SparseMatrixCSC,
                           real_grid::AbstractVector,
                           imag_grid::AbstractVector;
                           site0::Int=0,
                           n_odd::Int=500,
                           scale::Real=12.0,
                           verbose::Bool=true)
    values = Matrix{ComplexF64}(undef, length(imag_grid), length(real_grid))
    total = length(real_grid) * length(imag_grid)
    completed = 0

    for (ix, re_omega) in pairs(real_grid)
        for (iy, im_omega) in pairs(imag_grid)
            omega = ComplexF64(re_omega, im_omega)
            result = nhkpm_local_sparse(
                H, omega; site0=site0, n_odd=n_odd, scale=scale
            )
            values[iy, ix] = result.scaled_density
            completed += 1
            verbose && @printf("[%d/%d] omega = %.8g %+.8gi, F = %.12g %+.12gi\n",
                               completed, total, real(omega), imag(omega),
                               real(values[iy, ix]), imag(values[iy, ix]))
        end
    end
    return values
end

"""Write a complex-energy grid in long CSV form."""
function write_grid_csv(path::AbstractString,
                        real_grid::AbstractVector,
                        imag_grid::AbstractVector,
                        values::AbstractMatrix)
    size(values) == (length(imag_grid), length(real_grid)) ||
        throw(DimensionMismatch("grid dimensions do not match values"))
    open(path, "w") do io
        println(io, "omega_re,omega_im,scaled_density_re,scaled_density_im")
        for (ix, re_omega) in pairs(real_grid), (iy, im_omega) in pairs(imag_grid)
            value = values[iy, ix]
            println(io, join((re_omega, im_omega, real(value), imag(value)), ","))
        end
    end
    return path
end

end # module
