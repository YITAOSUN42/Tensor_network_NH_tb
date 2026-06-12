using ITensors
using ITensorMPS
using QuanticsTCI
import TensorCrossInterpolation as TCI
using TCIITensorConversion
using Quantics

include("2D_lattice.jl")
include("NHtk.jl")
include("extra_util.jl")

# This script mirrors the notebook demo in cubic_trial.ipynb.
# It builds the tensor-network Hamiltonian, hermitizes omega*I - H,
# and evaluates one real-space spectral point with KPM.

const L = 10
const tx1 = 1.0
const tx2 = 1.4
const beta1 = 1 / 4
const beta2 = 1 / 4
const beta3 = 1 / 4
const phi = pi / 4
const lambda_strength = 5.0
const kx = 0.0
const ky = 1.0
const kpm_order = 500
const spectrum_scale = 12.0
const site_index = 0

function alternating_row_hopping(Ltot, tx_odd, tx_even, sites)
    xvals = range(0, 2^Ltot - 1; length=2^Ltot)

    function hopping_value(x)
        x_in_row = x % 2^L
        return iseven(x_in_row) ? tx_odd : tx_even
    end

    qtt, ranks, errors = quanticscrossinterpolate(Float64, hopping_value, xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    hopping_mps = MPS(tt; sites)
    return mps_to_diagonal_mpo(hopping_mps, sites)
end

function staggered_column_hopping(Ltot, ty_odd, ty_even, sites)
    xvals = range(0, 2^Ltot - 1; length=2^Ltot)

    function hopping_value(x)
        x_in_row = x % 2^L
        row = div(x, 2^L)

        if iseven(row)
            return iseven(x_in_row) ? -ty_odd : ty_odd
        else
            return iseven(x_in_row) ? -ty_even : ty_even
        end
    end

    qtt, ranks, errors = quanticscrossinterpolate(Float64, hopping_value, xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    hopping_mps = MPS(tt; sites)
    return mps_to_diagonal_mpo(hopping_mps, sites)
end

function identity_mpo(sites, nsites)
    ops = OpSum()
    for i in 1:nsites
        ops += 1 / nsites, "Id", i
    end
    return MPO(ComplexF64, ops, sites)
end

function vertical_hopping(Ltot, tz_odd, tz_even, sites)
    xvals = range(0, 2^Ltot - 1; length=2^Ltot)

    function hopping_value(x)
        z = div(x, 2^(2L))
        x_coord = mod(x, 2^L)
        y_coord = mod(div(x, 2^L), 2^L)

        sign_xy = iseven(x_coord + y_coord) ? 1.0 : -1.0
        hopping = iseven(z) ? tz_odd : tz_even
        return sign_xy * hopping
    end

    qtt, ranks, errors = quanticscrossinterpolate(Float64, hopping_value, xvals; tolerance=1e-8)
    tt = TCI.tensortrain(qtt.tci)
    hopping_mps = MPS(tt; sites)
    return mps_to_diagonal_mpo(hopping_mps, sites)
end

const Nx = 2^L
const Ny = 2^L
const Nz = 2^L

@inline function xyz_from_linear_index(i)
    i0 = i - 1
    x = mod(i0, Nx) + 1
    y = mod(div(i0, Nx), Ny) + 1
    z = div(i0, Nx * Ny) + 1
    return x, y, z
end

function onsite_envelope(i)
    x, y, z = xyz_from_linear_index(i)

    vx = cos(2pi * beta1 * (x - 1) + phi)
    vy = cos(2pi * beta2 * (y - 1) + phi)
    vz = cos(2pi * beta3 * (z - 1) + phi)

    return 2sqrt(2) * vx * vy * vz
end

function onsite_unit_cell_value(i)
    x, y, z = xyz_from_linear_index(i)

    cx = div(x - 1, 4) + 1
    cy = div(y - 1, 4) + 1
    cz = div(z - 1, 4) + 1

    vx = cos(2pi * beta1 * (cx - 1) + phi)
    vy = cos(2pi * beta2 * (cy - 1) + phi)
    vz = cos(2pi * beta3 * (cz - 1) + phi)

    return 2sqrt(2) * lambda_strength * vx * vy * vz
end

site_xy = siteinds("Qubit", 2L; conserve_qns=false)
site_z = siteinds("Qubit", L; conserve_qns=false)
sites = vcat(site_z, site_xy)

id_xy = identity_mpo(site_xy, 2L)
id_z = identity_mpo(site_z, L)
id_all = identity_mpo(sites, 3L)

hop_x = alternating_row_hopping(2L, tx1, tx2, site_xy)
hop_y = staggered_column_hopping(2L, tx1, tx2, site_xy)
intra_mpo = intrachain_hopping(2^L, hop_x, 2^(2L), site_xy, id_xy)
inter_mpo = interchain_hopping_square(2^L, hop_y, 2^(2L), site_xy)
xy_mpo = intra_mpo + inter_mpo
xy_mpo_embedded, _ = concatenate_MPOs(id_z, site_z, xy_mpo, site_xy)

hop_z = vertical_hopping(3L, tx1, tx2, sites)
z_mpo = interchain_hopping_square(2^(2L), hop_z, 2^(3L), sites)

xvals = range(1, 2^(3L); length=2^(3L))
qtt_cell, _, _ = quanticscrossinterpolate(Float64, onsite_unit_cell_value, xvals; tolerance=1e-8)
onsite_cell_mps = MPS(TCI.tensortrain(qtt_cell.tci); sites)

qtt_env, _, _ = quanticscrossinterpolate(Float64, onsite_envelope, xvals; tolerance=1e-8)
onsite_envelope_mps = MPS(TCI.tensortrain(qtt_env.tci); sites)

loss_mpo = 1im * apply(
    mps_to_diagonal_mpo(onsite_cell_mps, sites),
    mps_to_diagonal_mpo(onsite_envelope_mps, sites),
)

tot_mpo = xy_mpo_embedded + z_mpo + loss_mpo

omega = kx + 1im * ky
upper_block = omega * id_all - tot_mpo
lower_block = dag(upper_block)

aux_site = siteinds("Qubit", 1; conserve_qns=false)
up_ops = OpSum()
up_ops += 1, "sigma_plus", 1
up_mpo = MPO(up_ops, aux_site)

down_ops = OpSum()
down_ops += 1, "sigma_minus", 1
down_mpo = MPO(down_ops, aux_site)

ham_u, hermitized_sites = concatenate_MPOs(up_mpo, aux_site, upper_block, sites)
ham_d, _ = concatenate_MPOs(down_mpo, aux_site, lower_block, sites)
hermitized_ham = ham_u + ham_d

I_ldn, _ = concatenate_MPOs(down_mpo, aux_site, id_all, sites)

moments = KPM_Tn_NH_bysite(
    hermitized_ham,
    kpm_order,
    site_index,
    spectrum_scale,
    hermitized_sites,
    I_ldn,
)

dos_point = get_energy_from_T_MPS(moments, kpm_order, site_index, hermitized_sites)
println("DOS at omega = $(omega): $(dos_point)")
