module RevisedTN

using LinearAlgebra, Random
using ITensors, ITensorMPS, QuanticsTCI, TCIITensorConversion, Quantics
import TensorCrossInterpolation as TCI
include(joinpath(@__DIR__, "..", "NHtk.jl"))
include("tensor_utils.jl")

export ModelConfig, build_model, hermitize_model, local_spectrum,
       density_from_moments, tanh_field, site_index, coordinates,
       maxlinkdim, KPM_ldos_bysite_streaming

"""L bits per coordinate: side=2^L, physical sites=2^(3L)."""
Base.@kwdef struct ModelConfig
    L::Int = 4
    lambda::Float64 = 2.0
    t1::Float64 = 1.0
    t2::Float64 = 1.4
    tci_tolerance::Float64 = 1e-8
end

function validate_model_config(c::ModelConfig)
    # QTCI uses exactly represented Float64 integer grid labels.
    1 <= c.L <= 17 || throw(ArgumentError("require 1 <= L <= 17"))
    all(isfinite, (c.lambda, c.t1, c.t2)) ||
        throw(ArgumentError("model coefficients must be finite"))
    isfinite(c.tci_tolerance) && c.tci_tolerance > 0 ||
        throw(ArgumentError("TCI tolerance must be finite and positive"))
    return c
end

"""Zero-based coordinates and site index; x varies fastest."""
function site_index(x::Integer, y::Integer, z::Integer, side::Integer)
    side > 0 && all(q -> 0 <= q < side, (x, y, z)) ||
        throw(ArgumentError("coordinates must be in 0:side-1"))
    return Int(x + side*y + side^2*z)
end

function coordinates(site0::Integer, side::Integer)
    side > 0 && 0 <= site0 < side^3 ||
        throw(ArgumentError("site0 must be in 0:side^3-1"))
    return (mod(site0, side), mod(site0 ÷ side, side), site0 ÷ side^2)
end

struct TanhField
    side::Int
    gamma::Float64
    x0::Float64
    R::Float64
end
(f::TanhField)(i0) = f.gamma * tanh((mod(Int(i0), f.side) - f.x0) / f.R)

"""Real callback for V=i*gamma*tanh((x-x0)/R), replacing the periodic loss."""
function tanh_field(side::Integer; gamma::Real=1.4, x0::Real=6.0, R::Real=3.0)
    side > 0 && all(isfinite, (gamma, x0, R)) && R > 0 ||
        throw(ArgumentError("require finite gamma,x0 and positive side,R"))
    return TanhField(Int(side),Float64(gamma),Float64(x0),Float64(R))
end

include("model.jl")

"""
Reweight stored odd scalar moments at a smaller Nc. Recurrence parameters,
energy and probe must be unchanged. Recompute the kernel for each Nc;
do not truncate a previously weighted sum. Returns complex densities.
"""
function density_from_moments(moments::AbstractVector, n_odd::Integer=length(moments);
                              scale::Real=12.0)
    1 <= n_odd <= length(moments) || throw(ArgumentError("not enough moments"))
    isfinite(scale) && scale > 0 || throw(ArgumentError("scale must be positive"))
    N = 2 * n_odd
    angle = pi / (N + 1)
    weighted_sum = zero(ComplexF64)
    for m in 1:n_odd
        # Production list-index convention, including the common prefactor.
        k = 2*m - 2
        kernel = (N-k+1)*cos(angle*k) + sin(angle*k)/tan(angle)
        weighted_sum += (isodd(m) ? 1 : -1) * kernel * moments[m]
    end
    scaled_density = 2 / (pi^2 * (N+1)) * weighted_sum
    physical_density = scaled_density / Float64(scale)^2
    all(isfinite, (scaled_density, physical_density)) || error("Non-finite density")
    return (; scaled_density, physical_density)
end

"""
Construct the unscaled Hermitized MPO, run the once-rescaled MPS recurrence,
and return both scaled and physical complex densities. D_H/D_herm are MPO
dimensions, distinct from the propagated MPS cap maxdim=chi. The fixed scale
must bound the Hermitized spectrum over the chosen energy window; this
example does not perform a DMRG bound calculation.
"""
function local_spectrum(model, omega; site0::Integer=0, n_odd::Integer=500,
                        scale::Real=12.0, maxdim::Integer=100, cutoff::Real=1e-8,
                        return_moments::Bool=false, observer=nothing,
                        progress_every::Integer=0)
    0 <= site0 < model.total_sites || throw(ArgumentError("invalid site0"))
    Hherm, sites, D = hermitize_model(model, omega)
    output = KPM_ldos_bysite_streaming(Hherm, n_odd, site0, scale, sites, D;
        maxdim, cutoff, return_moments, observer, progress_every)
    scaled_density = return_moments ? output.scaled_density : output
    result = (scaled_density=scaled_density,
              physical_density=scaled_density/Float64(scale)^2,
              hamiltonian_bond=maxlinkdim(model.hamiltonian),
              hermitized_bond=maxlinkdim(Hherm))
    return return_moments ? merge(result, (moments=output.moments,)) : result
end
end
