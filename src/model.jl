"""Build the identity MPO using the convention of the original calculation."""
function identity_mpo(sites)
    nsites = length(sites)
    operators = OpSum()
    for site in 1:nsites
        operators += 1 / nsites, "Id", site
    end
    return MPO(ComplexF64, operators, sites)
end

function quantics_diagonal_mpo(f, total_bits::Int, sites; tolerance::Real)
    grid_size = 2^total_bits
    grid = range(0, grid_size - 1; length=grid_size)
    qtt, _, _ = quanticscrossinterpolate(Float64, f, grid; tolerance=tolerance)
    tensor_train = TCI.tensortrain(qtt.tci)
    diagonal_mps = MPS(tensor_train; sites=sites)
    return mps_to_diagonal_mpo(diagonal_mps, sites)
end

function novel_hop_row(total_bits, coordinate_bits, t1, t2, sites; tolerance)
    side = 2^coordinate_bits
    hopping(index) = iseven(mod(Int(index), side)) ? t1 : t2
    return quantics_diagonal_mpo(hopping, total_bits, sites; tolerance=tolerance)
end

function novel_hop_column(total_bits, coordinate_bits, t1, t2, sites; tolerance)
    side = 2^coordinate_bits
    function hopping(index)
        linear_index = Int(index)
        x0 = mod(linear_index, side)
        y0 = div(linear_index, side)
        amplitude = iseven(y0) ? t1 : t2
        sign = iseven(x0) ? -1.0 : 1.0
        return sign * amplitude
    end
    return quantics_diagonal_mpo(hopping, total_bits, sites; tolerance=tolerance)
end

function novel_hop_vertical(total_bits, coordinate_bits, t1, t2, sites; tolerance)
    side = 2^coordinate_bits
    side_squared = side^2
    function hopping(index)
        linear_index = Int(index)
        x0 = mod(linear_index, side)
        y0 = mod(div(linear_index, side), side)
        z0 = div(linear_index, side_squared)
        amplitude = iseven(z0) ? t1 : t2
        sign = iseven(x0 + y0) ? 1.0 : -1.0
        return sign * amplitude
    end
    return quantics_diagonal_mpo(hopping, total_bits, sites; tolerance=tolerance)
end

# sqrt(2) * cos(pi*m/2 + pi/4), evaluated exactly for integer m.
@inline period_four_sign(m::Int) = mod(m, 4) in (0, 3) ? 1.0 : -1.0

function build_model(config::ModelConfig; loss_field=nothing)
    validate_model_config(config)
    coordinate_bits = config.L
    side = 2^coordinate_bits
    total_sites = side^3

    xy_sites = siteinds("Qubit", 2 * coordinate_bits; conserve_qns=false)
    z_sites = siteinds("Qubit", coordinate_bits; conserve_qns=false)
    sites = vcat(z_sites, xy_sites)

    identity_xy = identity_mpo(xy_sites)
    identity_z = identity_mpo(z_sites)
    identity_all = identity_mpo(sites)

    hop_x = novel_hop_row(
        2 * coordinate_bits,
        coordinate_bits,
        config.t1,
        config.t2,
        xy_sites;
        tolerance=config.tci_tolerance,
    )
    hop_y = novel_hop_column(
        2 * coordinate_bits,
        coordinate_bits,
        config.t1,
        config.t2,
        xy_sites;
        tolerance=config.tci_tolerance,
    )

    intra_xy = intrachain_hopping(
        side, hop_x, side^2, xy_sites, identity_xy
    )
    inter_xy = interchain_hopping_square(side, hop_y, side^2, xy_sites)
    hamiltonian_xy = intra_xy + inter_xy
    embedded_xy, _ = concatenate_MPOs(
        identity_z, z_sites, hamiltonian_xy, xy_sites
    )

    hop_z = novel_hop_vertical(
        3 * coordinate_bits,
        coordinate_bits,
        config.t1,
        config.t2,
        sites;
        tolerance=config.tci_tolerance,
    )
    inter_z = interchain_hopping_square(side^2, hop_z, total_sites, sites)

    # The callback takes i0 = x + side*y + side^2*z and returns a REAL field.
    # Convert QTCI -> scalar MPS -> diagonal MPO, then multiply by i once.
    # QTCI requires a nonzero initial pivot. A known zero-amplitude tanh
    # field is represented exactly without interpolation.
    if loss_field isa TanhField && iszero(loss_field.gamma)
        loss_mpo = 0 * identity_all
    elseif loss_field !== nothing
        loss_mpo = 1im * quantics_diagonal_mpo(
            loss_field, 3 * coordinate_bits, sites; tolerance=config.tci_tolerance,
        )
    elseif iszero(config.lambda)
        loss_mpo = 0 * identity_all
    else
        function cell_pattern(index)
            i0 = Int(index)
            x0 = mod(i0, side)
            y0 = mod(div(i0, side), side)
            z0 = div(i0, side^2)
            return config.lambda * (
                period_four_sign(div(x0, 4)) *
                period_four_sign(div(y0, 4)) *
                period_four_sign(div(z0, 4))
            )
        end

        function envelope_pattern(index)
            i0 = Int(index)
            x0 = mod(i0, side)
            y0 = mod(div(i0, side), side)
            z0 = div(i0, side^2)
            return (
                period_four_sign(x0) *
                period_four_sign(y0) *
                period_four_sign(z0)
            )
        end

        cell_mpo = quantics_diagonal_mpo(
            cell_pattern,
            3 * coordinate_bits,
            sites;
            tolerance=config.tci_tolerance,
        )
        envelope_mpo = quantics_diagonal_mpo(
            envelope_pattern,
            3 * coordinate_bits,
            sites;
            tolerance=config.tci_tolerance,
        )
        loss_mpo = 1im * apply(cell_mpo, envelope_mpo)
    end

    total_hamiltonian = embedded_xy + inter_z + loss_mpo
    return (
        hamiltonian=total_hamiltonian,
        loss_mpo=loss_mpo,
        config=config,
        identity=identity_all,
        sites=sites,
        side=side,
        total_sites=total_sites,
    )
end

function hermitize_model(model, omega)
    isfinite(omega) || throw(ArgumentError("omega must be finite"))
    upper_block = omega * model.identity - model.hamiltonian
    # MPO adjoint: conjugate and exchange input/output site indices.
    lower_block = swapprime(dag(upper_block), 0 => 1)

    auxiliary_site = siteinds("Qubit", 1; conserve_qns=false)
    upper_operators = OpSum()
    upper_operators += 1, "sigma_plus", 1
    upper_mpo = MPO(upper_operators, auxiliary_site)

    lower_operators = OpSum()
    lower_operators += 1, "sigma_minus", 1
    lower_mpo = MPO(lower_operators, auxiliary_site)

    upper_hamiltonian, hermitized_sites = concatenate_MPOs(
        upper_mpo, auxiliary_site, upper_block, model.sites
    )
    lower_hamiltonian, _ = concatenate_MPOs(
        lower_mpo, auxiliary_site, lower_block, model.sites
    )
    hermitized_hamiltonian = upper_hamiltonian + lower_hamiltonian
    derivative_mpo, _ = concatenate_MPOs(
        lower_mpo, auxiliary_site, model.identity, model.sites
    )
    return hermitized_hamiltonian, hermitized_sites, derivative_mpo
end

