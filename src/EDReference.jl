module EDReference
using LinearAlgebra
export local_ed_modes, broaden_local_modes

"""
Diagonalize a small dense Hamiltonian once. Biorthogonal local residues are
R[l,n] * inv(R)[n,l], not abs2(R[l,n]). These weights may be signed/complex.
For a defective matrix a modal expansion is not reliable; the right-basis
condition number and residual are returned so the caller can inspect them.
"""
function local_ed_modes(H::AbstractMatrix, sites0::AbstractVector{<:Integer};
                        max_sites::Integer=4096)
    n = size(H,1)
    size(H,2) == n || throw(DimensionMismatch("H must be square"))
    n <= max_sites || throw(ArgumentError("dense ED restricted to max_sites=$max_sites"))
    all(s -> 0 <= s < n, sites0) || throw(ArgumentError("invalid zero-based site"))
    A = Matrix{ComplexF64}(H)
    E, right = eigen(A)
    # Solve only the inverse columns needed for the probes.
    probes = zeros(ComplexF64,n,length(sites0))
    for (p,s) in enumerate(sites0)
        probes[s+1,p] = 1
    end
    inverse_columns = right \ probes
    residues = [right[s+1,m]*inverse_columns[m,p] for m in 1:n, (p,s) in enumerate(sites0)]
    condition = cond(right)
    residual = norm(A*right-right*Diagonal(E))/max(norm(A)*norm(right),eps())
    condition > 1e10 && @warn "Ill-conditioned eigenvector basis" condition
    return (energies=E, residues=residues, sites0=collect(sites0),
            condition=condition, relative_residual=residual)
end

"""Normalized two-dimensional Gaussian broadening in the physical energy plane."""
function broaden_local_modes(modes, omega; eta::Real=0.1)
    isfinite(eta) && eta > 0 || throw(ArgumentError("eta must be positive"))
    isfinite(omega) || throw(ArgumentError("omega must be finite"))
    kernel = exp.(-abs2.(omega .- modes.energies)/(2*eta^2))/(2*pi*eta^2)
    return vec(transpose(kernel)*modes.residues)
end
end
