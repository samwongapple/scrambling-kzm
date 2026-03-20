"""
    energy(psi, H_mpo)

Compute ⟨ψ|H|ψ⟩.
"""
function energy(psi::MPS, H_mpo::MPO)::Float64
    return real(inner(psi', H_mpo, psi))
end

"""
    energy_variance(psi, H_mpo)

Compute ΔE² = ⟨H²⟩ - ⟨H⟩².
"""
function energy_variance(psi::MPS, H_mpo::MPO)::Float64
    E = energy(psi, H_mpo)
    E2 = real(inner(H_mpo, psi, H_mpo, psi))
    return E2 - E^2
end
