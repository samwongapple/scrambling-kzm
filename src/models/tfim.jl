"""
    TFIM(L, J, bc, sites)

1D Transverse-Field Ising Model:
    H(t) = -f(t/τ_Q) J Σ_i σ^z_i σ^z_{i+1} - g(t/τ_Q) Σ_j σ^x_j
"""
struct TFIM <: AbstractModel
    L::Int
    J::Float64
    bc::Symbol      # :periodic or :open
    sites::Vector{<:Index}
end

function TFIM(L::Int; J::Float64=1.0, bc::Symbol=:periodic)
    s = siteinds("S=1/2", L; conserve_qns=false)
    return TFIM(L, J, bc, s)
end

sites(model::TFIM) = model.sites
num_sites(model::TFIM) = model.L

"""
    hz_terms(model::TFIM)

Returns OpSum for H_Z = Σ_i J σ^z_i σ^z_{i+1} (without the -f prefactor).
"""
function hz_terms(model::TFIM)::OpSum
    os = OpSum()
    L = model.L
    for i in 1:(L-1)
        os += model.J, "Sz", i, "Sz", i+1
    end
    if model.bc == :periodic && L > 2
        os += model.J, "Sz", L, "Sz", 1
    end
    return os
end

"""
    hx_terms(model::TFIM)

Returns OpSum for H_X = Σ_j σ^x_j (without the -g prefactor).
"""
function hx_terms(model::TFIM)::OpSum
    os = OpSum()
    for j in 1:model.L
        os += 1.0, "Sx", j
    end
    return os
end

"""
    hamiltonian_mpo(model::TFIM, f_val, g_val)

Build H(t) = -f * H_Z - g * H_X as MPO.
Uses 4*Sz*Sz convention (σ^z = 2*Sz).
"""
function hamiltonian_mpo(model::TFIM, f_val::Real, g_val::Real)::MPO
    os = OpSum()
    L = model.L

    # -f * J * Σ σ^z_i σ^z_{i+1}
    # σ^z = 2*Sz, so σ^z_i σ^z_{i+1} = 4*Sz_i*Sz_{i+1}
    for i in 1:(L-1)
        os += -4.0 * f_val * model.J, "Sz", i, "Sz", i+1
    end
    if model.bc == :periodic && L > 2
        os += -4.0 * f_val * model.J, "Sz", L, "Sz", 1
    end

    # -g * Σ σ^x_j = -g * Σ 2*Sx_j
    for j in 1:L
        os += -2.0 * g_val, "Sx", j
    end

    return MPO(os, model.sites)
end

"""
    initial_state(model::TFIM)

Returns |+>^L = ground state of -Σ σ^x_j.
Each site is |+> = (|↑> + |↓>)/√2.
"""
function initial_state(model::TFIM)::MPS
    states = ["X+" for _ in 1:model.L]
    return MPS(model.sites, states)
end
