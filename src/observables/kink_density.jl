"""
    kink_density(psi, model)

Compute mean kink density: ⟨n⟩ = (1/2N_bonds) Σ_i (1 - ⟨σ^z_i σ^z_{i+1}⟩)
where σ^z = 2Sz, so σ^z_i σ^z_{i+1} = 4⟨Sz_i Sz_{i+1}⟩.
"""
function kink_density(psi::MPS, model::AbstractModel)::Float64
    per_bond = kink_density_per_bond(psi, model)
    return mean(per_bond)
end

"""
    kink_density_per_bond(psi, model)

Returns K_i = (1 - ⟨σ^z_i σ^z_{i+1}⟩)/2 for each bond.
"""
function kink_density_per_bond(psi::MPS, model::AbstractModel)::Vector{Float64}
    L = num_sites(model)
    bonds = Tuple{Int,Int}[]

    for i in 1:(L-1)
        push!(bonds, (i, i+1))
    end
    if hasproperty(model, :bc) && model.bc == :periodic && L > 2
        push!(bonds, (L, 1))
    end

    K = Float64[]
    for (i, j) in bonds
        szz = _expect_zz(psi, i, j)
        push!(K, (1.0 - szz) / 2.0)
    end

    return K
end

"""
Compute ⟨σ^z_i σ^z_j⟩ for sites i,j using correlation_matrix.
"""
function _expect_zz(psi::MPS, i::Int, j::Int)::Float64
    if abs(i - j) == 1
        lo = min(i, j)
        hi = max(i, j)
        zz = correlation_matrix(psi, "Sz", "Sz"; sites=lo:hi)
        return 4.0 * real(zz[1, 2])
    end
    # For PBC boundary bond (L,1) or general case
    lo = min(i, j)
    hi = max(i, j)
    zz = correlation_matrix(psi, "Sz", "Sz"; sites=lo:hi)
    return 4.0 * real(zz[1, end])
end

"""
    _zz_sum_mpo(model)

Build MPO for C = Σ_bonds σ^z_i σ^z_{i+1} = 4 Σ_bonds Sz_i Sz_{i+1}.
N_hat = N_bonds/2 - C/2.
"""
function _zz_sum_mpo(model::AbstractModel)
    L = num_sites(model)
    s = sites(model)
    os = OpSum()

    for i in 1:(L-1)
        os += 4.0, "Sz", i, "Sz", i+1
    end
    if hasproperty(model, :bc) && model.bc == :periodic && L > 2
        os += 4.0, "Sz", L, "Sz", 1
    end

    return MPO(os, s)
end

"""
    _num_bonds(model)

Return the number of bonds in the model.
"""
function _num_bonds(model::AbstractModel)::Int
    L = num_sites(model)
    if hasproperty(model, :bc) && model.bc == :periodic && L > 2
        return L
    end
    return L - 1
end

"""
    kink_cumulants(psi, model; max_order=3, chi_max=256, cutoff=1e-12)

Compute cumulants of the total kink number operator N = Σ_i K_i
using the MPO for C = Σ σ^z_i σ^z_{i+1}, since N = N_bonds/2 - C/2.

- κ_1 = ⟨N⟩ = N_bonds/2 - ⟨C⟩/2
- κ_2 = Var(N) = Var(C)/4 = (⟨C²⟩ - ⟨C⟩²)/4
- κ_3 = -⟨(C-⟨C⟩)³⟩/8 = -(⟨C³⟩ - 3⟨C²⟩⟨C⟩ + 2⟨C⟩³)/8

Returns (κ_1, κ_2, κ_3) as total kink number cumulants.
"""
function kink_cumulants(psi::MPS, model::AbstractModel;
                        max_order::Int=3, chi_max::Int=256, cutoff::Float64=1e-12)
    Nb = _num_bonds(model)
    C_mpo = _zz_sum_mpo(model)

    # ⟨C⟩
    C1 = real(inner(psi', C_mpo, psi))

    # κ_1 = N_bonds/2 - ⟨C⟩/2
    kappa_1 = Nb / 2.0 - C1 / 2.0

    # ⟨C²⟩ = ⟨ψ|C†C|ψ⟩
    C2 = real(inner(C_mpo, psi, C_mpo, psi))

    # κ_2 = Var(C)/4
    kappa_2 = (C2 - C1^2) / 4.0

    kappa_3 = 0.0
    if max_order >= 3
        # ⟨C³⟩ = ⟨Cψ|C|Cψ⟩ where Cψ = C|ψ⟩
        Cpsi = apply(C_mpo, psi; maxdim=chi_max, cutoff=cutoff)
        noprime!(Cpsi)
        C3 = real(inner(Cpsi', C_mpo, Cpsi))

        # κ_3 = -⟨(C-⟨C⟩)³⟩/8 = -(⟨C³⟩ - 3⟨C²⟩⟨C⟩ + 2⟨C⟩³)/8
        kappa_3 = -(C3 - 3.0 * C2 * C1 + 2.0 * C1^3) / 8.0
    end

    return (kappa_1, kappa_2, kappa_3)
end
