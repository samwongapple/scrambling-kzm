"""
Build Trotter error operator [H_Z, H_X] and scrambling commutators as MPOs.
"""

"""
    commutator_mpo(A::MPO, B::MPO; cutoff=1e-10, maxdim=200)

Compute [A, B] = AB - BA as an MPO using ITensor's contract/apply.
"""
function commutator_mpo(A::MPO, B::MPO; cutoff::Float64=1e-10, maxdim::Int=200)::MPO
    # AB: contract A' * B, then relabel primes 2->1 to get standard MPO form
    AB = replaceprime(contract(A', B; cutoff=cutoff, maxdim=maxdim), 2 => 1)
    # BA
    BA = replaceprime(contract(B', A; cutoff=cutoff, maxdim=maxdim), 2 => 1)
    # [A,B] = AB - BA
    return +(AB, -1.0 * BA; cutoff=cutoff, maxdim=maxdim)
end

"""
    build_hz_hx_commutator(model::AbstractModel)

Build [H_Z, H_X] as MPO directly from OpSum.

For TFIM: [H_Z, H_X] = 2i J Σ_i (σ^y_i σ^z_{i+1} + σ^z_i σ^y_{i+1})
Using σ^y = 2Sy, σ^z = 2Sz:
    = 2i J Σ_i (4 Sy_i Sz_{i+1} + 4 Sz_i Sy_{i+1})
    = 8i J Σ_i (Sy_i Sz_{i+1} + Sz_i Sy_{i+1})
"""
function build_hz_hx_commutator(model::TFIM)::MPO
    L = num_sites(model)
    s = sites(model)
    os = OpSum()

    for i in 1:(L-1)
        os += 8.0im * model.J, "Sy", i, "Sz", i+1
        os += 8.0im * model.J, "Sz", i, "Sy", i+1
    end
    if model.bc == :periodic && L > 2
        os += 8.0im * model.J, "Sy", L, "Sz", 1
        os += 8.0im * model.J, "Sz", L, "Sy", 1
    end

    return MPO(os, s)
end

"""
    build_kink_number_mpo(model)

Build the kink number operator N_hat = Σ (1 - σ^z_i σ^z_{i+1})/2
as an MPO (without the constant N_bonds/2 term, just the -2 Σ Sz_i Sz_{i+1} part).

Actually returns the full operator including identity using:
    N_hat = N_bonds/2 * I - 2 Σ Sz_i Sz_{i+1}
which can't be represented as a simple OpSum with a constant.

Instead, returns the ZZ-sum MPO: C = 4 Σ Sz_i Sz_{i+1} = Σ σ^z_i σ^z_{i+1},
so that kink density ⟨n⟩ = 1/2 - ⟨C⟩/(2*N_bonds).

For commutators [N_hat, X] = [-C/2, X] = -[C,X]/2, we can use C directly.
"""
function build_kink_zz_mpo(model::AbstractModel)::MPO
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
    build_scrambling_operator(O_mpo::MPO, C_mpo::MPO; cutoff=1e-10, maxdim=200)

Compute [O, C] where C = [H_Z, H_X]. Returns the commutator MPO.
"""
function build_scrambling_operator(O_mpo::MPO, C_mpo::MPO;
                                    cutoff::Float64=1e-10, maxdim::Int=200)::MPO
    return commutator_mpo(O_mpo, C_mpo; cutoff=cutoff, maxdim=maxdim)
end
