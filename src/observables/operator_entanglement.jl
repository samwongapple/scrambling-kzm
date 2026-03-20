"""
Operator-induced entanglement: entropy of O|ψ⟩/||O|ψ⟩||.
"""

"""
    operator_induced_entropy(psi::MPS, O_mpo::MPO; maxdim=256, cutoff=1e-12)

Apply operator O to state |ψ⟩, normalize the result, and compute
half-chain entanglement entropy.

Returns the von Neumann entropy of |φ⟩ = O|ψ⟩/||O|ψ⟩|| across the middle cut.
"""
function operator_induced_entropy(psi::MPS, O_mpo::MPO;
                                   maxdim::Int=256, cutoff::Float64=1e-12)::Float64
    # Apply O to psi
    phi = apply(O_mpo, psi; maxdim=maxdim, cutoff=cutoff)
    noprime!(phi)

    # Normalize by scaling a single tensor (not broadcasting over all tensors)
    nrm = norm(phi)
    if nrm < 1e-15
        return 0.0
    end
    phi[1] = phi[1] / nrm

    # Compute half-chain entropy
    return half_chain_entropy(phi)
end
