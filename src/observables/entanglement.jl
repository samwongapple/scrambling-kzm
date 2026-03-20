"""
    half_chain_entropy(psi)

Compute the von Neumann entanglement entropy across the middle bond of the MPS.
"""
function half_chain_entropy(psi::MPS)::Float64
    L = length(psi)
    b = div(L, 2)
    return _bond_entropy(psi, b)
end

"""
    bond_entropies(psi)

Compute the von Neumann entanglement entropy at every bond.
Returns a vector of length L-1.
"""
function bond_entropies(psi::MPS)::Vector{Float64}
    L = length(psi)
    return [_bond_entropy(psi, b) for b in 1:(L-1)]
end

"""
Compute entanglement entropy at bond b (between sites b and b+1)
by performing SVD on the MPS at that bond.
"""
function _bond_entropy(psi::MPS, b::Int)::Float64
    # Orthogonalize the MPS to bond b
    psi_orth = orthogonalize(psi, b)

    # Collect the left indices for the SVD of psi_orth[b]
    # These are: site index of b, plus any link index from b-1
    linds = Index[]
    if b > 1
        li = linkind(psi_orth, b - 1)
        if li !== nothing
            push!(linds, li)
        end
    end
    push!(linds, siteind(psi_orth, b))

    # SVD
    _, S, _ = svd(psi_orth[b], linds;
                   lefttags="Link,l=$b", righttags="Link,r=$b")

    # Extract singular values and compute entropy
    entropy = 0.0
    for i in 1:dim(S, 1)
        sv = S[i, i]
        p = real(sv)^2
        if p > 1e-15
            entropy -= p * log(p)
        end
    end
    return entropy
end
