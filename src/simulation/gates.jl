"""
    make_pf1_gates(model, f_val, g_val, dt)

Build first-order product formula (PF1) gates for one Trotter step:
    U_PF1 = exp(-i f H_Z dt) * exp(-i g H_X dt)

For TFIM: H_Z = J Σ σ^z_i σ^z_{i+1}, H_X = Σ σ^x_j
Using σ^z = 2Sz, σ^x = 2Sx.

Returns a Vector{ITensor} of gates to be applied in order.
First the ZZ two-site gates, then the X single-site gates.
"""
function make_pf1_gates(model::TFIM, f_val::Float64, g_val::Float64, dt::Float64)
    s = sites(model)
    L = num_sites(model)
    gates = ITensor[]

    # Two-site ZZ gates: exp(-i f J σ^z_i σ^z_{i+1} dt)
    # σ^z_i σ^z_{i+1} = 4 Sz_i Sz_{i+1}
    # Gate = exp(-i * 4 * f * J * Sz_i Sz_{i+1} * dt)
    for i in 1:(L-1)
        hj = 4.0 * f_val * model.J * op("Sz", s[i]) * op("Sz", s[i+1])
        push!(gates, exp(-im * dt * hj))
    end

    # Periodic boundary: bond (L, 1) — need to swap to make adjacent
    # For PBC with TEBD, we handle the (L,1) bond specially
    if model.bc == :periodic && L > 2
        # Build the ZZ gate for sites L and 1
        # We need to use swap gates to bring sites L and 1 adjacent
        # For simplicity in TEBD, we apply this as an operator on sites L and 1
        # ITensor can handle non-adjacent two-site gates via the apply function
        hj = 4.0 * f_val * model.J * op("Sz", s[L]) * op("Sz", s[1])
        push!(gates, exp(-im * dt * hj))
    end

    # Single-site X gates: exp(-i g σ^x_j dt) = exp(-i 2g Sx_j dt)
    for j in 1:L
        hj = 2.0 * g_val * op("Sx", s[j])
        push!(gates, exp(-im * dt * hj))
    end

    return gates
end
