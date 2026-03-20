"""
Evaluate operator scrambling bounds on MPS states.
"""

"""
    scrambling_bound(psi::MPS, comm_OC::MPO)

Compute ⟨ψ| C†C |ψ⟩ where C = [O, [H_Z, H_X]] (the scrambling commutator MPO).
This gives the state-dependent scrambling bound (without dt prefactors).
"""
function scrambling_bound(psi::MPS, comm_OC::MPO)::Float64
    # ⟨ψ| C†C |ψ⟩ = inner(C_mpo, psi, C_mpo, psi)
    # inner(A::MPO, psi::MPS, B::MPO, phi::MPS) computes ⟨ψ|A†B|φ⟩
    return real(inner(comm_OC, psi, comm_OC, psi))
end

"""
    scrambling_profile(psi_snapshots, t_values, comm_nC, model, schedule)

Compute B_kink(t) and B_energy(t) at each snapshot time.

B_kink(t) = ⟨ψ(t)| [n̂, C]† [n̂, C] |ψ(t)⟩  (time-independent commutator)
B_energy(t) = ⟨ψ(t)| [H(t), C]† [H(t), C] |ψ(t)⟩  (time-dependent: H depends on f,g)

where C = [H_Z, H_X].
"""
function scrambling_profile(
    psi_snapshots::Vector{MPS},
    t_values::Vector{Float64},
    comm_nC::MPO,
    model::AbstractModel,
    schedule::AbstractQuenchSchedule;
    C_mpo::MPO = build_hz_hx_commutator(model),
    cutoff::Float64 = 1e-10,
    maxdim::Int = 200
)
    N = length(t_values)
    B_kink = zeros(N)
    B_energy = zeros(N)

    for i in 1:N
        psi = psi_snapshots[i]
        t = t_values[i]

        # B_kink: time-independent commutator
        B_kink[i] = scrambling_bound(psi, comm_nC)

        # B_energy: time-dependent — need [H(t), C] at this t
        f_val = f(schedule, t)
        g_val = g(schedule, t)
        H_t = hamiltonian_mpo(model, f_val, g_val)
        comm_HC = build_scrambling_operator(H_t, C_mpo; cutoff=cutoff, maxdim=maxdim)
        B_energy[i] = scrambling_bound(psi, comm_HC)
    end

    return B_kink, B_energy
end
