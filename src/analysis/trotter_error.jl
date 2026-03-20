"""
Per-step and time-resolved Trotter error analysis with full verification hierarchy.
"""

"""
    StepErrorResult

Full verification hierarchy for one Trotter step: energy and kink density errors
plus scrambling bounds.
"""
struct StepErrorResult
    t::Float64
    dt::Float64

    # Energy quantities
    E_before::Float64
    E_after_trotter::Float64
    E_after_exact::Float64
    delta_E::Float64             # |E_after_trotter - E_before|
    epsilon_H::Float64           # |E_after_exact - E_after_trotter|
    conservation_check::Float64  # |E_after_exact - E_before|
    epsilon_H_bound::Float64     # scrambling bound for energy

    # Kink density quantities
    n_before::Float64
    n_after_trotter::Float64
    n_after_exact::Float64
    epsilon_n::Float64           # |n_exact - n_trotter|
    epsilon_n_bound::Float64     # scrambling bound for kink density
end

"""
    compute_step_errors(psi_ref, model, schedule, t_n, dt; kwargs...)

Apply one PF1 Trotter step and one "exact" step (fine TEBD) to copies of `psi_ref`,
then measure all six quantities from the verification hierarchy.

Optionally computes scrambling bounds if `comm_nC` is provided.
"""
function compute_step_errors(
    psi_ref::MPS,
    model::AbstractModel,
    schedule::AbstractQuenchSchedule,
    t_n::Float64,
    dt::Float64;
    chi_max::Int = 256,
    chi_max_ref::Int = 256,
    cutoff::Float64 = 1e-12,
    n_substeps::Int = 100,
    comm_nC::Union{MPO,Nothing} = nothing,   # [n_hat, [H_Z, H_X]]
    C_mpo::Union{MPO,Nothing} = nothing      # [H_Z, H_X]
)::StepErrorResult
    # Build Hamiltonian at midpoint
    t_mid = t_n + dt / 2
    f_mid = f(schedule, t_mid)
    g_mid = g(schedule, t_mid)
    H_mpo = hamiltonian_mpo(model, f_mid, g_mid)

    # Measurements before
    E_before = energy(psi_ref, H_mpo)
    n_before = kink_density(psi_ref, model)

    # 1. One PF1 Trotter step
    psi_trotter = copy(psi_ref)
    gates = make_pf1_gates(model, f_mid, g_mid, dt)
    psi_trotter = apply(gates, psi_trotter; maxdim=chi_max, cutoff=cutoff)
    noprime!(psi_trotter)

    E_after_trotter = energy(psi_trotter, H_mpo)
    n_after_trotter = kink_density(psi_trotter, model)

    # 2. "Exact" step: fine TEBD with many substeps
    psi_exact = copy(psi_ref)
    fine_schedule = uniform_schedule(dt, n_substeps)
    fine_t_points = fine_schedule.t_points .+ t_n
    fine_ts = TimeSchedule(fine_t_points, fine_schedule.dt_values)
    psi_exact = evolve_tebd!(psi_exact, model, schedule, fine_ts;
                             chi_max=chi_max_ref, cutoff=cutoff)

    E_after_exact = energy(psi_exact, H_mpo)
    n_after_exact = kink_density(psi_exact, model)

    # 3. Derived quantities
    delta_E = abs(E_after_trotter - E_before)
    epsilon_H = abs(E_after_exact - E_after_trotter)
    conservation_check = abs(E_after_exact - E_before)
    epsilon_n = abs(n_after_exact - n_after_trotter)

    # 4. Scrambling bounds (if commutator MPOs provided)
    epsilon_H_bound = NaN
    epsilon_n_bound = NaN

    fg_dt2 = f_mid * g_mid * dt^2 / 2.0  # prefactor from M = (fg/2)[H_Z,H_X]dt^2

    if comm_nC !== nothing
        B_n = scrambling_bound(psi_ref, comm_nC)
        epsilon_n_bound = sqrt(max(0.0, B_n)) * abs(fg_dt2)
    end

    if C_mpo !== nothing
        # Build [H(t_mid), C] for the energy scrambling bound
        comm_HC = build_scrambling_operator(H_mpo, C_mpo; cutoff=1e-10, maxdim=200)
        B_H = scrambling_bound(psi_ref, comm_HC)
        epsilon_H_bound = sqrt(max(0.0, B_H)) * abs(fg_dt2)
    end

    return StepErrorResult(
        t_n, dt,
        E_before, E_after_trotter, E_after_exact,
        delta_E, epsilon_H, conservation_check, epsilon_H_bound,
        n_before, n_after_trotter, n_after_exact, epsilon_n, epsilon_n_bound
    )
end

"""
    TimeResolvedResult

Holds time-resolved comparison between exact and Trotter evolution.
"""
struct TimeResolvedResult
    t_snapshots::Vector{Float64}
    n_exact::Vector{Float64}
    n_trotter::Vector{Float64}
    E_exact::Vector{Float64}
    E_trotter::Vector{Float64}
    S_exact::Vector{Float64}
    S_trotter::Vector{Float64}
end

"""
    time_resolved_errors(model, schedule, tau_Q, N_steps; kwargs...)

Run both exact (fine TEBD) and Trotter evolution, recording observables at
snapshot times. Returns a TimeResolvedResult.
"""
function time_resolved_errors(
    model::AbstractModel,
    schedule::AbstractQuenchSchedule,
    tau_Q::Float64,
    N_steps::Int;
    n_snapshots::Int = 30,
    chi_max::Int = 64,
    chi_max_ref::Int = 128,
    n_steps_ref::Int = 2000,
    cutoff::Float64 = 1e-12
)::TimeResolvedResult

    t_snapshots = collect(range(tau_Q / n_snapshots, tau_Q, length=n_snapshots))

    n_exact    = zeros(n_snapshots)
    n_trotter  = zeros(n_snapshots)
    E_exact    = zeros(n_snapshots)
    E_trotter  = zeros(n_snapshots)
    S_exact    = zeros(n_snapshots)
    S_trotter  = zeros(n_snapshots)

    # Reference evolution
    psi_ref = initial_state(model)
    ref_schedule = uniform_schedule(tau_Q, n_steps_ref)

    snap_idx = 1
    function ref_observer(psi, t, step)
        while snap_idx <= n_snapshots && t >= t_snapshots[snap_idx] - 1e-10
            f_val = f(schedule, t_snapshots[snap_idx])
            g_val = g(schedule, t_snapshots[snap_idx])
            H = hamiltonian_mpo(model, f_val, g_val)
            n_exact[snap_idx] = kink_density(psi, model)
            E_exact[snap_idx] = energy(psi, H)
            S_exact[snap_idx] = half_chain_entropy(psi)
            snap_idx += 1
        end
    end

    psi_ref = evolve_tebd!(psi_ref, model, schedule, ref_schedule;
                           chi_max=chi_max_ref, cutoff=cutoff,
                           observer_fn=ref_observer)

    # Trotter evolution
    psi_trot = initial_state(model)
    trot_schedule = uniform_schedule(tau_Q, N_steps)

    snap_idx_t = 1
    function trot_observer(psi, t, step)
        while snap_idx_t <= n_snapshots && t >= t_snapshots[snap_idx_t] - 1e-10
            f_val = f(schedule, t_snapshots[snap_idx_t])
            g_val = g(schedule, t_snapshots[snap_idx_t])
            H = hamiltonian_mpo(model, f_val, g_val)
            n_trotter[snap_idx_t] = kink_density(psi, model)
            E_trotter[snap_idx_t] = energy(psi, H)
            S_trotter[snap_idx_t] = half_chain_entropy(psi)
            snap_idx_t += 1
        end
    end

    psi_trot = trotter_evolve!(psi_trot, model, schedule, trot_schedule;
                               chi_max=chi_max, cutoff=cutoff,
                               observer_fn=trot_observer)

    return TimeResolvedResult(t_snapshots, n_exact, n_trotter,
                              E_exact, E_trotter, S_exact, S_trotter)
end
