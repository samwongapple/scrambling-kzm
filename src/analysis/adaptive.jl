"""
Adaptive Trotterization schedule optimization.

Three methods:
A) Energy-adaptive (Zhao et al.) — iterative, based on direct ΔE measurement
B) Kink-bound-adaptive — from scrambling bound profile B_kink(t)
C) Kink-direct-adaptive — from actual per-step ε_n(t) profile
"""

"""
    _discretize_density(rho_values, t_profile, tau_Q, N_steps)

Given a continuous step density ρ(t) sampled at `t_profile`, produce a TimeSchedule
with `N_steps` steps where step boundaries are placed at cumulative integral crossings.
"""
function _discretize_density(rho_values::Vector{Float64}, t_profile::Vector{Float64},
                             tau_Q::Float64, N_steps::Int)::TimeSchedule
    # Ensure positivity
    rho = max.(rho_values, 1e-20)

    # Compute cumulative integral via trapezoidal rule
    N_prof = length(t_profile)
    cum = zeros(N_prof)
    for i in 2:N_prof
        dt_p = t_profile[i] - t_profile[i-1]
        cum[i] = cum[i-1] + 0.5 * (rho[i] + rho[i-1]) * dt_p
    end
    total = cum[end]

    # Place step boundaries where cum crosses k/N_steps * total
    t_points = [0.0]
    target_idx = 1
    for k in 1:N_steps-1
        target = k / N_steps * total
        # Find the interval where cum crosses target
        while target_idx < N_prof && cum[target_idx+1] < target
            target_idx += 1
        end
        if target_idx >= N_prof
            push!(t_points, tau_Q)
        else
            # Linear interpolation
            frac = (target - cum[target_idx]) / max(cum[target_idx+1] - cum[target_idx], 1e-20)
            t_new = t_profile[target_idx] + frac * (t_profile[min(target_idx+1, N_prof)] - t_profile[target_idx])
            push!(t_points, clamp(t_new, t_points[end] + 1e-14, tau_Q - 1e-14))
        end
    end
    push!(t_points, tau_Q)

    return custom_schedule(t_points)
end

"""
    energy_adaptive_schedule(tau_Q, N_steps, model, quench_schedule; kwargs...)

Method A: Energy-adaptive (Zhao et al. style). Iteratively refines step sizes
by measuring ΔE_n at each Trotter step.

Returns (schedule, diagnostics) where diagnostics contains ΔE profiles.
"""
function energy_adaptive_schedule(
    tau_Q::Float64,
    N_steps::Int,
    model::AbstractModel,
    quench_schedule::AbstractQuenchSchedule;
    chi_max::Int = 128,
    cutoff::Float64 = 1e-12,
    n_iterations::Int = 3
)::Tuple{TimeSchedule, Dict}
    current_schedule = uniform_schedule(tau_Q, N_steps)
    diagnostics = Dict{String,Any}()
    all_delta_E = Vector{Vector{Float64}}()

    for iter in 1:n_iterations
        # Run Trotter evolution, measuring ΔE at each step
        delta_E_vals = zeros(N_steps)
        t_mids = zeros(N_steps)

        psi = initial_state(model)
        for step in 1:N_steps
            t_n = current_schedule.t_points[step]
            dt = current_schedule.dt_values[step]
            t_mid = t_n + dt / 2
            t_mids[step] = t_mid

            f_val = f(quench_schedule, t_mid)
            g_val = g(quench_schedule, t_mid)
            H_mpo = hamiltonian_mpo(model, f_val, g_val)

            E_before = energy(psi, H_mpo)

            gates = make_pf1_gates(model, f_val, g_val, dt)
            psi = apply(gates, psi; maxdim=chi_max, cutoff=cutoff)
            noprime!(psi)

            E_after = energy(psi, H_mpo)
            delta_E_vals[step] = abs(E_after - E_before)
        end

        push!(all_delta_E, copy(delta_E_vals))

        # Redistribute: new dt ∝ 1/|ΔE|^{1/2}
        # Since per-step error ~ dt^2, equalizing error requires dt ∝ 1/sensitivity^{1/2}
        sensitivity = max.(delta_E_vals ./ (current_schedule.dt_values .^ 2), 1e-20)
        raw_dt = 1.0 ./ sensitivity .^ 0.5
        raw_dt = raw_dt ./ sum(raw_dt) .* tau_Q

        # Build new schedule from these dt values
        t_pts = [0.0]
        for k in 1:N_steps
            push!(t_pts, t_pts[end] + raw_dt[k])
        end
        # Fix endpoint to exactly tau_Q
        t_pts[end] = tau_Q
        current_schedule = custom_schedule(t_pts)
    end

    diagnostics["delta_E_iterations"] = all_delta_E
    return current_schedule, diagnostics
end

"""
    kink_bound_adaptive_schedule(tau_Q, N_steps, B_kink_profile, t_profile, quench_schedule)

Method B: Kink-adaptive via scrambling bound.
Step density ρ(t) ∝ B_kink(t)^{1/4} * (f(t)g(t))^{1/2}.
"""
function kink_bound_adaptive_schedule(
    tau_Q::Float64,
    N_steps::Int,
    B_kink_profile::Vector{Float64},
    t_profile::Vector{Float64},
    quench_schedule::AbstractQuenchSchedule
)::TimeSchedule
    fg = [f(quench_schedule, t) * g(quench_schedule, t) for t in t_profile]
    rho = max.(B_kink_profile, 0.0) .^ 0.25 .* max.(fg, 0.0) .^ 0.5
    return _discretize_density(rho, t_profile, tau_Q, N_steps)
end

"""
    kink_direct_adaptive_schedule(tau_Q, N_steps, epsilon_n_profile, t_profile, quench_schedule)

Method C: Kink-adaptive via direct per-step error profile.
Step density ρ(t) ∝ ε_n(t)^{1/2} (to equalize per-step error which scales as dt^2).
"""
function kink_direct_adaptive_schedule(
    tau_Q::Float64,
    N_steps::Int,
    epsilon_n_profile::Vector{Float64},
    t_profile::Vector{Float64},
    quench_schedule::AbstractQuenchSchedule
)::TimeSchedule
    rho = max.(epsilon_n_profile, 0.0) .^ 0.5
    return _discretize_density(rho, t_profile, tau_Q, N_steps)
end
