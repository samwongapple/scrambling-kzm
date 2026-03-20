"""
    trotter_evolve!(psi, model, schedule, time_schedule; kwargs...)

Trotterized TEBD evolution using PF1 gates. Semantically identical to evolve_tebd!
but intended for coarse Trotter steps (the "simulation" side of the comparison).

At each step n:
1. Evaluate f(t_mid), g(t_mid) at midpoint t_mid = (t_n + t_{n+1})/2
2. Build PF1 gates: exp(-i f H_Z dt) exp(-i g H_X dt)
3. Apply gates to MPS with truncation

# Keyword arguments
- `chi_max::Int = 256`: Maximum bond dimension
- `cutoff::Float64 = 1e-12`: SVD cutoff
- `observer_fn = nothing`: Called as `observer_fn(psi, t, step)` after each step

Returns the evolved MPS.
"""
function trotter_evolve!(
    psi::MPS,
    model::AbstractModel,
    schedule::AbstractQuenchSchedule,
    time_schedule::TimeSchedule;
    chi_max::Int = 256,
    cutoff::Float64 = 1e-12,
    observer_fn = nothing
)::MPS
    N_steps = length(time_schedule)

    for step in 1:N_steps
        t_n = time_schedule.t_points[step]
        dt = time_schedule.dt_values[step]

        # Midpoint evaluation
        t_mid = t_n + dt / 2
        f_val = f(schedule, t_mid)
        g_val = g(schedule, t_mid)

        # PF1 gates
        gates = make_pf1_gates(model, f_val, g_val, dt)
        psi = apply(gates, psi; maxdim=chi_max, cutoff=cutoff)
        noprime!(psi)

        # Observer callback at every step
        if observer_fn !== nothing
            t_after = time_schedule.t_points[step + 1]
            observer_fn(psi, t_after, step)
        end
    end

    return psi
end
