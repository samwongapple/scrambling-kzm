"""
    evolve_tebd!(psi, model, schedule, time_schedule; kwargs...)

Run TEBD evolution of MPS `psi` under `model` with quench `schedule`
using the time discretization in `time_schedule`.

# Keyword arguments
- `chi_max::Int = 256`: Maximum bond dimension for SVD truncation
- `cutoff::Float64 = 1e-12`: SVD cutoff
- `observer_fn = nothing`: Callback `observer_fn(psi, t, step)` called at each step
- `observer_times::Vector{Float64} = Float64[]`: If non-empty, only call observer at these times (nearest step)

Returns the evolved MPS.
"""
function evolve_tebd!(
    psi::MPS,
    model::AbstractModel,
    schedule::AbstractQuenchSchedule,
    time_schedule::TimeSchedule;
    chi_max::Int = 256,
    cutoff::Float64 = 1e-12,
    observer_fn = nothing,
    observer_times::Vector{Float64} = Float64[]
)::MPS
    N_steps = length(time_schedule)

    # Precompute which steps to observe
    observe_at_step = Set{Int}()
    if observer_fn !== nothing
        if isempty(observer_times)
            # Observe at every step
            observe_at_step = Set(1:N_steps)
        else
            # Find nearest step for each observer time
            for t_obs in observer_times
                # Find the step whose midpoint is closest
                best_step = 1
                best_dist = Inf
                for k in 1:N_steps
                    t_mid = (time_schedule.t_points[k] + time_schedule.t_points[k+1]) / 2
                    d = abs(t_mid - t_obs)
                    if d < best_dist
                        best_dist = d
                        best_step = k
                    end
                end
                push!(observe_at_step, best_step)
            end
        end
    end

    for step in 1:N_steps
        t_n = time_schedule.t_points[step]
        dt = time_schedule.dt_values[step]

        # Evaluate schedule at midpoint for better accuracy
        t_mid = t_n + dt / 2
        f_val = f(schedule, t_mid)
        g_val = g(schedule, t_mid)

        # Build and apply PF1 gates
        gates = make_pf1_gates(model, f_val, g_val, dt)
        psi = apply(gates, psi; maxdim=chi_max, cutoff=cutoff)
        noprime!(psi)

        # Observer callback
        if step in observe_at_step
            t_after = time_schedule.t_points[step + 1]
            observer_fn(psi, t_after, step)
        end
    end

    return psi
end
