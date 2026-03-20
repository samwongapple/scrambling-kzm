"""
    TimeSchedule

Stores the time discretization for a simulation.
- `t_points`: [t_0, t_1, ..., t_N] with t_0=0, t_N=tau_Q
- `dt_values`: dt[k] = t_{k+1} - t_k, length N
"""
struct TimeSchedule
    t_points::Vector{Float64}
    dt_values::Vector{Float64}
end

"""
    uniform_schedule(tau_Q, N_steps)

Create a uniform time schedule with equal step sizes.
"""
function uniform_schedule(tau_Q::Float64, N_steps::Int)::TimeSchedule
    dt = tau_Q / N_steps
    t_points = collect(range(0.0, tau_Q, length=N_steps+1))
    dt_values = fill(dt, N_steps)
    return TimeSchedule(t_points, dt_values)
end

"""
    custom_schedule(t_points)

Create a time schedule from arbitrary time points.
"""
function custom_schedule(t_points::Vector{Float64})::TimeSchedule
    dt_values = diff(t_points)
    @assert all(dt_values .> 0) "Time points must be strictly increasing"
    return TimeSchedule(t_points, dt_values)
end

Base.length(ts::TimeSchedule) = length(ts.dt_values)
