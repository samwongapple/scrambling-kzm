"""
Head-to-head comparison of uniform, energy-adaptive, and kink-adaptive schedules.
"""

"""
    ComparisonResult

Results for one (scheme, N_steps) combination.
"""
struct ComparisonResult
    scheme::String
    N_steps::Int
    n_kink::Float64
    kappa_1::Float64
    kappa_2::Float64
    kappa_3::Float64
    err_n::Float64
    err_ratio::Float64
end

"""
    run_single_schedule(model, quench_schedule, time_schedule, ref_n, ref_ratio; chi_max=128)

Run Trotter evolution with a given schedule and measure KZM observables.
"""
function run_single_schedule(
    model::AbstractModel,
    quench_schedule::AbstractQuenchSchedule,
    time_schedule::TimeSchedule,
    ref_n::Float64,
    ref_ratio::Float64;
    chi_max::Int = 128,
    cutoff::Float64 = 1e-12
)::ComparisonResult
    psi = initial_state(model)
    psi = trotter_evolve!(psi, model, quench_schedule, time_schedule;
                          chi_max=chi_max, cutoff=cutoff)

    n_kink = kink_density(psi, model)
    k1, k2, k3 = kink_cumulants(psi, model; chi_max=chi_max)
    ratio = k2 / k1

    return ComparisonResult(
        "custom", length(time_schedule),
        n_kink, k1, k2, k3,
        abs(n_kink - ref_n), abs(ratio - ref_ratio)
    )
end
