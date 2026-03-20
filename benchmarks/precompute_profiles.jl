"""
Pre-compute error profiles for adaptive schedule construction.

For each (L, tau_Q), compute:
- B_kink(t) = scrambling bound for kink density
- epsilon_n(t) = actual per-step kink density error (with fixed dt)
- epsilon_H(t) = actual per-step energy error

Usage:
    julia --project=. benchmarks/precompute_profiles.jl
"""

using Printf, Dates
using ITensorMPS

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM

const DATA_DIR = "data/profiles"

function precompute(; L::Int=20, tau_Q::Float64=10.0, J::Float64=1.0,
                     n_points::Int=60, dt_test::Float64=0.25,
                     chi_max::Int=128, n_steps_ref::Int=2000, n_substeps::Int=200)
    mkpath(DATA_DIR)
    model = TFIM(L; J=J, bc=:periodic)
    schedule = LinearQuench(tau_Q, J)

    println("  Pre-computing profiles: L=$L, τ_Q=$tau_Q, $n_points points, dt=$dt_test")
    flush(stdout)

    # Build commutator MPOs
    println("  Building MPOs...")
    flush(stdout)
    C_mpo = build_hz_hx_commutator(model)
    C_zz = build_kink_zz_mpo(model)
    comm_nC = -0.5 * build_scrambling_operator(C_zz, C_mpo; cutoff=1e-10, maxdim=200)
    println("  MPOs built: [n̂,C] bond dims = $(linkdims(comm_nC))")

    # Reference evolution with snapshots
    println("  Running reference evolution...")
    flush(stdout)
    t0 = now()

    sample_times = collect(range(dt_test, tau_Q - dt_test, length=n_points))
    psi_snapshots = Vector{MPS}(undef, n_points)

    psi = initial_state(model)
    ref_ts = uniform_schedule(tau_Q, n_steps_ref)
    snap_idx = 1
    function obs(psi_obs, t, step)
        while snap_idx <= n_points && t >= sample_times[snap_idx] - 1e-10
            psi_snapshots[snap_idx] = copy(psi_obs)
            snap_idx += 1
        end
    end
    psi = evolve_tebd!(psi, model, schedule, ref_ts; chi_max=chi_max, cutoff=1e-12, observer_fn=obs)
    elapsed = Dates.value(now() - t0) / 1000.0
    println("  Reference done ($(round(elapsed, digits=1))s), $(snap_idx-1) snapshots")
    flush(stdout)

    # Compute profiles
    B_kink = zeros(n_points)
    eps_n = zeros(n_points)
    eps_H = zeros(n_points)

    println("  Computing per-step errors and scrambling bounds...")
    flush(stdout)
    for i in 1:n_points
        t_n = sample_times[i]

        # Scrambling bound
        B_kink[i] = scrambling_bound(psi_snapshots[i], comm_nC)

        # Per-step errors
        res = compute_step_errors(psi_snapshots[i], model, schedule, t_n, dt_test;
                                  chi_max=chi_max, chi_max_ref=chi_max,
                                  cutoff=1e-12, n_substeps=n_substeps)
        eps_n[i] = res.epsilon_n
        eps_H[i] = res.epsilon_H

        if i % 15 == 0 || i == 1
            @printf("  [%2d/%d] t/τ=%.3f  B_kink=%.2e  εn=%.2e  εH=%.2e\n",
                    i, n_points, t_n/tau_Q, B_kink[i], eps_n[i], eps_H[i])
            flush(stdout)
        end
    end

    # Also compute reference final observables
    n_ref = kink_density(psi, model)
    k1_ref, k2_ref, k3_ref = kink_cumulants(psi, model; chi_max=chi_max)
    ratio_ref = k2_ref / k1_ref
    @printf("  Reference: ⟨n⟩=%.6f  κ₂/κ₁=%.4f\n", n_ref, ratio_ref)

    # Save
    outfile = joinpath(DATA_DIR, "L$(L)_tauQ$(Int(tau_Q))_profiles.h5")
    save_results(outfile, Dict{String,Any}(
        "L" => Float64(L), "tau_Q" => tau_Q, "J" => J, "dt_test" => dt_test,
        "t_sample" => sample_times,
        "B_kink" => B_kink,
        "epsilon_n" => eps_n,
        "epsilon_H" => eps_H,
        "n_ref" => n_ref,
        "kappa1_ref" => k1_ref,
        "kappa2_ref" => k2_ref,
        "kappa3_ref" => k3_ref,
        "ratio_ref" => ratio_ref,
    ))
    println("  Saved to $outfile")
    flush(stdout)

    return sample_times, B_kink, eps_n, eps_H, n_ref, ratio_ref
end

function main()
    println("=" ^ 65)
    println("  Pre-computing Error Profiles for Adaptive Schedules")
    println("=" ^ 65)

    # Primary: L=20, tau_Q=10
    t0 = now()
    precompute(L=20, tau_Q=10.0)
    elapsed = Dates.value(now() - t0) / 1000.0
    println("  L=20, τ_Q=10 total: $(round(elapsed, digits=1))s\n")

    # Additional for KZM scaling plot: tau_Q=5 and tau_Q=20
    for tau_Q in [5.0, 20.0]
        t0 = now()
        chi = tau_Q <= 10 ? 128 : 64
        n_ref_steps = tau_Q <= 10 ? 2000 : 1000
        precompute(L=20, tau_Q=tau_Q, chi_max=chi, n_steps_ref=n_ref_steps)
        elapsed = Dates.value(now() - t0) / 1000.0
        println("  L=20, τ_Q=$tau_Q total: $(round(elapsed, digits=1))s\n")
    end

    println("  All profiles computed.")
end

main()
