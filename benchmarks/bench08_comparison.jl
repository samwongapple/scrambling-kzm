"""
Benchmark 8 — Head-to-Head Comparison (incorporates Bench 6 & 7)

Compare four Trotter schedules:
  1. Uniform
  2. Energy-adaptive (Zhao et al.)
  3. Kink-bound-adaptive (scrambling bound)
  4. Kink-direct-adaptive (actual ε_n profile)

Usage:
    julia --project=. benchmarks/bench08_comparison.jl
"""

using Printf, Dates, Statistics
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM
using Plots, LaTeXStrings

const L             = 20
const J             = 1.0
const BC            = :periodic
const TAU_Q_PRIMARY = 10.0
const TAU_Q_SWEEP   = [5.0, 10.0, 20.0]
const N_STEPS_LIST  = [10, 20, 50, 100, 200]
const CHI_MAX       = 128
const CUTOFF        = 1e-12
const DATA_DIR      = "data/bench08"
const FIG_DIR       = "figures/bench08"
const PROFILE_DIR   = "data/profiles"

# ═══════════════════════════════════════════════════════════════════
# Load pre-computed profiles
# ═══════════════════════════════════════════════════════════════════
function load_profiles(tau_Q)
    path = joinpath(PROFILE_DIR, "L$(L)_tauQ$(Int(tau_Q))_profiles.h5")
    if !isfile(path)
        error("Profile not found: $path — run precompute_profiles.jl first")
    end
    d = load_results(path)
    return d
end

# ═══════════════════════════════════════════════════════════════════
# Run all four schemes for one (tau_Q, N_steps)
# ═══════════════════════════════════════════════════════════════════
function run_four_schemes(tau_Q, N_steps, profiles)
    model = TFIM(L; J=J, bc=BC)
    schedule = LinearQuench(tau_Q, J)

    t_prof = profiles["t_sample"]
    B_kink = profiles["B_kink"]
    eps_n_prof = profiles["epsilon_n"]
    n_ref = profiles["n_ref"]
    ratio_ref = profiles["ratio_ref"]

    results = Dict{String, ComparisonResult}()

    # 1. Uniform
    ts_uni = uniform_schedule(tau_Q, N_steps)
    results["Uniform"] = run_single_schedule(model, schedule, ts_uni, n_ref, ratio_ref;
                                              chi_max=CHI_MAX)

    # 2. Energy-adaptive
    ts_E, _ = energy_adaptive_schedule(tau_Q, N_steps, model, schedule;
                                        chi_max=CHI_MAX, n_iterations=3)
    r_E = run_single_schedule(model, schedule, ts_E, n_ref, ratio_ref; chi_max=CHI_MAX)
    results["Energy"] = ComparisonResult("Energy", r_E.N_steps, r_E.n_kink,
                                          r_E.kappa_1, r_E.kappa_2, r_E.kappa_3,
                                          r_E.err_n, r_E.err_ratio)

    # 3. Kink-bound-adaptive
    ts_KB = kink_bound_adaptive_schedule(tau_Q, N_steps, B_kink, t_prof, schedule)
    r_KB = run_single_schedule(model, schedule, ts_KB, n_ref, ratio_ref; chi_max=CHI_MAX)
    results["Kink-bound"] = ComparisonResult("Kink-bound", r_KB.N_steps, r_KB.n_kink,
                                              r_KB.kappa_1, r_KB.kappa_2, r_KB.kappa_3,
                                              r_KB.err_n, r_KB.err_ratio)

    # 4. Kink-direct-adaptive
    ts_KD = kink_direct_adaptive_schedule(tau_Q, N_steps, eps_n_prof, t_prof, schedule)
    r_KD = run_single_schedule(model, schedule, ts_KD, n_ref, ratio_ref; chi_max=CHI_MAX)
    results["Kink-direct"] = ComparisonResult("Kink-direct", r_KD.N_steps, r_KD.n_kink,
                                               r_KD.kappa_1, r_KD.kappa_2, r_KD.kappa_3,
                                               r_KD.err_n, r_KD.err_ratio)

    schedules = Dict("Uniform"=>ts_uni, "Energy"=>ts_E,
                     "Kink-bound"=>ts_KB, "Kink-direct"=>ts_KD)
    return results, schedules
end

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
function main()
    mkpath(DATA_DIR); mkpath(FIG_DIR)

    println("=" ^ 70)
    println("  Benchmark 8: Head-to-Head Comparison")
    println("=" ^ 70)
    println("  L=$L, τ_Q primary=$TAU_Q_PRIMARY")
    println("  N_steps: $N_STEPS_LIST")
    println("  Schemes: Uniform, Energy, Kink-bound, Kink-direct")
    println("=" ^ 70)
    flush(stdout)

    profiles_10 = load_profiles(10.0)
    println("  Loaded profiles for τ_Q=10")
    flush(stdout)

    # ── N_steps sweep at tau_Q=10 ──────────────────────────────
    scheme_names = ["Uniform", "Energy", "Kink-bound", "Kink-direct"]
    all_results = Dict{String, Vector{ComparisonResult}}()
    for name in scheme_names
        all_results[name] = ComparisonResult[]
    end
    all_schedules_50 = nothing  # Save schedules for N=50 plot

    println("\n  N_steps sweep (τ_Q=$TAU_Q_PRIMARY):")
    @printf("  %8s  %12s  %12s  %12s  %12s  %12s\n",
            "N_steps", "Uniform", "Energy", "Kink-bound", "Kink-direct", "time")
    println("  " * "-" ^ 72)

    for N_steps in N_STEPS_LIST
        t0 = now()
        results, scheds = run_four_schemes(TAU_Q_PRIMARY, N_steps, profiles_10)
        elapsed = Dates.value(now() - t0) / 1000.0

        if N_steps == 50
            all_schedules_50 = scheds
        end

        for name in scheme_names
            push!(all_results[name], results[name])
        end

        @printf("  %8d  %12.2e  %12.2e  %12.2e  %12.2e  %10.1fs\n",
                N_steps,
                results["Uniform"].err_n,
                results["Energy"].err_n,
                results["Kink-bound"].err_n,
                results["Kink-direct"].err_n,
                elapsed)
        flush(stdout)
    end

    # ── tau_Q sweep at N=50 ────────────────────────────────────
    println("\n  τ_Q sweep (N_steps=50):")
    tau_Q_results = Dict{Float64, Dict{String, ComparisonResult}}()

    for tau_Q in TAU_Q_SWEEP
        println("  τ_Q = $tau_Q ...")
        flush(stdout)
        t0 = now()
        prof = load_profiles(tau_Q)
        results, _ = run_four_schemes(tau_Q, 50, prof)
        tau_Q_results[tau_Q] = results
        elapsed = Dates.value(now() - t0) / 1000.0

        @printf("    ⟨n⟩ exact=%.6f: Uni=%.6f  E=%.6f  KB=%.6f  KD=%.6f  (%.1fs)\n",
                prof["n_ref"],
                results["Uniform"].n_kink, results["Energy"].n_kink,
                results["Kink-bound"].n_kink, results["Kink-direct"].n_kink, elapsed)
        flush(stdout)
    end

    # ── Summary table ──────────────────────────────────────────
    println("\n" * "=" ^ 70)
    println("  Summary Table (τ_Q=$TAU_Q_PRIMARY)")
    println("=" ^ 70)
    @printf("  %-12s  %8s  %12s  %14s  %10s\n",
            "Scheme", "N_steps", "err(⟨n⟩)", "err(κ₂/κ₁)", "N_gates")
    println("  " * "-" ^ 60)
    for name in scheme_names
        for r in all_results[name]
            N_gates = r.N_steps * (L + L)  # PBC: L bonds + L sites
            @printf("  %-12s  %8d  %12.2e  %14.2e  %10d\n",
                    name, r.N_steps, r.err_n, r.err_ratio, N_gates)
        end
    end

    # Min N_steps for thresholds
    println("\n  Minimum N_steps for error thresholds:")
    for name in scheme_names
        results_vec = all_results[name]
        min_n = ">200"
        min_r = ">200"
        for r in results_vec
            if r.err_n < 0.005
                min_n = string(r.N_steps)
                break
            end
        end
        for r in results_vec
            if r.err_ratio < 0.01
                min_r = string(r.N_steps)
                break
            end
        end
        println("    $name: err(⟨n⟩)<0.005 at N=$min_n, err(κ₂/κ₁)<0.01 at N=$min_r")
    end

    # ── Plots ──────────────────────────────────────────────────
    println("\n  Generating plots...")
    flush(stdout)

    default(; fontfamily="Computer Modern", titlefontsize=13, guidefontsize=12,
              tickfontsize=10, legendfontsize=9, framestyle=:box, grid=false,
              foreground_color_legend=nothing, background_color_legend=:white, dpi=300)

    colors = Dict("Uniform"=>:gray50, "Energy"=>:royalblue,
                   "Kink-bound"=>:firebrick, "Kink-direct"=>:forestgreen)
    markers = Dict("Uniform"=>:circle, "Energy"=>:square,
                    "Kink-bound"=>:diamond, "Kink-direct"=>:utriangle)

    # ── Plot 1: Schedules (N=50) ──────────────────────────────
    if all_schedules_50 !== nothing
        p1 = plot(; xlabel=L"t/\tau_Q", ylabel=L"\delta t",
                  title=latexstring("Step sizes for \$N=50\$, \$\\tau_Q=$TAU_Q_PRIMARY\$"),
                  size=(700, 500), legend=:topright)

        for name in scheme_names
            ts = all_schedules_50[name]
            t_mids = [(ts.t_points[k] + ts.t_points[k+1])/2 for k in 1:length(ts)]
            plot!(p1, t_mids ./ TAU_Q_PRIMARY, ts.dt_values;
                  lw=2, color=colors[name], label=name)
        end
        vline!(p1, [0.5]; lw=1, ls=:dot, color=:gray40, label=L"t_c")

        savefig(p1, joinpath(FIG_DIR, "schedules.png"))
        savefig(p1, joinpath(FIG_DIR, "schedules.pdf"))
        println("  Plot 1 saved: schedules")
    end

    # ── Plot 2: Error in ⟨n⟩ vs N_steps ──────────────────────
    p2 = plot(; xlabel=L"N_\mathrm{steps}",
              ylabel=latexstring("Error in \\langle\\hat{n}\\rangle"),
              title=latexstring("Kink density error (\$L=$L\$, \$\\tau_Q=$TAU_Q_PRIMARY\$)"),
              xscale=:log10, yscale=:log10, size=(700, 500), legend=:topright)

    for name in scheme_names
        errs = [r.err_n for r in all_results[name]]
        scatter!(p2, N_STEPS_LIST, errs; marker=markers[name], ms=6,
                 mc=colors[name], msc=colors[name], label=name)
        plot!(p2, N_STEPS_LIST, errs; lw=1.5, color=colors[name], alpha=0.5, label="")
    end

    savefig(p2, joinpath(FIG_DIR, "kink_error_vs_Nsteps.png"))
    savefig(p2, joinpath(FIG_DIR, "kink_error_vs_Nsteps.pdf"))
    println("  Plot 2 saved: kink_error_vs_Nsteps")

    # ── Plot 3: Error in κ₂/κ₁ vs N_steps ────────────────────
    p3 = plot(; xlabel=L"N_\mathrm{steps}",
              ylabel=latexstring("Error in \\kappa_2/\\kappa_1"),
              title=latexstring("Cumulant ratio error (\$L=$L\$, \$\\tau_Q=$TAU_Q_PRIMARY\$)"),
              xscale=:log10, yscale=:log10, size=(700, 500), legend=:topright)

    for name in scheme_names
        errs = [r.err_ratio for r in all_results[name]]
        scatter!(p3, N_STEPS_LIST, errs; marker=markers[name], ms=6,
                 mc=colors[name], msc=colors[name], label=name)
        plot!(p3, N_STEPS_LIST, errs; lw=1.5, color=colors[name], alpha=0.5, label="")
    end

    savefig(p3, joinpath(FIG_DIR, "cumulant_error_vs_Nsteps.png"))
    savefig(p3, joinpath(FIG_DIR, "cumulant_error_vs_Nsteps.pdf"))
    println("  Plot 3 saved: cumulant_error_vs_Nsteps")

    # ── Plot 4: KZM scaling (N=50, three tau_Q) ──────────────
    p4 = plot(; xlabel=L"\tau_Q", ylabel=L"\langle \hat{n} \rangle",
              title=latexstring("KZM scaling with \$N=50\$ Trotter steps"),
              xscale=:log10, yscale=:log10, size=(700, 500), legend=:topright)

    # Exact reference
    exact_n = [load_profiles(tq)["n_ref"] for tq in TAU_Q_SWEEP]
    scatter!(p4, TAU_Q_SWEEP, exact_n; marker=:star5, ms=10, mc=:black, msc=:black,
             label="Exact")

    for name in scheme_names
        n_vals = [tau_Q_results[tq][name].n_kink for tq in TAU_Q_SWEEP]
        scatter!(p4, TAU_Q_SWEEP, n_vals; marker=markers[name], ms=6,
                 mc=colors[name], msc=colors[name], label=name)
        plot!(p4, TAU_Q_SWEEP, n_vals; lw=1.5, color=colors[name], alpha=0.5, label="")
    end

    # Reference slope -0.5
    tau_fit = range(4, 25; length=30)
    plot!(p4, tau_fit, exact_n[2] * (tau_fit ./ TAU_Q_SWEEP[2]) .^ (-0.5);
          lw=1.5, ls=:dot, color=:gray40, label=L"\propto \tau_Q^{-0.5}")

    savefig(p4, joinpath(FIG_DIR, "kzm_scaling.png"))
    savefig(p4, joinpath(FIG_DIR, "kzm_scaling.pdf"))
    println("  Plot 4 saved: kzm_scaling")

    # ── Plot 5: Three-panel comparison (N=50) ─────────────────
    p5 = plot(layout=(1,3), size=(1400, 450), margin=5Plots.mm)

    # Left: schedules
    if all_schedules_50 !== nothing
        for name in scheme_names
            ts = all_schedules_50[name]
            t_mids = [(ts.t_points[k] + ts.t_points[k+1])/2 for k in 1:length(ts)]
            plot!(p5[1], t_mids ./ TAU_Q_PRIMARY, ts.dt_values;
                  lw=2, color=colors[name], label=name,
                  xlabel=L"t/\tau_Q", ylabel=L"\delta t",
                  title=latexstring("Schedules (\$N=50\$)"))
        end
        vline!(p5[1], [0.5]; lw=1, ls=:dot, color=:gray40, label="")
    end

    # Middle: <n>(t) trajectories (run time-resolved for N=50 with each schedule)
    # For speed, just show final errors as a bar comparison
    model = TFIM(L; J=J, bc=BC)
    schedule = LinearQuench(TAU_Q_PRIMARY, J)
    n_ref = profiles_10["n_ref"]
    ratio_ref = profiles_10["ratio_ref"]

    # Run time-resolved for each schedule to get trajectories
    n_snap = 20
    t_snaps = collect(range(TAU_Q_PRIMARY/n_snap, TAU_Q_PRIMARY, length=n_snap))
    for name in scheme_names
        ts = all_schedules_50[name]
        n_vals = Float64[]
        psi = initial_state(model)
        snap_k = 1
        for step in 1:length(ts)
            t_n = ts.t_points[step]; dt = ts.dt_values[step]
            t_mid = t_n + dt/2
            f_val = f(schedule, t_mid); g_val = g(schedule, t_mid)
            gates = make_pf1_gates(model, f_val, g_val, dt)
            psi = apply(gates, psi; maxdim=CHI_MAX, cutoff=CUTOFF)
            noprime!(psi)
            t_after = ts.t_points[step+1]
            while snap_k <= n_snap && t_after >= t_snaps[snap_k] - 1e-10
                push!(n_vals, kink_density(psi, model))
                snap_k += 1
            end
        end
        t_plot = t_snaps[1:length(n_vals)] ./ TAU_Q_PRIMARY
        plot!(p5[2], t_plot, n_vals; lw=2, color=colors[name], label=name,
              xlabel=L"t/\tau_Q", ylabel=L"\langle \hat{n} \rangle",
              title=latexstring("\\langle\\hat{n}\\rangle(t) trajectories"))
    end
    vline!(p5[2], [0.5]; lw=1, ls=:dot, color=:gray40, label="")

    # Right: bar chart of final errors
    bar_names = scheme_names
    err_n_50 = [all_results[name][3].err_n for name in bar_names]  # index 3 = N=50
    err_r_50 = [all_results[name][3].err_ratio for name in bar_names]
    bar_colors = [colors[name] for name in bar_names]

    bar!(p5[3], bar_names, err_n_50; color=bar_colors, alpha=0.8,
         ylabel="Error", title=latexstring("Final errors (\$N=50\$)"),
         label=latexstring("err(\\langle\\hat{n}\\rangle)"), bar_width=0.4)

    savefig(p5, joinpath(FIG_DIR, "schedule_comparison.png"))
    savefig(p5, joinpath(FIG_DIR, "schedule_comparison.pdf"))
    println("  Plot 5 saved: schedule_comparison")

    # ── Save data ──────────────────────────────────────────────
    save_dict = Dict{String,Any}()
    for name in scheme_names
        for (i, N) in enumerate(N_STEPS_LIST)
            r = all_results[name][i]
            save_dict["$(name)_N$(N)_err_n"] = r.err_n
            save_dict["$(name)_N$(N)_err_ratio"] = r.err_ratio
            save_dict["$(name)_N$(N)_n_kink"] = r.n_kink
        end
    end
    save_results(joinpath(DATA_DIR, "bench08_results.h5"), save_dict)
    println("  Data saved to $DATA_DIR/bench08_results.h5")

    println("\n  Benchmark 8 complete.")
end

main()
