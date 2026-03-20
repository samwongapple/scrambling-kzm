"""
Benchmark 2 — Uniform Trotter Errors

Quantify how uniform PF1 Trotterization distorts KZM observables at various
gate budgets. Time-resolved error profiles show where errors concentrate.

Usage:
    julia --project=. benchmarks/bench02_trotter_errors.jl
"""

using Printf, Dates, Statistics

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM
using Plots, LaTeXStrings

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
const L             = 20
const TAU_Q         = 10.0
const J             = 1.0
const BC            = :periodic
const N_STEPS_LIST  = [10, 20, 50, 100, 200, 500]
const N_STEPS_REF   = 2000
const CHI_MAX_REF   = 128
const CHI_MAX_TROT  = 64
const CUTOFF        = 1e-12
const N_SNAPSHOTS   = 30
const N_STEPS_TR    = 50   # N_steps for time-resolved plot
const DATA_DIR      = "data/bench02"
const FIG_DIR       = "figures/bench02"

# ═══════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════
function loglog_fit(x, y)
    lx = log.(x); ly = log.(y)
    n = length(lx)
    sx = sum(lx); sy = sum(ly); sxx = sum(lx.^2); sxy = sum(lx .* ly)
    slope = (n * sxy - sx * sy) / (n * sxx - sx^2)
    intercept = (sy - slope * sx) / n
    return slope, exp(intercept)
end

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
function main()
    mkpath(DATA_DIR); mkpath(FIG_DIR)

    model = TFIM(L; J=J, bc=BC)
    schedule = LinearQuench(TAU_Q, J)
    t_c = t_critical(schedule)

    println("=" ^ 70)
    println("  Benchmark 2: Uniform Trotter Errors")
    println("=" ^ 70)
    println("  L=$L, τ_Q=$TAU_Q, J=$J, BC=$BC")
    println("  N_steps sweep: $N_STEPS_LIST")
    println("  Reference: N_ref=$N_STEPS_REF, χ_ref=$CHI_MAX_REF")
    println("  t_c = $t_c")
    println("=" ^ 70)
    flush(stdout)

    # ── Part 1: Reference evolution ────────────────────────────
    println("\n  Running reference evolution...")
    flush(stdout)
    t0 = now()
    psi_ref = initial_state(model)
    ref_ts = uniform_schedule(TAU_Q, N_STEPS_REF)
    psi_ref = evolve_tebd!(psi_ref, model, schedule, ref_ts;
                           chi_max=CHI_MAX_REF, cutoff=CUTOFF)
    n_ref = kink_density(psi_ref, model)
    k1_ref, k2_ref, k3_ref = kink_cumulants(psi_ref, model; chi_max=CHI_MAX_REF)
    ratio_ref = k2_ref / k1_ref
    elapsed = Dates.value(now() - t0) / 1000.0
    @printf("  Reference: ⟨n⟩=%.6f  κ₂/κ₁=%.4f  (%.1fs)\n\n", n_ref, ratio_ref, elapsed)
    flush(stdout)

    # ── Part 2: N_steps sweep ──────────────────────────────────
    n_trot_vals   = Float64[]
    ratio_vals    = Float64[]
    n_errors      = Float64[]
    ratio_errors  = Float64[]

    println("  N_steps sweep:")
    @printf("  %8s  %10s  %10s  %10s  %10s  %6s\n",
            "N_steps", "⟨n⟩_trot", "err_n", "κ₂/κ₁", "err_ratio", "time")
    println("  " * "-" ^ 62)

    for N_steps in N_STEPS_LIST
        t0 = now()
        psi = initial_state(model)
        ts = uniform_schedule(TAU_Q, N_steps)
        psi = trotter_evolve!(psi, model, schedule, ts;
                              chi_max=CHI_MAX_TROT, cutoff=CUTOFF)
        n_trot = kink_density(psi, model)
        k1, k2, k3 = kink_cumulants(psi, model; chi_max=CHI_MAX_TROT)
        ratio = k2 / k1
        elapsed = Dates.value(now() - t0) / 1000.0

        err_n = abs(n_trot - n_ref)
        err_ratio = abs(ratio - ratio_ref)

        push!(n_trot_vals, n_trot)
        push!(ratio_vals, ratio)
        push!(n_errors, err_n)
        push!(ratio_errors, err_ratio)

        N_gates = N_steps * (L + (BC == :periodic ? L : L-1))
        @printf("  %8d  %10.6f  %10.2e  %10.4f  %10.2e  %5.1fs\n",
                N_steps, n_trot, err_n, ratio, err_ratio, elapsed)
        flush(stdout)
    end

    # ── Part 3: Time-resolved errors (N_steps=50) ─────────────
    println("\n  Running time-resolved comparison (N_steps=$N_STEPS_TR)...")
    flush(stdout)
    t0 = now()
    tr = time_resolved_errors(model, schedule, TAU_Q, N_STEPS_TR;
                              n_snapshots=N_SNAPSHOTS,
                              chi_max=CHI_MAX_TROT,
                              chi_max_ref=CHI_MAX_REF,
                              n_steps_ref=N_STEPS_REF,
                              cutoff=CUTOFF)
    elapsed = Dates.value(now() - t0) / 1000.0
    println("  Time-resolved done ($(round(elapsed, digits=1))s)")
    flush(stdout)

    # ── Save HDF5 ─────────────────────────────────────────────
    results = Dict{String,Any}(
        "N_steps_list" => Float64.(N_STEPS_LIST),
        "n_ref" => n_ref,
        "ratio_ref" => ratio_ref,
        "n_trot" => n_trot_vals,
        "n_errors" => n_errors,
        "ratio_vals" => ratio_vals,
        "ratio_errors" => ratio_errors,
        "tr_t" => tr.t_snapshots,
        "tr_n_exact" => tr.n_exact,
        "tr_n_trotter" => tr.n_trotter,
        "tr_E_exact" => tr.E_exact,
        "tr_E_trotter" => tr.E_trotter,
    )
    save_results(joinpath(DATA_DIR, "bench02_results.h5"), results)
    println("  Data saved to $DATA_DIR/bench02_results.h5")

    # ── Summary table ─────────────────────────────────────────
    println("\n" * "=" ^ 70)
    println("  Summary")
    println("=" ^ 70)
    @printf("  %8s  %12s  %14s  %10s\n",
            "N_steps", "err(⟨n⟩)", "err(κ₂/κ₁)", "N_gates")
    println("  " * "-" ^ 50)
    for (i, N_steps) in enumerate(N_STEPS_LIST)
        N_gates = N_steps * (L + (BC == :periodic ? L : L-1))
        @printf("  %8d  %12.2e  %14.2e  %10d\n",
                N_steps, n_errors[i], ratio_errors[i], N_gates)
    end

    # Convergence rate
    slope_n, _ = loglog_fit(Float64.(N_STEPS_LIST), n_errors)
    @printf("\n  Convergence rate: err(⟨n⟩) ∝ N_steps^{%.2f}\n", slope_n)

    # ── Plots ─────────────────────────────────────────────────
    println("\n  Generating plots...")
    flush(stdout)

    default(; fontfamily="Computer Modern",
              titlefontsize=14, guidefontsize=13, tickfontsize=11,
              legendfontsize=10, framestyle=:box, grid=false,
              foreground_color_legend=nothing,
              background_color_legend=:white, dpi=300)

    # Plot 1: Error in ⟨n⟩ vs N_steps
    slope_n, A_n = loglog_fit(Float64.(N_STEPS_LIST), n_errors)
    Ns_fit = range(N_STEPS_LIST[1]*0.7, N_STEPS_LIST[end]*1.5; length=50)

    p1 = plot(; xlabel=L"N_\mathrm{steps}", ylabel=latexstring("Error in \\langle\\hat{n}\\rangle"),
              title=latexstring("Trotter error in kink density (\$L=$L\$, \$\\tau_Q=$TAU_Q\$)"),
              xscale=:log10, yscale=:log10, size=(650, 500), legend=:topright)

    scatter!(p1, N_STEPS_LIST, n_errors; marker=:circle, ms=7, mc=:royalblue, msc=:royalblue,
             label="PF1 Trotter")
    plot!(p1, Ns_fit, A_n .* Ns_fit .^ slope_n; lw=2, ls=:dash, color=:firebrick,
          label=latexstring("\\propto N^{$(@sprintf("%.2f", slope_n))}"))

    savefig(p1, joinpath(FIG_DIR, "error_n_vs_Nsteps.png"))
    savefig(p1, joinpath(FIG_DIR, "error_n_vs_Nsteps.pdf"))
    println("  Plot 1 saved: error_n_vs_Nsteps")

    # Plot 2: Error in κ₂/κ₁ vs N_steps
    valid = ratio_errors .> 0
    slope_r, A_r = loglog_fit(Float64.(N_STEPS_LIST[valid]), ratio_errors[valid])

    p2 = plot(; xlabel=L"N_\mathrm{steps}", ylabel=latexstring("Error in \\kappa_2/\\kappa_1"),
              title=latexstring("Trotter error in cumulant ratio (\$L=$L\$, \$\\tau_Q=$TAU_Q\$)"),
              xscale=:log10, yscale=:log10, size=(650, 500), legend=:topright)

    scatter!(p2, N_STEPS_LIST, ratio_errors; marker=:square, ms=7, mc=:royalblue, msc=:royalblue,
             label="PF1 Trotter")
    plot!(p2, Ns_fit, A_r .* Ns_fit .^ slope_r; lw=2, ls=:dash, color=:firebrick,
          label=latexstring("\\propto N^{$(@sprintf("%.2f", slope_r))}"))

    savefig(p2, joinpath(FIG_DIR, "error_ratio_vs_Nsteps.png"))
    savefig(p2, joinpath(FIG_DIR, "error_ratio_vs_Nsteps.pdf"))
    println("  Plot 2 saved: error_ratio_vs_Nsteps")

    # Plot 3: Time-resolved error profile (N_steps=50)
    err_n_t = abs.(tr.n_exact .- tr.n_trotter)
    err_E_t = abs.(tr.E_exact .- tr.E_trotter)
    ts = tr.t_snapshots ./ TAU_Q  # normalized time

    p3 = plot(; xlabel=L"t / \tau_Q",
              ylabel="Error",
              title=latexstring("Time-resolved Trotter errors (\$N=$N_STEPS_TR\$)"),
              size=(700, 500), legend=:topleft, yscale=:identity)

    plot!(p3, ts, err_n_t; lw=2.5, color=:royalblue,
          label=latexstring("|\\langle\\hat{n}\\rangle_\\mathrm{Trotter} - \\langle\\hat{n}\\rangle_\\mathrm{exact}|"))
    plot!(p3, ts, err_E_t; lw=2.5, color=:firebrick, ls=:dash,
          label=latexstring("|\\langle H\\rangle_\\mathrm{Trotter} - \\langle H\\rangle_\\mathrm{exact}|"))

    vline!(p3, [t_c / TAU_Q]; lw=1.5, ls=:dot, color=:gray40,
           label=latexstring("t_c/\\tau_Q = $(@sprintf("%.2f", t_c/TAU_Q))"))

    savefig(p3, joinpath(FIG_DIR, "time_resolved_errors.png"))
    savefig(p3, joinpath(FIG_DIR, "time_resolved_errors.pdf"))
    println("  Plot 3 saved: time_resolved_errors")

    # Plot 4: Actual ⟨n⟩(t) for exact vs Trotter
    p4 = plot(; xlabel=L"t / \tau_Q",
              ylabel=L"\langle \hat{n} \rangle",
              title=latexstring("Kink density: exact vs Trotter (\$N=$N_STEPS_TR\$)"),
              size=(700, 500), legend=:topleft)

    plot!(p4, ts, tr.n_exact; lw=2.5, color=:royalblue, label="Exact (TEBD)")
    plot!(p4, ts, tr.n_trotter; lw=2.5, color=:firebrick, ls=:dash, label="Trotter (PF1)")

    vline!(p4, [t_c / TAU_Q]; lw=1.5, ls=:dot, color=:gray40,
           label=latexstring("t_c/\\tau_Q"))

    savefig(p4, joinpath(FIG_DIR, "n_exact_vs_trotter.png"))
    savefig(p4, joinpath(FIG_DIR, "n_exact_vs_trotter.pdf"))
    println("  Plot 4 saved: n_exact_vs_trotter")

    println("\n  Benchmark 2 complete.")
end

main()
