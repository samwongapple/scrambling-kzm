"""
Benchmark 4 Diagnostics — Additional validation of scrambling bound framework.

Test A: Shape correspondence between actual error and scrambling bound
Test B: System size dependence of the blind spot
Test C: Tighten conservation check (200 substeps)
Test D: tau_Q dependence of error profiles

Usage:
    julia --project=. benchmarks/bench04_diagnostics.jl
"""

using Printf, Dates, Statistics
using ITensors, ITensorMPS

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM
using Plots, LaTeXStrings

const FIG_DIR = "figures/bench04"
const DATA_DIR = "data/bench04"

function setup_plot_defaults()
    default(; fontfamily="Computer Modern", titlefontsize=13, guidefontsize=12,
              tickfontsize=10, legendfontsize=9, framestyle=:box, grid=false,
              foreground_color_legend=nothing, background_color_legend=:white, dpi=300)
end

# ═══════════════════════════════════════════════════════════════════
# Helper: run per-step errors at sampled time points
# ═══════════════════════════════════════════════════════════════════
function run_perstep_errors(;
    L::Int, tau_Q::Float64, dt_test::Float64,
    n_points::Int=30, chi_max::Int=128, n_steps_ref::Int=2000,
    n_substeps::Int=100, compute_bounds::Bool=false
)
    model = TFIM(L; J=1.0, bc=:periodic)
    schedule = LinearQuench(tau_Q, 1.0)

    # Build commutator MPOs if needed
    C_mpo = nothing; comm_nC = nothing
    if compute_bounds
        C_mpo = build_hz_hx_commutator(model)
        C_zz = build_kink_zz_mpo(model)
        comm_nC = -0.5 * build_scrambling_operator(C_zz, C_mpo; cutoff=1e-10, maxdim=200)
    end

    # Reference evolution with snapshots
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
    evolve_tebd!(psi, model, schedule, ref_ts; chi_max=chi_max, cutoff=1e-12, observer_fn=obs)

    # Per-step errors
    delta_E = zeros(n_points)
    eps_H = zeros(n_points)
    cons = zeros(n_points)
    eps_H_bound = fill(NaN, n_points)
    eps_n = zeros(n_points)
    eps_n_bound = fill(NaN, n_points)

    for i in 1:n_points
        res = compute_step_errors(psi_snapshots[i], model, schedule,
                                  sample_times[i], dt_test;
                                  chi_max=chi_max, chi_max_ref=chi_max,
                                  cutoff=1e-12, n_substeps=n_substeps,
                                  comm_nC=compute_bounds ? comm_nC : nothing,
                                  C_mpo=compute_bounds ? C_mpo : nothing)
        delta_E[i] = res.delta_E
        eps_H[i] = res.epsilon_H
        cons[i] = res.conservation_check
        eps_H_bound[i] = res.epsilon_H_bound
        eps_n[i] = res.epsilon_n
        eps_n_bound[i] = res.epsilon_n_bound
    end

    return (t=sample_times, t_norm=sample_times ./ tau_Q,
            delta_E=delta_E, eps_H=eps_H, cons=cons,
            eps_H_bound=eps_H_bound, eps_n=eps_n, eps_n_bound=eps_n_bound)
end

# ═══════════════════════════════════════════════════════════════════
# Test A: Shape correspondence
# ═══════════════════════════════════════════════════════════════════
function test_A()
    println("\n" * "=" ^ 65)
    println("  Test A: Shape Correspondence")
    println("=" ^ 65)
    flush(stdout)

    # Load existing bench04 data or recompute with bounds
    println("  Computing per-step errors with scrambling bounds (L=20, τ_Q=10)...")
    flush(stdout)
    t0 = now()
    r = run_perstep_errors(L=20, tau_Q=10.0, dt_test=0.5, n_points=40,
                           n_substeps=100, compute_bounds=true)
    elapsed = Dates.value(now() - t0) / 1000.0
    println("  Done ($(round(elapsed, digits=1))s)")

    # Normalize to [0,1]
    norm_eps_n = r.eps_n ./ maximum(r.eps_n)
    norm_bound_n = r.eps_n_bound ./ maximum(r.eps_n_bound)
    norm_eps_H = r.eps_H ./ maximum(r.eps_H)
    norm_bound_H = r.eps_H_bound ./ maximum(r.eps_H_bound)

    # Correlation
    corr_n = cor(norm_eps_n, norm_bound_n)
    corr_H = cor(norm_eps_H, norm_bound_H)
    @printf("  Kink density:  shape correlation = %.4f\n", corr_n)
    @printf("  Energy:        shape correlation = %.4f\n", corr_H)

    # Peak locations
    idx_peak_eps_n = argmax(r.eps_n)
    idx_peak_bnd_n = argmax(r.eps_n_bound)
    idx_peak_eps_H = argmax(r.eps_H)
    idx_peak_bnd_H = argmax(r.eps_H_bound)
    @printf("  εn peak: t/τ=%.3f,  bound peak: t/τ=%.3f\n",
            r.t_norm[idx_peak_eps_n], r.t_norm[idx_peak_bnd_n])
    @printf("  εH peak: t/τ=%.3f,  bound peak: t/τ=%.3f\n",
            r.t_norm[idx_peak_eps_H], r.t_norm[idx_peak_bnd_H])

    setup_plot_defaults()
    t_c_norm = 0.5

    p = plot(layout=(1,2), size=(1100, 450), margin=5Plots.mm)

    # Panel 1: kink density
    plot!(p[1], r.t_norm, norm_eps_n; lw=2.5, color=:firebrick,
          label=latexstring("\\epsilon_{\\hat{n}}(t)\\ (\\mathrm{actual})"),
          xlabel=L"t/\tau_Q", ylabel="Normalized error",
          title=latexstring("Kink density (corr = $(@sprintf("%.3f", corr_n)))"))
    plot!(p[1], r.t_norm, norm_bound_n; lw=2.5, ls=:dash, color=:firebrick, alpha=0.6,
          label=latexstring("\\epsilon_{\\hat{n}}^{\\mathrm{bound}}(t)"))
    vline!(p[1], [t_c_norm]; lw=1, ls=:dot, color=:gray40, label=L"t_c")

    # Panel 2: energy
    plot!(p[2], r.t_norm, norm_eps_H; lw=2.5, color=:royalblue,
          label=latexstring("\\epsilon_H(t)\\ (\\mathrm{actual})"),
          xlabel=L"t/\tau_Q", ylabel="Normalized error",
          title=latexstring("Energy (corr = $(@sprintf("%.3f", corr_H)))"))
    plot!(p[2], r.t_norm, norm_bound_H; lw=2.5, ls=:dash, color=:royalblue, alpha=0.6,
          label=latexstring("\\epsilon_H^{\\mathrm{bound}}(t)"))
    vline!(p[2], [t_c_norm]; lw=1, ls=:dot, color=:gray40, label=L"t_c")

    savefig(p, joinpath(FIG_DIR, "shape_correspondence.png"))
    savefig(p, joinpath(FIG_DIR, "shape_correspondence.pdf"))
    println("  Plot saved: shape_correspondence.{png,pdf}")
    flush(stdout)

    return r
end

# ═══════════════════════════════════════════════════════════════════
# Test B: System size dependence
# ═══════════════════════════════════════════════════════════════════
function test_B()
    println("\n" * "=" ^ 65)
    println("  Test B: System Size Dependence")
    println("=" ^ 65)
    flush(stdout)

    tau_Q = 10.0; dt_test = 0.5
    Ls = [10, 20, 40]
    colors_L = [:forestgreen, :royalblue, :firebrick]
    results = Dict{Int,Any}()

    for L in Ls
        println("  Running L=$L...")
        flush(stdout)
        t0 = now()
        # L=40 needs less steps for speed
        n_ref = L <= 20 ? 2000 : 1000
        chi = L <= 20 ? 128 : 64
        r = run_perstep_errors(L=L, tau_Q=tau_Q, dt_test=dt_test, n_points=30,
                               chi_max=chi, n_steps_ref=n_ref, n_substeps=100)
        elapsed = Dates.value(now() - t0) / 1000.0
        results[L] = r
        @printf("  L=%d done (%.1fs): max εn=%.2e at t/τ=%.3f, max εH=%.2e at t/τ=%.3f\n",
                L, elapsed, maximum(r.eps_n), r.t_norm[argmax(r.eps_n)],
                maximum(r.eps_H), r.t_norm[argmax(r.eps_H)])
        flush(stdout)
    end

    setup_plot_defaults()
    t_c_norm = 0.5

    p = plot(layout=(1,2), size=(1100, 450), margin=5Plots.mm)

    for (k, L) in enumerate(Ls)
        r = results[L]
        plot!(p[1], r.t_norm, r.eps_n; lw=2, color=colors_L[k], label="L=$L",
              xlabel=L"t/\tau_Q", ylabel=latexstring("\\epsilon_{\\hat{n}}"),
              title="Kink density Trotter error", yscale=:log10)
        plot!(p[2], r.t_norm, r.eps_H; lw=2, color=colors_L[k], label="L=$L",
              xlabel=L"t/\tau_Q", ylabel=latexstring("\\epsilon_H"),
              title="Energy Trotter error", yscale=:log10)
    end
    vline!(p[1], [t_c_norm]; lw=1, ls=:dot, color=:gray40, label=L"t_c")
    vline!(p[2], [t_c_norm]; lw=1, ls=:dot, color=:gray40, label=L"t_c")

    savefig(p, joinpath(FIG_DIR, "size_dependence.png"))
    savefig(p, joinpath(FIG_DIR, "size_dependence.pdf"))
    println("  Plot saved: size_dependence.{png,pdf}")
    flush(stdout)
end

# ═══════════════════════════════════════════════════════════════════
# Test C: Tighten conservation check
# ═══════════════════════════════════════════════════════════════════
function test_C()
    println("\n" * "=" ^ 65)
    println("  Test C: Tightened Conservation Check (200 substeps)")
    println("=" ^ 65)
    flush(stdout)

    t0 = now()
    r = run_perstep_errors(L=20, tau_Q=10.0, dt_test=0.5, n_points=40,
                           n_substeps=200, compute_bounds=false)
    elapsed = Dates.value(now() - t0) / 1000.0
    println("  Done ($(round(elapsed, digits=1))s)")

    @printf("  Max conservation check:  %.2e (was 1.24e-02 with 100 substeps)\n",
            maximum(r.cons))
    @printf("  Mean conservation check: %.2e\n", mean(r.cons))
    @printf("  Mean |ΔE - εH| / εH:    %.2e\n",
            mean(abs.(r.delta_E .- r.eps_H) ./ max.(r.eps_H, 1e-15)))

    setup_plot_defaults()

    p = plot(; xlabel=L"t/\tau_Q", ylabel="Conservation check",
              title=latexstring("Piecewise conservation (200 substeps, \$L=20\$)"),
              size=(650, 450))
    plot!(p, r.t_norm, r.cons; lw=2, color=:royalblue,
          label=latexstring("|E_{\\mathrm{exact}} - E_{\\mathrm{before}}|"))
    plot!(p, r.t_norm, abs.(r.delta_E .- r.eps_H); lw=2, ls=:dash, color=:firebrick,
          label=latexstring("|\\Delta E - \\epsilon_H|"))
    vline!(p, [0.5]; lw=1, ls=:dot, color=:gray40, label=L"t_c")

    savefig(p, joinpath(FIG_DIR, "conservation_check_tight.png"))
    savefig(p, joinpath(FIG_DIR, "conservation_check_tight.pdf"))
    println("  Plot saved: conservation_check_tight.{png,pdf}")
    flush(stdout)

    return r
end

# ═══════════════════════════════════════════════════════════════════
# Test D: tau_Q dependence
# ═══════════════════════════════════════════════════════════════════
function test_D()
    println("\n" * "=" ^ 65)
    println("  Test D: τ_Q Dependence")
    println("=" ^ 65)
    flush(stdout)

    L = 20
    tau_Qs = [5.0, 10.0, 20.0]
    colors_tau = [:forestgreen, :royalblue, :firebrick]
    results = Dict{Float64,Any}()

    for tau_Q in tau_Qs
        dt_test = tau_Q / 20.0
        println("  Running τ_Q=$tau_Q (dt=$dt_test)...")
        flush(stdout)
        t0 = now()
        n_ref = tau_Q <= 10 ? 2000 : 1000
        chi = tau_Q <= 10 ? 128 : 64
        r = run_perstep_errors(L=L, tau_Q=tau_Q, dt_test=dt_test, n_points=30,
                               chi_max=chi, n_steps_ref=n_ref, n_substeps=100)
        elapsed = Dates.value(now() - t0) / 1000.0
        results[tau_Q] = r
        @printf("  τ_Q=%.1f done (%.1fs): max εn=%.2e at t/τ=%.3f\n",
                tau_Q, elapsed, maximum(r.eps_n), r.t_norm[argmax(r.eps_n)])
        flush(stdout)
    end

    setup_plot_defaults()

    p = plot(; xlabel=L"t/\tau_Q",
              ylabel=latexstring("\\epsilon_{\\hat{n}}"),
              title=latexstring("Kink density error: \$\\tau_Q\$ dependence (\$L=$L\$)"),
              size=(700, 500), legend=:topright)

    for (k, tau_Q) in enumerate(tau_Qs)
        r = results[tau_Q]
        # Normalize epsilon_n by dt^2 to compare shapes (dt differs across tau_Q)
        dt_test = tau_Q / 20.0
        plot!(p, r.t_norm, r.eps_n; lw=2.5, color=colors_tau[k],
              label=latexstring("\\tau_Q = $(@sprintf("%.0f", tau_Q))"))
    end
    vline!(p, [0.5]; lw=1, ls=:dot, color=:gray40, label=L"t_c")

    savefig(p, joinpath(FIG_DIR, "tau_Q_dependence.png"))
    savefig(p, joinpath(FIG_DIR, "tau_Q_dependence.pdf"))
    println("  Plot saved: tau_Q_dependence.{png,pdf}")

    # Also plot normalized (divide by max) to compare shapes
    p2 = plot(; xlabel=L"t/\tau_Q",
              ylabel="Normalized error",
              title=latexstring("Normalized \\epsilon_{\\hat{n}}: shape vs \$\\tau_Q\$ (\$L=$L\$)"),
              size=(700, 500), legend=:topright)

    for (k, tau_Q) in enumerate(tau_Qs)
        r = results[tau_Q]
        eps_norm = r.eps_n ./ maximum(r.eps_n)
        plot!(p2, r.t_norm, eps_norm; lw=2.5, color=colors_tau[k],
              label=latexstring("\\tau_Q = $(@sprintf("%.0f", tau_Q))"))
    end
    vline!(p2, [0.5]; lw=1, ls=:dot, color=:gray40, label=L"t_c")

    savefig(p2, joinpath(FIG_DIR, "tau_Q_dependence_normalized.png"))
    savefig(p2, joinpath(FIG_DIR, "tau_Q_dependence_normalized.pdf"))
    println("  Plot saved: tau_Q_dependence_normalized.{png,pdf}")
    flush(stdout)
end

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
function main()
    mkpath(FIG_DIR); mkpath(DATA_DIR)

    println("=" ^ 65)
    println("  Benchmark 4 Diagnostics: Scrambling Bound Validation")
    println("=" ^ 65)

    r_A = test_A()
    test_C()
    test_D()
    test_B()

    # ── Final summary ─────────────────────────────────────────
    println("\n" * "=" ^ 65)
    println("  FINDINGS SUMMARY")
    println("=" ^ 65)

    println("""
  Test A (Shape Correspondence):
    The scrambling bound profiles predict WHERE errors are largest.
    High shape correlation means the bound-based adaptive schedule
    will concentrate steps in the right time regions.

  Test B (Size Dependence):
    The blind spot (differential sensitivity between εH and εn
    near t_c) should persist or sharpen with increasing L, confirming
    it is a genuine physical effect, not a finite-size artifact.

  Test C (Conservation Check):
    With 200 substeps, the conservation check should be well below
    1e-03, confirming that ΔE = εH to high precision. This validates
    the piecewise conservation property.

  Test D (τ_Q Dependence):
    The kink density error peak should narrow and sharpen as τ_Q
    increases, consistent with the KZM impulse window scaling as
    τ_Q^{1/2}.
""")
    println("  All diagnostics complete.")
end

main()
