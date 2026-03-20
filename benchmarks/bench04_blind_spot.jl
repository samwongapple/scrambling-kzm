"""
Benchmark 4 — The Blind Spot and Verification Hierarchy

At each time point, take the reference state, apply ONE Trotter step and ONE exact
step, measure all six quantities from the verification hierarchy. Shows:
- Energy error is suppressed near t_c (blind spot)
- Kink density error peaks near t_c
- Scrambling bounds are valid
- Piecewise conservation: delta_E ≈ epsilon_H

Usage:
    julia --project=. benchmarks/bench04_blind_spot.jl
"""

using Printf, Dates, Statistics
using ITensorMPS

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM
using Plots, LaTeXStrings

const L           = 20
const TAU_Q       = 10.0
const J           = 1.0
const BC          = :periodic
const DT_TEST     = 0.5       # tau_Q/20, coarse step
const N_POINTS    = 40
const N_SUBSTEPS  = 100
const CHI_MAX     = 128
const CHI_REF     = 128
const N_STEPS_REF = 2000
const DATA_DIR    = "data/bench04"
const FIG_DIR     = "figures/bench04"

function main()
    mkpath(DATA_DIR); mkpath(FIG_DIR)

    model = TFIM(L; J=J, bc=BC)
    schedule = LinearQuench(TAU_Q, J)
    t_c = t_critical(schedule)

    println("=" ^ 70)
    println("  Benchmark 4: The Blind Spot & Verification Hierarchy")
    println("=" ^ 70)
    println("  L=$L, τ_Q=$TAU_Q, dt_test=$DT_TEST, t_c=$t_c")
    println("  Sampling at $N_POINTS time points")
    println("=" ^ 70)
    flush(stdout)

    # ── Step 1: Build commutator MPOs ──────────────────────────
    println("\n  Building commutator MPOs...")
    t0 = now()
    C_mpo = build_hz_hx_commutator(model)
    println("  [H_Z, H_X] built: bond dims = $(linkdims(C_mpo))")

    # Build kink ZZ sum MPO and commutator [C_zz, C_mpo]
    # N_hat = N_bonds/2 - C_zz/2, so [N_hat, X] = -[C_zz, X]/2
    C_zz = build_kink_zz_mpo(model)
    comm_nC = build_scrambling_operator(C_zz, C_mpo; cutoff=1e-10, maxdim=200)
    # Scale by -1/2 for the kink number: [N_hat, C] = -[C_zz, C]/2
    comm_nC = -0.5 * comm_nC
    println("  [N_hat, [H_Z,H_X]] built: bond dims = $(linkdims(comm_nC))")
    elapsed = Dates.value(now() - t0) / 1000.0
    println("  MPO construction: $(round(elapsed, digits=1))s")
    flush(stdout)

    # ── Step 2: Reference evolution, collecting snapshots ──────
    println("\n  Running reference evolution ($N_STEPS_REF steps)...")
    flush(stdout)
    t0 = now()

    sample_times = collect(range(DT_TEST, TAU_Q - DT_TEST, length=N_POINTS))
    psi_snapshots = Vector{MPS}(undef, N_POINTS)

    psi_ref = initial_state(model)
    ref_ts = uniform_schedule(TAU_Q, N_STEPS_REF)

    snap_idx = 1
    function ref_obs(psi, t, step)
        while snap_idx <= N_POINTS && t >= sample_times[snap_idx] - 1e-10
            psi_snapshots[snap_idx] = copy(psi)
            snap_idx += 1
        end
    end

    evolve_tebd!(psi_ref, model, schedule, ref_ts;
                 chi_max=CHI_REF, cutoff=1e-12, observer_fn=ref_obs)
    elapsed = Dates.value(now() - t0) / 1000.0
    println("  Reference done ($(round(elapsed, digits=1))s), collected $(snap_idx-1) snapshots")
    flush(stdout)

    # ── Step 3: Per-step errors at each snapshot ───────────────
    println("\n  Computing per-step errors at each time point...")
    flush(stdout)

    results_delta_E = zeros(N_POINTS)
    results_eps_H = zeros(N_POINTS)
    results_cons = zeros(N_POINTS)
    results_eps_H_bound = zeros(N_POINTS)
    results_eps_n = zeros(N_POINTS)
    results_eps_n_bound = zeros(N_POINTS)

    for i in 1:N_POINTS
        t_n = sample_times[i]
        psi_snap = psi_snapshots[i]

        res = compute_step_errors(psi_snap, model, schedule, t_n, DT_TEST;
                                  chi_max=CHI_MAX, chi_max_ref=CHI_REF,
                                  cutoff=1e-12, n_substeps=N_SUBSTEPS,
                                  comm_nC=comm_nC, C_mpo=C_mpo)

        results_delta_E[i] = res.delta_E
        results_eps_H[i] = res.epsilon_H
        results_cons[i] = res.conservation_check
        results_eps_H_bound[i] = res.epsilon_H_bound
        results_eps_n[i] = res.epsilon_n
        results_eps_n_bound[i] = res.epsilon_n_bound

        if i % 10 == 0 || i == 1
            @printf("  [%2d/%d] t/τ=%.3f  ΔE=%.2e  εH=%.2e  εH_bnd=%.2e  εn=%.2e  εn_bnd=%.2e  cons=%.2e\n",
                    i, N_POINTS, t_n/TAU_Q, res.delta_E, res.epsilon_H,
                    res.epsilon_H_bound, res.epsilon_n, res.epsilon_n_bound,
                    res.conservation_check)
            flush(stdout)
        end
    end

    # ── Save results ───────────────────────────────────────────
    ts_norm = sample_times ./ TAU_Q
    save_results(joinpath(DATA_DIR, "bench04_results.h5"), Dict{String,Any}(
        "t_norm" => ts_norm,
        "delta_E" => results_delta_E,
        "epsilon_H" => results_eps_H,
        "conservation_check" => results_cons,
        "epsilon_H_bound" => results_eps_H_bound,
        "epsilon_n" => results_eps_n,
        "epsilon_n_bound" => results_eps_n_bound,
    ))
    println("\n  Data saved to $DATA_DIR/bench04_results.h5")

    # ── Summary statistics ─────────────────────────────────────
    println("\n" * "=" ^ 70)
    println("  Verification Summary")
    println("=" ^ 70)
    @printf("  Max conservation check:  %.2e (should be ≈ 0)\n", maximum(results_cons))
    @printf("  Mean |ΔE - εH| / εH:    %.2e (should be ≈ 0)\n",
            mean(abs.(results_delta_E .- results_eps_H) ./ max.(results_eps_H, 1e-15)))

    # Check bounds
    n_H_violations = sum(results_eps_H .> results_eps_H_bound .* 1.1)  # 10% tolerance
    n_n_violations = sum(results_eps_n .> results_eps_n_bound .* 1.1)
    println("  Energy bound violations: $n_H_violations / $N_POINTS")
    println("  Kink bound violations:   $n_n_violations / $N_POINTS")

    # Peak locations
    idx_peak_n = argmax(results_eps_n)
    idx_peak_H = argmax(results_eps_H)
    @printf("  εn peaks at t/τ = %.3f (t_c/τ = %.3f)\n", ts_norm[idx_peak_n], t_c/TAU_Q)
    @printf("  εH peaks at t/τ = %.3f\n", ts_norm[idx_peak_H])

    # ── Plots ──────────────────────────────────────────────────
    println("\n  Generating plots...")
    flush(stdout)

    default(; fontfamily="Computer Modern", titlefontsize=14, guidefontsize=13,
              tickfontsize=11, legendfontsize=10, framestyle=:box, grid=false,
              foreground_color_legend=nothing, background_color_legend=:white, dpi=300)

    # Plot 1: The Blind Spot + Verification Hierarchy
    p1 = plot(; xlabel=L"t/\tau_Q", ylabel="Error",
              title=latexstring("Verification hierarchy (\$L=$L\$, \$\\delta t=$DT_TEST\$)"),
              size=(750, 550), yscale=:log10, legend=:topleft)

    # Energy group
    scatter!(p1, ts_norm, results_delta_E; marker=:circle, ms=4, mc=:steelblue,
             msc=:steelblue, label=latexstring("\\Delta E_n\\ (\\mathrm{direct})"))
    scatter!(p1, ts_norm, results_eps_H; marker=:square, ms=4, mc=:royalblue,
             msc=:royalblue, label=latexstring("\\epsilon_{H,n}\\ (\\mathrm{actual})"))
    plot!(p1, ts_norm, results_eps_H_bound; lw=2, ls=:dash, color=:royalblue,
          alpha=0.7, label=latexstring("\\epsilon_{H,n}^{\\mathrm{bound}}"))

    # Kink density group
    scatter!(p1, ts_norm, results_eps_n; marker=:square, ms=4, mc=:firebrick,
             msc=:firebrick, label=latexstring("\\epsilon_{\\hat{n},n}\\ (\\mathrm{actual})"))
    plot!(p1, ts_norm, results_eps_n_bound; lw=2, ls=:dash, color=:firebrick,
          alpha=0.7, label=latexstring("\\epsilon_{\\hat{n},n}^{\\mathrm{bound}}"))

    vline!(p1, [t_c/TAU_Q]; lw=1.5, ls=:dot, color=:gray40, label=L"t_c")

    savefig(p1, joinpath(FIG_DIR, "blind_spot_hierarchy.png"))
    savefig(p1, joinpath(FIG_DIR, "blind_spot_hierarchy.pdf"))
    println("  Plot 1 saved: blind_spot_hierarchy")

    # Plot 2: Conservation check
    p2 = plot(; xlabel=L"t/\tau_Q", ylabel="Conservation check",
              title=latexstring("Piecewise conservation: \$|E_{\\mathrm{exact}} - E_{\\mathrm{before}}|\$"),
              size=(650, 450))

    plot!(p2, ts_norm, results_cons; lw=2, color=:royalblue,
          label=latexstring("|\\langle H_n\\rangle_{\\mathrm{exact}} - \\langle H_n\\rangle_{\\mathrm{before}}|"))
    vline!(p2, [t_c/TAU_Q]; lw=1.5, ls=:dot, color=:gray40, label=L"t_c")

    savefig(p2, joinpath(FIG_DIR, "conservation_check.png"))
    savefig(p2, joinpath(FIG_DIR, "conservation_check.pdf"))
    println("  Plot 2 saved: conservation_check")

    # Plot 3: Bound tightness (ratios)
    ratio_H = results_eps_H ./ max.(results_eps_H_bound, 1e-15)
    ratio_n = results_eps_n ./ max.(results_eps_n_bound, 1e-15)

    p3 = plot(; xlabel=L"t/\tau_Q", ylabel="Tightness ratio (actual / bound)",
              title="Scrambling bound tightness",
              size=(650, 450), legend=:topright)

    plot!(p3, ts_norm, ratio_H; lw=2, color=:royalblue,
          label=latexstring("\\epsilon_H / \\epsilon_H^{\\mathrm{bound}}"))
    plot!(p3, ts_norm, ratio_n; lw=2, color=:firebrick,
          label=latexstring("\\epsilon_{\\hat{n}} / \\epsilon_{\\hat{n}}^{\\mathrm{bound}}"))
    hline!(p3, [1.0]; lw=1, ls=:dash, color=:gray40, label="bound = actual")
    vline!(p3, [t_c/TAU_Q]; lw=1.5, ls=:dot, color=:gray40, label=L"t_c")

    savefig(p3, joinpath(FIG_DIR, "bound_tightness.png"))
    savefig(p3, joinpath(FIG_DIR, "bound_tightness.pdf"))
    println("  Plot 3 saved: bound_tightness")

    println("\n  Benchmark 4 complete.")
end

main()
