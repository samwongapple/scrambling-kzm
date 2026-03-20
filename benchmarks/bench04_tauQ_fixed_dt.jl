"""
τ_Q dependence with FIXED dt_test = 0.25 to isolate physics from step-size effects.

With fixed dt, M is the same size — what changes is WHERE the system is sensitive.
For larger τ_Q, the KZM impulse window narrows as τ_Q^{-1/2} in rescaled time.

Usage:
    julia --project=. benchmarks/bench04_tauQ_fixed_dt.jl
"""

using Printf, Dates, Statistics
using ITensorMPS

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM
using Plots, LaTeXStrings

const L           = 20
const DT_TEST     = 0.25
const TAU_QS      = [5.0, 10.0, 20.0]
const N_POINTS    = 40
const N_SUBSTEPS  = 200
const CHI_MAX     = 128
const N_STEPS_REF = 2000
const FIG_DIR     = "figures/bench04"

function run_one(tau_Q::Float64)
    model = TFIM(L; J=1.0, bc=:periodic)
    schedule = LinearQuench(tau_Q, 1.0)

    # Snapshot times: evenly spaced, leaving room for dt at edges
    sample_times = collect(range(DT_TEST, tau_Q - DT_TEST, length=N_POINTS))

    # Reference evolution
    psi_snapshots = Vector{MPS}(undef, N_POINTS)
    psi = initial_state(model)

    n_ref = tau_Q <= 10 ? N_STEPS_REF : 1000
    chi = tau_Q <= 10 ? CHI_MAX : 64
    ref_ts = uniform_schedule(tau_Q, n_ref)

    snap_idx = 1
    function obs(psi_obs, t, step)
        while snap_idx <= N_POINTS && t >= sample_times[snap_idx] - 1e-10
            psi_snapshots[snap_idx] = copy(psi_obs)
            snap_idx += 1
        end
    end
    evolve_tebd!(psi, model, schedule, ref_ts; chi_max=chi, cutoff=1e-12, observer_fn=obs)

    # Per-step errors with FIXED dt
    eps_n = zeros(N_POINTS)
    eps_H = zeros(N_POINTS)

    for i in 1:N_POINTS
        t_n = sample_times[i]
        # Skip if dt would overshoot
        if t_n + DT_TEST > tau_Q + 1e-10
            continue
        end
        res = compute_step_errors(psi_snapshots[i], model, schedule, t_n, DT_TEST;
                                  chi_max=chi, chi_max_ref=chi,
                                  cutoff=1e-12, n_substeps=N_SUBSTEPS)
        eps_n[i] = res.epsilon_n
        eps_H[i] = res.epsilon_H
    end

    return (t=sample_times, t_norm=sample_times ./ tau_Q, eps_n=eps_n, eps_H=eps_H)
end

function compute_fwhm(t_norm, vals)
    peak_val = maximum(vals)
    half_max = peak_val / 2
    idx_peak = argmax(vals)

    # Find left crossing
    left = 1
    for i in idx_peak:-1:1
        if vals[i] < half_max
            left = i
            break
        end
    end
    # Find right crossing
    right = length(vals)
    for i in idx_peak:length(vals)
        if vals[i] < half_max
            right = i
            break
        end
    end

    fwhm = t_norm[right] - t_norm[left]
    return fwhm
end

function main()
    mkpath(FIG_DIR)

    println("=" ^ 65)
    println("  τ_Q Dependence with Fixed dt = $DT_TEST")
    println("=" ^ 65)
    println("  L=$L, τ_Q = $TAU_QS, dt_test=$DT_TEST")
    println("  N_points=$N_POINTS, N_substeps=$N_SUBSTEPS")
    println("=" ^ 65)
    flush(stdout)

    results = Dict{Float64, Any}()
    colors = [:forestgreen, :royalblue, :firebrick]

    for (k, tau_Q) in enumerate(TAU_QS)
        println("\n  Running τ_Q = $tau_Q...")
        flush(stdout)
        t0 = now()
        r = run_one(tau_Q)
        elapsed = Dates.value(now() - t0) / 1000.0
        results[tau_Q] = r

        idx_peak = argmax(r.eps_n)
        fwhm = compute_fwhm(r.t_norm, r.eps_n)
        @printf("  τ_Q=%5.1f: max εn = %.2e at t/τ = %.3f, FWHM = %.3f  (%.1fs)\n",
                tau_Q, maximum(r.eps_n), r.t_norm[idx_peak], fwhm, elapsed)
        flush(stdout)
    end

    # ── Summary table ─────────────────────────────────────────
    println("\n" * "=" ^ 65)
    println("  Summary: εn peak properties with fixed dt=$DT_TEST")
    println("=" ^ 65)
    @printf("  %8s  %12s  %12s  %12s\n", "τ_Q", "peak height", "peak t/τ_Q", "FWHM(t/τ_Q)")
    println("  " * "-" ^ 50)
    for tau_Q in TAU_QS
        r = results[tau_Q]
        idx = argmax(r.eps_n)
        fwhm = compute_fwhm(r.t_norm, r.eps_n)
        @printf("  %8.1f  %12.2e  %12.3f  %12.3f\n",
                tau_Q, maximum(r.eps_n), r.t_norm[idx], fwhm)
    end

    # KZM prediction: FWHM ∝ τ_Q^{-1/2} in rescaled time
    fwhms = [compute_fwhm(results[tq].t_norm, results[tq].eps_n) for tq in TAU_QS]
    if length(fwhms) >= 2 && all(fwhms .> 0)
        log_tau = log.(TAU_QS)
        log_fwhm = log.(fwhms)
        n = length(log_tau)
        sx = sum(log_tau); sy = sum(log_fwhm); sxx = sum(log_tau.^2); sxy = sum(log_tau .* log_fwhm)
        slope = (n * sxy - sx * sy) / (n * sxx - sx^2)
        @printf("\n  FWHM scaling: FWHM ∝ τ_Q^{%.2f}  (KZM prediction: -0.50)\n", slope)
    end

    # ── Plots ─────────────────────────────────────────────────
    println("\n  Generating plots...")
    flush(stdout)

    default(; fontfamily="Computer Modern", titlefontsize=13, guidefontsize=12,
              tickfontsize=10, legendfontsize=9, framestyle=:box, grid=false,
              foreground_color_legend=nothing, background_color_legend=:white, dpi=300)

    # Plot 1: Raw errors
    p1 = plot(layout=(1,2), size=(1100, 450), margin=5Plots.mm)

    for (k, tau_Q) in enumerate(TAU_QS)
        r = results[tau_Q]
        mask = r.eps_n .> 0
        plot!(p1[1], r.t_norm[mask], r.eps_n[mask]; lw=2.5, color=colors[k],
              label=latexstring("\\tau_Q = $(Int(tau_Q))"),
              xlabel=L"t/\tau_Q", ylabel=latexstring("\\epsilon_{\\hat{n}}"),
              title=latexstring("Kink density error (\\delta t = $DT_TEST)"),
              yscale=:log10)
        mask_H = r.eps_H .> 0
        plot!(p1[2], r.t_norm[mask_H], r.eps_H[mask_H]; lw=2.5, color=colors[k],
              label=latexstring("\\tau_Q = $(Int(tau_Q))"),
              xlabel=L"t/\tau_Q", ylabel=latexstring("\\epsilon_H"),
              title=latexstring("Energy error (\\delta t = $DT_TEST)"),
              yscale=:log10)
    end
    vline!(p1[1], [0.5]; lw=1, ls=:dot, color=:gray40, label=L"t_c")
    vline!(p1[2], [0.5]; lw=1, ls=:dot, color=:gray40, label=L"t_c")

    savefig(p1, joinpath(FIG_DIR, "tau_Q_dependence_fixed_dt.png"))
    savefig(p1, joinpath(FIG_DIR, "tau_Q_dependence_fixed_dt.pdf"))
    println("  Plot 1 saved: tau_Q_dependence_fixed_dt")

    # Plot 2: Normalized
    p2 = plot(layout=(1,2), size=(1100, 450), margin=5Plots.mm)

    for (k, tau_Q) in enumerate(TAU_QS)
        r = results[tau_Q]
        norm_n = r.eps_n ./ maximum(r.eps_n)
        norm_H = r.eps_H ./ maximum(r.eps_H)
        plot!(p2[1], r.t_norm, norm_n; lw=2.5, color=colors[k],
              label=latexstring("\\tau_Q = $(Int(tau_Q))"),
              xlabel=L"t/\tau_Q", ylabel="Normalized error",
              title=latexstring("Kink density (normalized, \\delta t = $DT_TEST)"))
        plot!(p2[2], r.t_norm, norm_H; lw=2.5, color=colors[k],
              label=latexstring("\\tau_Q = $(Int(tau_Q))"),
              xlabel=L"t/\tau_Q", ylabel="Normalized error",
              title=latexstring("Energy (normalized, \\delta t = $DT_TEST)"))
    end
    vline!(p2[1], [0.5]; lw=1, ls=:dot, color=:gray40, label=L"t_c")
    vline!(p2[2], [0.5]; lw=1, ls=:dot, color=:gray40, label=L"t_c")

    savefig(p2, joinpath(FIG_DIR, "tau_Q_dependence_fixed_dt_normalized.png"))
    savefig(p2, joinpath(FIG_DIR, "tau_Q_dependence_fixed_dt_normalized.pdf"))
    println("  Plot 2 saved: tau_Q_dependence_fixed_dt_normalized")

    println("\n  Done.")
end

main()
