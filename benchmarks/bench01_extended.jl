"""
Benchmark 1 Extended — Publication-Quality KZM Plots

Wider τ_Q range to capture both fast-quench plateau and KZM scaling regime.
Three publication-style plots matching del Campo (2018), Zeng et al. (2023).

Usage:
    julia --project=. benchmarks/bench01_extended.jl
"""

using Printf, Dates, Statistics

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM
using Plots, LaTeXStrings

# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════
const SYSTEM_SIZES   = [40, 100]
const TAU_Q_VALUES   = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
const J              = 1.0
const BC             = :periodic
const N_STEPS_DEFAULT = 2000
const CHI_MAX_DEFAULT = 128
const MAX_TIME        = 600.0   # 10 min per run
const DATA_DIR        = "data/bench01_extended"
const FIG_DIR         = "figures/bench01_extended"

# ═══════════════════════════════════════════════════════════════════
# Run all simulations
# ═══════════════════════════════════════════════════════════════════
function run_all_simulations()
    mkpath(DATA_DIR); mkpath(FIG_DIR)

    n_L   = length(SYSTEM_SIZES)
    n_tau = length(TAU_Q_VALUES)

    kink_data = zeros(n_L, n_tau)
    k1_data   = zeros(n_L, n_tau)
    k2_data   = zeros(n_L, n_tau)
    k3_data   = zeros(n_L, n_tau)
    time_data = zeros(n_L, n_tau)
    nsteps_used = fill(N_STEPS_DEFAULT, n_L, n_tau)
    chi_used    = fill(CHI_MAX_DEFAULT, n_L, n_tau)

    println("=" ^ 70)
    println("  Benchmark 1 Extended: KZM Scaling + Fast-Quench Plateau")
    println("=" ^ 70)
    println("  L = $SYSTEM_SIZES,  τ_Q = $TAU_Q_VALUES")
    println("  Default: n_steps=$N_STEPS_DEFAULT, χ_max=$CHI_MAX_DEFAULT")
    println("=" ^ 70)
    flush(stdout)

    # Per-L adaptive parameter reduction
    reduced = Dict{Int, Tuple{Int,Int}}()

    for (i_L, L) in enumerate(SYSTEM_SIZES)
        # For L=100, start with reduced parameters immediately
        if L >= 100
            reduced[L] = (1000, 64)
            println("  [L=$L] Using reduced parameters: n_steps=1000, χ=64")
            flush(stdout)
        end

        for (i_tau, tau_Q) in enumerate(TAU_Q_VALUES)
            n_steps = N_STEPS_DEFAULT
            chi_max = CHI_MAX_DEFAULT
            if haskey(reduced, L)
                n_steps, chi_max = reduced[L]
            end
            nsteps_used[i_L, i_tau] = n_steps
            chi_used[i_L, i_tau] = chi_max

            t_start = now()

            model    = TFIM(L; J=J, bc=BC)
            schedule = LinearQuench(tau_Q, J)
            tsched   = uniform_schedule(tau_Q, n_steps)
            psi      = initial_state(model)

            psi = evolve_tebd!(psi, model, schedule, tsched;
                               chi_max=chi_max, cutoff=1e-12)

            n_kink = kink_density(psi, model)
            k1, k2, k3 = kink_cumulants(psi, model; chi_max=chi_max)

            elapsed = Dates.value(now() - t_start) / 1000.0

            kink_data[i_L, i_tau] = n_kink
            k1_data[i_L, i_tau]   = k1
            k2_data[i_L, i_tau]   = k2
            k3_data[i_L, i_tau]   = k3
            time_data[i_L, i_tau] = elapsed

            tag = haskey(reduced, L) ? " [reduced]" : ""
            @printf("  L=%3d  τ_Q=%6.2f  ⟨n⟩=%.6f  κ₁=%7.3f  κ₂=%7.3f  κ₃=%7.3f  κ₂/κ₁=%6.4f  (%5.1fs)%s\n",
                    L, tau_Q, n_kink, k1, k2, k3, k2/k1, elapsed, tag)
            flush(stdout)

            if elapsed > MAX_TIME && !haskey(reduced, L)
                reduced[L] = (1000, 64)
                println("  ⚠ Exceeded $(Int(MAX_TIME))s — reducing params for L=$L")
                flush(stdout)
            end
        end
        println()
    end

    return kink_data, k1_data, k2_data, k3_data, time_data, nsteps_used, chi_used
end

# ═══════════════════════════════════════════════════════════════════
# Least-squares power-law fit on log-log data
# ═══════════════════════════════════════════════════════════════════
function loglog_fit(x, y; x_min=-Inf, x_max=Inf)
    mask = (x .>= x_min) .& (x .<= x_max) .& (y .> 0)
    lx = log.(x[mask]); ly = log.(y[mask])
    n = length(lx)
    sx = sum(lx); sy = sum(ly); sxx = sum(lx.^2); sxy = sum(lx .* ly)
    slope = (n * sxy - sx * sy) / (n * sxx - sx^2)
    intercept = (sy - slope * sx) / n
    return slope, exp(intercept)   # y = A * x^slope
end

# ═══════════════════════════════════════════════════════════════════
# Publication plots
# ═══════════════════════════════════════════════════════════════════
function make_plots(k1_data, k2_data, k3_data, kink_data)
    # Use the largest available L (last row)
    i_best = size(k1_data, 1)
    L_best = SYSTEM_SIZES[i_best]

    k1 = k1_data[i_best, :]
    k2 = k2_data[i_best, :]
    k3 = k3_data[i_best, :]
    nd = kink_data[i_best, :]
    taus = TAU_Q_VALUES

    # Common plot defaults
    default(; fontfamily="Computer Modern",
              titlefontsize=14, guidefontsize=13, tickfontsize=11,
              legendfontsize=10, framestyle=:box, grid=false,
              foreground_color_legend=nothing,
              background_color_legend=:white, dpi=300)

    # ───────────────────────────────────────────────────────────────
    # Plot 1 — del Campo style: all three cumulants vs τ_Q
    # ───────────────────────────────────────────────────────────────
    # Fit in scaling regime (τ_Q ∈ [1, 50])
    α1, A1 = loglog_fit(taus, k1; x_min=1.0, x_max=50.0)
    α2, A2 = loglog_fit(taus, k2; x_min=1.0, x_max=50.0)
    α3, A3 = loglog_fit(taus, abs.(k3); x_min=1.0, x_max=50.0)

    tau_fit = range(0.5, 150; length=100)

    p1 = plot(; xlabel=L"\tau_Q", ylabel=L"\kappa_n",
              title=latexstring("Kink number cumulants (\$L=$L_best\$)"),
              xscale=:log10, yscale=:log10, size=(650, 500),
              xlims=(0.05, 200), legend=:topright)

    scatter!(p1, taus, k1; marker=:circle, ms=6, mc=:royalblue, msc=:royalblue,
             label=latexstring("\\kappa_1\\ (\\alpha=$(@sprintf("%.2f", α1)))"))
    scatter!(p1, taus, k2; marker=:square, ms=6, mc=:firebrick, msc=:firebrick,
             label=latexstring("\\kappa_2\\ (\\alpha=$(@sprintf("%.2f", α2)))"))
    # κ₃ can be negative; plot |κ₃| for log scale
    k3_pos = [k > 0 ? k : NaN for k in k3]
    scatter!(p1, taus, k3_pos; marker=:utriangle, ms=6, mc=:forestgreen, msc=:forestgreen,
             label=latexstring("|\\kappa_3|\\ (\\alpha=$(@sprintf("%.2f", α3)))"))

    plot!(p1, tau_fit, A1 .* tau_fit .^ α1; lw=1.5, ls=:dash, color=:royalblue,  label="")
    plot!(p1, tau_fit, A2 .* tau_fit .^ α2; lw=1.5, ls=:dash, color=:firebrick,  label="")
    plot!(p1, tau_fit, A3 .* tau_fit .^ α3; lw=1.5, ls=:dash, color=:forestgreen, label="")

    # Reference slope -0.5
    plot!(p1, tau_fit, 0.8*maximum(k1) .* (tau_fit ./ tau_fit[1]) .^ (-0.5);
          lw=2, ls=:dot, color=:gray40, label=L"\propto \tau_Q^{-1/2}")

    savefig(p1, joinpath(FIG_DIR, "cumulants_delcampo.png"))
    savefig(p1, joinpath(FIG_DIR, "cumulants_delcampo.pdf"))
    println("  Plot 1 saved: cumulants_delcampo.{png,pdf}")

    # ───────────────────────────────────────────────────────────────
    # Plot 2 — Zeng et al. style: plateau + scaling in ⟨n⟩
    # ───────────────────────────────────────────────────────────────
    α_n, A_n = loglog_fit(taus, nd; x_min=1.0, x_max=50.0)
    # Plateau value ≈ mean of fastest quenches
    n_plateau = mean(nd[taus .< 0.5])

    p2 = plot(; xlabel=L"\tau_Q", ylabel=L"\langle \hat{n} \rangle",
              title=latexstring("Kink density (\$L=$L_best\$)"),
              xscale=:log10, yscale=:log10, size=(650, 500),
              xlims=(0.05, 200), legend=:topright)

    scatter!(p2, taus, nd; marker=:circle, ms=7, mc=:royalblue, msc=:royalblue,
             label=latexstring("TEBD (\$L=$L_best\$)"))

    # Scaling fit line
    tau_scale = range(0.8, 150; length=100)
    plot!(p2, tau_scale, A_n .* tau_scale .^ α_n;
          lw=2, ls=:dash, color=:firebrick,
          label=latexstring("\\propto \\tau_Q^{$(@sprintf("%.2f", α_n))}"))

    # Plateau line
    hline!(p2, [n_plateau]; lw=1.5, ls=:dot, color=:gray50,
           label=@sprintf("plateau ≈ %.3f", n_plateau))

    savefig(p2, joinpath(FIG_DIR, "kink_density_zeng.png"))
    savefig(p2, joinpath(FIG_DIR, "kink_density_zeng.pdf"))
    println("  Plot 2 saved: kink_density_zeng.{png,pdf}")

    # ───────────────────────────────────────────────────────────────
    # Plot 3 — Universal cumulant ratios
    # ───────────────────────────────────────────────────────────────
    r21 = k2 ./ k1
    r31 = k3 ./ k1

    p3 = plot(; xlabel=L"\tau_Q",
              ylabel="Cumulant ratio",
              title=latexstring("Universal cumulant ratios (\$L=$L_best\$)"),
              xscale=:log10, size=(650, 500),
              xlims=(0.05, 200), legend=:topright)

    scatter!(p3, taus, r21; marker=:circle, ms=7, mc=:royalblue, msc=:royalblue,
             label=L"\kappa_2/\kappa_1")
    scatter!(p3, taus, r31; marker=:square, ms=7, mc=:firebrick, msc=:firebrick,
             label=L"\kappa_3/\kappa_1")

    # Universal predictions
    κ21_pred = 2 - sqrt(2)
    κ31_pred = 4 * (1 - 3/sqrt(2) + 2/sqrt(3))
    hline!(p3, [κ21_pred]; lw=2, ls=:dash, color=:royalblue, alpha=0.6,
           label=@sprintf("%.4f  (2−√2)", κ21_pred))
    hline!(p3, [κ31_pred]; lw=2, ls=:dash, color=:firebrick, alpha=0.6,
           label=@sprintf("%.4f  (universal)", κ31_pred))

    savefig(p3, joinpath(FIG_DIR, "cumulant_ratios_universal.png"))
    savefig(p3, joinpath(FIG_DIR, "cumulant_ratios_universal.pdf"))
    println("  Plot 3 saved: cumulant_ratios_universal.{png,pdf}")

    return p1, p2, p3
end

# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
function main()
    kink_data, k1_data, k2_data, k3_data, time_data, nsteps_used, chi_used =
        run_all_simulations()

    # ── Raw data table ─────────────────────────────────────────────
    println("=" ^ 70)
    println("  Raw Data Table")
    println("=" ^ 70)
    @printf("  %5s  %7s  %9s  %9s  %9s  %9s  %9s\n",
            "L", "τ_Q", "κ₁", "κ₂", "κ₃", "κ₂/κ₁", "κ₃/κ₁")
    println("  " * "-" ^ 65)
    for (i_L, L) in enumerate(SYSTEM_SIZES)
        for (i_tau, tau_Q) in enumerate(TAU_Q_VALUES)
            k1 = k1_data[i_L, i_tau]
            k2 = k2_data[i_L, i_tau]
            k3 = k3_data[i_L, i_tau]
            @printf("  %5d  %7.2f  %9.4f  %9.4f  %9.4f  %9.4f  %9.4f\n",
                    L, tau_Q, k1, k2, k3, k2/k1, k3/k1)
        end
        println()
    end

    # ── Save HDF5 ──────────────────────────────────────────────────
    results = Dict{String,Any}(
        "system_sizes" => Float64.(SYSTEM_SIZES),
        "tau_Q_values" => TAU_Q_VALUES,
        "kink_density" => kink_data,
        "kappa1" => k1_data, "kappa2" => k2_data, "kappa3" => k3_data,
        "elapsed_times" => time_data,
        "n_steps_used" => Float64.(nsteps_used),
        "chi_max_used" => Float64.(chi_used),
    )
    outfile = joinpath(DATA_DIR, "bench01_extended.h5")
    save_results(outfile, results)
    println("  HDF5 saved to $outfile")

    # ── Plots ──────────────────────────────────────────────────────
    println("\n  Generating publication plots...")
    flush(stdout)
    make_plots(k1_data, k2_data, k3_data, kink_data)

    println("\n  Benchmark 1 Extended complete.")
end

main()
