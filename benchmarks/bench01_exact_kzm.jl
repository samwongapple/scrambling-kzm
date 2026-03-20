"""
Benchmark 1 — Exact KZM Scaling

Establish reference results by running TEBD with fine time steps and large
bond dimension. Verify KZM scaling ⟨n⟩ ~ τ_Q^{-1/2} and universal cumulant ratios.

Produces:
  - data/bench01/bench01_results.h5   (HDF5 results)
  - figures/bench01/kzm_scaling.png   (Plot 1: log-log ⟨n⟩ vs τ_Q)
  - figures/bench01/cumulant_ratios.png (Plot 2: κ₂/κ₁ vs τ_Q)
  - figures/bench01/alpha_vs_L.png    (Plot 3: fitted α vs L)

Usage:
    julia --project=. benchmarks/bench01_exact_kzm.jl [config_path]
"""

using Printf
using Dates

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM
using Plots

function run_benchmark(;
    system_sizes = [10, 20, 40],
    tau_Q_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0],
    J = 1.0,
    bc = :periodic,
    n_steps_default = 2000,
    chi_max_default = 128,
    max_time_per_run = 600.0,  # 10 minutes
    data_dir = "data/bench01",
    figure_dir = "figures/bench01"
)
    mkpath(data_dir)
    mkpath(figure_dir)

    println("=" ^ 65)
    println("  Benchmark 1: Exact KZM Scaling")
    println("=" ^ 65)
    println("  System sizes:  $system_sizes")
    println("  τ_Q values:    $tau_Q_values")
    println("  Default N_steps=$n_steps_default, χ_max=$chi_max_default")
    println("=" ^ 65)
    flush(stdout)

    n_L = length(system_sizes)
    n_tau = length(tau_Q_values)

    kink_data = zeros(n_L, n_tau)
    k1_data   = zeros(n_L, n_tau)
    k2_data   = zeros(n_L, n_tau)
    k3_data   = zeros(n_L, n_tau)
    time_data = zeros(n_L, n_tau)
    nsteps_used = fill(n_steps_default, n_L, n_tau)
    chi_used    = fill(chi_max_default, n_L, n_tau)

    # Track if we need to reduce parameters for large L
    reduced_params = Dict{Int, Tuple{Int,Int}}()  # L => (n_steps, chi_max) if reduced

    for (i_L, L) in enumerate(system_sizes)
        for (i_tau, tau_Q) in enumerate(tau_Q_values)
            # Determine parameters — reduce if previous run at this L exceeded limit
            n_steps = n_steps_default
            chi_max = chi_max_default
            if haskey(reduced_params, L)
                n_steps, chi_max = reduced_params[L]
            end
            nsteps_used[i_L, i_tau] = n_steps
            chi_used[i_L, i_tau] = chi_max

            t_start = now()

            model = TFIM(L; J=J, bc=bc)
            schedule = LinearQuench(tau_Q, J)
            time_sched = uniform_schedule(tau_Q, n_steps)
            psi = initial_state(model)

            psi = evolve_tebd!(psi, model, schedule, time_sched;
                              chi_max=chi_max, cutoff=1e-12)

            n = kink_density(psi, model)
            k1, k2, k3 = kink_cumulants(psi, model; chi_max=chi_max)

            elapsed = Dates.value(now() - t_start) / 1000.0

            kink_data[i_L, i_tau] = n
            k1_data[i_L, i_tau] = k1
            k2_data[i_L, i_tau] = k2
            k3_data[i_L, i_tau] = k3
            time_data[i_L, i_tau] = elapsed

            reduced_tag = haskey(reduced_params, L) ? " [reduced]" : ""
            @printf("  L=%3d  τ_Q=%5.1f  ⟨n⟩=%.6f  κ₂/κ₁=%7.4f  κ₃/κ₁=%8.5f  (%5.1fs)%s\n",
                    L, tau_Q, n, k2/k1, k3/k1, elapsed, reduced_tag)
            flush(stdout)

            # If this run exceeded limit, reduce params for remaining runs at this L
            if elapsed > max_time_per_run && !haskey(reduced_params, L)
                reduced_params[L] = (1000, 64)
                println("  ⚠ Run exceeded $(Int(max_time_per_run))s — reducing to n_steps=1000, χ=64 for L=$L")
                flush(stdout)
            end
        end
    end

    # ═══════════════════════════════════════════════════════════════
    # KZM Scaling Analysis
    # ═══════════════════════════════════════════════════════════════
    println("\n" * "=" ^ 65)
    println("  KZM Scaling Analysis: ⟨n⟩ = A · τ_Q^α")
    println("=" ^ 65)

    alphas = Float64[]
    for (i_L, L) in enumerate(system_sizes)
        log_tau = log.(tau_Q_values)
        log_n = log.(kink_data[i_L, :])
        n_pts = length(log_tau)
        sx = sum(log_tau); sy = sum(log_n)
        sxx = sum(log_tau .^ 2); sxy = sum(log_tau .* log_n)
        alpha = (n_pts * sxy - sx * sy) / (n_pts * sxx - sx^2)
        push!(alphas, alpha)
        @printf("  L=%3d: α = %.4f (expected: -0.50)\n", L, alpha)
    end

    # ═══════════════════════════════════════════════════════════════
    # Summary Table
    # ═══════════════════════════════════════════════════════════════
    println("\n" * "=" ^ 65)
    println("  Summary Table")
    println("=" ^ 65)
    @printf("  %5s  %6s  %10s  %9s  %9s  %10s\n",
            "L", "τ_Q", "⟨n⟩", "κ₂/κ₁", "κ₃/κ₁", "α_fitted")
    println("  " * "-" ^ 60)
    for (i_L, L) in enumerate(system_sizes)
        for (i_tau, tau_Q) in enumerate(tau_Q_values)
            @printf("  %5d  %6.1f  %10.6f  %9.4f  %9.5f  %10.4f\n",
                    L, tau_Q, kink_data[i_L, i_tau],
                    k2_data[i_L, i_tau] / k1_data[i_L, i_tau],
                    k3_data[i_L, i_tau] / k1_data[i_L, i_tau],
                    alphas[i_L])
        end
    end

    # ═══════════════════════════════════════════════════════════════
    # Save results to HDF5
    # ═══════════════════════════════════════════════════════════════
    results = Dict{String,Any}(
        "system_sizes" => Float64.(system_sizes),
        "tau_Q_values" => tau_Q_values,
        "J" => J,
        "kink_density" => kink_data,
        "kappa1" => k1_data,
        "kappa2" => k2_data,
        "kappa3" => k3_data,
        "alphas" => alphas,
        "elapsed_times" => time_data,
        "n_steps_used" => Float64.(nsteps_used),
        "chi_max_used" => Float64.(chi_used),
    )
    outfile = joinpath(data_dir, "bench01_results.h5")
    save_results(outfile, results)
    println("\n  Results saved to $outfile")

    # ═══════════════════════════════════════════════════════════════
    # Plot 1: KZM Scaling (log-log ⟨n⟩ vs τ_Q)
    # ═══════════════════════════════════════════════════════════════
    p1 = plot(; xlabel="τ_Q", ylabel="⟨n⟩ (kink density)",
              title="KZM Scaling: ⟨n⟩ vs τ_Q",
              xscale=:log10, yscale=:log10,
              legend=:topright, size=(700, 500), dpi=150)

    colors = [:blue, :red, :green, :orange, :purple]
    for (i_L, L) in enumerate(system_sizes)
        alpha = alphas[i_L]
        label = @sprintf("L=%d (α=%.3f)", L, alpha)
        plot!(p1, tau_Q_values, kink_data[i_L, :];
              marker=:circle, markersize=5, lw=2,
              color=colors[i_L], label=label)
    end

    # Reference line with slope -0.5
    tau_ref = range(tau_Q_values[1], tau_Q_values[end], length=50)
    n_ref = 0.5 .* tau_ref .^ (-0.5)  # approximate amplitude
    plot!(p1, tau_ref, n_ref; ls=:dash, lw=2, color=:black,
          label="∝ τ_Q^{-0.5} (KZM)")

    savefig(p1, joinpath(figure_dir, "kzm_scaling.png"))
    println("  Plot 1 saved to $(joinpath(figure_dir, "kzm_scaling.png"))")

    # ═══════════════════════════════════════════════════════════════
    # Plot 2: Cumulant Ratios (κ₂/κ₁ vs τ_Q)
    # ═══════════════════════════════════════════════════════════════
    p2 = plot(; xlabel="τ_Q", ylabel="κ₂/κ₁",
              title="Cumulant Ratio κ₂/κ₁ vs τ_Q",
              xscale=:log10, legend=:topright, size=(700, 500), dpi=150)

    for (i_L, L) in enumerate(system_sizes)
        ratios = k2_data[i_L, :] ./ k1_data[i_L, :]
        plot!(p2, tau_Q_values, ratios;
              marker=:circle, markersize=5, lw=2,
              color=colors[i_L], label="L=$L")
    end

    # Universal prediction
    hline!(p2, [2 - sqrt(2)]; ls=:dash, lw=2, color=:black,
           label=@sprintf("2-√2 = %.4f", 2-sqrt(2)))

    savefig(p2, joinpath(figure_dir, "cumulant_ratios.png"))
    println("  Plot 2 saved to $(joinpath(figure_dir, "cumulant_ratios.png"))")

    # ═══════════════════════════════════════════════════════════════
    # Plot 3: Finite-Size Convergence (α vs L)
    # ═══════════════════════════════════════════════════════════════
    p3 = plot(; xlabel="L (system size)", ylabel="α (fitted KZM exponent)",
              title="Finite-Size Convergence of KZM Exponent",
              legend=:topright, size=(700, 500), dpi=150)

    plot!(p3, system_sizes, abs.(alphas);
          marker=:circle, markersize=8, lw=2, color=:blue,
          label="|α| (fitted)")

    hline!(p3, [0.5]; ls=:dash, lw=2, color=:black,
           label="α = 0.5 (KZM prediction)")

    savefig(p3, joinpath(figure_dir, "alpha_vs_L.png"))
    println("  Plot 3 saved to $(joinpath(figure_dir, "alpha_vs_L.png"))")

    println("\n  Benchmark 1 complete.")
    return results
end

# Main entry point
if abspath(PROGRAM_FILE) == @__FILE__
    run_benchmark()
end
