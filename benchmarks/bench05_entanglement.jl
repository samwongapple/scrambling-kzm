"""
Benchmark 5 — Entanglement Profile

Map entanglement dynamics during KZM quench:
- Physical half-chain entropy S(ψ(t))
- Operator-induced entropy from n_hat and [H_Z, H_X]

Usage:
    julia --project=. benchmarks/bench05_entanglement.jl
"""

using Printf, Dates
using ITensorMPS

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM
using Plots, LaTeXStrings

const L           = 20
const TAU_Q       = 10.0
const J           = 1.0
const BC          = :periodic
const N_POINTS    = 40
const CHI_MAX     = 128
const N_STEPS_REF = 2000
const DATA_DIR    = "data/bench05"
const FIG_DIR     = "figures/bench05"

function main()
    mkpath(DATA_DIR); mkpath(FIG_DIR)

    model = TFIM(L; J=J, bc=BC)
    schedule = LinearQuench(TAU_Q, J)
    t_c = t_critical(schedule)

    println("=" ^ 65)
    println("  Benchmark 5: Entanglement Profile")
    println("=" ^ 65)
    println("  L=$L, τ_Q=$TAU_Q, $N_POINTS time points")
    println("=" ^ 65)
    flush(stdout)

    # Build operator MPOs
    println("\n  Building operator MPOs...")
    C_mpo = build_hz_hx_commutator(model)
    C_zz = build_kink_zz_mpo(model)
    println("  [H_Z, H_X]: bond dims = $(linkdims(C_mpo))")
    println("  C_zz:        bond dims = $(linkdims(C_zz))")
    flush(stdout)

    # Reference evolution, collect snapshots
    println("\n  Running reference evolution...")
    flush(stdout)
    t0 = now()

    sample_times = collect(range(TAU_Q / N_POINTS, TAU_Q, length=N_POINTS))
    psi_snapshots = Vector{MPS}(undef, N_POINTS)

    psi = initial_state(model)
    ref_ts = uniform_schedule(TAU_Q, N_STEPS_REF)

    snap_idx = 1
    function obs(psi_obs, t, step)
        while snap_idx <= N_POINTS && t >= sample_times[snap_idx] - 1e-10
            psi_snapshots[snap_idx] = copy(psi_obs)
            snap_idx += 1
        end
    end

    evolve_tebd!(psi, model, schedule, ref_ts;
                 chi_max=CHI_MAX, cutoff=1e-12, observer_fn=obs)
    elapsed = Dates.value(now() - t0) / 1000.0
    println("  Reference done ($(round(elapsed, digits=1))s)")
    flush(stdout)

    # Compute entropies at each snapshot
    println("\n  Computing entropies...")
    flush(stdout)

    S_physical = zeros(N_POINTS)
    S_nhat     = zeros(N_POINTS)
    S_comm     = zeros(N_POINTS)

    for i in 1:N_POINTS
        psi_snap = psi_snapshots[i]

        S_physical[i] = half_chain_entropy(psi_snap)
        S_nhat[i] = operator_induced_entropy(psi_snap, C_zz; maxdim=CHI_MAX)
        S_comm[i] = operator_induced_entropy(psi_snap, C_mpo; maxdim=CHI_MAX)

        if i % 10 == 0 || i == 1
            @printf("  [%2d/%d] t/τ=%.3f  S=%.4f  S(n̂|ψ⟩)=%.4f  S(C|ψ⟩)=%.4f\n",
                    i, N_POINTS, sample_times[i]/TAU_Q,
                    S_physical[i], S_nhat[i], S_comm[i])
            flush(stdout)
        end
    end

    # Save results
    ts_norm = sample_times ./ TAU_Q
    save_results(joinpath(DATA_DIR, "bench05_results.h5"), Dict{String,Any}(
        "t_norm" => ts_norm,
        "S_physical" => S_physical,
        "S_nhat" => S_nhat,
        "S_comm" => S_comm,
    ))
    println("\n  Data saved to $DATA_DIR/")

    # Plot
    println("  Generating plot...")
    flush(stdout)

    default(; fontfamily="Computer Modern", titlefontsize=14, guidefontsize=13,
              tickfontsize=11, legendfontsize=10, framestyle=:box, grid=false,
              foreground_color_legend=nothing, background_color_legend=:white, dpi=300)

    p1 = plot(; xlabel=L"t/\tau_Q",
              ylabel="Von Neumann entropy",
              title=latexstring("Entanglement dynamics (\$L=$L\$, \$\\tau_Q=$TAU_Q\$)"),
              size=(700, 500), legend=:topleft)

    plot!(p1, ts_norm, S_physical; lw=2.5, color=:royalblue,
          label=latexstring("S(|\\psi(t)\\rangle)"))
    plot!(p1, ts_norm, S_nhat; lw=2.5, color=:firebrick, ls=:dash,
          label=latexstring("S(\\hat{n}|\\psi\\rangle)"))
    plot!(p1, ts_norm, S_comm; lw=2.5, color=:forestgreen, ls=:dashdot,
          label=latexstring("S([H_Z,H_X]|\\psi\\rangle)"))

    vline!(p1, [t_c/TAU_Q]; lw=1.5, ls=:dot, color=:gray40,
           label=latexstring("t_c/\\tau_Q"))

    savefig(p1, joinpath(FIG_DIR, "entanglement_profile.png"))
    savefig(p1, joinpath(FIG_DIR, "entanglement_profile.pdf"))
    println("  Plot saved: entanglement_profile.{png,pdf}")

    println("\n  Benchmark 5 complete.")
end

main()
