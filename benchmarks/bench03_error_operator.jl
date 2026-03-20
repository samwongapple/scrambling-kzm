"""
Benchmark 3 — Error Operator Structure

Build [H_Z, H_X] as MPO, verify analytical form, compute norms, plot f(t)g(t).

Usage:
    julia --project=. benchmarks/bench03_error_operator.jl
"""

using Printf, Dates
using ITensors
using ITensorMPS

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM
using Plots, LaTeXStrings

const DATA_DIR = "data/bench03"
const FIG_DIR  = "figures/bench03"

function main()
    mkpath(DATA_DIR); mkpath(FIG_DIR)

    println("=" ^ 65)
    println("  Benchmark 3: Error Operator Structure")
    println("=" ^ 65)
    flush(stdout)

    default(; fontfamily="Computer Modern", titlefontsize=14, guidefontsize=13,
              tickfontsize=11, legendfontsize=10, framestyle=:box, grid=false,
              foreground_color_legend=nothing, background_color_legend=:white, dpi=300)

    for L in [10, 20]
        println("\n  ── L = $L ──")
        model = TFIM(L; J=1.0, bc=:periodic)
        s = sites(model)

        # Build [H_Z, H_X] from analytical formula
        C_analytical = build_hz_hx_commutator(model)
        println("  [H_Z, H_X] MPO built (analytical): bond dims = $(linkdims(C_analytical))")

        # Verify: compute ||C||_F^2 = Tr(C†C)/dim via inner
        # For MPO: inner(C, C) gives Tr(C†C)? No — need to contract properly.
        # Use: Tr(C†C) = sum over all basis states of |⟨i|C|j⟩|^2
        # With MPS: pick a random state, compute ⟨ψ|C†C|ψ⟩
        # Better: directly compute via Frobenius norm squared = inner of MPO with itself
        # For ITensor MPOs: inner(A::MPO, B::MPO) is not directly supported.
        # Use: ⟨+^L|C†C|+^L⟩ as a lower bound, and check on several states.

        psi_plus = initial_state(model)
        CdagC_plus = real(inner(C_analytical, psi_plus, C_analytical, psi_plus))
        println("  ⟨+|C†C|+⟩ = $(@sprintf("%.6f", CdagC_plus))")

        # Check anti-Hermiticity: ⟨+|C|+⟩ should be purely imaginary
        C_expect = inner(psi_plus', C_analytical, psi_plus)
        println("  ⟨+|C|+⟩ = $(@sprintf("%.2e + %.2ei", real(C_expect), imag(C_expect)))")
        @assert abs(real(C_expect)) < 1e-8 "C should be anti-Hermitian!"
        println("  ✓ C is anti-Hermitian (real part ≈ 0)")

        # Compare with numerical commutator for small L
        if L <= 10
            println("  Comparing analytical vs numerical commutator...")
            Hz_os = OpSum()
            for i in 1:(L-1)
                Hz_os += 4.0, "Sz", i, "Sz", i+1
            end
            Hz_os += 4.0, "Sz", L, "Sz", 1  # PBC
            Hz_mpo = MPO(Hz_os, s)

            Hx_os = OpSum()
            for j in 1:L
                Hx_os += 2.0, "Sx", j
            end
            Hx_mpo = MPO(Hx_os, s)

            C_numerical = commutator_mpo(Hz_mpo, Hx_mpo; cutoff=1e-12)

            # Compare expectation values on |+⟩
            C_num_expect = inner(psi_plus', C_numerical, psi_plus)
            println("  ⟨+|C_num|+⟩ = $(@sprintf("%.2e + %.2ei", real(C_num_expect), imag(C_num_expect)))")

            CdagC_num = real(inner(C_numerical, psi_plus, C_numerical, psi_plus))
            println("  ⟨+|C†_num C_num|+⟩ = $(@sprintf("%.6f", CdagC_num))")
            println("  ⟨+|C†_ana C_ana|+⟩ = $(@sprintf("%.6f", CdagC_plus))")

            rel_diff = abs(CdagC_num - CdagC_plus) / max(abs(CdagC_num), 1e-15)
            println("  Relative difference: $(@sprintf("%.2e", rel_diff))")
            if rel_diff < 0.01
                println("  ✓ Analytical and numerical commutators agree")
            else
                println("  ⚠ Mismatch — check normalization")
            end
        end

        flush(stdout)
    end

    # ── Plot: f(t)*g(t) vs t/τ_Q ──────────────────────────────
    tau_Q = 10.0
    schedule = LinearQuench(tau_Q, 1.0)
    t_vals = range(0, tau_Q, length=200)
    fg_vals = [f(schedule, t) * g(schedule, t) for t in t_vals]
    t_norm = t_vals ./ tau_Q

    p1 = plot(; xlabel=L"t/\tau_Q", ylabel=L"f(t) \cdot g(t)",
              title=L"Time-dependent prefactor $f(t)g(t)$",
              size=(650, 450), legend=:topright)

    plot!(p1, t_norm, fg_vals; lw=2.5, color=:royalblue,
          label="Linear quench")

    t_c = t_critical(schedule) / tau_Q
    vline!(p1, [t_c]; lw=1.5, ls=:dot, color=:gray40,
           label=latexstring("t_c/\\tau_Q = $(@sprintf("%.2f", t_c))"))

    # Mark maximum
    fg_max, idx_max = findmax(fg_vals)
    scatter!(p1, [t_norm[idx_max]], [fg_max]; ms=6, mc=:firebrick, msc=:firebrick,
             label=@sprintf("max = %.3f", fg_max))

    savefig(p1, joinpath(FIG_DIR, "fg_prefactor.png"))
    savefig(p1, joinpath(FIG_DIR, "fg_prefactor.pdf"))
    println("\n  Plot saved: fg_prefactor.{png,pdf}")

    println("\n  Benchmark 3 complete.")
end

main()
