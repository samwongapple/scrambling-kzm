using Test
using ITensors
using ITensorMPS

include(joinpath(@__DIR__, "..", "src", "ScramblKZM.jl"))
using .ScramblKZM

@testset "ScramblKZM Tests" begin

    @testset "Quench Schedules" begin
        tau_Q = 10.0; J = 1.0
        lq = LinearQuench(tau_Q, J)
        @test f(lq, 0.0) ≈ 0.0
        @test f(lq, tau_Q) ≈ 1.0
        @test g(lq, 0.0) ≈ 1.0
        @test g(lq, tau_Q) ≈ 0.0
        @test f(lq, tau_Q/2) ≈ 0.5
        @test g(lq, tau_Q/2) ≈ 0.5
        @test t_critical(lq) ≈ tau_Q / 2

        qq = QuadraticQuench(tau_Q, J)
        @test f(qq, 0.0) ≈ 0.0
        @test g(qq, 0.0) ≈ 1.0
        @test f(qq, tau_Q) ≈ 1.0
        @test g(qq, tau_Q) ≈ 0.0
    end

    @testset "TFIM Model" begin
        L = 6
        model = TFIM(L; J=1.0, bc=:periodic)
        @test num_sites(model) == L
        @test length(sites(model)) == L

        psi0 = initial_state(model)
        @test length(psi0) == L
        for j in 1:L
            sx = expect(psi0, "Sx"; sites=j)
            @test abs(sx - 0.5) < 1e-10
        end

        H = hamiltonian_mpo(model, 0.0, 1.0)
        E0 = energy(psi0, H)
        @test abs(E0 - (-L)) < 1e-8
    end

    @testset "TimeSchedule" begin
        tau_Q = 10.0; N = 100
        ts = uniform_schedule(tau_Q, N)
        @test length(ts) == N
        @test ts.t_points[1] ≈ 0.0
        @test ts.t_points[end] ≈ tau_Q
        @test all(dt -> abs(dt - tau_Q/N) < 1e-14, ts.dt_values)

        t_pts = [0.0, 1.0, 3.0, 6.0, 10.0]
        cs = custom_schedule(t_pts)
        @test length(cs) == 4
        @test cs.dt_values ≈ [1.0, 2.0, 3.0, 4.0]
    end

    @testset "Gate Construction" begin
        model = TFIM(4; J=1.0, bc=:open)
        gates = make_pf1_gates(model, 0.5, 0.5, 0.01)
        @test length(gates) == 7  # 3 ZZ + 4 X
    end

    @testset "Kink Density" begin
        L = 6; model = TFIM(L; J=1.0, bc=:open)
        psi = initial_state(model)
        @test abs(kink_density(psi, model) - 0.5) < 1e-10
        per_bond = kink_density_per_bond(psi, model)
        @test length(per_bond) == L - 1
        @test all(k -> abs(k - 0.5) < 1e-10, per_bond)
    end

    @testset "Kink Cumulants" begin
        L = 6; model = TFIM(L; J=1.0, bc=:open)
        psi = initial_state(model)
        k1, k2, k3 = kink_cumulants(psi, model)
        N_bonds = L - 1
        @test abs(k1 - N_bonds * 0.5) < 1e-10
        @test abs(k2 - N_bonds * 0.25) < 1e-10
        @test abs(k3) < 1e-10
    end

    @testset "Energy Observable" begin
        model = TFIM(4; J=1.0, bc=:open)
        psi = initial_state(model)
        H = hamiltonian_mpo(model, 0.5, 0.5)
        @test isa(energy(psi, H), Float64)
        @test energy_variance(psi, H) >= -1e-10
    end

    @testset "Entanglement" begin
        L = 6; model = TFIM(L; J=1.0, bc=:open)
        psi = initial_state(model)
        @test abs(half_chain_entropy(psi)) < 1e-10
        S_bonds = bond_entropies(psi)
        @test length(S_bonds) == L - 1
        @test all(s -> abs(s) < 1e-10, S_bonds)
    end

    @testset "TEBD Evolution" begin
        model = TFIM(4; J=1.0, bc=:open)
        schedule = LinearQuench(1.0, 1.0)
        ts = uniform_schedule(1.0, 10)
        psi = initial_state(model)
        psi = evolve_tebd!(psi, model, schedule, ts; chi_max=16, cutoff=1e-10)
        @test abs(norm(psi) - 1.0) < 1e-6
        @test 0.0 < kink_density(psi, model) < 1.0
    end

    @testset "Trotter Evolve" begin
        model = TFIM(4; J=1.0, bc=:open)
        schedule = LinearQuench(1.0, 1.0)
        ts = uniform_schedule(1.0, 20)

        psi1 = evolve_tebd!(initial_state(model), model, schedule, ts; chi_max=16, cutoff=1e-10)
        psi2 = trotter_evolve!(initial_state(model), model, schedule, ts; chi_max=16, cutoff=1e-10)
        @test abs(kink_density(psi1, model) - kink_density(psi2, model)) < 1e-10

        times = Float64[]
        obs(psi, t, step) = push!(times, t)
        trotter_evolve!(initial_state(model), model, schedule, ts; chi_max=16, observer_fn=obs)
        @test length(times) == 20
    end

    @testset "Error Operator — [H_Z, H_X]" begin
        L = 6; model = TFIM(L; J=1.0, bc=:open)
        C = build_hz_hx_commutator(model)
        @test isa(C, MPO)
        @test length(C) == L

        # Check that it's anti-Hermitian (purely imaginary eigenvalues)
        # C† = -C for anti-Hermitian, so ⟨ψ|C|ψ⟩ should be purely imaginary
        psi = initial_state(model)
        val = inner(psi', C, psi)
        @test abs(real(val)) < 1e-10  # real part should vanish
    end

    @testset "Kink ZZ MPO" begin
        L = 6; model = TFIM(L; J=1.0, bc=:open)
        C_zz = build_kink_zz_mpo(model)
        psi = initial_state(model)
        # ⟨+|σ^z_i σ^z_{i+1}|+⟩ = 0 for product state
        @test abs(real(inner(psi', C_zz, psi))) < 1e-10
    end

    @testset "Scrambling Bound" begin
        L = 4; model = TFIM(L; J=1.0, bc=:open)
        C = build_hz_hx_commutator(model)
        psi = initial_state(model)

        # ⟨ψ|C†C|ψ⟩ should be non-negative
        B = scrambling_bound(psi, C)
        @test B >= -1e-10
    end

    @testset "Operator-Induced Entropy" begin
        L = 6; model = TFIM(L; J=1.0, bc=:open)
        psi = initial_state(model)

        # Apply transverse field H_X to product state — should create entanglement
        H_x = hamiltonian_mpo(model, 0.0, 1.0)
        S = operator_induced_entropy(psi, H_x; maxdim=32)
        @test S >= 0.0  # entropy is non-negative
    end

    @testset "Trotter Error with Scrambling Bounds" begin
        L = 4; model = TFIM(L; J=1.0, bc=:open)
        schedule = LinearQuench(2.0, 1.0)

        # Evolve to non-trivial state
        psi = initial_state(model)
        psi = evolve_tebd!(psi, model, schedule, uniform_schedule(1.0, 50);
                          chi_max=32, cutoff=1e-12)

        # Build commutator MPOs
        C_mpo = build_hz_hx_commutator(model)
        C_zz = build_kink_zz_mpo(model)
        # [n_hat, C] = [-C_zz/2, C] for the kink number
        comm_nC = build_scrambling_operator(C_zz, C_mpo; cutoff=1e-10)

        result = compute_step_errors(psi, model, schedule, 1.0, 0.2;
                                     chi_max=32, chi_max_ref=32, n_substeps=50,
                                     comm_nC=comm_nC, C_mpo=C_mpo)

        @test isa(result, StepErrorResult)
        @test result.conservation_check < 0.1
        @test !isnan(result.epsilon_H_bound)
        @test !isnan(result.epsilon_n_bound)
        @test result.epsilon_H_bound >= 0.0
        @test result.epsilon_n_bound >= 0.0
    end

    @testset "Time-Resolved Errors" begin
        model = TFIM(4; J=1.0, bc=:open)
        schedule = LinearQuench(1.0, 1.0)
        tr = time_resolved_errors(model, schedule, 1.0, 5;
                                  n_snapshots=5, chi_max=16,
                                  chi_max_ref=32, n_steps_ref=100)
        @test length(tr.t_snapshots) == 5
        @test abs(tr.n_exact[end] - tr.n_trotter[end]) > 0
    end

    @testset "Config Loading" begin
        tmp = tempname() * ".toml"
        open(tmp, "w") do io
            write(io, "[model]\ntype = \"TFIM\"\nJ = 1.0\n[simulation]\nchi_max = 128\n")
        end
        cfg = load_config(tmp)
        @test cfg["model"]["type"] == "TFIM"
        @test cfg["simulation"]["chi_max"] == 128
        rm(tmp)
    end

end
