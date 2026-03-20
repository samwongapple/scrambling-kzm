# ScramblKZM — Research Notes

## Phase 1: Core Infrastructure (Complete)
- 17 files created, 44 tests passing
- End-to-end pipeline verified with L=10 KZM quench

## Benchmark 1: Exact KZM Scaling (Complete)
- Tested L = 10, 20, 40 with tau_Q = 1, 2, 5, 10, 20, 50
- KZM exponent alpha converges toward 0.5 with increasing L:
  - L=10: alpha = -1.04 (strong finite-size effects)
  - L=20: alpha = -0.56
  - L=40: alpha = -0.53
- Cumulant ratio kappa_2/kappa_1 matches universal prediction 2-sqrt(2) = 0.586:
  - L=40, tau_Q=10: kappa_2/kappa_1 = 0.585 (excellent match)
  - Deviates at small L (finite-size) and small tau_Q (fast-quench breakdown)
- L=40 with tau_Q >= 20 used reduced parameters (n_steps=1000, chi_max=64) due to runtime

## Benchmark 2: Uniform Trotter Errors (Complete)
- L=20, tau_Q=10, N_steps = 10, 20, 50, 100, 200, 500
- Convergence rate: err(<n>) ~ N_steps^{-1.38} (faster than expected N^{-1} for PF1)
- Cumulant ratio convergence slower: needs N >= 200 for 0.05% accuracy
- Time-resolved errors show CUMULATIVE error profiles:
  - Energy error peaks post-critically (t/tau_Q ~ 0.6) due to growing energy scale f(t)
  - Kink density error stays small throughout — Trotter errors affect phases more than populations
  - NOTE: These are cumulative errors (two different evolution histories), NOT per-step errors

## Phase 3: Analysis Infrastructure (Complete)
- 4 new source files, 59 total tests passing
- error_operator.jl: [H_Z, H_X] analytical MPO matches numerical (relative error 7e-16)
- scrambling.jl: scrambling_bound and scrambling_profile working
- operator_entanglement.jl: operator_induced_entropy working

## Benchmark 3: Error Operator Structure (Complete)
- [H_Z, H_X] verified to match analytical form 2i * J * sum_i(sigma^y sigma^z + sigma^z sigma^y)
- Anti-Hermiticity verified: <psi|C|psi> is purely imaginary
- <+|C^dag C|+> = 16L (L=10: 160, L=20: 320)
- f(t)*g(t) prefactor peaks at t_c

## Benchmark 4: The Blind Spot — KEY RESULTS AND OBSERVATIONS

### What we verified:
- Conservation check: max 1.2e-02 (piecewise conservation holds)
  - Does NOT improve with more substeps (200 vs 100) — limited by frozen Hamiltonian
    approximation H_n vs Magnus H_[inf], not by substep accuracy
  - |Delta_E - epsilon_H| tracks |E_exact - E_before| perfectly, confirming both arise
    from the same source (Magnus correction)
- Delta_E ≈ epsilon_H: mean relative difference 1.8% (confirmed)
- Scrambling bounds valid: 0/40 energy violations, 0/40 kink violations
- Chain of trust established: bound works for energy (verifiable) → trust it for kink density

### What we observed (differs from initial predictions):
1. epsilon_H > epsilon_n EVERYWHERE (not just away from t_c)
   - Energy error is always larger than kink density error
   - This is because many non-critical modes contribute O(1) energies to epsilon_H
   - The "blind spot" is not that epsilon_H is small at t_c, but that epsilon_H at t_c is
     dominated by the WRONG modes (non-critical ones that don't produce defects)

2. Different temporal profiles (THIS IS THE KEY RESULT):
   - epsilon_n peaks at t/tau_Q ≈ 0.37 (before t_c) and drops sharply post-critically
   - epsilon_H peaks at t/tau_Q ≈ 0.65 (after t_c) and stays elevated post-critically
   - This means energy-adaptive schedule wastes steps post-critically where kink errors are tiny
   - Kink-adaptive schedule concentrates steps in pre-critical region where defects are decided

3. Scrambling bound looseness:
   - Energy bound moderately tight: epsilon_H / epsilon_H^bound ≈ 0.1-0.6
   - Kink bound very loose: epsilon_n / epsilon_n^bound ≈ 0.01-0.03 (30-100x overestimate)
   - Looseness likely from Cauchy-Schwarz step in Feng et al. derivation

### Shape correspondence test:
- Kink density: correlation = 0.936 between normalized actual error and normalized bound
  → Bound captures the correct shape, schedule optimization will work
- Energy: correlation = 0.859, but shapes visually quite different
  → Energy bound peaks early while actual energy error peaks late
  → Post-critical energy error has contributions beyond leading-order commutator

### Size dependence (L = 10, 20, 40, fixed dt=0.5, tau_Q=10):
- Kink density error profile is ROBUST across system sizes:
  - Same peak location (t/tau_Q ≈ 0.35-0.4), same sharp post-critical drop
  - Magnitude grows mildly with L
  - NOT a finite-size artifact — the differential sensitivity is physical
- Energy error grows EXTENSIVELY with L (proportional to L):
  - L=10: peak ~0.05, L=20: peak ~0.5, L=40: peak ~3.0
  - This means energy-adaptive becomes INCREASINGLY suboptimal at larger L
  - More and more steps wasted fighting extensive energy error in post-critical region

### tau_Q dependence — FIRST ATTEMPT (varying dt = tau_Q/20):
- CONFOUNDED by varying dt: tau_Q=5 used dt=0.25, tau_Q=20 used dt=1.0
- tau_Q=20 showed anomalous post-critical peak — artifact of large dt=1.0 being outside
  perturbative regime, not physical
- Lesson: must fix dt when comparing across tau_Q

### tau_Q dependence — SECOND ATTEMPT (fixed dt = 0.25, L=20):
- Kink density profiles are CLEAN and UNIVERSAL:
  - All three tau_Q values (5, 10, 20) peak at t/tau_Q ≈ 0.35-0.45
  - All drop 1-2 orders of magnitude post-critically
  - Normalized shapes nearly collapse on top of each other
  - Peak did NOT narrow with tau_Q (likely because L=20 is finite-size limited:
    xi_hat ~ tau_Q^{1/2} ~ 4-5 approaches L=20)
  - Peak magnitudes similar (~0.02-0.03) since dt is fixed
- Energy error profiles are NOISY and NON-UNIVERSAL:
  - Oscillatory, tau_Q-dependent structure
  - Peaked post-critically but shape changes across tau_Q
  - No clean collapse in normalized plot
  - NOT a reliable or universal guide for step allocation

### Revised understanding of "blind spot":
The blind spot is NOT "energy error is small at t_c." It IS:
"The kink density Trotter error has a clean, universal, pre-critical profile directly tied to the
KZM impulse region where defects are formed. The energy Trotter error has a noisy, non-universal,
post-critical profile reflecting the extensive energy scale, not defect physics. Equalizing energy
error wastes steps in a regime irrelevant to defect formation."

### Implications for Phase 4:
- Two options for kink-adaptive schedule:
  Option 1: Use scrambling bound profile B_kink(t) — justified by 0.936 shape correlation
  Option 2: Use actual epsilon_n(t) from per-step comparison — more accurate, needs exact sim
- Should implement both and compare
- The energy-adaptive schedule will over-allocate post-critically, increasingly so at larger L
- The kink error shape universality means one schedule shape works across tau_Q values

## Benchmark 5: Entanglement Profile (Complete)
- Physical entropy S(psi) grows from ~0 through t_c, plateaus at ~1.4
- Operator-induced entanglement nonzero from the start:
  - S([H_Z,H_X]|psi>) ≈ 1.0 even at t=0 (operators create entanglement on product states)
  - S(n_hat|psi>) starts at ~1.0, dips to ~0.7 mid-quench, then grows
- Confirms Feng et al. picture: operator-induced entanglement provides partial error suppression
  even before physical entanglement develops

## Summary of Evidence Supporting Phase 4:
1. epsilon_n(t) and epsilon_H(t) have fundamentally different profiles (pre-critical vs post-critical)
2. The epsilon_n(t) profile is robust and universal across tau_Q and L
3. The scrambling bound captures the kink error shape well (corr = 0.936)
4. Energy error grows extensively with L — energy-adaptive becomes worse at larger systems
5. Energy error profile is noisy and non-universal — unreliable guide for step allocation
