# ScramblKZM — Implementation Plan

This document specifies the code structure, module interfaces, and benchmark specifications. See `PROJECT_DESCRIPTION.md` for the full physics context and motivation.

## 1. Technical Stack

- **Language:** Julia (≥ 1.10)
- **Core dependency:** ITensors.jl / ITensorMPS.jl (all MPS/MPO/TEBD operations)
- **Config:** TOML (Julia stdlib)
- **Data persistence:** HDF5.jl
- **Visualization:** Plots.jl or Makie.jl

## 2. Simulation Engine Design

All simulations use **TEBD** (Time-Evolving Block Decimation) through ITensorMPS. This provides a unified framework:

- **"Exact" reference:** TEBD with very small $\delta t$ (e.g., $\tau_Q/4000$) and large bond dimension $\chi$ (e.g., 256), verified by convergence checks
- **Trotterized simulation:** TEBD with PF1 gates at step sizes determined by the chosen schedule

The only difference between "exact" and "Trotter" is the step-size schedule — same code path, same framework, different `TimeSchedule` input.

**Trotter method:** We start with PF1 ($U_1 = e^{-if_n H_Z \delta t} e^{-ig_n H_X \delta t}$) for simplicity. The multiplicative error is $M = \frac{f_n g_n}{2}[H_Z, H_X]\delta t^2 + O(\delta t^3)$, involving a single commutator (2-local terms for TFIM). Higher-order formulas (PF2, etc.) can be added later by extending the gate construction module.

## 3. The Three Comparison Schemes

**Scheme A — Uniform:** All steps have equal size $\delta t = \tau_Q / N_{\text{steps}}$.

**Scheme B — Energy-adaptive:** Steps distributed based on per-step Trotter error in energy. We use **both** approaches to ensure correctness:

1. **Direct measurement (Zhao et al. style, no $M$ needed):** At each Trotter step $n$, measure the energy change:
   $$\Delta E_n = \langle \psi_{\text{trotter}}(t_{n+1}) | H_n | \psi_{\text{trotter}}(t_{n+1}) \rangle - \langle \psi_{\text{trotter}}(t_n) | H_n | \psi_{\text{trotter}}(t_n) \rangle$$
   Because energy is piecewise conserved under exact evolution, this directly equals the Trotter error: $\Delta E_n = \epsilon_{H,n}$. The schedule is optimized so that $|\Delta E_n|$ is equalized across all steps.

2. **Scrambling bound (Feng et al. style, with $M$):** Compute:
   $$\epsilon_{H,n}^{\text{bound}} = \sqrt{\langle \psi(t_n) | [H_n, M_n]^\dagger [H_n, M_n] | \psi(t_n) \rangle}$$
   where $M_n = \frac{f_n g_n}{2}[H_Z, H_X]\delta t_n^2$.

We verify that these two approaches are consistent: $\Delta E_n = \epsilon_{H,n} \leq \epsilon_{H,n}^{\text{bound}}$. This validates the scrambling bound framework before trusting it for kink density, where direct measurement is not possible.

**Scheme C — Kink-density-adaptive (our method):** Steps distributed based on the scrambling bound for $\hat{n}$:
$$\epsilon_{\hat{n},n}^{\text{bound}} = \sqrt{\langle \psi(t_n) | [\hat{n}, M_n]^\dagger [\hat{n}, M_n] | \psi(t_n) \rangle}$$
Same total $N_{\text{steps}}$, different distribution. Schedule optimized to equalize $\epsilon_{\hat{n},n}^{\text{bound}}$ across all steps.

## 4. Code Architecture

```
ScramblKZM.jl/
├── Project.toml
├── README.md
├── PROJECT_DESCRIPTION.md
├── IMPLEMENTATION_PLAN.md
│
├── configs/
│   ├── defaults.toml
│   ├── models/
│   │   ├── tfim_integrable.toml
│   │   └── tfim_j2.toml
│   └── benchmarks/
│       ├── bench01_exact_kzm.toml
│       ├── bench02_trotter_errors.toml
│       ├── bench03_error_operator.toml
│       ├── bench04_blind_spot.toml
│       ├── bench05_entanglement.toml
│       ├── bench06_energy_adaptive.toml
│       ├── bench07_kink_adaptive.toml
│       └── bench08_comparison.toml
│
├── src/
│   ├── ScramblKZM.jl
│   ├── models/
│   │   ├── abstract.jl
│   │   ├── quench.jl
│   │   ├── tfim.jl
│   │   └── tfim_j2.jl
│   ├── simulation/
│   │   ├── gates.jl
│   │   ├── tebd.jl
│   │   ├── trotter_evolve.jl
│   │   └── schedule.jl
│   ├── observables/
│   │   ├── kink_density.jl
│   │   ├── correlators.jl
│   │   ├── energy.jl
│   │   ├── entanglement.jl
│   │   └── operator_entanglement.jl
│   ├── analysis/
│   │   ├── error_operator.jl
│   │   ├── scrambling.jl
│   │   ├── trotter_error.jl
│   │   ├── adaptive.jl
│   │   └── comparison.jl
│   └── utils/
│       ├── config.jl
│       ├── io.jl
│       └── plotting.jl
│
├── benchmarks/
│   ├── run_benchmark.jl
│   ├── bench01_exact_kzm.jl
│   ├── bench02_trotter_errors.jl
│   ├── bench03_error_operator.jl
│   ├── bench04_blind_spot.jl
│   ├── bench05_entanglement.jl
│   ├── bench06_energy_adaptive.jl
│   ├── bench07_kink_adaptive.jl
│   └── bench08_comparison.jl
│
├── test/
│   └── runtests.jl
├── notebooks/
├── data/
└── figures/
```

## 5. Module Specifications

### 5.1 Models (`src/models/`)

**`abstract.jl`** — Interface that all models implement:
```julia
abstract type AbstractModel end

# Required methods:
sites(model::AbstractModel)::Vector{Index}           # ITensor site indices
hz_terms(model::AbstractModel)::OpSum                 # Ising interaction part
hx_terms(model::AbstractModel)::OpSum                 # Transverse field part
hamiltonian_mpo(model::AbstractModel, f, g)::MPO      # Full H(t) = -f*H_Z - g*H_X as MPO
initial_state(model::AbstractModel)::MPS              # |+>^L ground state of H(t=0)
num_sites(model::AbstractModel)::Int
```

**`quench.jl`** — Quench schedule definitions:
```julia
abstract type AbstractQuenchSchedule end

struct LinearQuench <: AbstractQuenchSchedule
    tau_Q::Float64
    J::Float64
end

struct QuadraticQuench <: AbstractQuenchSchedule
    tau_Q::Float64
    J::Float64
end

# Required methods:
f(schedule, t)::Float64              # Ising coupling coefficient at time t
g(schedule, t)::Float64              # Transverse field coefficient at time t
t_critical(schedule)::Float64        # Time when critical point is crossed
```

**`tfim.jl`** — Standard nearest-neighbor TFIM:
```julia
struct TFIM <: AbstractModel
    L::Int
    J::Float64
    bc::Symbol   # :periodic or :open
    sites::Vector{Index}
end
```

**`tfim_j2.jl`** — Non-integrable TFIM with next-nearest-neighbor:
```julia
struct TFIMJ2 <: AbstractModel
    L::Int
    J1::Float64
    J2::Float64
    bc::Symbol
    sites::Vector{Index}
end
```

### 5.2 Simulation (`src/simulation/`)

**`schedule.jl`** — Time discretization:
```julia
struct TimeSchedule
    t_points::Vector{Float64}   # [t_0, t_1, ..., t_N] with t_0=0, t_N=tau_Q
    dt_values::Vector{Float64}  # dt[k] = t_{k+1} - t_k, length N
end

uniform_schedule(tau_Q::Float64, N_steps::Int)::TimeSchedule
custom_schedule(t_points::Vector{Float64})::TimeSchedule
```

**`gates.jl`** — Construct ITensor gates for one Trotter step:
```julia
function make_pf1_gates(model::AbstractModel, f_val::Float64, g_val::Float64, dt::Float64)
    # Returns Vector{ITensor} of two-site and one-site gates
    # For TFIM: exp(-i f J sigma^z_i sigma^z_{i+1} dt) and exp(-i g sigma^x_j dt)
    # Uses ITensor's op() function to build local operators
end
```

**`tebd.jl`** — TEBD evolution wrapper:
```julia
function evolve_tebd!(
    psi::MPS,
    model::AbstractModel,
    schedule::AbstractQuenchSchedule,
    time_schedule::TimeSchedule;
    chi_max::Int = 256,
    cutoff::Float64 = 1e-12,
    observer_fn = nothing,          # Called at each step: observer_fn(psi, t, step)
    observer_times::Vector{Float64} = Float64[]
)::MPS
```

**`trotter_evolve.jl`** — Trotterized evolution (wraps tebd.jl with PF1 gates):
```julia
function trotter_evolve!(
    psi::MPS,
    model::AbstractModel,
    schedule::AbstractQuenchSchedule,
    time_schedule::TimeSchedule;
    chi_max::Int = 256,
    cutoff::Float64 = 1e-12,
    observer_fn = nothing
)::MPS
    # At each step: build PF1 gates, apply via ITensor's apply() with truncation
end
```

### 5.3 Observables (`src/observables/`)

**`kink_density.jl`**:
```julia
function kink_density(psi::MPS, model::AbstractModel)::Float64
    # <n> = (1/2N) sum_i (1 - <sigma^z_i sigma^z_{i+1}>)
end

function kink_density_per_bond(psi::MPS, model::AbstractModel)::Vector{Float64}
    # Returns <K_i> = (1 - <sigma^z_i sigma^z_{i+1}>)/2 for each bond
end

function kink_cumulants(psi::MPS, model::AbstractModel; max_order::Int=3)
    # Returns kappa_1, kappa_2, kappa_3
end
```

**`correlators.jl`**:
```julia
function kink_kink_correlator(psi::MPS, model::AbstractModel, r::Int)::Float64
    # C^KK_r = (1/L) sum_i (<K_i K_{i+r}> - <K_i><K_{i+r}>)
end

function kink_kink_correlator_profile(psi::MPS, model::AbstractModel)::Vector{Float64}
    # Returns C^KK_r for r = 1, ..., L/4
end
```

**`energy.jl`**:
```julia
function energy(psi::MPS, H_mpo::MPO)::Float64
    # <H> = inner(psi', H, psi)
end

function energy_variance(psi::MPS, H_mpo::MPO)::Float64
    # delta_E^2 = <H^2> - <H>^2
end
```

**`entanglement.jl`**:
```julia
function half_chain_entropy(psi::MPS)::Float64
function bond_entropies(psi::MPS)::Vector{Float64}
```

**`operator_entanglement.jl`**:
```julia
function operator_induced_entropy(psi::MPS, O_mpo::MPO)::Float64
    # |phi> = O|psi>, normalize, compute half-chain entropy
end
```

### 5.4 Analysis (`src/analysis/`)

**`error_operator.jl`** — Build the Trotter error structure as MPOs:
```julia
function commutator_mpo(A::MPO, B::MPO; cutoff=1e-12)::MPO
    # [A, B] = AB - BA
end

function build_hz_hx_commutator(model::AbstractModel)::MPO
    # [H_Z, H_X] as MPO — can build directly from OpSum for efficiency
end

function build_scrambling_operator(O_mpo::MPO, C_mpo::MPO; cutoff=1e-12)::MPO
    # [O, C] where C = [H_Z, H_X]
end
```

**`scrambling.jl`** — Evaluate scrambling bounds:
```julia
function scrambling_bound(psi::MPS, comm_OC::MPO)::Float64
    # Compute <psi| [O,C]^dag [O,C] |psi>
end

function scrambling_profile(
    psi_snapshots::Vector{MPS},
    t_values::Vector{Float64},
    comm_nC::MPO,                    # [n_hat, [H_Z, H_X]] (time-independent)
    model::AbstractModel,
    schedule::AbstractQuenchSchedule
)
    # Returns B_kink(t) and B_energy(t) at each snapshot time
    # For B_energy: rebuild [H(t), [H_Z,H_X]] at each t (time-dependent)
end
```

**`trotter_error.jl`** — Per-step error analysis with full verification hierarchy:
```julia
struct StepErrorResult
    # Energy quantities
    E_before::Float64           # <psi(t_n) | H_n | psi(t_n)>
    E_after_trotter::Float64    # <psi_trotter(t_{n+1}) | H_n | psi_trotter(t_{n+1})>
    E_after_exact::Float64      # <psi_exact(t_{n+1}) | H_n | psi_exact(t_{n+1})>
    delta_E::Float64            # |E_after_trotter - E_before| (direct, no M needed)
    epsilon_H::Float64          # |E_after_exact - E_after_trotter| (actual Trotter error)
    conservation_check::Float64 # |E_after_exact - E_before| (should be ≈ 0)
    epsilon_H_bound::Float64    # sqrt(<psi|[H,M]†[H,M]|psi>) (scrambling bound)

    # Kink density quantities
    n_after_trotter::Float64    # <psi_trotter | n_hat | psi_trotter>
    n_after_exact::Float64      # <psi_exact | n_hat | psi_exact>
    epsilon_n::Float64          # |n_exact - n_trotter| (actual Trotter error)
    epsilon_n_bound::Float64    # sqrt(<psi|[n,M]†[n,M]|psi>) (scrambling bound)
end

function compute_step_errors(
    psi_ref::MPS,                    # Reference state at time t_n
    model::AbstractModel,
    schedule::AbstractQuenchSchedule,
    t_n::Float64,
    dt_test::Float64;
    chi_max_ref::Int = 512,          # Bond dim for "exact" single step
    comm_HC::MPO = nothing,          # [H_n, [H_Z, H_X]] for scrambling bound
    comm_nC::MPO = nothing           # [n_hat, [H_Z, H_X]] for scrambling bound
)::StepErrorResult
    #
    # Full procedure:
    #
    # 1. Record E_before = <psi_ref | H_n | psi_ref>
    #
    # 2. Apply one PF1 Trotter step to copy of psi_ref -> psi_trotter
    #    Record E_after_trotter = <psi_trotter | H_n | psi_trotter>
    #    Record n_after_trotter = <psi_trotter | n_hat | psi_trotter>
    #
    # 3. Apply one "exact" step (fine TEBD with tiny dt) to copy of psi_ref -> psi_exact
    #    Record E_after_exact = <psi_exact | H_n | psi_exact>
    #    Record n_after_exact = <psi_exact | n_hat | psi_exact>
    #
    # 4. Compute derived quantities:
    #    delta_E = |E_after_trotter - E_before|
    #    epsilon_H = |E_after_exact - E_after_trotter|
    #    conservation_check = |E_after_exact - E_before|
    #    epsilon_n = |n_after_exact - n_after_trotter|
    #
    # 5. Verify: conservation_check ≈ 0 (piecewise conservation)
    #    This implies: delta_E ≈ epsilon_H
    #
    # 6. If comm_HC and comm_nC provided, compute scrambling bounds:
    #    epsilon_H_bound = sqrt(scrambling_bound(psi_ref, comm_HC)) * fg * dt^2
    #    epsilon_n_bound = sqrt(scrambling_bound(psi_ref, comm_nC)) * fg * dt^2
    #
    # 7. Verify: epsilon_H <= epsilon_H_bound, epsilon_n <= epsilon_n_bound
    #
end
```

**`adaptive.jl`** — Schedule optimization:
```julia
function energy_adaptive_schedule(
    tau_Q::Float64,
    N_steps::Int,
    model::AbstractModel,
    schedule::AbstractQuenchSchedule;
    chi_max::Int = 256,
    n_iterations::Int = 3,
    also_compute_scrambling::Bool = true   # Also compute epsilon_H^bound for verification
)::Tuple{TimeSchedule, Dict}
    #
    # Iterative method using direct Delta_E measurement:
    # 1. Start with uniform schedule
    # 2. Run Trotter evolution
    # 3. At each step, measure:
    #    - Delta_E_n (direct: energy before minus after, equals epsilon_H_n)
    #    - Optionally: epsilon_H_n^bound (scrambling bound, for verification)
    # 4. Redistribute steps to equalize |Delta_E_n|
    # 5. Repeat for n_iterations
    #
    # Returns: (optimized_schedule, diagnostics_dict)
    # diagnostics_dict contains delta_E_profile, epsilon_H_bound_profile, etc.
    #
end

function kink_adaptive_schedule(
    tau_Q::Float64,
    N_steps::Int,
    B_kink_profile::Vector{Float64},
    t_profile::Vector{Float64},
    schedule::AbstractQuenchSchedule
)::TimeSchedule
    #
    # Direct computation from scrambling bound profile:
    # 1. Step density: rho(t) ∝ B_kink(t)^{1/4} * (f(t)*g(t))^{1/2}
    # 2. Discretize to N_steps
    #
end
```

**`comparison.jl`** — Head-to-head:
```julia
function run_comparison(
    model::AbstractModel,
    schedule::AbstractQuenchSchedule,
    N_steps_values::Vector{Int};
    reference_data::Dict,
    chi_max::Int = 256
)
    # For each N_steps: run uniform, energy-adaptive, kink-adaptive
    # Measure KZM observables, compute errors vs reference
    # Return structured results
end
```

### 5.5 Utilities (`src/utils/`)

**`config.jl`**: TOML loading with defaults merging.
**`io.jl`**: HDF5 save/load for simulation data.
**`plotting.jl`**: Standard plot recipes for KZM scaling plots, schedule comparisons, etc.

## 6. Config File Schema

Example: `configs/benchmarks/bench01_exact_kzm.toml`
```toml
[model]
type = "TFIM"
J = 1.0
bc = "periodic"

[quench]
schedule = "linear"

[sweep]
system_sizes = [20, 40, 60, 80, 100]
tau_Q_values = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

[simulation]
method = "tebd"
n_steps = 4000
chi_max = 256
cutoff = 1e-12

[observables]
compute = ["kink_density", "kink_cumulants", "kink_correlator", "entanglement"]

[convergence]
chi_max_check = [128, 256, 512]
n_steps_check = [2000, 4000, 8000]

[output]
data_dir = "data/bench01"
figure_dir = "figures/bench01"
save_states = false
save_observables = true
```

## 7. Benchmark Specifications

### Benchmark 1 — Exact KZM Scaling

**Purpose:** Establish reference results.

**Parameters:** $L \in \{20,40,60,80,100\}$, $\tau_Q \in \{1,2,5,10,20,50,100\}$, $J=1$, linear schedule, periodic BC, $N_{\text{ref}}=4000$, $\chi=256$.

**Computed:** $\langle\hat{n}\rangle$, $\kappa_2$, $\kappa_3$, $C_r^{KK}$ for $r=1,\ldots,L/4$.

**Expected:** Log-log slope of $\langle\hat{n}\rangle$ vs $\tau_Q$ gives $\alpha \approx 0.5$. $\kappa_2/\kappa_1 \to 2-\sqrt{2}$ as $L$ increases.

**Convergence check:** Verify results don't change when doubling $N_{\text{ref}}$ or $\chi$.

---

### Benchmark 2 — Uniform Trotter Errors

**Purpose:** Quantify how uniform Trotterization distorts KZM at various gate budgets.

**Parameters:** $L=40$, $\tau_Q=20$, $N_{\text{steps}} \in \{10,20,50,100,200,500,1000\}$, PF1.

**Computed:** Error in $\langle\hat{n}\rangle$ and $\kappa_2/\kappa_1$ vs reference. Time-resolved snapshot of errors at ~50 intermediate times.

**Expected:** Errors decrease as $N_{\text{steps}}^{-1}$. Time-resolved errors peak near $t_c$.

---

### Benchmark 3 — Error Operator Structure

**Purpose:** Understand the structure of $M \propto [H_Z, H_X]$.

**Parameters:** $L \in \{10,20,40\}$.

**Computed:** Build $[H_Z, H_X]$ as MPO. Verify structure matches $2i\sum_i J(\sigma_i^y\sigma_{i+1}^z + \sigma_i^z\sigma_{i+1}^y)$. Spectral norm, Frobenius norm, ratio. Plot $f(t)g(t)$.

---

### Benchmark 4 — The Blind Spot and Verification Hierarchy

**Purpose:** Central motivating demonstration. Show the critical-mode blind spot AND validate the scrambling bound framework by verifying $\Delta E = \epsilon_H \leq \epsilon_H^{\text{bound}}$.

**Parameters:** $L=40$, $\tau_Q=20$, $\delta t_{\text{test}} = \tau_Q/50$, sample at ~200 time points.

**Pre-computation (done once):**
- Build $[H_Z, H_X]$ as MPO (call it $C$)
- Build $[\hat{n}, C]$ as MPO (for kink scrambling bound)
- At each time $t_n$: build $[H(t_n), C]$ as MPO (for energy scrambling bound — time-dependent because $H$ depends on $f_n, g_n$)

**At each time $t_n$, compute the full hierarchy of six quantities:**

**Energy quantities (three, should satisfy $\Delta E_n = \epsilon_{H,n} \leq \epsilon_{H,n}^{\text{bound}}$):**

1. $\Delta E_n$ = $|\langle\psi_{\text{trotter}}(t_{n+1})|H_n|\psi_{\text{trotter}}(t_{n+1})\rangle - \langle\psi(t_n)|H_n|\psi(t_n)\rangle|$
   — Direct measurement, no $M$ needed, no exact evolution needed

2. $\epsilon_{H,n}$ = $|\langle\psi_{\text{exact}}(t_{n+1})|H_n|\psi_{\text{exact}}(t_{n+1})\rangle - \langle\psi_{\text{trotter}}(t_{n+1})|H_n|\psi_{\text{trotter}}(t_{n+1})\rangle|$
   — Actual Trotter error, needs exact evolution

3. $\epsilon_{H,n}^{\text{bound}}$ = $\sqrt{\langle\psi(t_n)|[H_n, M]^\dagger[H_n, M]|\psi(t_n)\rangle}$
   — Scrambling bound, needs $M$, no exact evolution needed

**Plus verification:** Conservation check = $|\langle\psi_{\text{exact}}(t_{n+1})|H_n|\psi_{\text{exact}}(t_{n+1})\rangle - \langle\psi(t_n)|H_n|\psi(t_n)\rangle| \approx 0$

**Kink density quantities (two, should satisfy $\epsilon_{\hat{n},n} \leq \epsilon_{\hat{n},n}^{\text{bound}}$):**

4. $\epsilon_{\hat{n},n}$ = $|\langle\psi_{\text{exact}}(t_{n+1})|\hat{n}|\psi_{\text{exact}}(t_{n+1})\rangle - \langle\psi_{\text{trotter}}(t_{n+1})|\hat{n}|\psi_{\text{trotter}}(t_{n+1})\rangle|$
   — Actual Trotter error, needs exact evolution

5. $\epsilon_{\hat{n},n}^{\text{bound}}$ = $\sqrt{\langle\psi(t_n)|[\hat{n}, M]^\dagger[\hat{n}, M]|\psi(t_n)\rangle}$
   — Scrambling bound, needs $M$

**The key plot (six curves vs $t/\tau_Q$):**

Energy group:
- $\Delta E_n$ (circles) — direct measurement
- $\epsilon_{H,n}$ (squares) — actual Trotter error
- $\epsilon_{H,n}^{\text{bound}}$ (dashed) — scrambling bound

Kink density group:
- $\epsilon_{\hat{n},n}$ (squares) — actual Trotter error
- $\epsilon_{\hat{n},n}^{\text{bound}}$ (dashed) — scrambling bound

**Expected results:**
- Conservation check $\approx 0$ at all times (verifies piecewise conservation)
- $\Delta E_n \approx \epsilon_{H,n}$ at all times (consequence of conservation)
- $\epsilon_{H,n} \leq \epsilon_{H,n}^{\text{bound}}$ and $\epsilon_{\hat{n},n} \leq \epsilon_{\hat{n},n}^{\text{bound}}$ (validates scrambling bound)
- **The blind spot:** $\epsilon_{\hat{n},n}$ peaks sharply at $t_c$ while $\epsilon_{H,n}$ (= $\Delta E_n$) is suppressed near $t_c$

**Chain of trust:** The energy verification ($\Delta E = \epsilon_H \leq \epsilon_H^{\text{bound}}$) builds confidence in the scrambling bound framework. Then $\epsilon_{\hat{n}} \leq \epsilon_{\hat{n}}^{\text{bound}}$ confirms it works for kink density too. This justifies using $\epsilon_{\hat{n}}^{\text{bound}}$ to design the adaptive schedule in Benchmark 7, where we can't verify against direct measurement.

---

### Benchmark 5 — Entanglement Profile

**Purpose:** Map entanglement structure and its connection to Trotter error suppression.

**Parameters:** $L \in \{20,40,60\}$, $\tau_Q \in \{10,20,50\}$, ~100 time points.

**Computed:** Physical half-chain entropy $S(|\psi(t)\rangle)$. Operator-induced entropies from $\hat{n}$ and $[H_Z,H_X]$. Overlay with scrambling bounds from Benchmark 4.

---

### Benchmark 6 — Energy-Adaptive Schedule

**Purpose:** Implement the energy-based competitor with dual verification.

**Parameters:** $L=40$, $\tau_Q=20$, $N_{\text{steps}} \in \{20,50,100,200,500\}$.

**Method:**
1. Start with uniform schedule
2. Run Trotter evolution, measuring at each step:
   - $\Delta E_n$ (direct: energy before minus after = $\epsilon_{H,n}$, the schedule driver)
   - $\epsilon_{H,n}^{\text{bound}}$ (scrambling bound, for verification that direct and bound agree)
3. Redistribute steps to equalize $|\Delta E_n|$
4. Iterate 2–3 times
5. Run final Trotter evolution with optimized schedule
6. Measure KZM observables

**Output:** Schedule $\delta t(t)$, KZM observables, errors vs reference, plus the $\Delta E$ and $\epsilon_H^{\text{bound}}$ profiles for the final schedule (verifying they're consistent).

---

### Benchmark 7 — Kink-Density-Adaptive Schedule

**Purpose:** Our proposed method.

**Parameters:** Same as Benchmark 6.

**Method:**
1. From reference evolution, compute $\mathcal{B}_{\hat{n}}(t) = \langle\psi(t)|[\hat{n},[H_Z,H_X]]^\dagger[\hat{n},[H_Z,H_X]]|\psi(t)\rangle$ at many time points
2. Optimal step density: $\rho(t) \propto \mathcal{B}_{\hat{n}}(t)^{1/4} \cdot (f(t)g(t))^{1/2}$
3. Discretize to $N_{\text{steps}}$
4. Run Trotter, measure KZM observables

**Output:** Same format as Benchmark 6.

---

### Benchmark 8 — Head-to-Head Comparison

**Purpose:** The main result.

**Parameters:** $L=40$, $\tau_Q \in \{5,10,20,50\}$, $N_{\text{steps}} \in \{20,50,100,200,500\}$.

**Plots:**
- (a) $\langle\hat{n}\rangle$ vs $\tau_Q$ at fixed $N_{\text{steps}}=100$: exact, uniform, energy-adaptive, kink-adaptive
- (b) $\kappa_2/\kappa_1$ vs $\tau_Q$ for same
- (c) Three schedules overlaid on $\mathcal{B}_{\hat{n}}(t)$
- (d) Error in $\langle\hat{n}\rangle$ vs $N_{\text{steps}}$ for three schemes
- (e) Kink-kink correlator at fixed $N_{\text{steps}}$

**Quantitative:** Min $N_{\text{steps}}$ for $|\alpha-0.5|<0.05$ and $|\kappa_2/\kappa_1-0.586|<0.02$.

---

### Extension — Non-Integrable Model

Repeat Benchmarks 1, 4, 7, 8 for TFIM+$J_2$ with $J_2 \in \{0.1, 0.3, 0.5\}$.

## 8. Implementation Phases

### Phase 1 — Core Infrastructure (→ Benchmark 1)

Build: `abstract.jl`, `quench.jl`, `tfim.jl`, `gates.jl`, `tebd.jl`, `schedule.jl`, `kink_density.jl`, `energy.jl`, `entanglement.jl`, `config.jl`, `io.jl`

Test: Reproduce KZM scaling with TEBD.

### Phase 2 — Trotter Comparison (→ Benchmark 2)

Build: `trotter_evolve.jl`, `trotter_error.jl`

Test: Quantify uniform Trotter errors, verify error scaling.

### Phase 3 — Analysis Infrastructure (→ Benchmarks 3, 4, 5)

Build: `error_operator.jl`, `scrambling.jl`, `operator_entanglement.jl`

Test: Demonstrate blind spot with full six-quantity verification hierarchy. Validate $\Delta E = \epsilon_H \leq \epsilon_H^{\text{bound}}$.

### Phase 4 — Adaptive Methods (→ Benchmarks 6, 7, 8)

Build: `adaptive.jl`, `comparison.jl`, `tfim_j2.jl`

Test: Head-to-head comparison with dual-verified energy-adaptive baseline. Non-integrable extension.
