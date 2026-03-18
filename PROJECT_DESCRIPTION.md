# ScramblKZM — Project Description

## 1. Research Goal

We develop an **observable-specific adaptive Trotterization** method for simulating the Kibble-Zurek mechanism (KZM) on digital quantum computers. The central contribution is showing that the choice of observable used to assess Trotter errors fundamentally affects the accuracy of quantum simulations near quantum critical points, and that targeting the physically relevant observable (defect density) rather than energy yields dramatically better results at fixed computational cost.

## 2. Physical Background

### 2.1 The Kibble-Zurek Mechanism

The KZM describes universal defect formation when a system is driven through a continuous phase transition at a finite rate. Originally proposed in cosmology (Kibble, 1976) and condensed matter (Zurek, 1985), the mechanism relies on three ingredients:

1. **Critical slowing down:** Near the critical point, the relaxation time $\tau \propto |\lambda - \lambda_c|^{-z\nu}$ diverges, where $\lambda$ is the control parameter, $z$ is the dynamical critical exponent, and $\nu$ is the correlation length exponent.

2. **Freeze-out:** When the system is driven through the critical point in finite time $\tau_Q$, it cannot respond fast enough. The freeze-out time $\hat{t} \sim (\tau_0 \tau_Q^{z\nu})^{1/(1+z\nu)}$ marks the boundary between adiabatic and impulse regimes.

3. **Defect formation:** Independent domains of size $\hat{\xi} \sim \tau_Q^{\nu/(1+z\nu)}$ form, with topological defects at domain boundaries. The defect density scales as $\langle \hat{n} \rangle \propto \tau_Q^{-d\nu/(1+z\nu)}$ where $d$ is the spatial dimension.

### 2.2 Universal Statistics of Defects

Beyond the mean defect density, del Campo (PRL 121, 200601, 2018) showed that for the quantum Ising model, the **full counting statistics** of kinks are universal. The kink number follows a binomial distribution, and all cumulants $\kappa_n$ scale with the same power law $\kappa_n \propto \tau_Q^{-1/2}$. The universal cumulant ratios are:

$$\kappa_2/\kappa_1 = 2 - \sqrt{2} \approx 0.586, \qquad \kappa_3/\kappa_1 = 4(1 - 3/\sqrt{2} + 2/\sqrt{3}) \approx 0.134$$

King et al. (Nature Physics 18, 1324, 2022) verified these on D-Wave's quantum annealer, and Kiss et al. (arXiv:2410.06250, 2025) confirmed them on digital superconducting processors up to 100 qubits.

### 2.3 Breakdown of KZM at Fast Quenches

Zeng et al. (PRL 130, 060402, 2023) showed that for rapid quenches ($\tau_Q < \tau_Q^{c1}$), the KZM power-law scaling breaks down and defect density saturates to a plateau $n \sim 1/\xi(\lambda_f)^d$, which is universal and independent of $\tau_Q$.

### 2.4 2D Extensions

Weinberg et al. (arXiv:2507.09273, 2025) studied the 2D TFIM and identified three distinct dynamical timescales: KZ criticality ($\propto L^{z+1/\nu}$ with 3D Ising exponents $z=1$, $\nu \approx 0.63$), coarsening of confined domains ($\propto L^2$), and system-spanning domain wall decay ($\propto L^3$). This richness makes 2D an important future target.

## 3. The Models

### 3.1 Transverse-Field Ising Model (Integrable, 1D)

The primary testbed is the 1D TFIM:

$$H(t) = -f(t/\tau_Q) \sum_{\langle i,j \rangle} J \sigma_i^z \sigma_{j}^z - g(t/\tau_Q) \sum_j \sigma_j^x$$

with nearest-neighbor coupling $J$, linear quench schedule $f(s) = s$, $g(s) = 1-s$ (or quadratic: $f(s) = s^2$, $g(s) = (1-s)^2$). The quantum phase transition occurs at $f/g = 1/J$.

**Key properties:**
- Exactly solvable via Jordan-Wigner transformation (maps to free fermions)
- Each momentum mode $k$ undergoes an independent Landau-Zener transition with excitation probability $p_k = \exp(-\pi \Delta_k^2 \tau_Q / 2|\dot{\Delta}_k|)$
- Universality class: 2D Ising ($z=1$, $\nu=1$), giving KZM exponent $\alpha = \nu/(1+z\nu) = 1/2$
- The kink density operator is $\hat{n} = \frac{1}{2N}\sum_{i=1}^{N-1}(1 - \sigma_i^z \sigma_{i+1}^z)$

**Why it matters for us:** The exact solvability allows analytical cross-checks of all numerical results. In the free-fermion picture, we can verify mode-by-mode that the kink density weights critical modes differently from the energy.

### 3.2 Non-Integrable TFIM + $J_2$ (1D)

To demonstrate our method works beyond integrable models:

$$H(t) = -f(t/\tau_Q)\left(J_1\sum_i \sigma_i^z \sigma_{i+1}^z + J_2\sum_i \sigma_i^z \sigma_{i+2}^z\right) - g(t/\tau_Q)\sum_j \sigma_j^x$$

The next-nearest-neighbor coupling $J_2$ breaks integrability (Jordan-Wigner no longer gives free fermions) while preserving the $\mathbb{Z}_2$ symmetry $\sigma^z \to -\sigma^z$. For small $J_2$, the quantum phase transition survives in the same 2D Ising universality class, so the KZM prediction $\alpha = 1/2$ should hold.

**Why it matters:** This is a genuinely interacting model where:
- The Eigenstate Thermalization Hypothesis (ETH) applies (unlike the integrable TFIM)
- Classical simulation via free fermions is impossible
- The universality of our method can be tested

### 3.3 2D TFIM (Future Extension)

$$H(t) = -f(t/\tau_Q) J\sum_{\langle ij \rangle} \sigma_i^z \sigma_j^z - g(t/\tau_Q)\sum_i \sigma_i^x$$

on a square lattice. The quantum critical point is in the 3D Ising universality class ($z=1$, $\nu \approx 0.63$), giving a different KZM exponent $\alpha = 2\nu/(1+z\nu) \approx 0.77$. Not integrable. Classically limited to small systems ($L \leq 6$ for exact Schrödinger, larger for TEBD/PEPS but with approximations).

## 4. Trotterization and Its Errors

### 4.1 The Trotterization Problem

To simulate $H(t) = f(t)H_Z + g(t)H_X$ on a digital quantum computer, we discretize time into steps and approximate each step using product formulas. The first-order product formula (PF1) for one step with frozen Hamiltonian $H_n = f_n H_Z + g_n H_X$ is:

$$U_{\text{PF1}} = e^{-if_n H_Z \delta t} \, e^{-ig_n H_X \delta t}$$

while the exact evolution is $U_0 = e^{-i(f_n H_Z + g_n H_X)\delta t}$.

### 4.2 Multiplicative vs Additive Error

The **additive error** $E = U_0 - U_{\text{PF1}}$ measures the distance between the two unitaries. For PF1, $\|E\| = O(\delta t^2)$.

The **multiplicative error** $M$ is defined by $U_{\text{PF1}} = U_0(I + M)$, i.e., $M = U_0^\dagger U_{\text{PF1}} - I$. It represents the "extra corruption" applied on top of the exact evolution. For PF1:

$$M_{\text{PF1}} = \frac{f_n g_n}{2}[H_Z, H_X]\delta t^2 + O(\delta t^3)$$

For the TFIM: $[H_Z, H_X] = 2i\sum_i J(\sigma_i^y \sigma_{i+1}^z + \sigma_i^z \sigma_{i+1}^y)$, which is a sum of 2-local operators.

### 4.3 Observable-Specific Trotter Error

The Trotter error in an observable $O$ is:

$$\epsilon_O = |\langle \psi | U_0^\dagger O \, U_0 | \psi \rangle - \langle \psi | U_{\text{PF1}}^\dagger O \, U_{\text{PF1}} | \psi \rangle|$$

This is the difference between the exact and Trotterized expectation values — purely a consequence of the Trotterization choice, nothing else.

## 5. The Two Adaptive Frameworks

### 5.1 Feng et al. — Operator Scrambling Bound (arXiv:2506.23345, 2025)

Feng et al. proved that the observable Trotter error is bounded by operator scrambling:

$$\epsilon_O^2 \leq \langle \psi | [O(\delta t), M]^\dagger [O(\delta t), M] | \psi \rangle$$

where $O(\delta t) = e^{iH\delta t} O e^{-iH\delta t}$ is the Heisenberg-evolved observable. This bound is:
- **State-dependent** — tighter than worst-case $\|[O(\delta t), M]\|$ (Lieb-Robinson bound)
- **Observable-specific** — different $O$ give different bounds
- **Connected to entanglement** — when the state is sufficiently entangled, the bound reduces from spectral norm to Frobenius norm scaling, giving a quadratic improvement

**Entanglement connection (Theorem III.1 of Feng et al.):** For operators $A = \sum_j A_j$ (sums of local terms), the vector norm satisfies:

$$\|A|\psi\rangle\|^2 \leq \|A\|_F^2 + \Delta_A(|\psi\rangle)$$

where $\|A\|_F^2 = \text{Tr}(A^\dagger A)/d$ is the normalized Frobenius norm and $\Delta_A$ involves entanglement entropies of subsystems where $A_j^\dagger A_{j'}$ acts:

$$\Delta_A(|\psi\rangle) = \sum_{j,j'} \|A_j^\dagger A_{j'}\| \sqrt{2\log d_{\text{sub}} - 2S(\rho_{j,j'})}$$

High entanglement ($S \to \log d_{\text{sub}}$) drives $\Delta_A \to 0$, suppressing the error to the Frobenius norm level.

**Operator-induced entanglement:** Even when $|\psi\rangle$ has low entanglement, the states $O|\psi\rangle/\|O|\psi\rangle\|$ and $M|\psi\rangle/\|M|\psi\rangle\|$ can have high entanglement because the operators spread information across subsystems.

### 5.2 Zhao et al. — Piecewise Energy Conservation (PRL 133, 010603, 2024; PRX Quantum 4, 030319, 2023)

Zhao et al. proposed adapting Trotter step sizes by monitoring **piecewise conservation laws**. The key construction:

1. The exact time evolution from $t$ to $t+\delta t$ can be generated by a static effective Hamiltonian $H_{[\infty]}$ via the Magnus expansion: $U(t+\delta t, t) = e^{-iH_{[\infty]}\delta t}$.

2. Since $[H_{[\infty]}, H_{[\infty]}] = 0$, the expectation value $\langle H_{[\infty]} \rangle$ is exactly conserved under exact evolution. Any change after a Trotter step is pure error.

3. In practice, $H_{[\infty]}$ is approximated by a truncated Magnus expansion $H_{[k]}$, which is approximately conserved.

4. At each step, measure $E_i = \langle\psi(t_n)|H_{[k]}|\psi(t_n)\rangle$ and $E_f = \langle\psi(t_{n+1})|H_{[k]}|\psi(t_{n+1})\rangle$. If $|E_f - E_i| > d_E$, shrink $\delta t$.

**Their models:** Non-integrable quantum Ising chain with longitudinal field ($h_z \sum_j \sigma_j^z$), which breaks $\mathbb{Z}_2$ symmetry — no phase transition, no gap closing, ETH applies throughout. Also demonstrated on U(1) lattice gauge theories.

**Validation:** Compared against exact diagonalization for $L = 18$–$24$, showing local observables (magnetization) track the exact dynamics. Explicitly noted that fidelity is not suitable as it overestimates errors in local observables.

### 5.3 How the Two Frameworks are Related

The Zhao et al. approach is a **special case** of the Feng et al. scrambling bound with $O = H_{[\infty]}$ (or its truncation $H_{[k]}$).

**Proof that $\Delta E = \epsilon_H$:**

The Trotter error in energy is defined as:

$$\epsilon_H = |\langle \psi | U_0^\dagger H_{[\infty]} U_0 | \psi \rangle - \langle \psi | U_p^\dagger H_{[\infty]} U_p | \psi \rangle|$$

Since $U_0 = e^{-iH_{[\infty]}\delta t}$ and $[H_{[\infty]}, U_0] = 0$:

$$\langle \psi | U_0^\dagger H_{[\infty]} U_0 | \psi \rangle = \langle \psi | H_{[\infty]} | \psi \rangle$$

Therefore:

$$\epsilon_H = |\langle \psi | H_{[\infty]} | \psi \rangle - \langle \psi | U_p^\dagger H_{[\infty]} U_p | \psi \rangle|$$

Writing this in terms of states before and after the Trotterized step:

$$\epsilon_H = |\underbrace{\langle \psi(t_n) | H_{[\infty]} | \psi(t_n) \rangle}_{E_i} - \underbrace{\langle \psi_{\text{trotter}}(t_{n+1}) | H_{[\infty]} | \psi_{\text{trotter}}(t_{n+1}) \rangle}_{E_f}| = |E_i - E_f| = |\Delta E|$$

**The Trotter error in energy equals the measured energy change.** This is what makes the energy approach special — you can detect the Trotter error directly by measuring the same quantity before and after, because the ideal evolution would have given zero change.

**Why this fails for general observables:** For $O = \hat{n}$ (kink density):

$$\epsilon_{\hat{n}} = |\langle \psi | U_0^\dagger \hat{n} \, U_0 | \psi \rangle - \langle \psi | U_p^\dagger \hat{n} \, U_p | \psi \rangle|$$

Since $[\hat{n}, H_{[\infty]}] \neq 0$, the exact evolution changes $\langle \hat{n} \rangle$:

$$\langle \psi | U_0^\dagger \hat{n} \, U_0 | \psi \rangle \neq \langle \psi | \hat{n} | \psi \rangle$$

So the measured change $\Delta\langle\hat{n}\rangle$ mixes physical dynamics with Trotter error, and you cannot extract $\epsilon_{\hat{n}}$ from before-and-after measurements alone. You need the scrambling bound with explicit $M$.

### 5.4 The Full Verification Hierarchy

For a single Trotter step at time $t_n$, we can compute the following quantities that form a hierarchy of relationships:

**For energy ($O = H_n$):**

| Quantity | Definition | Needs $M$? | Needs exact evolution? |
|---|---|---|---|
| $\Delta E_n$ (direct measurement) | $\langle\psi_{\text{trotter}}(t_{n+1})\|H_n\|\psi_{\text{trotter}}(t_{n+1})\rangle - \langle\psi(t_n)\|H_n\|\psi(t_n)\rangle$ | No | No |
| $\epsilon_{H,n}$ (actual Trotter error) | $\|\langle\psi_{\text{exact}}(t_{n+1})\|H_n\|\psi_{\text{exact}}(t_{n+1})\rangle - \langle\psi_{\text{trotter}}(t_{n+1})\|H_n\|\psi_{\text{trotter}}(t_{n+1})\rangle\|$ | No | Yes |
| Conservation check | $\|\langle\psi_{\text{exact}}(t_{n+1})\|H_n\|\psi_{\text{exact}}(t_{n+1})\rangle - \langle\psi(t_n)\|H_n\|\psi(t_n)\rangle\|$ | No | Yes |
| $\epsilon_{H,n}^{\text{bound}}$ (scrambling bound) | $\sqrt{\langle\psi(t_n)\|[H_n, M]^\dagger[H_n, M]\|\psi(t_n)\rangle}$ | Yes | No |

The expected relationships:

$$\underbrace{\text{Conservation check}}_{\approx 0} \implies \underbrace{\Delta E_n = \epsilon_{H,n}}_{\text{direct = actual error}} \leq \underbrace{\epsilon_{H,n}^{\text{bound}}}_{\text{scrambling bound}}$$

The conservation check confirms $\langle H_n \rangle$ is unchanged under exact evolution. This makes $\Delta E_n$ (which requires no knowledge of the exact solution) equal to $\epsilon_{H,n}$ (which does). The scrambling bound is then an upper bound on both.

**For kink density ($O = \hat{n}$):**

| Quantity | Definition | Needs $M$? | Needs exact evolution? |
|---|---|---|---|
| $\epsilon_{\hat{n},n}$ (actual Trotter error) | $\|\langle\psi_{\text{exact}}(t_{n+1})\|\hat{n}\|\psi_{\text{exact}}(t_{n+1})\rangle - \langle\psi_{\text{trotter}}(t_{n+1})\|\hat{n}\|\psi_{\text{trotter}}(t_{n+1})\rangle\|$ | No | Yes |
| $\epsilon_{\hat{n},n}^{\text{bound}}$ (scrambling bound) | $\sqrt{\langle\psi(t_n)\|[\hat{n}, M]^\dagger[\hat{n}, M]\|\psi(t_n)\rangle}$ | Yes | No |

Here we can only verify:

$$\epsilon_{\hat{n},n} \leq \epsilon_{\hat{n},n}^{\text{bound}}$$

No direct measurement shortcut is available because $\hat{n}$ is not piecewise conserved.

**Why computing the full hierarchy matters:** Verifying $\Delta E = \epsilon_H$ numerically confirms the piecewise conservation property. Verifying $\epsilon_H \leq \epsilon_H^{\text{bound}}$ validates the scrambling bound framework. Once both are confirmed for energy, we can trust the scrambling bound for kink density where independent verification via direct measurement is impossible. This builds a chain of trust from the measurable ($\Delta E$) to the computable ($\epsilon_{\hat{n}}^{\text{bound}}$).

### 5.5 Summary: Comparison of the Two Approaches

| Property | Energy approach ($O = H$) | Kink density approach ($O = \hat{n}$) |
|---|---|---|
| Trotter error equals measured change? | Yes: $\epsilon_H = \Delta E$ | No: $\epsilon_{\hat{n}} \neq \Delta\langle\hat{n}\rangle$ |
| Can detect error on-chip? | Yes (measure before/after) | No (need explicit $M$) |
| Where is it computed? | On quantum computer or classical | Classical pre-computation only |
| Sensitive to critical modes? | No ($\varepsilon_k \to 0$ suppression) | Yes ($w_k = O(1)$) |
| Theoretical justification | ETH (non-integrable systems) | Scrambling bound (general) |

### 5.6 The Critical-Mode Blind Spot

For the integrable TFIM in the free-fermion picture, the Hamiltonian is $H = \sum_k \varepsilon_k(t)(2\gamma_k^\dagger\gamma_k - 1)$. The commutator $[H, M]$ weights each mode by $\varepsilon_k$. Near the critical point, $\varepsilon_k \to 0$ for the critical modes (those near the gap minimum). These are precisely the modes undergoing Landau-Zener transitions that create topological defects.

The energy-based scrambling bound is:

$$\langle \psi | [H, M]^\dagger [H, M] | \psi \rangle \sim \sum_k \varepsilon_k^2 \cdot (\text{mode-}k\text{ error})^2$$

Critical modes contribute $\varepsilon_k^2 \to 0$ — the bound is **blind** to errors in the most important modes.

The kink-density-based scrambling bound is:

$$\langle \psi | [\hat{n}, M]^\dagger [\hat{n}, M] | \psi \rangle \sim \sum_k w_k^2 \cdot (\text{mode-}k\text{ error})^2$$

where $w_k = O(1)$ for critical modes — the bound **sees** these errors.

This blind spot is not specific to integrability — it persists for any model with a quantum phase transition where the gap closes, including the non-integrable TFIM+$J_2$ and the 2D TFIM.

## 6. Our Method: Observable-Specific Adaptive Trotterization

### 6.1 The Algorithm

1. **Reference simulation (classical):** Run TEBD with fine time steps and large bond dimension to generate reference states $|\psi_{\text{ref}}(t)\rangle$ across the quench.

2. **Scrambling bound profiling:** At many time points, compute:
   $$\mathcal{B}_{\hat{n}}(t) = \langle \psi(t) | [\hat{n}, [H_Z, H_X]]^\dagger [\hat{n}, [H_Z, H_X]] | \psi(t) \rangle$$
   This is an MPO-MPS expectation value computable with ITensor. The commutator MPO $C = [\hat{n}, [H_Z, H_X]]$ is built once; only the MPS contraction varies across time.

3. **Schedule optimization:** For PF1, the per-step error bound is:
   $$\epsilon_{\hat{n},n}^2 \leq \frac{f_n^2 g_n^2}{4} \cdot \mathcal{B}_{\hat{n}}(t_n) \cdot \delta t_n^4$$
   To equalize this across all steps at fixed total $N_{\text{steps}}$, the step density is:
   $$\rho(t) \propto \mathcal{B}_{\hat{n}}(t)^{1/4} \cdot (f(t)g(t))^{1/2}$$

4. **Quantum execution:** Compile the pre-computed schedule into a fixed circuit. Run on hardware. No on-chip adaptation needed.

5. **Validation:** The universal KZM predictions ($\alpha = 1/2$, $\kappa_2/\kappa_1 = 2 - \sqrt{2}$, positive kink-kink correlator peak) serve as parameter-free self-consistency checks.

### 6.2 Entanglement Dynamics During the Quench

The KZM quench creates a rich interplay between Trotter error amplification and entanglement suppression:

**Early quench (adiabatic):** Product state, zero entanglement. Trotter errors at worst-case (spectral norm) scaling, but the commutator $[\hat{n}, M]$ is small because the Hamiltonian changes slowly. Large steps are fine.

**Near critical point (impulse):** Entanglement growing rapidly. The commutator $[\hat{n}, M]$ is large (gap closing amplifies it). Competition between amplification and suppression. Step sizes must be small.

**Post-critical (ordered phase):** High entanglement. Frobenius norm suppression active. Errors naturally small. Large steps again.

Additionally, **operator-induced entanglement** (Feng et al., Section III) means that even when the physical state has low entanglement, applying $\hat{n}$ or $M = [H_Z, H_X]$ to the state can create entanglement, providing partial error suppression.

## 7. What We Demonstrate

### 7.1 The Blind Spot and Verification Hierarchy (Motivation)

At each point during the quench, we compute the full hierarchy of six quantities (Section 5.4) for both energy and kink density. This demonstrates:
- The blind spot: $\epsilon_{\hat{n}}$ peaks at $t_c$ while $\epsilon_H$ (= $\Delta E$) is suppressed
- The piecewise conservation: $\Delta E = \epsilon_H$ (verified by the conservation check $\approx 0$)
- The scrambling bound validity: both $\epsilon_H \leq \epsilon_H^{\text{bound}}$ and $\epsilon_{\hat{n}} \leq \epsilon_{\hat{n}}^{\text{bound}}$
- The chain of trust: if the scrambling bound works for energy (verifiable), we trust it for kink density (not independently verifiable)

### 7.2 The Main Result (Comparison)

At a fixed gate budget $N_{\text{steps}}$, compare three schedules:
- **Uniform:** Equal $\delta t$ everywhere
- **Energy-adaptive:** Steps distributed to equalize $\Delta E_n = \epsilon_{H,n}$ across all steps (we also compute $\epsilon_{H,n}^{\text{bound}}$ via the scrambling bound to verify consistency)
- **Kink-density-adaptive:** Steps distributed to equalize the scrambling bound $\epsilon_{\hat{n},n}^{\text{bound}}$ across all steps

Show that only the kink-density-adaptive schedule reproduces the correct KZM scaling exponent $\alpha$ and cumulant ratios at realistic gate budgets.

### 7.3 Extensions

- **Non-integrable model (TFIM + $J_2$):** Verify the blind spot persists and our method still outperforms
- **Open question:** Whether the universal cumulant ratios survive integrability breaking
- **Future: 2D TFIM** with different universality class ($\nu \approx 0.63$)

## 8. References

1. Feng, Cao, Zhao, "Trotterization, Operator Scrambling, and Entanglement," arXiv:2506.23345 (2025)
2. Zhao, Bukov, Heyl, Moessner, "Adaptive Trotterization for Time-Dependent Hamiltonian Quantum Dynamics Using Piecewise Conservation Laws," PRL 133, 010603 (2024)
3. Zhao, Bukov, Heyl, Moessner, "Making Trotterization Adaptive and Energy-Self-Correcting for NISQ Devices and Beyond," PRX Quantum 4, 030319 (2023)
4. del Campo, "Universal Statistics of Topological Defects Formed in a Quantum Phase Transition," PRL 121, 200601 (2018)
5. Kiss, Teplitskiy, Grossi, Mandarino, "Statistics of topological defects across a phase transition in a digital superconducting quantum processor," arXiv:2410.06250 (2025)
6. King et al., "Coherent quantum annealing in a programmable 2000-qubit Ising chain," Nature Physics 18, 1324 (2022)
7. Weinberg, Xu, Sandvik, "Defects and their Time Scales in Quantum and Classical Annealing of the Two-Dimensional Ising Model," arXiv:2507.09273 (2025)
8. Zeng, Xia, del Campo, "Universal Breakdown of Kibble-Zurek Scaling in Fast Quenches across a Phase Transition," PRL 130, 060402 (2023)
9. Ikeda, Abrar, Chuang, Sugiura, "Minimum Trotterization Formulas for a Time-Dependent Hamiltonian," Quantum 7, 1168 (2023)
10. Dziarmaga, "Dynamics of a quantum phase transition: Exact solution of the quantum Ising model," PRL 95, 245701 (2005)
11. Schmitt, Rams, Dziarmaga, Heyl, Zurek, "Quantum phase transition dynamics in the two-dimensional transverse-field Ising model," Science Advances 8, eabl6850 (2022)
