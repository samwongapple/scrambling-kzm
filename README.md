# ScramblKZM.jl

**Observable-specific operator scrambling bounds for adaptive Trotterization of quantum phase transitions — applied to the Kibble-Zurek mechanism on digital quantum computers.**

## The Problem

Simulating the Kibble-Zurek mechanism (KZM) on a digital quantum computer requires Trotterizing time evolution through a quantum critical point. The standard energy-based adaptive Trotterization (Zhao et al., PRL 2024) has a **critical-mode blind spot**: the spectral gap vanishes at the critical point, suppressing the sensitivity of the energy diagnostic to the modes that produce topological defects.

## Our Approach

We use the **operator scrambling bound** (Feng et al., arXiv:2506.23345) to adapt the Trotter step size based on the **kink density** observable rather than energy. The energy-based approach is a special case of this framework where $\Delta E = \epsilon_H$ (measured energy change equals Trotter error, due to piecewise conservation). For the kink density, physical dynamics mix with Trotter error, requiring explicit computation of the multiplicative error operator $M$ — done during classical pre-computation via TEBD.

## Documentation

- **[PROJECT_DESCRIPTION.md](PROJECT_DESCRIPTION.md)** — Full physics context, theoretical framework, model definitions, and research goals
- **[IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)** — Code architecture, module interfaces, benchmark specifications, and implementation phases

## Quick Start

```bash
cd ScramblKZM.jl
julia --project=. -e 'using Pkg; Pkg.instantiate()'
julia --project=. benchmarks/bench01_exact_kzm.jl configs/benchmarks/bench01_exact_kzm.toml
```

## Dependencies

- Julia ≥ 1.10
- ITensors.jl / ITensorMPS.jl
- HDF5.jl
- TOML (stdlib)
- Plots.jl
