module ScramblKZM

using ITensors
using ITensorMPS
using LinearAlgebra
using HDF5
using TOML
using Statistics: mean

# Models
include("models/abstract.jl")
include("models/quench.jl")
include("models/tfim.jl")

# Simulation
include("simulation/schedule.jl")
include("simulation/gates.jl")
include("simulation/tebd.jl")
include("simulation/trotter_evolve.jl")

# Observables
include("observables/kink_density.jl")
include("observables/energy.jl")
include("observables/entanglement.jl")
include("observables/operator_entanglement.jl")

# Analysis (order matters: error_operator before scrambling before trotter_error)
include("analysis/error_operator.jl")
include("analysis/scrambling.jl")
include("analysis/trotter_error.jl")

# Utilities
include("utils/config.jl")
include("utils/io.jl")

# Export public API

# Models
export AbstractModel, AbstractQuenchSchedule
export TFIM
export LinearQuench, QuadraticQuench
export sites, hz_terms, hx_terms, hamiltonian_mpo, initial_state, num_sites
export f, g, t_critical

# Simulation
export TimeSchedule, uniform_schedule, custom_schedule
export make_pf1_gates
export evolve_tebd!
export trotter_evolve!

# Observables
export kink_density, kink_density_per_bond, kink_cumulants
export energy, energy_variance
export half_chain_entropy, bond_entropies
export operator_induced_entropy

# Analysis — error operators
export commutator_mpo, build_hz_hx_commutator
export build_kink_zz_mpo, build_scrambling_operator
export scrambling_bound, scrambling_profile

# Analysis — Trotter errors
export StepErrorResult, compute_step_errors
export TimeResolvedResult, time_resolved_errors

# Utilities
export load_config
export save_results, load_results

end # module
