"""
    AbstractModel

Abstract type for quantum lattice models. All models must implement:
- `sites(model)` — ITensor site indices
- `hz_terms(model)` — Ising interaction OpSum (H_Z)
- `hx_terms(model)` — Transverse field OpSum (H_X)
- `hamiltonian_mpo(model, f, g)` — Full H(t) = -f*H_Z - g*H_X as MPO
- `initial_state(model)` — Initial MPS (|+>^L for KZM quench)
- `num_sites(model)` — Number of sites
"""
abstract type AbstractModel end

"""
    AbstractQuenchSchedule

Abstract type for quench schedules. All schedules must implement:
- `f(schedule, t)` — Ising coupling coefficient at time t
- `g(schedule, t)` — Transverse field coefficient at time t
- `t_critical(schedule)` — Time when critical point is crossed
"""
abstract type AbstractQuenchSchedule end

# Interface declarations (concrete methods in subtypes)
function sites end
function hz_terms end
function hx_terms end
function hamiltonian_mpo end
function initial_state end
function num_sites end

function f end
function g end
function t_critical end
