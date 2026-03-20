"""
    LinearQuench(tau_Q, J)

Linear quench schedule: f(s) = s, g(s) = 1-s where s = t/tau_Q.
Critical point at f/g = 1/J, i.e. s_c = 1/(1+J).
"""
struct LinearQuench <: AbstractQuenchSchedule
    tau_Q::Float64
    J::Float64
end

LinearQuench(tau_Q; J=1.0) = LinearQuench(tau_Q, J)

function f(schedule::LinearQuench, t::Real)::Float64
    s = t / schedule.tau_Q
    return clamp(s, 0.0, 1.0)
end

function g(schedule::LinearQuench, t::Real)::Float64
    s = t / schedule.tau_Q
    return clamp(1.0 - s, 0.0, 1.0)
end

function t_critical(schedule::LinearQuench)::Float64
    # f(t_c)/g(t_c) = 1/J => s_c/(1-s_c) = 1/J => s_c = 1/(1+J)
    s_c = 1.0 / (1.0 + schedule.J)
    return s_c * schedule.tau_Q
end

"""
    QuadraticQuench(tau_Q, J)

Quadratic quench schedule: f(s) = s^2, g(s) = (1-s)^2 where s = t/tau_Q.
"""
struct QuadraticQuench <: AbstractQuenchSchedule
    tau_Q::Float64
    J::Float64
end

QuadraticQuench(tau_Q; J=1.0) = QuadraticQuench(tau_Q, J)

function f(schedule::QuadraticQuench, t::Real)::Float64
    s = t / schedule.tau_Q
    s = clamp(s, 0.0, 1.0)
    return s^2
end

function g(schedule::QuadraticQuench, t::Real)::Float64
    s = t / schedule.tau_Q
    s = clamp(s, 0.0, 1.0)
    return (1.0 - s)^2
end

function t_critical(schedule::QuadraticQuench)::Float64
    # f(s)/g(s) = (s/(1-s))^2 = 1/J => s/(1-s) = 1/sqrt(J) => s_c = 1/(1+sqrt(J))
    s_c = 1.0 / (1.0 + sqrt(schedule.J))
    return s_c * schedule.tau_Q
end
