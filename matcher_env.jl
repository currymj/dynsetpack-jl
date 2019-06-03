using CSV
using DataFrames
#using Gurobi
using Distributions
using Plots
include("./matcher.jl")

const feasible_sets = collect(Int64.(convert(Matrix, CSV.File("/Users/curry/src/julia_setpack/bloodTypeVectors.csv") |> DataFrame))')
const arrival_means = Float64[32.3250498, 21.26307068, 9.998228937, 2.540044521, 14.54586926, 9.568116614, 4.499078324, 1.142988355, 4.531473682, 2.980754732, 1.401597571, 0.356075086, 0.548292189, 0.360660715, 0.169588318, 0.043083818]
const departure_means = Float64[14.39983629, 7.932098763, 4.343646604, 0.73473771, 6.479746736, 3.569345515, 1.954586798, 0.330622806, 2.01863507, 1.111957971, 0.608912295, 0.102998901, 0.24424766, 0.134542957, 0.073676221, 0.0124625]

abstract type MatcherEnv end

mutable struct BloodTypeMatcherEnv <: MatcherEnv
    state::Array{Float32}
    daily_arrival_dists::Array{Poisson}
    daily_departure_dists::Array{Poisson}
    matcher::SetPackMatcher
end

function BloodTypeMatcherEnv(feasible_sets::Array{Int64, 2}, daily_arrival_means, daily_departure_means)
    n_types = size(feasible_sets, 1)
    n_sets = size(feasible_sets, 2)

    initstate = zeros(Float32, n_types)
    matcher = SetPackMatcher(feasible_sets)
    BloodTypeMatcherEnv(initstate, Poisson.(daily_arrival_means), Poisson.(daily_departure_means), matcher)
end

function BloodTypeMatcherEnv()
    BloodTypeMatcherEnv(feasible_sets, arrival_means, departure_means)
end

state(m::BloodTypeMatcherEnv) = m.state

mutable struct ToyMatcherEnv <: MatcherEnv
    state::Array{Float32}
    matcher::SetPackMatcher
    time_step::Int64
end

state(m::ToyMatcherEnv) = m.state

function ToyMatcherEnv()
    n_types = 5
    n_sets = 2
    feasible_sets = [1 1; 1 1; 1 0 ; 1 0 ; 1 0]
    matcher = SetPackMatcher(feasible_sets)
    initstate = zeros(Float32, n_types)
    ToyMatcherEnv(initstate, matcher, 0)
end


function reset!(m::ToyMatcherEnv)
    m.time_step = 0
    m.state .= zeros(size(m.state))
    m.state
end

function reset!(m::BloodTypeMatcherEnv)
    m.state .= zeros(size(m.state))
    m.state
end

function _perform_match(m::BloodTypeMatcherEnv)
    perform_match(m.matcher, m.state)
end

function _perform_match(m::ToyMatcherEnv)
    perform_match(m.matcher, m.state)
end

function _arrive_and_depart!(m::ToyMatcherEnv)
    # arrive and depart here
    if m.time_step == 0
        m.state = [1.0, 1.0, 0.0, 0.0, 0.0]
        m.time_step = 1
    elseif m.time_step == 1
        m.state .+= [0.0, 0.0, 1.0, 1.0, 1.0]
        m.time_step = 2
    elseif m.time_step == 2
        m.state .= [0.0, 0.0, 0.0, 0.0, 0.0]
        m.time_step = 0
    end
end

function _arrive_and_depart!(m::BloodTypeMatcherEnv)
    for i=1:length(m.state)
        m.state[i] += rand(m.daily_arrival_dists[i])
        m.state[i] -= rand(m.daily_departure_dists[i])
        if m.state[i] < 0
            m.state[i] = 0
        end
    end
end

function _run_match!(m::BloodTypeMatcherEnv, match)
    total_match = m.matcher.feasible_sets * match
    match_card = sum(total_match)
    m.state .-= total_match
    match_card
end

function _run_match!(m::ToyMatcherEnv, match)
    total_match = m.matcher.feasible_sets * match
    match_card = sum(total_match)
    m.state .-= total_match
    match_card
end

function step!(m::MatcherEnv, action)
    reward = 0.0
    if action == 1
        soln = _perform_match(m)
        reward = _run_match!(m, soln)
    end
    _arrive_and_depart!(m)
    (state(m), reward, false)
end

greedypolicy(state) = 1
function greedy_episodeloop(m::MatcherEnv; nsteps=100)
    starttime = time()
    state = reset!(m)
    episodereward = 0.0
    reward = 0.0
    for step=1:nsteps
        action = greedypolicy(state)
        state, reward, _ = step!(m, action)
        episodereward += reward
        #println(state)
    end
    endtime = time()
    println(endtime - starttime)
    episodereward
end
