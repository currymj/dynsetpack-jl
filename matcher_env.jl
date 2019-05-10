using JuMP
using CSV
using DataFrames
using Gurobi
using GLPK
using ParameterJuMP
using Distributions
using Plots
feasible_sets = collect(Int64.(convert(Matrix, CSV.File("/Users/curry/src/julia_setpack/bloodTypeVectors.csv") |> DataFrame))')
arrival_means = Float64[32.3250498, 21.26307068, 9.998228937, 2.540044521, 14.54586926, 9.568116614, 4.499078324, 1.142988355, 4.531473682, 2.980754732, 1.401597571, 0.356075086, 0.548292189, 0.360660715, 0.169588318, 0.043083818]
departure_means = Float64[14.39983629, 7.932098763, 4.343646604, 0.73473771, 6.479746736, 3.569345515, 1.954586798, 0.330622806, 2.01863507, 1.111957971, 0.608912295, 0.102998901, 0.24424766, 0.134542957, 0.073676221, 0.0124625]

mutable struct MatcherEnv
    state::Array{Float32}
    constraintparams::Array{Parameter}
    solnvars::Array{VariableRef}
    feasible_sets::Array{Float32,2}
    daily_arrival_dists::Array{Poisson}
    daily_departure_dists::Array{Poisson}
    optmodel::Model
end

function MatcherEnv(feasible_sets::Array{Int64, 2}, daily_arrival_means, daily_departure_means)
    n_types = size(feasible_sets, 1)
    n_sets = size(feasible_sets, 2)

    model = ModelWithParams(with_optimizer(GLPK.Optimizer))
    #model = ModelWithParams(with_optimizer(Gurobi.Optimizer; OutputFlag=0))

    @variable(model, x[1:n_sets] >= 0, Int)

    row_sums = @expression(model, feasible_sets * x)

    constraint_vec = rand(0:2, 16)
    cons_rhs = Parameters(model, constraint_vec)
    conses = [@constraint(model, row_sums[k] <= cons_rhs[k]) for k=1:n_types]
    @objective(model, Max, sum(row_sums))
    initstate = zeros(Float32, n_types)
    MatcherEnv(initstate, cons_rhs, x, feasible_sets, Poisson.(daily_arrival_means), Poisson.(daily_departure_means), model)
end

function reset!(m::MatcherEnv)
    m.state .= zeros(size(m.state))
    m.state
end

function _perform_match(m::MatcherEnv)
    fix.(m.constraintparams, m.state)
    optimize!(m.optmodel)
    value.(m.solnvars)
end

function _arrive_and_depart!(m::MatcherEnv)
    for i=1:length(m.state)
        m.state[i] += rand(m.daily_arrival_dists[i])
        m.state[i] -= rand(m.daily_departure_dists[i])
        if m.state[i] < 0
            m.state[i] = 0
        end
    end
end

function _run_match!(m::MatcherEnv, match)
    total_match = m.feasible_sets * match
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
    (m.state, reward, false)
end

greedypolicy(state) = 1
function episodeloop(m::MatcherEnv, p; nsteps=100)
    starttime = time()
    state = reset!(m)
    episodereward = 0.0
    reward = 0.0
    for step=1:nsteps
        action = p(state)
        state, reward, _ = step!(m, action)
        episodereward += reward
        #println(state)
    end
    endtime = time()
    println(endtime - starttime)
    episodereward
end

matcher = MatcherEnv(feasible_sets, arrival_means, departure_means)

function trainloop(m::MatcherEnv, p; nsteps=100, nepisodes=10)
    runningreward = 0.0
    eprewards = Float32[]
    runningrewards = Float32[]

    for ep=1:nepisodes
        epreward = episodeloop(m, p; nsteps=nsteps)
        runningreward = 0.05 * epreward + (1 - 0.05) * runningreward
        push!(eprewards, epreward)
        push!(runningrewards, runningreward)
    end
    eprewards, runningrewards
end

(eprewards, runningrewards) = trainloop(matcher, greedypolicy; nsteps=10, nepisodes=100)

using Distributions
using Flux: params
using Flux.Tracker: update!
using Flux

mutable struct MLPAgent
    nnmodel::Chain
    opt::ADAM
    rewards::Array{Float32}
    saved_log_probs::Array{Tracker.TrackedReal}
end

MLPAgent(model) = MLPAgent(model, ADAM(0.1), Float32[], Tracker.TrackedReal[])
agent = MLPAgent(Chain(Dense(16, 128, relu), Dense(128, 2), softmax))

function Distributions.isprobvec(p::TrackedArray)
    sum(p).data ≈ 1.0
end

function select_action(state)
    probs = agent.nnmodel(state)
    dist = Categorical(probs)
    action = rand(dist)
    logprob = logpdf(dist, action)
    push!(agent.saved_log_probs, logprob)
    action - 1 # convert to 0/1
end


function remember_reward(reward)
    push!(agent.rewards, reward)
end

function finish_episode(gamma=0.99)
    # change gamma to be somewhere else later
    returns = Float32[]
    rr = 0.0
    for r in agent.rewards[end:-1:1]
        rr = r + gamma * rr
        insert!(returns, 1, rr)
    end
    returns = (returns .- mean(returns)) ./ (std(returns) + eps())

    # multiply returns and logprobs
    grads = Tracker.gradient(() ->sum(-agent.saved_log_probs .* returns), params(agent.nnmodel))
    for p in params(agent.nnmodel)
        update!(agent.opt, p, grads[p])
    end
    empty!(agent.saved_log_probs)
    empty!(agent.rewards)
end

function episodeloop(m::MatcherEnv, select_action, remember_reward; nsteps=100)
    starttime = time()
    state = reset!(m)
    episodereward = 0.0
    reward = 0.0
    for step=1:nsteps
        action = select_action(state)
        state, reward, _ = step!(m, action)
        episodereward += reward
        remember_reward(reward)
        #println(state)
    end
    endtime = time()
    #println(endtime - starttime)
    episodereward
end

using Printf: @printf
function trainloop(m::MatcherEnv, select_action, remember_reward, finish_episode; nsteps=100, nepisodes=10)
    runningreward = 0.0
    eprewards = Float32[]
    runningrewards = Float32[]

    for ep=1:nepisodes
        epreward = episodeloop(m, select_action, remember_reward; nsteps=nsteps)
        finish_episode()
        runningreward = 0.05 * epreward + (1 - 0.05) * runningreward
        push!(eprewards, epreward)
        push!(runningrewards, runningreward)
        if ep % 10 == 0
            @printf("Episode %d current reward %f running reward %f\n", ep, epreward, runningreward)
        end
    end
    eprewards, runningrewards
end

plotweights(agent) = heatmap(collect(params(agent.nnmodel[1]))[1].data)
(eprewards, runningrewards) = trainloop(matcher, select_action, remember_reward, finish_episode; nsteps=10, nepisodes=1000)
