include("./matcher_env.jl")
matcher = BloodTypeMatcherEnv(feasible_sets, arrival_means, departure_means)
matcher = ToyMatcherEnv()
using Distributions
import Flux.params
using Flux.Tracker: update!
using Flux
using Flux: testmode!

mutable struct MLPAgent
    nnmodel::Chain
    opt::ADAM
    rewards::Array{Float32}
    saved_log_probs::Array{Tracker.TrackedReal}
end

MLPAgent(model) = MLPAgent(model, ADAM(0.01), Float32[], Tracker.TrackedReal[])

function Distributions.isprobvec(p::TrackedArray)
    sum(p).data ≈ 1.0
end

function select_action(state)
    probs = agent.nnmodel(state)
    dist = Categorical(softmax(probs))
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

function status_callback(ep, epreward, runningreward)
    testmode!(agent.nnmodel)
    @printf("Episode %d current reward %f running reward %f\n", ep, epreward, runningreward)
    println(softmax(agent.nnmodel([1.0,1.0,1.0,1.0,1.0])))
    println(softmax(agent.nnmodel([1.0,1.0,0.0,0.0,0.0])))
    testmode!(agent.nnmodel, false)
end

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
            status_callback(ep, epreward, runningreward)
        end
    end
    eprewards, runningrewards
end

env = ToyMatcherEnv()

agent = MLPAgent(Chain(Dense(5, 128, relu), Dropout(0.6), Dense(128, 2)))
plotweights(agent) = heatmap(collect(params(agent.nnmodel[1]))[1].data)
(eprewards, runningrewards) = trainloop(env, select_action, remember_reward, finish_episode; nsteps=40, nepisodes=1000)
