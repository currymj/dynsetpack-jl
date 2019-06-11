include("./matcher_rl.jl")

env = ToyMatcherEnv()
agent = MLPAgent(Chain(Dense(5, 128, relu), Dropout(0.6), Dense(128, 2)))
plotweights(agent) = heatmap(collect(params(agent.nnmodel[1]))[1].data)
(eprewards, runningrewards) = trainloop(env, agent; nsteps=40, nepisodes=1000, status_callback=print_probs)
test_rewards = [evalepisode(env, agent; nsteps=40) for i=1:20]

env = ToyMatcherEnv()
agent = GreedyAgent()
(eprewards, runningrewards) = trainloop(env, agent; nsteps=40, nepisodes=1000)
test_rewards = [evalepisode(env, agent; nsteps=40) for i=1:20]

env = BloodTypeMatcherEnv()
agent = MLPAgent(Chain(Dense(16, 128, relu), Dropout(0.6), Dense(128, 2)))
(eprewards, runningrewards) = trainloop(env, agent; nsteps=40, nepisodes=100)

env = BloodTypeMatcherEnv()
agent = GreedyAgent()
(eprewards, runningrewards) = trainloop(env, agent; nsteps=40, nepisodes=100)

env = RandomObservableMatcherEnv(10, 100,100,0.2,0.05)
agent = GreedyAgent()
(eprewards, runningrewards) = trainloop(env, agent; nsteps=40, nepisodes=100)

env = RandomObservableMatcherEnv(10, 100,100,0.2,0.05)
agent = MLPAgent(Chain(Dense(10, 128, relu), Dropout(0.6), Dense(128, 2)))
(eprewards, runningrewards) = trainloop(env, agent; nsteps=40, nepisodes=100)
