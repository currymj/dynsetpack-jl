function twocycles(n_types; n_generate=100)
    candidates = zeros(Int64, n_types, n_generate)

    for i=1:n_generate
        inds = rand(1:n_types, 2)
        candidates[inds[1], i] += 1
        candidates[inds[2], i] += 1
    end

    feasible_sets = unique(candidates; dims=2)
end

function threecycles(n_types; n_generate=100)
    candidates = zeros(Int64, n_types, n_generate)

    for i=1:n_generate
        inds = rand(1:n_types, 3)
        candidates[inds[1], i] += 1
        candidates[inds[2], i] += 1
        candidates[inds[3], i] += 1
    end

    feasible_sets = unique(candidates; dims=2)
end
