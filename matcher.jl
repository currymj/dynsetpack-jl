using JuMP
using GLPK
using ParameterJuMP
struct SetPackMatcher
    constraintparams::Array{ParameterRef}
    solnvars::Array{VariableRef}
    feasible_sets::Array{Float32, 2}
    optmodel::Model
    function SetPackMatcher(feasible_sets::Array{Int64, 2})
        n_types = size(feasible_sets, 1)
        n_sets = size(feasible_sets, 2)

        model = ModelWithParams(with_optimizer(GLPK.Optimizer))
        #model = ModelWithParams(with_optimizer(Gurobi.Optimizer; OutputFlag=0))

        @variable(model, x[1:n_sets] >= 0, Int)

        row_sums = @expression(model, feasible_sets * x)

        constraint_vec = rand(0:2, n_types)
        cons_rhs = add_parameters(model, constraint_vec)
        conses = [@constraint(model, row_sums[k] <= cons_rhs[k]) for k=1:n_types]
        @objective(model, Max, sum(row_sums))
        new(cons_rhs, x, feasible_sets, model)
    end
end

function perform_match(s::SetPackMatcher, state)
    fix.(s.constraintparams, state)
    optimize!(s.optmodel)
    value.(s.solnvars)
end
