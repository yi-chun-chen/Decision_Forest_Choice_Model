include("../tcm_code/tcm_estimateForest_L1.jl")

function tcm_estimateForest_L1_RTS(N, assortments, choice_probs, depth, K)

	M = size(assortments,1)

	forest_left = Array{Int64,1}[]
    forest_right = Array{Int64,1}[]
    forest_product = Array{Int64,1}[]
    forest_isLeaf = Array{Bool,1}[]

	v_by_class = Array{Int64,2}[];

    start_time = time()

    for k in 1:K
        tree_left, tree_right, tree_product, tree_isLeaf = tcm_randomCompleteTree( N, depth-1 )

        v_one_class = tcm_treeToA( tree_left, tree_right, tree_product, tree_isLeaf, assortments)
        push!(v_by_class, v_one_class)


        push!(forest_left, tree_left)
        push!(forest_right, tree_right)
        push!(forest_product, tree_product)
        push!(forest_isLeaf, tree_isLeaf)
    end


	m_primal = Model(solver = GurobiSolver())
	@variable(m_primal, lambda_var[1:K] >= 0)
	@variable(m_primal, eps_plus[1:M,1:N] >= 0)
	@variable(m_primal, eps_minus[1:M,1:N] >= 0)

	for m in 1:M
		for i in 1:N
			@constraint(m_primal, sum( v_by_class[k][m,i] * lambda_var[k] for k in 1:K) + eps_plus[m,i] - eps_minus[m,i] == choice_probs[m,i])
		end
	end
	@constraint(m_primal, sum(lambda_var[k] for k in 1:K) == 1)

	@objective(m_primal, Min, sum( eps_minus[m,i] for m in 1:M, i in 1:N) + sum(eps_plus[m,i] for m in 1:M, i in 1:N))

	solve(m_primal)

    end_time = time()
    elapsed_time = end_time - start_time

	lambda_val = getvalue(lambda_var)

	primal_objval = getobjectivevalue(m_primal)


	return lambda_val, forest_left, forest_right, forest_product, forest_isLeaf, primal_objval, elapsed_time


end