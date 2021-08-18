include("../tcm_code/tcm_estimateForest.jl")

function tcm_estimateForest_L1(N, assortments, choice_probs, time_limit_overall, depth_limit, initial_left, initial_right, initial_product, initial_isLeaf)

	M = size(assortments,1)

	dim_V = sum(assortments)

	m_dual = Model(solver = GurobiSolver(OutputFlag = 0))
	@variable(m_dual, -1 <= p[1:M, 1:N] <= +1)
	@variable(m_dual, q)

	@objective(m_dual, Max, sum( p[m,i]* choice_probs[m,i] for m in 1:M, i in 1:N ) + q)


	forest_left = deepcopy(initial_left)
    forest_right = deepcopy(initial_right)
    forest_product = deepcopy(initial_product)
    forest_isLeaf = deepcopy(initial_isLeaf)

	K = length(forest_left)

	v_by_class = Array{Int64,2}[];


	for k in 1:K
		v_one_class = tcm_treeToA( forest_left[k], forest_right[k], forest_product[k], forest_isLeaf[k], assortments)
		push!(v_by_class, v_one_class)
		@constraint(m_dual, sum( v_one_class[m,i] * p[m,i] for m in 1:M, i in 1:N) + q <= 0)
	end

	tol_RC = -1e-4

	start_time = time()

	solve(m_dual)

	dual_objval = getobjectivevalue(m_dual)

	iter = 0;

    println("CG iter ", iter, " -- time elapsed = ", 0.0, " objective = ",dual_objval)

	while (time() - start_time < time_limit_overall)
        iter += 1

        p_val = getvalue(p)
        q_val = getvalue(q)

        single_left, single_right, single_product, single_isLeaf, sub_objval = tcm_solveTreeSubproblem_L1(N, p_val, assortments, depth_limit)

        # @show sub_objval
        v_one_class = tcm_treeToA( single_left, single_right, single_product, single_isLeaf, assortments)
        # @show dot(p_val, v_one_class)

        # @show p_val

        # error("Stop here")

        RC = -sub_objval - q_val

        @show RC
        @show sub_objval
        @show q_val
        @show sub_objval + q_val

        println("CG iter ", iter, " -- time elapsed = ", time() - start_time, " objective = ", dual_objval, " -- RC = ", RC)

        if (RC >= tol_RC)
        	# Stop here -- we are at optimality
        	println("\t RC = ", RC, " -- terminating ... ")
        	break;
        else
        	# Add the tree to the dual problem, and re-solve the master.
        	push!(forest_left, single_left)
        	push!(forest_right, single_right)
        	push!(forest_product, single_product)
        	push!(forest_isLeaf, single_isLeaf)

        	v_one_class = tcm_treeToA( single_left, single_right, single_product, single_isLeaf, assortments)
        	push!(v_by_class, v_one_class)
        	@constraint(m_dual, sum( v_one_class[m,i] * p[m,i] for m in 1:M, i in 1:N) + q <= 0)

        	solve(m_dual)
        	dual_objval = getobjectivevalue(m_dual)
        end

        # break;
	end

	end_time = time()
	elapsed_time = end_time - start_time

	# Have terminated; solve primal to get what the lambda's are. 
	K = length(forest_left)

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

	lambda_val = getvalue(lambda_var)

	primal_objval = getobjectivevalue(m_primal)


	return lambda_val, forest_left, forest_right, forest_product, forest_isLeaf, primal_objval, elapsed_time


end


function tcm_treeToA(tree_left, tree_right, tree_product, tree_isLeaf, assortments)
	M = size(assortments, 1)
	v_one_class = zeros(Float64,size(assortments))
    for m = 1 : M
        cn = 1;
        while (!tree_isLeaf[cn])
        	if (assortments[m, tree_product[cn]] > 0)
        		cn = tree_left[cn]
        	else
        		cn = tree_right[cn]
        	end
        end
        v_one_class[m, tree_product[cn]] = 1.0
    end

    return v_one_class
end



function tcm_solveTreeSubproblem_L1(N, u, assortments, depth_limit)
    # @objective(m_sub, Max, sum(u[m,p] * a[m,p] for m in 1:M, p in 1:N))
    # solve(m_sub)

    # x_val = getvalue(x)
    # objval = getobjectivevalue(m_sub)

    # single_ranking = sum(x_val, 2)
 #    single_ranking = single_ranking[:]
    # single_ordering = sortperm(single_ranking, rev = true)

    # return single_ordering, objval

    M = size(assortments,1)

    single_left = Int64[0]
    single_right = Int64[0]
    single_product = Int64[N]
    single_isLeaf = Bool[true]
    single_depth = Int64[1]
    single_include_set = Array{Int64}[ [N] ];
    single_assortment_binary = Array{Int64}[ ones(M) ]
    # single_reduced_cost = Float64[ sum(u[:,N]) ] #Float64[0.0];
    single_reduced_cost = Float64[ -Inf ] #Float64[0.0];
    single_isClosed = Bool[false]

    iter = 0 ;



    while (true)
        current_reduced_cost = sum( single_reduced_cost[single_isLeaf])
        println("\t\t Subproblem Iter : ", iter, " -- reduced cost: ", current_reduced_cost, " -- num leaves: ", sum(single_isLeaf))
        candidate_leaves = find(single_isLeaf .& (single_depth .< depth_limit) .&  (.!single_isClosed) )

        if (isempty(candidate_leaves))
            println("\t\t\t No more eligible leaves to split; exiting procedure...")
            break;
        end 

        num_candL = length(candidate_leaves)
        candidate_reduced_costs = -Inf* ones(num_candL)
        best_split_p_by_leaf = zeros(Int64, num_candL)
        best_left_leaf_p_by_leaf  = zeros(Int64, num_candL)
        best_right_leaf_p_by_leaf  = zeros(Int64, num_candL)
        best_left_reduced_cost_by_leaf  = zeros(Float64, num_candL)
        best_right_reduced_cost_by_leaf  = zeros(Float64, num_candL)
        best_left_assortment_binary_by_leaf = Array{Array{Int64}}(num_candL)
        best_right_assortment_binary_by_leaf = Array{Array{Int64}}(num_candL)


        for ell_ind in 1:num_candL

            leaf = candidate_leaves[ell_ind]

            assortment_binary = copy(single_assortment_binary[leaf])

            candidate_split_products = setdiff(collect(1:(N-1)), single_include_set[leaf])

            for split_p in candidate_split_products
                
                candidate_left_leaf_products = union(single_include_set[leaf], split_p)
                best_left_reduced_cost = -Inf
                best_left_p = 0; 
                best_left_assortment_binary = zeros(M)
                for left_p in candidate_left_leaf_products
                    left_assortment_binary = assortment_binary .* assortments[:, split_p][:]
                    temp = dot( left_assortment_binary, u[:, left_p])
                    if (temp > best_left_reduced_cost)
                        best_left_reduced_cost = temp
                        best_left_p = left_p
                        best_left_assortment_binary = copy(left_assortment_binary)
                    end
                end

                candidate_right_leaf_products = copy(single_include_set[leaf])
                best_right_reduced_cost = -Inf
                best_right_p = 0; 
                best_right_assortment_binary = zeros(M)
                for right_p in candidate_right_leaf_products
                    right_assortment_binary = assortment_binary .* (1 - assortments[:,split_p][:])
                    temp = dot(right_assortment_binary, u[:, right_p])
                    if (temp > best_right_reduced_cost)
                        best_right_reduced_cost = temp
                        best_right_p = right_p
                        best_right_assortment_binary = copy(right_assortment_binary)
                    end
                end

                new_reduced_cost = best_left_reduced_cost + best_right_reduced_cost

                

                if (new_reduced_cost > candidate_reduced_costs[ell_ind] )
                    candidate_reduced_costs[ell_ind] = new_reduced_cost
                    best_split_p_by_leaf[ell_ind] = split_p 
                    best_left_leaf_p_by_leaf[ell_ind] = best_left_p 
                    best_right_leaf_p_by_leaf[ell_ind] = best_right_p 
                    best_left_reduced_cost_by_leaf[ell_ind] = best_left_reduced_cost
                    best_right_reduced_cost_by_leaf[ell_ind] = best_right_reduced_cost
                    best_left_assortment_binary_by_leaf[ell_ind] = best_left_assortment_binary # VVM: should there be a copy here?
                    best_right_assortment_binary_by_leaf[ell_ind] = best_right_assortment_binary # VVM: should there be a copy here?
                end
            end
        end

        incumbent_reduced_costs = single_reduced_cost[candidate_leaves]

        tol_RC = 1e-7

        for ell_ind in 1:num_candL
            if (incumbent_reduced_costs[ell_ind] + tol_RC < candidate_reduced_costs[ell_ind])
                # Splitting this leaf does improve things, so go ahead with 

                leaf = candidate_leaves[ell_ind]

                # Grow tree. 
                numNodes = length(single_left)
                push!(single_left, 0, 0)
                push!(single_right, 0, 0)
                push!(single_isLeaf, true, true)
                push!(single_product, best_left_leaf_p_by_leaf[ell_ind], best_right_leaf_p_by_leaf[ell_ind])
                push!(single_depth, single_depth[leaf]+1, single_depth[leaf]+1)
                push!(single_include_set, union(single_include_set[leaf],best_split_p_by_leaf[ell_ind]), copy(single_include_set[leaf]) )
                push!(single_assortment_binary, copy(best_left_assortment_binary_by_leaf[ell_ind]), copy(best_right_assortment_binary_by_leaf[ell_ind]))
                push!(single_reduced_cost, best_left_reduced_cost_by_leaf[ell_ind], best_right_reduced_cost_by_leaf[ell_ind])
                push!(single_isClosed, false, false)

                # Update the leaf 
                single_left[leaf] = numNodes+1
                single_right[leaf] = numNodes+2
                single_product[leaf] = best_split_p_by_leaf[ell_ind]
                single_isClosed[leaf] = true
                single_isLeaf[leaf] = false
            else
                # Leaf does not improve 
                leaf = candidate_leaves[ell_ind]
                single_isClosed[leaf] = true
            end
        end
    end

    optimal_reduced_cost = sum( single_reduced_cost[single_isLeaf] )

    return single_left, single_right, single_product, single_isLeaf, optimal_reduced_cost

end