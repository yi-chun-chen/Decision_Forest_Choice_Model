using JuMP, Gurobi

function tcm_estimateForest(N, assortments, transaction_counts, time_limit_overall, perTol, depth_limit, initial_left, initial_right, initial_product, initial_isLeaf)
    M = size(assortments, 1)
    @show M

    time_limit_EM = 60.0

    start_time = time()

    forest_left = deepcopy(initial_left)
    forest_right = deepcopy(initial_right)
    forest_product = deepcopy(initial_product)
    forest_isLeaf = deepcopy(initial_isLeaf)

    K = length(forest_left)

    # 
    initial_lambda = 1/K * ones(K)
    lambda, v, loglik, elapsed_time = tcm_forest_EM(N, K, forest_left, forest_right, forest_product, forest_isLeaf, assortments, transaction_counts, time_limit_EM, initial_lambda)

    # @show loglik
    
	u = transaction_counts ./ v
	u[ isinf.(u)] = 0.0 
	u[ isnan.(u)] = 0.0

    # NB - re. isinf.(...), this should not be necessary, if the initial model is such that every item assortment pair has a positive choice probability;
    # this is guaranteed for the ranking model because it starts from a set of rankings corresponding to an independent demand model. If the forest
    # is initialized with the ranking model, then the same condition will hold.
    # for isnan.(...), this is definitely necessary, because transaction_counts and v will both be zero at items that are not part of the assortment.    

	# perTol = 1e-7
    iter = 0;

    println("CG iter ", iter, " -- time elapsed = ", 0.0, " log likelihood = ",loglik)

	while (time() - start_time < time_limit_overall)
        iter += 1

		u = transaction_counts ./ v
		u[ isinf.(u)] = 0.0
		u[ isnan.(u)] = 0.0
		single_left, single_right, single_product, single_isLeaf, objval = tcm_solveTreeSubproblem(N, u, assortments, depth_limit)
        # error("Stop here")

		#orderings = [orderings; single_ordering']
        push!(forest_left, single_left)
        push!(forest_right, single_right)
        push!(forest_product, single_product)
        push!(forest_isLeaf, single_isLeaf)

		K += 1

        initial_fudge = 0.999
        initial_lambda = [ initial_fudge*lambda; (1-initial_fudge)]
		lambda, v, new_loglik, elapsed_time = tcm_forest_EM(N, K, forest_left, forest_right, forest_product, forest_isLeaf, assortments, transaction_counts, time_limit_EM, initial_lambda)
		
        println("CG iter ", iter, " -- time elapsed = ", time() - start_time, " log likelihood = ",new_loglik)

        if ( (new_loglik - loglik) / abs(loglik) < perTol )
        	temp = (new_loglik - loglik) / abs(loglik)
        	@show temp 
        	@show new_loglik
        	@show loglik

        	loglik = new_loglik
			break
		end

        loglik = new_loglik


	end

	end_time = time()

	elapsed_time = end_time - start_time

	return lambda, forest_left, forest_right, forest_product, forest_isLeaf, loglik, elapsed_time
end




function tcm_forest_EM(N, K, forest_left, forest_right, forest_product, forest_isLeaf, assortments, transaction_counts, time_limit_EM, initial_lambda)
	lambda = copy(initial_lambda); #1/K * ones(K)
    v, v_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, assortments)
    logv = log.(v)
    logv[ isinf.(logv)] = 0.0; 
    loglik = dot( convert(Array{Float64,}, transaction_counts), logv)

    # @show v
    # @show loglik


	v = zeros(size(transaction_counts))

	perTol_EM = 1e-5
    iter = 0;

	start_time = time()

	while (time() - start_time < time_limit_EM)
		iter += 1;
		h, v, v_by_class = tcm_forest_E(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, assortments)
		lambda = tcm_forest_M_update_lambda(K, h, transaction_counts)

		v, v_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, assortments)
		logv = log.(v)
		logv[ isinf.(logv)] = 0.0; 
		new_loglik = dot( convert(Array{Float64,}, transaction_counts), logv)

		println("\t Ranking EM: iter ", iter, " -- time elapsed = ", time() - start_time, " log likelihood = ", new_loglik)

	    if ( (new_loglik - loglik) / abs(loglik) < perTol_EM )
	    	temp = (new_loglik - loglik) / abs(loglik)
	    	# @show temp 
	    	# @show new_loglik
	    	# @show loglik

	    	loglik = new_loglik
			break
		end

        # if (iter > 100)
        #     error("Stop here")
        # end

		loglik = new_loglik
	end

	elapsed_time = time() - start_time

	return lambda, v, loglik, elapsed_time

end



function tcm_forest_M_update_lambda(K, h, transaction_counts)
	sum_of_h = zeros(K)

	for k in 1:K
		sum_of_h[k] = dot( h[k], convert(Array{Float64}, transaction_counts))
	end

	lambda = sum_of_h ./ sum(sum_of_h)

	return lambda
end



function tcm_forest_E(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, assortments)
	v, v_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, assortments)

	h = Array{ Array{Float64,2}, 1}(K)
	for k in 1:K
		single_h = lambda[k] * v_by_class[k] ./ v
		single_h[ isnan.(single_h)] = 0.0;

		h[k] = single_h
	end

	return h, v, v_by_class
end


function tcm_solveTreeSubproblem(N, u, assortments, depth_limit)
    # @objective(m_sub, Max, sum(u[m,p] * a[m,p] for m in 1:M, p in 1:N))
    # solve(m_sub)

    # x_val = getvalue(x)
    # objval = getobjectivevalue(m_sub)

    # single_ranking = sum(x_val, 2)
 #    single_ranking = single_ranking[:]
    # single_ordering = sortperm(single_ranking, rev = true)

    # return single_ordering, objval

    tol_RC = 1e-4

    M = size(assortments,1)

    single_left = Int64[0]
    single_right = Int64[0]
    single_product = Int64[N]
    single_isLeaf = Bool[true]
    single_depth = Int64[1]
    single_include_set = Array{Int64}[ [N] ];
    single_assortment_binary = Array{Int64}[ ones(M) ]
    single_reduced_cost = Float64[0.0];
    # single_reduced_cost = Float64[ sum(u[:,N]) - 2*tol_RC ] #
    single_isClosed = Bool[false]

    iter = 0 ;



    while (true)
        iter += 1
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

        tol_RC = 1e-4

        for ell_ind in 1:num_candL
            # @show incumbent_reduced_costs[ell_ind]
            # @show candidate_reduced_costs[ell_ind]

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


function tcm_convertRankingToForest(N, orderings)
    K = size(orderings,1)

    forest_left = Array{ Array{Int64,1} }(K)
    forest_right = Array{ Array{Int64,1} }(K)
    forest_product = Array{ Array{Int64,1} }(K)
    forest_isLeaf = Array{ Array{Bool,1} }(K)


    for k in 1:K
        cn = 1;
        single_left = Int64[]
        single_right = Int64[]
        single_product = Int64[]
        single_isLeaf = Bool[]

        for i in 1:N
            if (orderings[k,i] == N)
                push!(single_left, 0)
                push!(single_right, 0)
                push!(single_product, N)
                push!(single_isLeaf, true)
                break;
            else
                # Grow tree by one leaf. 
                push!(single_left, cn+1, 0 )
                push!(single_right, cn+2, 0 )
                push!(single_product, orderings[k,i], orderings[k,i])
                push!(single_isLeaf, false, true)
                cn += 2;
            end
        end

        forest_left[k] = single_left
        forest_right[k] = single_right
        forest_product[k] = single_product
        forest_isLeaf[k] = single_isLeaf
    end

    return forest_left, forest_right, forest_product, forest_isLeaf
end


function tcm_createBasicForest(N)
    K = N
    orderings = zeros(Int64, K, N)
    for k in 1:(N-1)
        single_ordering = [k; N; collect(1:(k-1)); collect( (k+1):(N-1) )]
        orderings[k,:] = single_ordering
    end
    orderings[N,:] = [N; collect(1:(N-1))]

    forest_left, forest_right, forest_product, forest_isLeaf = tcm_convertRankingToForest(N, orderings)
    return forest_left, forest_right, forest_product, forest_isLeaf
end

