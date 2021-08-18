function tcm_estimateForest_RTS(N, assortments, transaction_counts, K, depth, perTol_EM, time_limit_EM)

	forest_left = Array{Int64,1}[]
	forest_right = Array{Int64,1}[]
	forest_product = Array{Int64,1}[]
	forest_isLeaf = Array{Bool,1}[]

	for k in 1:K
		tree_left, tree_right, tree_product, tree_isLeaf = tcm_randomCompleteTree( N, depth-1 )
		push!(forest_left, tree_left)
		push!(forest_right, tree_right)
		push!(forest_product, tree_product)
		push!(forest_isLeaf, tree_isLeaf)
	end

	initial_lambda = 1/K * ones(K)

	# time_limit_EM = 

	println("Starting EM...")
	# perTol_EM = 1e-5

    lambda, v, loglik, elapsed_time = tcm_forest_EM_RTS(N, K, forest_left, forest_right, forest_product, forest_isLeaf, assortments, transaction_counts, time_limit_EM, perTol_EM, initial_lambda)


    return lambda, forest_left, forest_right, forest_product, forest_isLeaf, loglik, elapsed_time
end

function tcm_estimateForest_RTS_withInitial(N, assortments, transaction_counts, K_sample, depth, perTol_EM, time_limit_EM,
											initial_left, initial_right, initial_product, initial_isLeaf)

	forest_left = deepcopy(initial_left) #Array{Int64,1}[]
	forest_right = deepcopy(initial_right) #Array{Int64,1}[]
	forest_product = deepcopy(initial_product) #Array{Int64,1}[]
	forest_isLeaf = deepcopy(initial_isLeaf) #Array{Bool,1}[]

	for k in 1:K_sample
		tree_left, tree_right, tree_product, tree_isLeaf = tcm_randomCompleteTree( N, depth-1 )
		push!(forest_left, tree_left)
		push!(forest_right, tree_right)
		push!(forest_product, tree_product)
		push!(forest_isLeaf, tree_isLeaf)
	end

	K = length(initial_left) + K_sample;

	initial_lambda = 1/K * ones(K)

	# time_limit_EM = 

	println("Starting EM...")
	# perTol_EM = 1e-5

    lambda, v, loglik, elapsed_time = tcm_forest_EM_RTS(N, K, forest_left, forest_right, forest_product, forest_isLeaf, assortments, transaction_counts, time_limit_EM, perTol_EM, initial_lambda)


    return lambda, forest_left, forest_right, forest_product, forest_isLeaf, loglik, elapsed_time
end



function tcm_forest_EM_RTS(N, K, forest_left, forest_right, forest_product, forest_isLeaf, assortments, transaction_counts, time_limit_EM, perTol_EM, initial_lambda)
	lambda = copy(initial_lambda); #1/K * ones(K)
    v, v_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, assortments)
    logv = log.(v)
    logv[ isinf.(logv)] = 0.0; 
    loglik = dot( convert(Array{Float64,}, transaction_counts), logv)

    # @show v
    # @show loglik


	v = zeros(size(transaction_counts))

	# perTol_EM = 1e-6
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

		# println("\t Ranking EM: iter ", iter, " -- time elapsed = ", time() - start_time, " log likelihood = ", new_loglik)

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