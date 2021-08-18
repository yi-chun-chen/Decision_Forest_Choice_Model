include("tcm_estimateForest.jl")

function tcm_estimateForest_leafLimit(N, assortments, transaction_counts, time_limit_overall, perTol, leaf_limit, initial_left, initial_right, initial_product, initial_isLeaf)
    M = size(assortments, 1)

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
        single_left, single_right, single_product, single_isLeaf, objval = tcm_solveTreeSubproblem_leafLimit(N, u, assortments, leaf_limit)
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


function tcm_solveTreeSubproblem_leafLimit(N, u, assortments, leaf_limit)
    M = size(assortments,1)

    single_left = Int64[0]
    single_right = Int64[0]
    single_product = Int64[N]
    single_isLeaf = Bool[true]
    single_depth = Int64[1]
    single_include_set = Array{Int64}[ [N] ];
    single_assortment_binary = Array{Int64}[ ones(M) ]
    single_reduced_cost = Float64[0.0];
    single_isClosed = Bool[false]

    iter = 0 ;



    while (true)
        current_reduced_cost = sum( single_reduced_cost[single_isLeaf])
        println("\t\t Subproblem Iter : ", iter, " -- reduced cost: ", current_reduced_cost, " -- num leaves: ", sum(single_isLeaf))
        candidate_leaves = find(single_isLeaf .&  (.!single_isClosed) )

        if (isempty(candidate_leaves))
            println("\t\t\t No more eligible leaves to split; exiting procedure...")
            break;
        end 

        if (sum(single_isLeaf) >= leaf_limit)
            println("\t\t\t Leaf limit reached; exiting procedure...")
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

                if (new_reduced_cost > candidate_reduced_costs[ell_ind])
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

        # Find most improving leaf: 
        ell_ind = indmax(candidate_reduced_costs - incumbent_reduced_costs)
        tol_RC = 1e-4

        if (incumbent_reduced_costs[ell_ind] + tol_RC < candidate_reduced_costs[ell_ind])
            # Splitting this leaf does improve things, so go ahead with it.

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
            # push!(single_isClosed, false, false)

            # Update the leaf 
            single_left[leaf] = numNodes+1
            single_right[leaf] = numNodes+2
            single_product[leaf] = best_split_p_by_leaf[ell_ind]
            # single_isClosed[leaf] = true
            single_isLeaf[leaf] = false
        else
            # Leaf does not improve 
            println("\t\t\t No more improvement; exiting procedure...")
            break;
        end
    end

    optimal_reduced_cost = sum( single_reduced_cost[single_isLeaf] )

    return single_left, single_right, single_product, single_isLeaf, optimal_reduced_cost

end