include("tcm_estimateForest_L1.jl")

function tcm_estimateForest_L1_MIO(N, assortments, choice_probs, time_limit_overall, time_limit_sub, depth_limit, initial_left, initial_right, initial_product, initial_isLeaf)

	M = size(assortments,1)

	dim_V = sum(assortments)


	# START SUBPROBLEM MIP

	tree_left, tree_right, tree_isLeaf, left_leaves, right_leaves = tcm_createCompleteTree(depth_limit)
	num_nodes = length(tree_isLeaf)
	leaves = find(tree_isLeaf)
	splits = find(!tree_isLeaf)

	m_sub = Model(solver = GurobiSolver(OutputFlag = 0, TimeLimit = time_limit_sub))
	@variable(m_sub, y_var[1:num_nodes, 1:N], Bin)
	@variable(m_sub, w_var[leaves, 1:M], Bin)
	@variable(m_sub, u_var[leaves, 1:M, 1:N], Bin) # u_ell,m,o. 
	@variable(m_sub, A_var[1:M, 1:N], Bin)

	for cn in 1:num_nodes
		if (!tree_isLeaf[cn])
			@constraint(m_sub, sum(y_var[cn,i] for i in 1:(N-1)) == 1) #EC.51b
			@constraint(m_sub, y_var[cn,N] == 0)
		else 
			@constraint(m_sub, sum(y_var[cn,i] for i in 1:N) == 1) #EC.51c
		end
	end

	for m in 1:M
		@constraint(m_sub, sum(w_var[cn,m] for cn in leaves) == 1) # EC.51d

		for cn in splits
			@constraint(m_sub, sum(w_var[ell,m] for ell in left_leaves[cn]) <= sum( assortments[m,i] * y_var[cn,i] for i in 1:(N-1)) )
			@constraint(m_sub, sum(w_var[ell,m] for ell in right_leaves[cn]) <= 1 - sum( assortments[m,i] * y_var[cn,i] for i in 1:(N-1)) )
		end

		for ell in leaves
			@constraint(m_sub, w_var[ell,m] == sum(u_var[ell,m,i] for i in 1:N))

			for i in 1:N
				@constraint(m_sub, u_var[ell,m,i] <= y_var[ell,i])
			end
		end

		for i in 1:N
			@constraint(m_sub, A_var[m,i] == sum( u_var[ell, m, i] for ell in leaves))
		end
	end


	# VVM: we need another constraint here. 
	# Need to ensure tree is well-defined. 
	for ell in leaves
		for i in 1:(N-1)
			@constraint(m_sub, y_var[ell,i] <= sum( y_var[s,i] for s in splits if ell in left_leaves[s]))
		end
	end

	# VVM: yet another constraint. This one is to ensure that a product 
	# appears in a split only once along any given path. (This is to prevent
	# weird cases where e.g., we split on product 3, then the left child again splits
	# on product 3.) 
	for ell in leaves
		left_splits = filter( s -> (ell in left_leaves[s]) , splits)
		right_splits = filter( s -> (ell in right_leaves[s]) , splits)
		@show left_splits
		@show right_splits
		@show leaves
		@show splits 
		for i in 1:(N-1)
			@constraint(m_sub, sum(y_var[s,i] for s in left_splits) + sum(y_var[s,i] for s in right_splits) <= 1)
		end
	end

	#@objective(m_sub, Max)

	# tree_product = getvalue(y_var) * collect(1:N) 


	# END SUBPROBLEM MIP





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

        #single_left, single_right, single_product, single_isLeaf, sub_objval = tcm_solveTreeSubproblem_L1(N, p_val, assortments, depth_limit)

        @objective(m_sub, Max, sum( p_val[m,i] * A_var[m,i] for m in 1:M, i in 1:N))
        solve(m_sub)
        sub_objval = getobjectivevalue(m_sub)
        
        single_left = copy(tree_left)
        single_right = copy(tree_right)
        # temp = getvalue(y_var) * collect(1:N)
        single_product = zeros(Int64, num_nodes)
		y_val = getvalue(y_var)
		for cn in 1:num_nodes
			single_product[cn] = find(  y_val[cn,:] .> 0.5 )[1]
		end
        # @show temp 
        # single_product = convert( Array{Int64,1}, temp )
        single_isLeaf = copy(tree_isLeaf)


        # @show sub_objval
        v_one_class = tcm_treeToA( single_left, single_right, single_product, single_isLeaf, assortments)

        # @show v_one_class
        # @show getvalue(A_var)

        # assert( sum( abs.(v_one_class - getvalue(A_var) ) ) < 1e-7 )
        @show sum( abs.(v_one_class - getvalue(A_var) ) )

        if ( sum( abs.(v_one_class - getvalue(A_var) ) ) >= 1e-7)
        	println("Problem step is here")
        end
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





function tcm_createCompleteTree(depth_limit)

	depth = depth_limit - 1
	# Step 1: create complete tree of given depth
	tree_left = Int64[]
	tree_right = Int64[]
	tree_isLeaf = Bool[]
	
	counter = 1;
	num_nodes = 0;

	for d in 0:(depth-1)
		num_d_nodes = 2^d;

		for i in 1:num_d_nodes
			push!(tree_left, counter + 1)
			push!(tree_right, counter + 2)

			push!(tree_isLeaf, false)
			counter += 2
			num_nodes += 1
		end
	end

	num_d_nodes = 2^depth;
	for i in 1:num_d_nodes
		push!(tree_left, 0)
		push!(tree_right, 0)
		push!(tree_isLeaf, true)
		counter += 2
		num_nodes += 1
	end

	closed = zeros(Bool, num_nodes)
	predecessor = zeros(Int64, num_nodes)
	left_leaves = Array{Array{Int64,1}}(num_nodes)
	right_leaves = Array{Array{Int64,1}}(num_nodes)

	@show tree_left
	@show tree_right
	@show tree_isLeaf
	@show num_nodes

	cn = 1
	while (!all(closed))
		ln = tree_left[cn]
		predecessor[ln] = cn;
		leftIsCleared = true;
		if (tree_isLeaf[ln])
			left_leaves[cn] = [ln];
			closed[ln] = true;
		elseif (closed[ln])
			left_leaves[cn] = [ left_leaves[ln]; right_leaves[ln]];
		else
			leftIsCleared = false;
			cn = ln 
			continue
		end

		rn = tree_right[cn]
		predecessor[rn] = cn;
		rightIsCleared = true;
		if (tree_isLeaf[rn])
			right_leaves[cn] = [rn];
			closed[rn] = true;
		elseif (closed[rn])
			right_leaves[cn] = [left_leaves[rn]; right_leaves[rn]];
		else
			rightIsCleared = false;
			cn = rn 
			continue
		end

		if (leftIsCleared & rightIsCleared)
			closed[cn] = true;
			cn = predecessor[cn];
		end
	end

	return tree_left, tree_right, tree_isLeaf, left_leaves, right_leaves
end