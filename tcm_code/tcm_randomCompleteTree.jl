function tcm_randomCompleteTree( N, depth )

	tree_left = Int64[]
	tree_right = Int64[]
	tree_product = Int64[]
	tree_isLeaf = Bool[]
	#tree_choice_prediction = Int64[]

	tree_unused_products = Array{ Array{Int64,1} }( 2^(depth+1) - 1)
	tree_include_set = Array{ Array{Int64,1} }( 2^(depth+1) - 1)

	tree_unused_products[1] = collect(1:(N-1))
	tree_include_set[1] = [N]

	counter = 1;
	num_nodes = 1;

	for d in 0:(depth-1)
		num_d_nodes = 2^d;

		#next_children_are_leaves = d == n_levels ? true : false;

		for i in 1:num_d_nodes
			push!(tree_left, counter + 1)
			push!(tree_right, counter + 2)

			temp = rand(tree_unused_products[num_nodes])

			push!(tree_product, temp)
			push!(tree_isLeaf, false)
			tree_unused_products[counter+1] = setdiff(tree_unused_products[num_nodes], temp)
			tree_unused_products[counter+2] = setdiff(tree_unused_products[num_nodes], temp)
			tree_include_set[counter+1] = [tree_include_set[num_nodes]; temp]
			tree_include_set[counter+2] = copy(tree_include_set[num_nodes])

			counter += 2
			num_nodes += 1
		end
	end

	num_d_nodes = 2^depth;
	for i in 1:num_d_nodes
		push!(tree_left, 0)
		push!(tree_right, 0)
		push!(tree_product, rand(tree_include_set[num_nodes]))
		push!(tree_isLeaf, true)
		counter += 2
		num_nodes += 1
	end

	return tree_left, tree_right, tree_product, tree_isLeaf 
end