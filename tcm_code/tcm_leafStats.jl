function tcm_leafStats(lambda, forest_isLeaf)
	K = length(forest_isLeaf)

	numLeaves_by_tree = [ sum(forest_isLeaf[k]) for k in 1:K ]

	# @show numLeaves_by_tree

	simple_avg_leaves = mean(numLeaves_by_tree)
	weighted_avg_leaves = dot(lambda, numLeaves_by_tree)

	max_leaves = maximum(numLeaves_by_tree)

	return simple_avg_leaves, weighted_avg_leaves, max_leaves
end



function tcm_depthStats(lambda, forest_left, forest_right, forest_isLeaf)

	K = length(forest_isLeaf)
	depth_by_tree = zeros(K)

	for k in 1:K
		depth_vec = zeros(size(forest_left[k]))

		depth_vec[1] = 1

		for cn in 1:length(depth_vec)
			if (!forest_isLeaf[k][cn])
				depth_vec[forest_left[k][cn]] = depth_vec[cn] + 1
				depth_vec[forest_right[k][cn]] = depth_vec[cn] + 1
			end
		end

		depth_by_tree[k] = maximum(depth_vec)
	end

	simple_avg_depth = mean(depth_by_tree)
	weighted_avg_depth = dot(lambda, depth_by_tree)
	max_depth = maximum(depth_by_tree)

	return simple_avg_depth, weighted_avg_depth, max_depth
end