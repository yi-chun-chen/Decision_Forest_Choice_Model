function tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, assortments)
	M = size(assortments,1)
	v = zeros(size(assortments))
	v_by_class = Array{ Array{Float64,2}, 1}(K)
	for t in 1:K
		v_one_class = zeros(Float64,size(assortments))
	    for m = 1 : M
	        cn = 1;
	        while (!forest_isLeaf[t][cn])
	        	if (assortments[m, forest_product[t][cn]] > 0)
	        		cn = forest_left[t][cn]
	        	else
	        		cn = forest_right[t][cn]
	        	end
	        end
	        v_one_class[m, forest_product[t][cn]] = 1.0
	    end
		v += lambda[t] * v_one_class

		v_by_class[t] = v_one_class
	end

	return v, v_by_class
end