function tcm_predictRanking(N, K, lambda, orderings, assortments)
	M = size(assortments,1)
	v = zeros(size(assortments))
	v_by_class = Array{ Array{Float64,2}, 1}(K)
	for k in 1:K
		v_one_class = zeros(Float64,size(assortments))
	    for m = 1 : M
	        for j in 1:N
	        	if (assortments[m, orderings[k,j]] > 0)
	        		v_one_class[m, orderings[k,j]] = 1
	        		break;
	        	end
	        end
	    end
		v += lambda[k] * v_one_class

		v_by_class[k] = v_one_class
	end

	return v, v_by_class
end