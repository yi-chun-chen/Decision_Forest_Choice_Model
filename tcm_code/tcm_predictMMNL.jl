function tcm_predictMMNL(N, K, class_probs, u, assortments)
	M = size(assortments,1)
	v = zeros(size(assortments))
	v_by_class = Array{ Array{Float64,2}, 1}(K)
	for k in 1:K
		v_one_class = zeros(Float64,size(assortments))
	    for m = 1 : M
	        v_one_class[m,N] = 1.0
	        for p = 1 : N-1
	            v_one_class[m,p] = exp(u[k,p]) * assortments[m,p]
	        end
	        v_one_class[m,:] = v_one_class[m,:] / sum(v_one_class[m,:])
	    end
		v += class_probs[k] * v_one_class

		v_by_class[k] = v_one_class
	end

	return v, v_by_class
end