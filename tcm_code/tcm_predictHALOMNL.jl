function tcm_predictHALOMNL(N, u, alpha_val, assortments)
	M = size(assortments,1)
	v = zeros(size(assortments))
	for m = 1 : M
		v[m,N] = 1.0
		for p in 1:N-1
			if (assortments[m,p] > 0)
				v[m,p] = exp(u[p] + dot( (1 - assortments[m,1:N-1]), alpha_val[:, p] )   )
			end
		end
		v[m,:] = v[m,:] / sum(v[m,:])
	end

	return v
end

