function tcm_predictMNL(N, u, assortments)
	M = size(assortments,1)

	v = zeros(Float64,size(assortments))

    for m = 1 : M
        v[m,N] = 1.0
        for p = 1 : N-1
            v[m,p] = exp(u[p]) * assortments[m,p]
        end
        v[m,:] = v[m,:] / sum(v[m,:])
    end

    return v
end