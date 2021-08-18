function tcm_createRandomDistinctAssortments(M, N)
	temp = randperm(2^(N-1))[1:M] - 1

	assortments = zeros(M, N)
	for m in 1:M
		# @show bin(temp[m],N-1)
		assortments[m,1:(N-1)] = map(x -> parse(string(x)), [bin(temp[m],N-1)...] )
	end

	assortments[:,N] = 1

	return assortments
end