function tcm_createGrandAssortments(N)
	ass_ints = collect(0:(2^(N-1) - 1))
	M = length(ass_ints)
	assortments = zeros(M, N)
	for m in 1:M
		# @show bin(temp[m],N-1)
		assortments[m,1:(N-1)] = map(x -> parse(string(x)), [bin(ass_ints[m],N-1)...] )
	end

	assortments[:,N] = 1

	return assortments
end