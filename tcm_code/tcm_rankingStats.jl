function tcm_rankingStats(N, lambda, orderings)
	CS_size_by_ranking = (orderings .== N) * collect(0:(N-1))
	avg_CS_size = mean(CS_size_by_ranking)
	weighted_avg_CS_size = dot(lambda, CS_size_by_ranking)
	max_CS_size = maximum(CS_size_by_ranking)

	return avg_CS_size, weighted_avg_CS_size, max_CS_size
end