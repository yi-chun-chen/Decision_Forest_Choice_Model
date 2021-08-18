function tcm_estimateHALOMNL(N, assortments, transaction_counts)
	M = size(assortments, 1)

	total_trans_by_assortment = sum(transaction_counts, 2)

	m_HALOMNL = Model(solver = IpoptSolver(print_level=1))

	@variable(m_HALOMNL, -20 <= u[1:N-1] <= 20)
	@variable(m_HALOMNL, -20 <= alph[1:N-1, 1:N-1] <= 20)


	@NLobjective(m_HALOMNL, Max, sum(sum( transaction_counts[m,p] * (u[p] + sum(alph[q,p]*(1 - assortments[m,q]) for q in 1:N-1 ) ) for p in 1:N-1) for m in 1:M) 
							- sum(total_trans_by_assortment[m]*log(1+sum(assortments[m,p]* exp(u[p] + sum(alph[q,p]*(1 - assortments[m,q]) for q in 1:N-1 )) for p in 1:N-1)) for m in 1:M) )

	start_time = time()
	solve(m_HALOMNL)
	end_time = time()
	elapsed_time = end_time - start_time

	u_val = getvalue(u)
	alpha_val = getvalue(alph)

	obj = getobjectivevalue(m_HALOMNL)

	return u_val, alpha_val, obj, elapsed_time
end