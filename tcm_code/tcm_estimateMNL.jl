using JuMP, Ipopt 

function tcm_estimateMNL(N, assortments, transaction_counts)

	M = size(assortments, 1)

	total_trans_by_assortment = sum(transaction_counts, 2)

	m_MNL = Model(solver = IpoptSolver(print_level=1))

	@variable(m_MNL, -20 <= u[1:N-1] <= 20)

	@NLobjective(m_MNL, Max, sum(sum( transaction_counts[m,p] * u[p] for p in 1:N-1) for m in 1:M) - sum(total_trans_by_assortment[m]*log(1+sum(assortments[m,p]* exp(u[p]) for p in 1:N-1)) for m in 1:M) )

	start_time = time()
	solve(m_MNL)
	end_time = time()
	elapsed_time = end_time - start_time

	u_val = getvalue(u)

	obj = getobjectivevalue(m_MNL)

	return u_val, obj, elapsed_time
end