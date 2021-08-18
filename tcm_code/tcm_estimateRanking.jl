using JuMP, Gurobi

function tcm_estimateRanking(N, assortments, transaction_counts, time_limit_overall, perTol)
    M = size(assortments, 1)

	m_sub = Model(solver = GurobiSolver(OutputFlag = 0))
	@variable(m_sub, x[1:N,1:N], Bin)
	@variable(m_sub, a[1:M,1:N], Bin)

	# VVM - constraint below is not in van Ryzin and Vulcano, but maybe its needed?
	for m in 1:M
        @constraint(m_sub,sum( a[m,p1] for p1 in 1:N) == 1)
    end


	for m in 1:M
        for p1 in 1:N
            @constraint(m_sub,a[m,p1] <= assortments[m,p1])
            for p2 in 1:N
                if ( (p1 != p2) & (assortments[m,p1] == 1) & (assortments[m,p2] == 1) )
    	            @constraint(m_sub, a[m,p1] <= x[p1,p2])
                end
           end
        end
    end

    for p1 in 1:N
        for p2 in 1:N
            if p1 != p2
                @constraint(m_sub, x[p1,p2] + x[p2,p1] == 1)
            end
        end
    end


    for p1 in 1:N
        for p2 in 1:N
            for p3 in 1:N
                if ( (p1 != p2) & (p2 != p3) & (p1 != p3) )
                    @constraint(m_sub, x[p1,p2] + x[p2,p3] + x[p3,p1] <= 2)
                end
            end
        end
    end

    function solve_subproblem(u)
    	@objective(m_sub, Max, sum(u[m,p] * a[m,p] for m in 1:M, p in 1:N))
    	solve(m_sub)

    	x_val = getvalue(x)
    	objval = getobjectivevalue(m_sub)

    	single_ranking = sum(x_val, 2)
        single_ranking = single_ranking[:]
    	single_ordering = sortperm(single_ranking, rev = true)

    	return single_ordering, objval
    end

    time_limit_EM = 60.0

    start_time = time()

    K = N
    orderings = zeros(Int64, K, N)
    for k in 1:(N-1)
    	single_ordering = [k; N; collect(1:(k-1)); collect( (k+1):(N-1) )]
    	orderings[k,:] = single_ordering
    end
    orderings[N,:] = [N; collect(1:(N-1))]

    # 
    initial_lambda = 1/K * ones(K)
    lambda, v, loglik, elapsed_time = tcm_ranking_EM(N, K, orderings, assortments, transaction_counts, time_limit_EM, initial_lambda)
	u = transaction_counts ./ v
    # if (any(isinf.(u)) | any(isnan.(u)))
    #     @show u
    #     @show lambda
    #     @show orderings
    #     error("Stop here - pre loop")
    # end
	u[ isinf.(u)] = 0.0
	u[ isnan.(u)] = 0.0



	#perTol = 1e-7
    iter = 0;

    #@show loglik
    println("CG iter ", iter, " -- time elapsed = ", 0.0, " log likelihood = ",loglik)

	while (time() - start_time < time_limit_overall)
        iter += 1

		u = transaction_counts ./ v
        # if (any(isinf.(u)) | any(isnan.(u)))
        #     @show u
        #     error("Stop here")
        # end
		u[ isinf.(u)] = 0.0
		u[ isnan.(u)] = 0.0
		single_ordering, objval = solve_subproblem(u)

		orderings = [orderings; single_ordering']
		K += 1

        initial_fudge = 0.999
        initial_lambda = [ initial_fudge*lambda; (1-initial_fudge)]
		lambda, v, new_loglik, elapsed_time = tcm_ranking_EM(N, K, orderings, assortments, transaction_counts, time_limit_EM, initial_lambda)
		
        println("CG iter ", iter, " -- time elapsed = ", time() - start_time, " log likelihood = ",new_loglik)

        if ( (new_loglik - loglik) / abs(loglik) < perTol )
        	temp = (new_loglik - loglik) / abs(loglik)
        	@show temp 
        	@show new_loglik
        	@show loglik

        	loglik = new_loglik
			break
		end

        loglik = new_loglik
	end

	end_time = time()

	elapsed_time = end_time - start_time

	return lambda, orderings, loglik, elapsed_time
end




function tcm_ranking_EM(N, K, orderings, assortments, transaction_counts, time_limit_EM, initial_lambda)
	lambda = copy(initial_lambda); #1/K * ones(K)
    v, v_by_class = tcm_predictRanking(N, K, lambda, orderings, assortments)
    logv = log.(v)
    logv[ isinf.(logv)] = 0.0; 
    loglik = dot( convert(Array{Float64,}, transaction_counts), logv)

    # @show v
    # @show loglik


	v = zeros(size(transaction_counts))

	perTol_EM = 1e-5
    iter = 0;

	start_time = time()

	while (time() - start_time < time_limit_EM)
		iter += 1;
		h, v, v_by_class = tcm_ranking_E(N, K, lambda, orderings, assortments)
        # @show lambda 
		lambda = tcm_ranking_M_update_lambda(K, h, transaction_counts)

        # @show lambda 

        # error("Stop here - EM loop")

		v, v_by_class = tcm_predictRanking(N, K, lambda, orderings, assortments)
		logv = log.(v)
		logv[ isinf.(logv)] = 0.0; 
		new_loglik = dot( convert(Array{Float64,}, transaction_counts), logv)

		# println("\t Ranking EM: iter ", iter, " -- time elapsed = ", time() - start_time, " log likelihood = ", new_loglik)

	    if ( (new_loglik - loglik) / abs(loglik) < perTol_EM )
	    	temp = (new_loglik - loglik) / abs(loglik)
	    	# @show temp 
	    	# @show new_loglik
	    	# @show loglik

	    	loglik = new_loglik
			break
		end

        # if (iter > 100)
        #     error("Stop here")
        # end

		loglik = new_loglik
	end

	elapsed_time = time() - start_time

	return lambda, v, loglik, elapsed_time

end



function tcm_ranking_M_update_lambda(K, h, transaction_counts)
	sum_of_h = zeros(K)

	for k in 1:K
		sum_of_h[k] = dot( h[k], convert(Array{Float64}, transaction_counts))
	end

	lambda = sum_of_h ./ sum(sum_of_h)

	return lambda
end



function tcm_ranking_E(N, K, lambda, orderings, assortments)
	v, v_by_class = tcm_predictRanking(N, K, lambda, orderings, assortments)

	h = Array{ Array{Float64,2}, 1}(K)
	for k in 1:K
		single_h = lambda[k] * v_by_class[k] ./ v
		single_h[ isnan.(single_h)] = 0.0;

		h[k] = single_h
	end

	return h, v, v_by_class
end