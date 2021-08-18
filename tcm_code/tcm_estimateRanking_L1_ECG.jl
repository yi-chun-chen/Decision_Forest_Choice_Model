include("../tcm_code/tcm_estimateForest.jl")

function tcm_estimateRanking_L1_ECG(N, assortments, choice_probs, time_limit_overall)

	M = size(assortments,1)

	dim_V = sum(assortments)

	m_dual = Model(solver = GurobiSolver(OutputFlag = 0))
	@variable(m_dual, -1 <= p[1:M, 1:N] <= +1)
	@variable(m_dual, q)

	@objective(m_dual, Max, sum( p[m,i]* choice_probs[m,i] for m in 1:M, i in 1:N ) + q)


    orderings = zeros(Int64,0,N)
	v_by_class = Array{Int64,2}[];

	tol_RC = -1e-4

    # numInitializations_LS = 5


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






	start_time = time()

	solve(m_dual)

	dual_objval = getobjectivevalue(m_dual)

	iter = 0;

    println("Ranking CG iter ", iter, " -- time elapsed = ", 0.0, " objective = ",dual_objval)

	while (time() - start_time < time_limit_overall)
        iter += 1

        p_val = getvalue(p)
        q_val = getvalue(q)

        # single_ordering, sub_objval = tcm_solveRankingSubproblem_L1(N, p_val, assortments, numInitializations_LS)
        single_ordering, sub_objval = solve_subproblem(p_val)

        # @show single_ordering
        # @show typeof(single_ordering)

        RC = -sub_objval - q_val

        println("Ranking CG iter ", iter, " -- time elapsed = ", time() - start_time, " objective = ", dual_objval, " -- RC = ", RC)

        if (RC >= tol_RC)
        	# Stop here -- we are at optimality
        	println("\t RC = ", RC, " -- terminating ... ")
        	break;
        else
        	# Add the ranking to the dual problem, and re-solve the master.
        	orderings = [orderings; single_ordering']

        	v_one_class = tcm_orderingToA( single_ordering, assortments)
        	push!(v_by_class, v_one_class)
        	@constraint(m_dual, sum( v_one_class[m,i] * p[m,i] for m in 1:M, i in 1:N) + q <= 0)

        	solve(m_dual)
        	dual_objval = getobjectivevalue(m_dual)
        end

        # break;
	end

	end_time = time()
	elapsed_time = end_time - start_time

	# Have terminated; solve primal to get what the lambda's are. 
	K = size(orderings,1)

	m_primal = Model(solver = GurobiSolver())
	@variable(m_primal, lambda_var[1:K] >= 0)
	@variable(m_primal, eps_plus[1:M,1:N] >= 0)
	@variable(m_primal, eps_minus[1:M,1:N] >= 0)

	for m in 1:M
		for i in 1:N
			@constraint(m_primal, sum( v_by_class[k][m,i] * lambda_var[k] for k in 1:K) + eps_plus[m,i] - eps_minus[m,i] == choice_probs[m,i])
		end
	end
	@constraint(m_primal, sum(lambda_var[k] for k in 1:K) == 1)

	@objective(m_primal, Min, sum( eps_minus[m,i] for m in 1:M, i in 1:N) + sum(eps_plus[m,i] for m in 1:M, i in 1:N))

	solve(m_primal)

	lambda_val = getvalue(lambda_var)

	primal_objval = getobjectivevalue(m_primal)


	return lambda_val, orderings, primal_objval, elapsed_time


end


function tcm_orderingToA(single_ordering, assortments)
	M, N = size(assortments)

	v_one_class = zeros(Float64,size(assortments))
        for m = 1 : M
            for j in 1:N
                if (assortments[m, single_ordering[j]] > 0)
                    v_one_class[m, single_ordering[j]] = 1
                    break;
                end
            end
        end

    return v_one_class
end



# function tcm_solveRankingSubproblem_L1(N, u, assortments, numInitializations_LS)

#     orderings_by_rep = zeros(Int64,numInitializations_LS, N)
#     objval_by_rep = zeros(numInitializations_LS)

#     for rep in 1:numInitializations_LS
#         current_solution = randperm(N)
#         v = tcm_orderingToA(current_solution, assortments)
#         current_obj = dot(u, v)

#         while (true)
#             best_solution = copy(current_solution)
#             best_obj = current_obj

#             for i in 1:(N-1)
#                 for j in (i+1):N
#                     candidate_solution = copy(current_solution)
#                     temp = candidate_solution[i]
#                     candidate_solution[i] = candidate_solution[j]
#                     candidate_solution[j] = temp

#                     v = tcm_orderingToA(candidate_solution, assortments)
#                     candidate_obj = dot(u, v)

#                     if (candidate_obj > best_obj)
#                         best_solution = candidate_solution
#                         best_obj = candidate_obj
#                     end
#                 end
#             end

#             if ( best_obj > current_obj)
#                 current_solution = best_solution
#                 current_obj = best_obj
#             else
#                 break;
#             end
#         end

#         orderings_by_rep[rep,:] = current_solution
#         objval_by_rep[rep] = current_obj
#     end

#     best_rep = indmax(objval_by_rep)
#     best_ordering = orderings_by_rep[best_rep, :]
#     best_objval = objval_by_rep[best_rep]

#     return best_ordering, best_objval
# end