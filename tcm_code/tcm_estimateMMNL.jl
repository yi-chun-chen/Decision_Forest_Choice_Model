function tcm_estimateMMNL(N, K, assortments, transaction_counts, numInitializations, time_limit, perTol)

	results_class_probs = Array{Array{Float64}, 1}(numInitializations)
	results_u = Array{Array{Float64}, 1}(numInitializations)
	results_loglik = zeros(numInitializations)

	#perTol = 1e-7;

	grand_start_time = time()

	for i in 1:numInitializations
		start_time = time()

		u = 10*(rand(K,N) - 0.5)
		u[:,N] = 0
		class_probs = 1/K * ones(K) #rand(K)
		#class_probs = class_probs / sum(class_probs)

		v, v_by_class = tcm_predictMMNL(N, K, class_probs, u, assortments)
		logv = log.(v)
		logv[ isinf.(logv)] = 0.0; 
		loglik = dot( convert(Array{Float64,}, transaction_counts), logv)
		iter = 0;
		println("Init ", i, " -- iter ", iter, " MMNL iteration -- time elapsed = ", time() - start_time, " log likelihood = ",loglik)

		while ( (time() - start_time) < time_limit )
			iter += 1;
			h, v, v_by_class = tcm_MMNL_E(N, K, class_probs, u, assortments)
			u = tcm_MMNL_M_update_u(N, K, h, assortments, transaction_counts)
			class_probs = tcm_MMNL_M_update_class_probs(K, h, transaction_counts)

			v, v_by_class = tcm_predictMMNL(N, K, class_probs, u, assortments)
			logv = log.(v)
			logv[ isinf.(logv)] = 0.0; 
			new_loglik = dot( convert(Array{Float64,}, transaction_counts), logv)

	        println("Init ", i, " -- iter ", iter, " MMNL iteration -- time elapsed = ", time() - start_time, " log likelihood = ",loglik)

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

		results_loglik[i] = loglik
		results_u[i] = u
		results_class_probs[i] = class_probs
	end

	grand_end_time = time()

	best_i = indmax(results_loglik)
	u = results_u[best_i]
	class_probs = results_class_probs[best_i]
	loglik = results_loglik[best_i]

	grand_elapsed_time = grand_end_time - grand_start_time

	return u, class_probs, loglik, grand_elapsed_time
		
end



function tcm_MMNL_M_update_u(N, K, h, assortments, transaction_counts)
	u = zeros(K,N)

	for k in 1:K
		# @show h[k]
		# @show size(h[k])
		#@show size(h[k])
		#@show size(transaction_counts)
		weighted_transaction_counts = transaction_counts .* h[k]
		single_u, obj, elapsed_time = tcm_estimateMNL(N, assortments, weighted_transaction_counts)
		#@show single_u
		u[k,1:N-1] = single_u
	end

	return u 
end



function tcm_MMNL_M_update_class_probs(K, h, transaction_counts)
	sum_of_h = zeros(K)

	for k in 1:K
		sum_of_h[k] = dot( h[k], convert(Array{Float64}, transaction_counts))
	end

	class_probs = sum_of_h ./ sum(sum_of_h)

	return class_probs
end

function tcm_MMNL_E(N, K, class_probs, u, assortments)

	v, v_by_class = tcm_predictMMNL(N, K, class_probs, u, assortments)

	#@show size(v)
	#@show size(v_by_class[1])

	h = Array{ Array{Float64,2}, 1}(K)
	for k in 1:K
		single_h = class_probs[k] * v_by_class[k] ./ v
		single_h[ isnan.(single_h)] = 0.0;

		h[k] = single_h
	end

	return h, v, v_by_class
end
