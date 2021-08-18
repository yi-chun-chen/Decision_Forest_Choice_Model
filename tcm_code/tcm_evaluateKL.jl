# function tcm_evaluateKL(N, v, transaction_counts)
# 	M = size(transaction_counts,1)

# 	empirical_probs = transaction_counts ./ sum(transaction_counts, 2)

# 	KL = 0.0; 
	
# 	for m in 1:M
# 		for i in 1:N
# 			if (transaction_counts[m,i] > 0)
# 				temp = transaction_counts[m,i] * log( v[m,i] / empirical_probs[m,i] ) 
# 				KL += temp
# 			end
# 		end
# 	end
	
# 	KL_avg = KL / sum(transaction_counts)

# 	return -KL_avg
# end


function tcm_evaluateKL(N, v, transaction_counts)
	M = size(transaction_counts,1)

	empirical_probs = transaction_counts ./ sum(transaction_counts, 2)


	KL_noinf = 0.0
	total_noinf_transactions = 0;
	
	for m in 1:M
		for i in 1:N
			if (transaction_counts[m,i] > 0)
				temp = transaction_counts[m,i] * log( v[m,i] / empirical_probs[m,i] ) 
				if (!isinf(temp))
					KL_noinf += temp
					total_noinf_transactions += transaction_counts[m,i]
				end
			end
		end
	end
	KL_noinf = -KL_noinf
	
	KL_noinf_avg = KL_noinf / total_noinf_transactions * 100

	return KL_noinf, total_noinf_transactions, KL_noinf_avg
end



# function tcm_evaluateKL_debug(N, v, transaction_counts)
# 	M = size(transaction_counts,1)

# 	empirical_probs = transaction_counts ./ sum(transaction_counts, 2)

# 	KL = 0.0;

# 	KL_matrix = zeros(M, N);
	
# 	for m in 1:M
# 		for i in 1:N
# 			if (transaction_counts[m,i] > 0)
# 				KL += transaction_counts[m,i] * log( v[m,i] / empirical_probs[m,i] )
# 				KL_matrix[m,i] = transaction_counts[m,i] * log( v[m,i] / empirical_probs[m,i] )
# 			end
# 		end
# 	end
# 	return -KL, KL_matrix
# end
