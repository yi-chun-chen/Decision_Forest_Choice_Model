function tcm_transactionsToCounts(N, transactions)
	T = size(transactions,1)
	assortments = unique(transactions[:,1:N], 1)
	M = size(assortments,1)

	transaction_counts = zeros(Int64,M,N)

	for t = 1 : T
	    for m in 1:M
	    	if (transactions[t,1:N] == assortments[m,:] )
	    		transaction_counts[m, transactions[t,N+1] ] += 1
	    	end
	    end
	end


	return assortments, transaction_counts
end