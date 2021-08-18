# function tcm_estimateMMNL_kcv(N, K_list, nfolds, transaction_folds, transactions, numInitializations, time_limit, perTol)

# 	unique_fold_ids = unique(transaction_folds)

# 	KL_avg_holdout_fold = zeros(length(K_list), nfolds)

# 	start_time = time()

# 	for fold_ind in 1:nfolds
# 		fold = unique_fold_ids[fold_ind]

# 		@show fold

# 		@show unique_fold_ids

# 		println("Pre-subsetting ")

# 		#@show transaction_folds
# 		#temp = find(transaction_folds .!= fold)
# 		#@show temp
# 		@show size(transaction_folds)
# 		@show size(transactions)
# 		@show fold 

# 		temp = find(transaction_folds .!= fold)
# 		@show temp[1:5]
# 		trainfolds_transactions = transactions[ find(transaction_folds .!= fold), :]
# 		println("Post-subsetting")
# 		testfold_transactions = transactions[ find(transaction_folds .== fold), :]

		

# 		trainfolds_assortments, trainfolds_transaction_counts = tcm_transactionsToCounts(N, trainfolds_transactions)
# 		testfold_assortments, testfold_transaction_counts = tcm_transactionsToCounts(N, testfold_transactions)


# 		for Kind in 1:length(K_list)
# 			K = K_list[Kind]

# 			u, class_probs, loglik, grand_elapsed_time = tcm_estimateMMNL(N, K, trainfolds_assortments, trainfolds_transaction_counts, numInitializations, time_limit, perTol)

# 			testfold_MMNL_predict, testfold_MMNL_predict_by_class = tcm_predictMMNL(N, K, class_probs, u, testfold_assortments)

# 			testfold_KL, testfold_noinf_count = tcm_evaluateKL(N, testfold_MMNL_predict, testfold_transaction_counts)

# 			KL_avg_holdout_fold[Kind, fold_ind] = testfold_KL / testfold_noinf_count # sum(testfold_transaction_counts)
# 		end
# 	end

# 	end_time = time()
# 	elapsed_time = end_time - start_time

# 	KL_avg_by_K = mean(KL_avg_holdout_fold, 2)

# 	best_Kind = indmin(KL_avg_by_K)
# 	best_K = K_list[best_Kind]
# 	best_KL_avg_cv = KL_avg_by_K[best_Kind]

# 	return best_K, best_KL_avg_cv, elapsed_time
# end



function tcm_estimateMMNL_kcv_weighted(N, K_list, nfolds, transaction_folds, transactions, numInitializations, time_limit, perTol)

	unique_fold_ids = unique(transaction_folds)

	KL_total_holdout_fold = zeros(length(K_list), nfolds)
	noinf_count_holdout_fold = zeros(length(K_list), nfolds)

	start_time = time()

	for fold_ind in 1:nfolds
		fold = unique_fold_ids[fold_ind]

		@show fold

		@show unique_fold_ids

		println("Pre-subsetting ")

		#@show transaction_folds
		#temp = find(transaction_folds .!= fold)
		#@show temp
		@show size(transaction_folds)
		@show size(transactions)
		@show fold 

		temp = find(transaction_folds .!= fold)
		@show temp[1:5]
		trainfolds_transactions = transactions[ find(transaction_folds .!= fold), :]
		println("Post-subsetting")
		testfold_transactions = transactions[ find(transaction_folds .== fold), :]

		

		trainfolds_assortments, trainfolds_transaction_counts = tcm_transactionsToCounts(N, trainfolds_transactions)
		testfold_assortments, testfold_transaction_counts = tcm_transactionsToCounts(N, testfold_transactions)


		for Kind in 1:length(K_list)
			K = K_list[Kind]

			u, class_probs, loglik, grand_elapsed_time = tcm_estimateMMNL(N, K, trainfolds_assortments, trainfolds_transaction_counts, numInitializations, time_limit, perTol)

			testfold_MMNL_predict, testfold_MMNL_predict_by_class = tcm_predictMMNL(N, K, class_probs, u, testfold_assortments)

			testfold_KL, testfold_noinf_count, testfold_KL_avg = tcm_evaluateKL(N, testfold_MMNL_predict, testfold_transaction_counts)

			@show testfold_KL
			@show testfold_noinf_count
			@show testfold_KL_avg
			
			KL_total_holdout_fold[Kind, fold_ind] = testfold_KL #testfold_KL / sum(testfold_transaction_counts)
			noinf_count_holdout_fold[Kind, fold_ind] = testfold_noinf_count

			@show KL_total_holdout_fold
			@show noinf_count_holdout_fold
		end
	end

	end_time = time()
	elapsed_time = end_time - start_time

	KL_avg_by_K = sum(KL_total_holdout_fold, 2) ./ sum(noinf_count_holdout_fold, 2)  * 100

	best_Kind = indmin(KL_avg_by_K)
	best_K = K_list[best_Kind]
	best_KL_avg_cv = KL_avg_by_K[best_Kind]

	@show KL_avg_by_K
	@show KL_total_holdout_fold
	@show noinf_count_holdout_fold

	return best_K, best_KL_avg_cv, elapsed_time
end


function tcm_estimateMMNL_kcv_weighted_ass(N, K_list, nfolds, assortment_folds, assortments, transaction_counts, numInitializations, time_limit, perTol)

	unique_fold_ids = unique(assortment_folds)

	KL_total_holdout_fold = zeros(length(K_list), nfolds)
	noinf_count_holdout_fold = zeros(length(K_list), nfolds)

	start_time = time()

	for fold_ind in 1:nfolds
		fold = unique_fold_ids[fold_ind]

		# trainfolds_assortments, trainfolds_transaction_counts = tcm_transactionsToCounts(N, trainfolds_transactions)
		# testfold_assortments, testfold_transaction_counts = tcm_transactionsToCounts(N, testfold_transactions)

		trainfolds_assortments = assortments[ find(assortment_folds .!= fold), :]
		testfold_assortments = assortments[ find(assortment_folds .== fold), :]

		trainfolds_transaction_counts = transaction_counts[ find(assortment_folds .!= fold), :]
		testfold_transaction_counts = transaction_counts[ find(assortment_folds .== fold), :]


		for Kind in 1:length(K_list)
			K = K_list[Kind]

			u, class_probs, loglik, grand_elapsed_time = tcm_estimateMMNL(N, K, trainfolds_assortments, trainfolds_transaction_counts, numInitializations, time_limit, perTol)

			testfold_MMNL_predict, testfold_MMNL_predict_by_class = tcm_predictMMNL(N, K, class_probs, u, testfold_assortments)

			testfold_KL, testfold_noinf_count, testfold_KL_avg = tcm_evaluateKL(N, testfold_MMNL_predict, testfold_transaction_counts)
			
			KL_total_holdout_fold[Kind, fold_ind] = testfold_KL #testfold_KL / sum(testfold_transaction_counts)
			noinf_count_holdout_fold[Kind, fold_ind] = testfold_noinf_count
		end
	end

	end_time = time()
	elapsed_time = end_time - start_time

	KL_avg_by_K = sum(KL_total_holdout_fold, 2) ./ sum(noinf_count_holdout_fold, 2)  * 100

	best_Kind = indmin(KL_avg_by_K)
	best_K = K_list[best_Kind]
	best_KL_avg_cv = KL_avg_by_K[best_Kind]

	return best_K, best_KL_avg_cv, elapsed_time
end