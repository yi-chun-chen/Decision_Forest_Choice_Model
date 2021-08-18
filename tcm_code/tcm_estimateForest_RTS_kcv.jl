

function tcm_estimateForest_RTS_kcv_weighted(N, K_sample, d_list, nfolds, transaction_folds, transactions, time_limit_EM, perTol_EM, initial_left, initial_right, initial_product, initial_isLeaf)
	unique_fold_ids = unique(transaction_folds)

	KL_total_holdout_fold = zeros(length(d_list), nfolds)
	noinf_count_holdout_fold = zeros(length(d_list), nfolds)

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

		for i in 1:length(d_list)
			println("Depth ", d_list[i], ", fold ", fold)
			depth_limit = d_list[i]

			if (isempty(initial_left))
				lambda, forest_left, forest_right, forest_product, forest_isLeaf, loglik, elapsed_time = tcm_estimateForest_RTS(N, trainfolds_assortments, trainfolds_transaction_counts, K_sample, depth_limit, perTol_EM, time_limit_EM)

			else 
				lambda, forest_left, forest_right, forest_product, forest_isLeaf, loglik, elapsed_time = tcm_estimateForest_RTS_withInitial(N, trainfolds_assortments, trainfolds_transaction_counts, K_sample, depth_limit, perTol_EM, time_limit_EM,
																																initial_left, initial_right, initial_product, initial_isLeaf)
			end
			
			K = length(lambda)
			testfold_forest_predict, testfold_forest_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, testfold_assortments)

			testfold_KL, testfold_noinf_count, testfold_KL_avg = tcm_evaluateKL(N, testfold_forest_predict, testfold_transaction_counts)

			KL_total_holdout_fold[i, fold_ind] = testfold_KL #testfold_KL / sum(testfold_transaction_counts)
			noinf_count_holdout_fold[i, fold_ind] = testfold_noinf_count
		end
	end

	end_time = time()
	elapsed_time = end_time - start_time

	KL_avg_by_d = sum(KL_total_holdout_fold, 2) ./ sum(noinf_count_holdout_fold, 2)  * 100

	@show KL_total_holdout_fold
	@show noinf_count_holdout_fold
	@show KL_avg_by_d

	best_i = indmin(KL_avg_by_d)
	best_d = d_list[best_i]
	best_KL_avg_cv = KL_avg_by_d[best_i]

	return best_d, best_KL_avg_cv, elapsed_time

end





function tcm_estimateForest_RTS_kcv_weighted_ass(N, K_sample, d_list, nfolds, assortment_folds, assortments, transaction_counts, time_limit_EM, perTol_EM, initial_left, initial_right, initial_product, initial_isLeaf)
	unique_fold_ids = unique(assortment_folds)

	KL_total_holdout_fold = zeros(length(d_list), nfolds)
	noinf_count_holdout_fold = zeros(length(d_list), nfolds)

	start_time = time()

	for fold_ind in 1:nfolds
		fold = unique_fold_ids[fold_ind]

		# trainfolds_assortments, trainfolds_transaction_counts = tcm_transactionsToCounts(N, trainfolds_transactions)
		# testfold_assortments, testfold_transaction_counts = tcm_transactionsToCounts(N, testfold_transactions)

		trainfolds_assortments = assortments[ find(assortment_folds .!= fold), :]
		testfold_assortments = assortments[ find(assortment_folds .== fold), :]

		trainfolds_transaction_counts = transaction_counts[ find(assortment_folds .!= fold), :]
		testfold_transaction_counts = transaction_counts[ find(assortment_folds .== fold), :]

		for i in 1:length(d_list)
			println("Depth ", d_list[i], ", fold ", fold)
			depth_limit = d_list[i]

			if (isempty(initial_left))
				lambda, forest_left, forest_right, forest_product, forest_isLeaf, loglik, elapsed_time = tcm_estimateForest_RTS(N, trainfolds_assortments, trainfolds_transaction_counts, K_sample, depth_limit, perTol_EM, time_limit_EM)

			else 
				lambda, forest_left, forest_right, forest_product, forest_isLeaf, loglik, elapsed_time = tcm_estimateForest_RTS_withInitial(N, trainfolds_assortments, trainfolds_transaction_counts, K_sample, depth_limit, perTol_EM, time_limit_EM,
																																initial_left, initial_right, initial_product, initial_isLeaf)
			end
			
			K = length(lambda)
			testfold_forest_predict, testfold_forest_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, testfold_assortments)

			testfold_KL, testfold_noinf_count, testfold_KL_avg = tcm_evaluateKL(N, testfold_forest_predict, testfold_transaction_counts)

			KL_total_holdout_fold[i, fold_ind] = testfold_KL #testfold_KL / sum(testfold_transaction_counts)
			noinf_count_holdout_fold[i, fold_ind] = testfold_noinf_count
		end
	end

	end_time = time()
	elapsed_time = end_time - start_time

	KL_avg_by_d = sum(KL_total_holdout_fold, 2) ./ sum(noinf_count_holdout_fold, 2)  * 100

	best_i = indmin(KL_avg_by_d)
	best_d = d_list[best_i]
	best_KL_avg_cv = KL_avg_by_d[best_i]

	return best_d, best_KL_avg_cv, elapsed_time

end





function tcm_estimateForest_RTS_RMredo_kcv_weighted_ass(N, K_sample, d_list, nfolds, assortment_folds, assortments, transaction_counts, time_limit, perTol, perTol_EM)
	unique_fold_ids = unique(assortment_folds)

	KL_total_holdout_fold = zeros(length(d_list), nfolds)
	noinf_count_holdout_fold = zeros(length(d_list), nfolds)

	start_time = time()

	for fold_ind in 1:nfolds
		fold = unique_fold_ids[fold_ind]

		# trainfolds_assortments, trainfolds_transaction_counts = tcm_transactionsToCounts(N, trainfolds_transactions)
		# testfold_assortments, testfold_transaction_counts = tcm_transactionsToCounts(N, testfold_transactions)

		trainfolds_assortments = assortments[ find(assortment_folds .!= fold), :]
		testfold_assortments = assortments[ find(assortment_folds .== fold), :]

		trainfolds_transaction_counts = transaction_counts[ find(assortment_folds .!= fold), :]
		testfold_transaction_counts = transaction_counts[ find(assortment_folds .== fold), :]

		for i in 1:length(d_list)
			println("Depth ", d_list[i], ", fold ", fold)
			depth_limit = d_list[i]

			ranking_lambda, orderings, ranking_loglik, ranking_elapsed_time = tcm_estimateRanking(N, trainfolds_assortments, trainfolds_transaction_counts, time_limit, perTol)
			initial_left, initial_right, initial_product, initial_isLeaf = tcm_convertRankingToForest(N, orderings)

			lambda, forest_left, forest_right, forest_product, forest_isLeaf, loglik, elapsed_time = tcm_estimateForest_RTS_withInitial(N, trainfolds_assortments, trainfolds_transaction_counts, K_sample, depth_limit, perTol_EM, time_limit,
																															initial_left, initial_right, initial_product, initial_isLeaf)
		
			
			K = length(lambda)
			testfold_forest_predict, testfold_forest_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, testfold_assortments)

			testfold_KL, testfold_noinf_count, testfold_KL_avg = tcm_evaluateKL(N, testfold_forest_predict, testfold_transaction_counts)

			KL_total_holdout_fold[i, fold_ind] = testfold_KL #testfold_KL / sum(testfold_transaction_counts)
			noinf_count_holdout_fold[i, fold_ind] = testfold_noinf_count
		end
	end

	end_time = time()
	elapsed_time = end_time - start_time

	KL_avg_by_d = sum(KL_total_holdout_fold, 2) ./ sum(noinf_count_holdout_fold, 2)  * 100

	best_i = indmin(KL_avg_by_d)
	best_d = d_list[best_i]
	best_KL_avg_cv = KL_avg_by_d[best_i]

	return best_d, best_KL_avg_cv, elapsed_time

end