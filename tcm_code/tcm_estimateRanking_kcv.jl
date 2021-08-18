

function tcm_estimateRanking_kcv_weighted_ass(N, CS_list, nfolds, assortment_folds, assortments, transaction_counts, numInitializations, time_limit, perTol)

	unique_fold_ids = unique(assortment_folds)

	KL_total_holdout_fold = zeros(length(CS_list), nfolds)
	noinf_count_holdout_fold = zeros(length(CS_list), nfolds)

	start_time = time()

	for fold_ind in 1:nfolds
		fold = unique_fold_ids[fold_ind]

		# trainfolds_assortments, trainfolds_transaction_counts = tcm_transactionsToCounts(N, trainfolds_transactions)
		# testfold_assortments, testfold_transaction_counts = tcm_transactionsToCounts(N, testfold_transactions)

		trainfolds_assortments = assortments[ find(assortment_folds .!= fold), :]
		testfold_assortments = assortments[ find(assortment_folds .== fold), :]

		trainfolds_transaction_counts = transaction_counts[ find(assortment_folds .!= fold), :]
		testfold_transaction_counts = transaction_counts[ find(assortment_folds .== fold), :]


		for CSind in 1:length(CS_list)
			println("CS size ", CS_list[CSind], ", fold ", fold)
			CS_size = CS_list[CSind]

			ranking_lambda, orderings, ranking_loglik, ranking_elapsed_time = tcm_estimateRanking_CS(N, trainfolds_assortments, trainfolds_transaction_counts, time_limit, perTol, CS_size)
			ranking_K = length(ranking_lambda)
			
			testfold_ranking_predict, testfold_ranking_predict_by_class = tcm_predictRanking(N, ranking_K, ranking_lambda, orderings, testfold_assortments)

			testfold_KL, testfold_noinf_count, testfold_KL_avg = tcm_evaluateKL(N, testfold_ranking_predict, testfold_transaction_counts)
			
			KL_total_holdout_fold[CSind, fold_ind] = testfold_KL #testfold_KL / sum(testfold_transaction_counts)
			noinf_count_holdout_fold[CSind, fold_ind] = testfold_noinf_count
		end
	end

	end_time = time()
	elapsed_time = end_time - start_time

	KL_avg_by_CS = sum(KL_total_holdout_fold, 2) ./ sum(noinf_count_holdout_fold, 2)  * 100

	best_CSind = indmin(KL_avg_by_CS)
	best_CS_size = CS_list[best_CSind]
	best_KL_avg_cv = KL_avg_by_CS[best_CSind]

	@show KL_avg_by_CS

	return best_CS_size, best_KL_avg_cv, elapsed_time
end


function tcm_estimateRanking_kcv_weighted(N, CS_list, nfolds, transaction_folds, transactions, time_limit, perTol)
	unique_fold_ids = unique(transaction_folds)

	KL_total_holdout_fold = zeros(length(CS_list), nfolds)
	noinf_count_holdout_fold = zeros(length(CS_list), nfolds)

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

		for i in 1:length(CS_list)
			println("CS size ", CS_list[i], ", fold ", fold)
			input_CS_size = CS_list[i]

			lambda, orderings, loglik, elapsed_time = tcm_estimateRanking_CS(N, trainfolds_assortments, trainfolds_transaction_counts, time_limit, perTol, input_CS_size)
			K = length(lambda)
			testfold_ranking_predict, testfold_ranking_predict_by_class = tcm_predictRanking(N, K, lambda, orderings, testfold_assortments)

			testfold_KL, testfold_noinf_count, testfold_KL_avg = tcm_evaluateKL(N, testfold_ranking_predict, testfold_transaction_counts)

			KL_total_holdout_fold[i, fold_ind] = testfold_KL #testfold_KL / sum(testfold_transaction_counts)
			noinf_count_holdout_fold[i, fold_ind] = testfold_noinf_count
		end
	end

	end_time = time()
	elapsed_time = end_time - start_time

	KL_avg_by_CS_size = sum(KL_total_holdout_fold, 2) ./ sum(noinf_count_holdout_fold, 2)  * 100

	@show KL_total_holdout_fold
	@show noinf_count_holdout_fold
	@show KL_avg_by_CS_size

	best_i = indmin(KL_avg_by_CS_size)
	best_CS_size = CS_list[best_i]
	best_KL_avg_cv = KL_avg_by_CS_size[best_i]

	return best_CS_size, best_KL_avg_cv, elapsed_time

end
