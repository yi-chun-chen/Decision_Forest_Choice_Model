function helper_fn(z, attempt)
	single_data_name, test_fold = z
	proc_id = myid();

	helper_start_time = time()

	transaction_counts = convert(Array{Int64}, readcsv( string("../tcm_data/IRI_data_assortments/", single_data_name, "_transaction_counts.csv") ) )
	assortments = convert(Array{Int64}, readcsv( string("../tcm_data/IRI_data_assortments/", single_data_name, "_assortments.csv") ) )
	folds  = convert(Array{Int64}, readcsv( string("../tcm_data/IRI_data_assortments/", single_data_name, "_folds.csv") ) )

	train_assortments = assortments[ find(folds .!= test_fold), :]
	test_assortments = assortments[ find(folds .== test_fold), :]

	train_transaction_counts = transaction_counts[ find(folds .!= test_fold), :]
	test_transaction_counts = transaction_counts[ find(folds .== test_fold), :]

	num_train_transactions = sum(train_transaction_counts)
	num_test_transactions = sum(test_transaction_counts)

	num_train_assortments = size(train_assortments, 1)
	num_test_assortments = size(test_assortments, 1)

	N = 10

	time_limit = 1200

	perTol = 1e-6


	## RANKING-BASED MODEL ##
	CS_size_vec = [2,3,4,5,6,7,8,9]

	for CS_size in CS_size_vec

		ranking_lambda, orderings, ranking_loglik, ranking_elapsed_time = tcm_estimateRanking_CS(N, train_assortments, train_transaction_counts, time_limit, perTol, CS_size)
		ranking_K = length(ranking_lambda)

		test_ranking_predict, test_ranking_predict_by_class = tcm_predictRanking(N, ranking_K, ranking_lambda, orderings, test_assortments)
		ranking_KL, ranking_noinf_count, ranking_KL_avg = tcm_evaluateKL(N, test_ranking_predict, test_transaction_counts)

		avg_CS_size, weighted_avg_CS_size, max_CS_size = tcm_rankingStats(N, ranking_lambda, orderings)

		row_string = string(single_data_name,  ",",
								test_fold, ",",
								num_train_assortments, ",",
								num_test_assortments, ",",
								num_train_transactions, ",",
								num_test_transactions, ",",
								proc_id, ",", 
								attempt, ",",
								"ranking", ",",
								CS_size, ",",
								ranking_loglik, ",",
								ranking_K, ",",
								perTol, ",",
								ranking_KL, ",",
								ranking_noinf_count, ",",
								ranking_KL_avg, ",",
								ranking_elapsed_time, ",",
								avg_CS_size, ",", 
								weighted_avg_CS_size, ",", 
								max_CS_size)
		row_string = string(row_string, "\n");

		@spawnat 1 begin
			print(ranking_outcsvhandle, row_string);
			flush(ranking_outcsvhandle);
		end
	end

	## RANKING-CV MODEL ## 

	numInitializations = 5
	perTol_CV = 1e-6;
	CS_list = [2,3,4,5,6,7,8,9]
	nfolds = 4
	train_folds = folds[ find(folds .!= test_fold)]
	best_CS_size, best_KL_cv, cv_elapsed_time = tcm_estimateRanking_kcv_weighted_ass(N, CS_list, nfolds, train_folds, train_assortments, train_transaction_counts, numInitializations, time_limit, perTol_CV)

	row_string = string(single_data_name,  ",",
							test_fold, ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							num_train_transactions, ",",
							num_test_transactions, ",",
							proc_id, ",", 
							attempt, ",",
							"ranking-CV", ",",
							best_CS_size, ",",
							perTol_CV, ",",
							numInitializations, ",",
							best_KL_cv, ",",
							cv_elapsed_time)
	row_string = string(row_string, "\n");

	@spawnat 1 begin
		print(ranking_cv_outcsvhandle, row_string);
		flush(ranking_cv_outcsvhandle);
	end

	@show best_CS_size


	helper_end_time = time()

	return helper_end_time - helper_start_time;


end