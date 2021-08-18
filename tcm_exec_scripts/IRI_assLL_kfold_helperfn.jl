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

	ranking_lambda, orderings, ranking_loglik, ranking_elapsed_time = tcm_estimateRanking(N, train_assortments, train_transaction_counts, time_limit, perTol)
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

	## FOREST-LL, COLD-START ## 

	LL_list = [4,8,16,32,64]

	initial_left, initial_right, initial_product, initial_isLeaf = tcm_createBasicForest(N)

	for leaf_limit in LL_list
		
		lambda, forest_left, forest_right, forest_product, forest_isLeaf, forest_loglik, forest_elapsed_time = tcm_estimateForest_leafLimit(N, train_assortments, train_transaction_counts, time_limit, perTol, leaf_limit,
																												initial_left, initial_right, initial_product, initial_isLeaf)
		K = length(lambda)

		test_forest_predict, test_forest_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, test_assortments)
		forest_KL, forest_noinf_count, forest_KL_avg = tcm_evaluateKL(N, test_forest_predict, test_transaction_counts)
		#forest_KL_avg = forest_KL / sum(test_transaction_counts) * 100

		simple_avg_leaves, weighted_avg_leaves, max_leaves = tcm_leafStats(lambda, forest_isLeaf)
		simple_avg_depth, weighted_avg_depth, max_depth = tcm_depthStats(lambda, forest_left, forest_right, forest_isLeaf)


		row_string = string(single_data_name,  ",",
							test_fold, ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							num_train_transactions, ",",
							num_test_transactions, ",",
							proc_id, ",", 
							attempt, ",",
							"forest-LL-cold", ",",
							leaf_limit, ",",
							forest_loglik, ",",
							K, ",",
							perTol, ",",
							forest_KL, ",",
							forest_noinf_count, ",",
							forest_KL_avg, ",",
							forest_elapsed_time, ",",
							simple_avg_leaves, ",",
							weighted_avg_leaves, ",", 
							max_leaves, ",",
							simple_avg_depth, ",", 
							weighted_avg_depth, ",",
							max_depth)
		row_string = string(row_string, "\n");

		@spawnat 1 begin
			print(forest_outcsvhandle, row_string);
			flush(forest_outcsvhandle);


			# for k in 1:K
			# 	temp_tree = [forest_left[k] forest_right[k] forest_product[k] forest_isLeaf[k]]
			# 	forest_tree_path = string(expdirpath, "forest_cold_", single_data_name, "_", test_fold, "_depth", depth_limit, "_tree", k, ".csv")
			# 	writecsv(forest_tree_path, temp_tree)
			# end

			# forest_lambda_path = string(expdirpath, "forest_cold_", single_data_name, "_", test_fold, "_depth", depth_limit, "_lambda.csv")
			# writecsv(forest_lambda_path, lambda)
		end
	end


	## FOREST-LL, WARM-START ##
	LL_list = [4,8,16,32,64]

	initial_left, initial_right, initial_product, initial_isLeaf = tcm_convertRankingToForest(N, orderings)

	for leaf_limit in LL_list
		
		lambda, forest_left, forest_right, forest_product, forest_isLeaf, forest_loglik, forest_elapsed_time = tcm_estimateForest_leafLimit(N, train_assortments, train_transaction_counts, time_limit, perTol, leaf_limit,
																												initial_left, initial_right, initial_product, initial_isLeaf)
		K = length(lambda)

		test_forest_predict, test_forest_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, test_assortments)
		forest_KL, forest_noinf_count, forest_KL_avg = tcm_evaluateKL(N, test_forest_predict, test_transaction_counts)
		#forest_KL_avg = forest_KL / sum(test_transaction_counts) * 100

		simple_avg_leaves, weighted_avg_leaves, max_leaves = tcm_leafStats(lambda, forest_isLeaf)
		simple_avg_depth, weighted_avg_depth, max_depth = tcm_depthStats(lambda, forest_left, forest_right, forest_isLeaf)

		row_string = string(single_data_name,  ",",
							test_fold, ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							num_train_transactions, ",",
							num_test_transactions, ",",
							proc_id, ",", 
							attempt, ",",
							"forest-LL-ranking-warm", ",",
							leaf_limit, ",",
							forest_loglik, ",",
							K, ",",
							perTol, ",",
							forest_KL, ",",
							forest_noinf_count, ",",
							forest_KL_avg, ",",
							forest_elapsed_time, ",",
							simple_avg_leaves, ",",
							weighted_avg_leaves, ",", 
							max_leaves, ",",
							simple_avg_depth, ",", 
							weighted_avg_depth, ",",
							max_depth)
		row_string = string(row_string, "\n");

		@spawnat 1 begin
			print(forest_outcsvhandle, row_string);
			flush(forest_outcsvhandle);

			# for k in 1:K
			# 	temp_tree = [forest_left[k] forest_right[k] forest_product[k] forest_isLeaf[k]]
			# 	forest_tree_path = string(expdirpath, "forest_ranking_warm_", single_data_name, "_", test_fold, "_depth", depth_limit, "_tree", k, ".csv")
			# 	writecsv(forest_tree_path, temp_tree)
			# end

			# forest_lambda_path = string(expdirpath, "forest_ranking_warm_", single_data_name, "_", test_fold, "_depth", depth_limit, "_lambda.csv")
			# writecsv(forest_lambda_path, lambda)
		end
	end

	## FOREST-CV, WARM-START ##
	#depth_list = [3,4,5,6,7]
	LL_list = [4,8,16,32,64]

	initial_left, initial_right, initial_product, initial_isLeaf = tcm_convertRankingToForest(N, orderings)

	nfolds = 4
	train_folds = folds[ find(folds .!= test_fold)]
	best_LL, best_KL_avg_cv, elapsed_time_cv = tcm_estimateForest_leafLimit_kcv_weighted_ass(N, LL_list, nfolds, train_folds, train_assortments, train_transaction_counts, time_limit, perTol, initial_left, initial_right, initial_product, initial_isLeaf)

	row_string = string(single_data_name,  ",",
							test_fold, ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							num_train_transactions, ",",
							num_test_transactions, ",",
							proc_id, ",", 
							attempt, ",",
							"forest-LL-cv-ranking-warm", ",",
							best_LL, ",",
							perTol, ",",
							best_KL_avg_cv, ",",
							elapsed_time_cv)
	row_string = string(row_string, "\n");

	@spawnat 1 begin
		print(forest_cv_outcsvhandle, row_string);
		flush(forest_cv_outcsvhandle);
	end


	## FOREST-CV, COLD-START ##
	#depth_list = [3,4,5,6,7]
	LL_list = [4,8,16,32,64]

	initial_left, initial_right, initial_product, initial_isLeaf = tcm_createBasicForest(N)

	nfolds = 4
	train_folds = folds[ find(folds .!= test_fold)]
	best_LL, best_KL_avg_cv, elapsed_time_cv = tcm_estimateForest_leafLimit_kcv_weighted_ass(N, LL_list, nfolds, train_folds, train_assortments, train_transaction_counts, time_limit, perTol, initial_left, initial_right, initial_product, initial_isLeaf)

	row_string = string(single_data_name,  ",",
							test_fold, ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							num_train_transactions, ",",
							num_test_transactions, ",",
							proc_id, ",", 
							attempt, ",",
							"forest-LL-cv-cold", ",",
							best_LL, ",",
							perTol, ",",
							best_KL_avg_cv, ",",
							elapsed_time_cv)
	row_string = string(row_string, "\n");

	@spawnat 1 begin
		print(forest_cv_outcsvhandle, row_string);
		flush(forest_cv_outcsvhandle);
	end




	helper_end_time = time()

	return helper_end_time - helper_start_time;


end