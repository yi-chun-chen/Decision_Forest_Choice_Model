function helper_fn(z, attempt)
	single_data_name, test_fold = z
	proc_id = myid();

	helper_start_time = time()

	transaction_counts = convert(Array{Int64}, readcsv( string("../tcm_data/synthetic_data_assortments/", single_data_name, "_transaction_counts.csv") ) )
	assortments = convert(Array{Int64}, readcsv( string("../tcm_data/synthetic_data_assortments/", single_data_name, "_assortments.csv") ) )
	
	grand_M = Int64(50)
	folds = Int64[ 1*ones(grand_M/5); 2*ones(grand_M/5); 3*ones(grand_M/5); 4*ones(grand_M/5); 5*ones(grand_M/5) ]

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


	## LC-MNL MODEL ##

	K_list = [5]
	numInitializations = 5

	for K in K_list
		LCMNL_u_val, LCMNL_class_probs, LCMNL_loglik, LCMNL_grand_elapsed_time = tcm_estimateMMNL(N, K, train_assortments, train_transaction_counts, numInitializations, time_limit, perTol)

		test_LCMNL_predict, test_LCMNL_predict_by_class = tcm_predictMMNL(N, K, LCMNL_class_probs, LCMNL_u_val, test_assortments)
		LCMNL_KL, LCMNL_noinf_count, LCMNL_KL_avg = tcm_evaluateKL(N, test_LCMNL_predict, test_transaction_counts)


		row_string = string(single_data_name,  ",",
							test_fold, ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							num_train_transactions, ",",
							num_test_transactions, ",",
							proc_id, ",", 
							attempt, ",",
							"LCMNL", ",",
							LCMNL_loglik, ",",
							K, ",",
							perTol, ",",
							numInitializations, ",",
							LCMNL_KL, ",",
							LCMNL_noinf_count, ",",
							LCMNL_KL_avg, ",",
							LCMNL_grand_elapsed_time)
		row_string = string(row_string, "\n");

		@spawnat 1 begin
			print(MNL_outcsvhandle, row_string);
			flush(MNL_outcsvhandle);
		end
	end


	## MNL MODEL ## 

	MNL_u_val, MNL_loglik, MNL_elapsed_time = tcm_estimateMNL(N, train_assortments, train_transaction_counts)
	test_MNL_predict = tcm_predictMNL(N, MNL_u_val, test_assortments)
	MNL_KL, MNL_noinf_count, MNL_KL_avg = tcm_evaluateKL(N, test_MNL_predict, test_transaction_counts)

	row_string = string(single_data_name,  ",",
							test_fold, ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							num_train_transactions, ",",
							num_test_transactions, ",",
							proc_id, ",", 
							attempt, ",",
							"MNL", ",",
							MNL_loglik, ",",
							1, ",",
							0, ",",
							1, ",",
							MNL_KL, ",",
							MNL_noinf_count, ",",
							MNL_KL_avg, ",",
							MNL_elapsed_time)
	row_string = string(row_string, "\n");

	@spawnat 1 begin
		print(MNL_outcsvhandle, row_string);
		flush(MNL_outcsvhandle);
	end


	## HALO-MNL MODEL ## 

	HALOMNL_u_val, HALOMNL_alpha_val, HALOMNL_loglik, HALOMNL_elapsed_time = tcm_estimateHALOMNL(N, train_assortments, train_transaction_counts)

	test_HALOMNL_predict = tcm_predictHALOMNL(N, HALOMNL_u_val, HALOMNL_alpha_val, test_assortments)

	HALOMNL_KL, HALOMNL_noinf_count, HALOMNL_KL_avg = tcm_evaluateKL(N, test_HALOMNL_predict, test_transaction_counts)

	row_string = string(single_data_name,  ",",
							test_fold, ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							num_train_transactions, ",",
							num_test_transactions, ",",
							proc_id, ",", 
							attempt, ",",
							"HALOMNL", ",",
							HALOMNL_loglik, ",",
							1, ",",
							0, ",",
							1, ",",
							HALOMNL_KL, ",",
							HALOMNL_noinf_count, ",",
							HALOMNL_KL_avg, ",",
							HALOMNL_elapsed_time)
	row_string = string(row_string, "\n");

	@spawnat 1 begin
		print(MNL_outcsvhandle, row_string);
		flush(MNL_outcsvhandle);
	end


	## FOREST, COLD-START ## 

	depth_list = [4]

	initial_left, initial_right, initial_product, initial_isLeaf = tcm_createBasicForest(N)

	for depth_limit in depth_list
		
		lambda, forest_left, forest_right, forest_product, forest_isLeaf, forest_loglik, forest_elapsed_time = tcm_estimateForest(N, train_assortments, train_transaction_counts, time_limit, perTol, depth_limit,
																												initial_left, initial_right, initial_product, initial_isLeaf)
		K = length(lambda)

		test_forest_predict, test_forest_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, test_assortments)
		forest_KL, forest_noinf_count, forest_KL_avg = tcm_evaluateKL(N, test_forest_predict, test_transaction_counts)

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
							"forest-cold", ",",
							depth_limit, ",",
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
		end

		@show forest_KL
	end

	## FOREST, WARM-START ##

	depth_list = [4]

	initial_left, initial_right, initial_product, initial_isLeaf = tcm_convertRankingToForest(N, orderings)

	for depth_limit in depth_list

		lambda, forest_left, forest_right, forest_product, forest_isLeaf, forest_loglik, forest_elapsed_time = tcm_estimateForest(N, train_assortments, train_transaction_counts, time_limit, perTol, depth_limit,
																												initial_left, initial_right, initial_product, initial_isLeaf)
		K = length(lambda)

		test_forest_predict, test_forest_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, test_assortments)
		forest_KL, forest_noinf_count, forest_KL_avg = tcm_evaluateKL(N, test_forest_predict, test_transaction_counts)

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
							"forest-ranking-warm", ",",
							depth_limit, ",",
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
		end
	end

	helper_end_time = time()

	return helper_end_time - helper_start_time;


end