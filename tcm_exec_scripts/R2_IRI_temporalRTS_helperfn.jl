function helper_fn(z, attempt)
	single_data_name = z
	proc_id = myid();

	helper_start_time = time()

	train_transaction_counts = convert(Array{Int64}, readcsv( string("../tcm_data/IRI_data_assortments/", single_data_name, "_transaction_counts.csv") ) )
	train_assortments = convert(Array{Int64}, readcsv( string("../tcm_data/IRI_data_assortments/", single_data_name, "_assortments.csv") ) )

	test_transaction_counts = convert(Array{Int64}, readcsv( string("../tcm_data/IRI_data_R2_temporally_split/week3_to_week6/", single_data_name, "_transaction_counts.csv") ) )
	test_assortments = convert(Array{Int64}, readcsv( string("../tcm_data/IRI_data_R2_temporally_split/week3_to_week6/", single_data_name, "_assortments.csv") ) )
	
	num_train_transactions = sum(train_transaction_counts)
	num_test_transactions = sum(test_transaction_counts)

	num_train_assortments = size(train_assortments, 1)
	num_test_assortments = size(test_assortments, 1)

	N = 10

	time_limit = 1200

	perTol = 1e-6
	perTol_EM = 1e-5;
	time_limit_EM = time_limit;
	K_sample = 2000;


	## RANKING-BASED MODEL ##
	ranking_lambda, orderings, ranking_loglik, ranking_elapsed_time = tcm_estimateRanking(N, train_assortments, train_transaction_counts, time_limit, perTol)
	ranking_K = length(ranking_lambda)

	test_ranking_predict, test_ranking_predict_by_class = tcm_predictRanking(N, ranking_K, ranking_lambda, orderings, test_assortments)
	ranking_KL, ranking_noinf_count, ranking_KL_avg = tcm_evaluateKL(N, test_ranking_predict, test_transaction_counts)

	avg_CS_size, weighted_avg_CS_size, max_CS_size = tcm_rankingStats(N, ranking_lambda, orderings)

	row_string = string(single_data_name,  ",",
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



	## FOREST, RTS, COLD-START ## 

	depth_list = [3,4,5,6,7]

	# initial_left, initial_right, initial_product, initial_isLeaf = tcm_createBasicForest(N)

	for depth_limit in depth_list
		
		lambda, forest_left, forest_right, forest_product, forest_isLeaf, forest_loglik, forest_elapsed_time = tcm_estimateForest_RTS(N, train_assortments, train_transaction_counts, K_sample, depth_limit, perTol_EM, time_limit_EM)
		K = length(lambda)

		test_forest_predict, test_forest_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, test_assortments)
		forest_KL, forest_noinf_count, forest_KL_avg = tcm_evaluateKL(N, test_forest_predict, test_transaction_counts)

		simple_avg_leaves, weighted_avg_leaves, max_leaves = tcm_leafStats(lambda, forest_isLeaf)
		simple_avg_depth, weighted_avg_depth, max_depth = tcm_depthStats(lambda, forest_left, forest_right, forest_isLeaf)


		row_string = string(single_data_name,  ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							num_train_transactions, ",",
							num_test_transactions, ",",
							proc_id, ",", 
							attempt, ",",
							"forest-cold-RTS", ",",
							depth_limit, ",",
							K_sample, ",",
							perTol_EM, ",",
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



	## FOREST, WARM-START ##
	depth_list = [3,4,5,6,7]

	initial_left, initial_right, initial_product, initial_isLeaf = tcm_convertRankingToForest(N, orderings)

	for depth_limit in depth_list

		lambda, forest_left, forest_right, forest_product, forest_isLeaf, forest_loglik, forest_elapsed_time = tcm_estimateForest_RTS_withInitial(N, train_assortments, train_transaction_counts, K_sample, depth_limit, perTol_EM, time_limit_EM,
																												initial_left, initial_right, initial_product, initial_isLeaf)
		K = length(lambda)

		test_forest_predict, test_forest_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, test_assortments)
		forest_KL, forest_noinf_count, forest_KL_avg = tcm_evaluateKL(N, test_forest_predict, test_transaction_counts)

		simple_avg_leaves, weighted_avg_leaves, max_leaves = tcm_leafStats(lambda, forest_isLeaf)
		simple_avg_depth, weighted_avg_depth, max_depth = tcm_depthStats(lambda, forest_left, forest_right, forest_isLeaf)


		row_string = string(single_data_name,  ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							num_train_transactions, ",",
							num_test_transactions, ",",
							proc_id, ",", 
							attempt, ",",
							"forest-ranking-warm-RTS", ",",
							depth_limit, ",",
							K_sample, ",",
							perTol_EM, ",",
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