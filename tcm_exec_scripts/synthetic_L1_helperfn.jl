function helper_fn(z, attempt)
	single_data_name = z
	proc_id = myid();

	helper_start_time = time()

	choice_probs = readcsv( string("../tcm_data/synthetic_data_models/", single_data_name, "_model_predict.csv") )
	grand_choice_probs = readcsv( string("../tcm_data/synthetic_data_models/", single_data_name, "_grand_model_predict.csv") )

	train_assortments = convert(Array{Int64}, readcsv( string("../tcm_data/synthetic_data_assortments/", single_data_name, "_assortments.csv") ) )

	N = 10
	grand_assortments = tcm_createGrandAssortments(N)
	time_limit = 1200

	num_train_assortments = size(train_assortments,1)
	num_test_assortments = size(grand_assortments, 1)

	revenues = collect(90:-10:0)

	grand_best_r = maximum( grand_choice_probs * revenues )



	## RANKING-BASED MODEL ##
	numInitializations_LS = 5
	ranking_lambda, orderings, ranking_L1_train_error, ranking_elapsed_time = tcm_estimateRanking_L1(N, train_assortments, choice_probs, numInitializations_LS, time_limit )
	ranking_K = length(ranking_lambda)

	ranking_grand_predict, ranking_grand_predict_by_class = tcm_predictRanking(N, ranking_K, ranking_lambda, orderings, grand_assortments)
	ranking_L1_test_error = sum( abs.( ranking_grand_predict - grand_choice_probs  )  ) / 2^(N-1)

	ranking_best_a = indmax( ranking_grand_predict * revenues)
	ranking_best_r = dot(grand_choice_probs[ranking_best_a,:], revenues)

	ranking_approx_rate = ranking_best_r / grand_best_r

	row_string = string(single_data_name,  ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							proc_id, ",", 
							attempt, ",",
							"ranking", ",",
							ranking_L1_train_error, ",",
							ranking_K, ",",
							ranking_L1_test_error, ",",
							ranking_approx_rate, ",",
							ranking_elapsed_time)
	row_string = string(row_string, "\n");

	@spawnat 1 begin
		print(ranking_outcsvhandle, row_string);
		flush(ranking_outcsvhandle);
	end


	## LC-MNL MODEL ##
	perTol = 1e-6
	K_list = [5]
	numInitializations = 5

	for K in K_list
		LCMNL_u_val, LCMNL_class_probs, LCMNL_loglik, LCMNL_grand_elapsed_time = tcm_estimateMMNL(N, K, train_assortments, choice_probs, numInitializations, time_limit, perTol)

		LCMNL_train_predict, LCMNL_train_predict_by_class = tcm_predictMMNL(N, K, LCMNL_class_probs, LCMNL_u_val, train_assortments)
		LCMNL_L1_train_error = sum( abs.( LCMNL_train_predict - choice_probs  )  )

		LCMNL_grand_predict, LCMNL_grand_predict_by_class = tcm_predictMMNL(N, K, LCMNL_class_probs, LCMNL_u_val, grand_assortments)
		LCMNL_L1_test_error = sum( abs.( LCMNL_grand_predict - grand_choice_probs  )  ) / 2^(N-1)

		LCMNL_best_a = indmax( LCMNL_grand_predict * revenues)
		LCMNL_best_r = dot(grand_choice_probs[LCMNL_best_a,:], revenues)

		LCMNL_approx_rate = LCMNL_best_r / grand_best_r

		row_string = string(single_data_name,  ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							proc_id, ",", 
							attempt, ",",
							"LCMNL", ",",
							LCMNL_loglik, ",",
							K, ",",
							perTol, ",",
							numInitializations, ",",
							LCMNL_L1_train_error, ",",
							LCMNL_L1_test_error, ",",
							LCMNL_approx_rate, ",",
							LCMNL_grand_elapsed_time)
		row_string = string(row_string, "\n");

		@spawnat 1 begin
			print(MNL_outcsvhandle, row_string);
			flush(MNL_outcsvhandle);
		end
	end


	## MNL MODEL ## 

	MNL_u_val, MNL_loglik, MNL_elapsed_time = tcm_estimateMNL(N, train_assortments, choice_probs)
	MNL_train_predict = tcm_predictMNL(N, MNL_u_val, train_assortments)
	MNL_L1_train_error = sum( abs.( MNL_train_predict - choice_probs  )  )

	MNL_grand_predict = tcm_predictMNL(N, MNL_u_val, grand_assortments)
	MNL_L1_test_error = sum( abs.( MNL_grand_predict - grand_choice_probs  )  ) / 2^(N-1)

	MNL_best_a = indmax( MNL_grand_predict * revenues)
	MNL_best_r = dot(grand_choice_probs[MNL_best_a,:], revenues)
	MNL_approx_rate = MNL_best_r / grand_best_r

	row_string = string(single_data_name,  ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							proc_id, ",", 
							attempt, ",",
							"MNL", ",",
							MNL_loglik, ",",
							1, ",",
							0, ",",
							1, ",",
							MNL_L1_train_error, ",",
							MNL_L1_test_error, ",",
							MNL_approx_rate, ",",
							MNL_elapsed_time)
	row_string = string(row_string, "\n");

	@spawnat 1 begin
		print(MNL_outcsvhandle, row_string);
		flush(MNL_outcsvhandle);
	end

	## HALO-MNL MODEL ## 

	HALOMNL_u_val, HALOMNL_alpha_val, HALOMNL_loglik, HALOMNL_elapsed_time = tcm_estimateHALOMNL(N, train_assortments, choice_probs)
	HALOMNL_train_predict = tcm_predictHALOMNL(N, HALOMNL_u_val, HALOMNL_alpha_val, train_assortments)
	HALOMNL_L1_train_error = sum( abs.( HALOMNL_train_predict - choice_probs  )  )

	HALOMNL_grand_predict = tcm_predictHALOMNL(N, HALOMNL_u_val, HALOMNL_alpha_val, grand_assortments)
	HALOMNL_L1_test_error = sum( abs.( HALOMNL_grand_predict - grand_choice_probs  )  ) / 2^(N-1)

	HALOMNL_best_a = indmax( HALOMNL_grand_predict * revenues)
	HALOMNL_best_r = dot(grand_choice_probs[MNL_best_a,:], revenues)
	HALOMNL_approx_rate = HALOMNL_best_r / grand_best_r

	row_string = string(single_data_name,  ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							proc_id, ",", 
							attempt, ",",
							"HALOMNL", ",",
							HALOMNL_loglik, ",",
							1, ",",
							0, ",",
							1, ",",
							HALOMNL_L1_train_error, ",",
							HALOMNL_L1_test_error, ",",
							HALOMNL_approx_rate, ",",
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
		
		lambda, forest_left, forest_right, forest_product, forest_isLeaf, forest_L1_train_error, forest_elapsed_time = tcm_estimateForest_L1(N, train_assortments, choice_probs, time_limit, depth_limit,
																												initial_left, initial_right, initial_product, initial_isLeaf)
		K = length(lambda)

		forest_grand_predict, forest_grand_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, grand_assortments)
		forest_L1_test_error = sum( abs.( forest_grand_predict - grand_choice_probs  )  ) / 2^(N-1)

		forest_best_a = indmax( forest_grand_predict * revenues)
		forest_best_r = dot(grand_choice_probs[forest_best_a,:], revenues)
		forest_approx_rate = forest_best_r / grand_best_r


		row_string = string(single_data_name,  ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							proc_id, ",", 
							attempt, ",",
							"forest-cold", ",",
							depth_limit, ",",
							forest_L1_train_error, ",",
							K, ",",
							forest_L1_test_error, ",",
							forest_approx_rate, ",",
							forest_elapsed_time)
		row_string = string(row_string, "\n");

		@spawnat 1 begin
			print(forest_outcsvhandle, row_string);
			flush(forest_outcsvhandle);
		end
	end


	## FOREST, COLD-START (EMPTY INITIAL FOREST) ## 

	depth_list = [4]

	initial_left, initial_right, initial_product, initial_isLeaf = Array{Int64,1}[], Array{Int64,1}[], Array{Int64,1}[], Array{Bool,1}[]; #tcm_createBasicForest(N)

	for depth_limit in depth_list
		
		lambda, forest_left, forest_right, forest_product, forest_isLeaf, forest_L1_train_error, forest_elapsed_time = tcm_estimateForest_L1(N, train_assortments, choice_probs, time_limit, depth_limit,
																												initial_left, initial_right, initial_product, initial_isLeaf)
		K = length(lambda)

		forest_grand_predict, forest_grand_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, grand_assortments)
		forest_L1_test_error = sum( abs.( forest_grand_predict - grand_choice_probs  )  ) / 2^(N-1)

		forest_best_a = indmax( forest_grand_predict * revenues)
		forest_best_r = dot(grand_choice_probs[forest_best_a,:], revenues)
		forest_approx_rate = forest_best_r / grand_best_r


		row_string = string(single_data_name,  ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							proc_id, ",", 
							attempt, ",",
							"forest-cold-empty", ",",
							depth_limit, ",",
							forest_L1_train_error, ",",
							K, ",",
							forest_L1_test_error, ",",
							forest_approx_rate, ",",
							forest_elapsed_time)
		row_string = string(row_string, "\n");

		@spawnat 1 begin
			print(forest_outcsvhandle, row_string);
			flush(forest_outcsvhandle);
		end
	end



	## FOREST, WARM-START ##

	depth_list = [4]

	initial_left, initial_right, initial_product, initial_isLeaf = tcm_convertRankingToForest(N, orderings)

	for depth_limit in depth_list
		
		lambda, forest_left, forest_right, forest_product, forest_isLeaf, forest_L1_train_error, forest_elapsed_time = tcm_estimateForest_L1(N, train_assortments, choice_probs, time_limit, depth_limit,
																												initial_left, initial_right, initial_product, initial_isLeaf)
		K = length(lambda)

		forest_grand_predict, forest_grand_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, grand_assortments)
		forest_L1_test_error = sum( abs.( forest_grand_predict - grand_choice_probs  )  ) / 2^(N-1)

		forest_best_a = indmax( forest_grand_predict * revenues)
		forest_best_r = dot(grand_choice_probs[forest_best_a,:], revenues)
		forest_approx_rate = forest_best_r / grand_best_r


		row_string = string(single_data_name,  ",",
							num_train_assortments, ",",
							num_test_assortments, ",",
							proc_id, ",", 
							attempt, ",",
							"forest-ranking-warm", ",",
							depth_limit, ",",
							forest_L1_train_error, ",",
							K, ",",
							forest_L1_test_error, ",",
							forest_approx_rate, ",",
							forest_elapsed_time)
		row_string = string(row_string, "\n");

		@spawnat 1 begin
			print(forest_outcsvhandle, row_string);
			flush(forest_outcsvhandle);
		end
	end

	helper_end_time = time()

	return helper_end_time - helper_start_time;



end