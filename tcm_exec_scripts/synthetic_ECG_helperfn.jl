function helper_fn(z, attempt)
	method, N, M, d = z
	proc_id = myid();

	helper_start_time = time()

	time_limit_overall = 6 * 60 * 60 # 6 hours
	time_limit_sub = 30 * 60 # 30 minutes
	depth_limit = d

	grand_choice_probs = readcsv(string("../tcm_data/synthetic_data_ECG_models/MNL_ECG2_Neq", N, "_grand_model_predict.csv"))
	grand_assortments = convert(Array{Int64}, readcsv(string("../tcm_data/synthetic_data_ECG_models/MNL_ECG2_Neq", N, "_grand_assortments.csv")))
	
	assortments = copy(grand_assortments)
	choice_probs = copy(grand_choice_probs)

	test_assortments = copy(grand_assortments)
	test_choice_probs = copy(grand_choice_probs)

	println("Before if")

	if (M < 2^(N-1))
		ass_ord_inds = convert(Array{Int64}, readcsv(string("../tcm_data/synthetic_data_ECG_models/Neq", N, "_randperm.csv")) )
		@show typeof(ass_ord_inds)
		assortments = grand_assortments[ ass_ord_inds[1:M], :]
		choice_probs = grand_choice_probs[ ass_ord_inds[1:M], :]

		test_assortments = grand_assortments[ ass_ord_inds[(M+1):end], :]
		test_choice_probs = grand_choice_probs[ ass_ord_inds[(M+1):end], :]
	end



	println("After if")
	num_train_assortments = size(assortments,1)
	num_test_assortments = size(test_assortments, 1)

	# FOREST, L1, COLD-START, SUBPROBLEM VIA TOP DOWN INDUCTION 
	if (method == "HCG")
		initial_left = Array{Int64,1}[]
		initial_right = Array{Int64,1}[]
		initial_product = Array{Int64,1}[]
		initial_isLeaf = Array{Bool,1}[]

		println("Starting forest L1 HCG")
		lambda, forest_left, forest_right, forest_product, forest_isLeaf, primal_objval, elapsed_time = tcm_estimateForest_L1(N, assortments, choice_probs, time_limit_overall, depth_limit, initial_left, initial_right, initial_product, initial_isLeaf)
		L1_error = primal_objval / size(assortments,1)

		K = length(lambda)
		forest_predict, forest_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, assortments)
		forest_train_error = sum( abs.( forest_predict - choice_probs  )  ) / size(assortments,1)

		forest_all_predict, forest_all_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, grand_assortments)
		forest_all_error = sum( abs.( forest_all_predict - grand_choice_probs  )  ) / size(grand_assortments,1)

		forest_test_predict, forest_test_predict_by_class = tcm_predictForest(N, K, lambda, forest_left, forest_right, forest_product, forest_isLeaf, test_assortments)
		forest_test_error = sum( abs.( forest_test_predict - test_choice_probs  )  ) / size(test_assortments,1)
		

		
		row_string = string(N,  ",",
							num_train_assortments, ",",
							d, ",",
							"forest-cold-HCG", ",",
							proc_id, ",",
							attempt, ",",
							num_test_assortments, ",",
							primal_objval, ",",
							L1_error, ",",
							forest_train_error, ",",
							forest_test_error, ",",
							forest_all_error, ",",
							K, ",",
							elapsed_time)
		row_string = string(row_string, "\n")

		@spawnat 1 begin
			print(forest_outcsvhandle, row_string)
			flush(forest_outcsvhandle);
		end
	end


	# FOREST, L1, COLD-START, SUBPROBLEM VIA MIO 
	if (method == "ECG")
		initial_left = Array{Int64,1}[]
		initial_right = Array{Int64,1}[]
		initial_product = Array{Int64,1}[]
		initial_isLeaf = Array{Bool,1}[]

		println("Starting forest L1 ECG")
		lambda_MIO, forest_left_MIO, forest_right_MIO, forest_product_MIO, forest_isLeaf_MIO, primal_objval_MIO, elapsed_time_MIO = tcm_estimateForest_L1_MIO(N, assortments, choice_probs, time_limit_overall, time_limit_sub, depth_limit, initial_left, initial_right, initial_product, initial_isLeaf)

		L1_error_MIO = primal_objval_MIO / size(assortments,1)

		K_MIO = length(lambda_MIO)
		forest_MIO_predict, forest_MIO_predict_by_class = tcm_predictForest(N, K_MIO, lambda_MIO, forest_left_MIO, forest_right_MIO, forest_product_MIO, forest_isLeaf_MIO, assortments)
		forest_MIO_train_error = sum( abs.( forest_MIO_predict - choice_probs  )  ) / size(assortments,1)

		forest_MIO_all_predict, forest_MIO_all_predict_by_class = tcm_predictForest(N, K_MIO, lambda_MIO, forest_left_MIO, forest_right_MIO, forest_product_MIO, forest_isLeaf_MIO, grand_assortments)
		forest_MIO_all_error = sum( abs.( forest_MIO_all_predict - grand_choice_probs  )  ) / size(grand_assortments,1)

		forest_MIO_test_predict, forest_MIO_test_predict_by_class = tcm_predictForest(N, K_MIO, lambda_MIO, forest_left_MIO, forest_right_MIO, forest_product_MIO, forest_isLeaf_MIO, test_assortments)
		forest_MIO_test_error = sum( abs.( forest_MIO_test_predict - test_choice_probs  )  ) / size(test_assortments,1)



		row_string = string(N,  ",",
							num_train_assortments, ",",
							d, ",",
							"forest-cold-ECG", ",",
							proc_id, ",",
							attempt, ",",
							num_test_assortments, ",",
							primal_objval_MIO, ",",
							L1_error_MIO, ",",
							forest_MIO_train_error, ",",
							forest_MIO_test_error, ",",
							forest_MIO_all_error, ",",
							K_MIO, ",",
							elapsed_time_MIO)
		row_string = string(row_string, "\n")

		@spawnat 1 begin
			print(forest_outcsvhandle, row_string)
			flush(forest_outcsvhandle);
		end
	end


	if (method == "RTS")
		initial_left = Array{Int64,1}[]
		initial_right = Array{Int64,1}[]
		initial_product = Array{Int64,1}[]
		initial_isLeaf = Array{Bool,1}[]

		K_vec = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 100000 ]

		for K in K_vec 

			println("Starting forest L1 RTS")
			lambda_RTS, forest_left_RTS, forest_right_RTS, forest_product_RTS, forest_isLeaf_RTS, primal_objval_RTS, elapsed_time_RTS = tcm_estimateForest_L1_RTS(N, assortments, choice_probs, depth_limit, K)

			L1_error_RTS = primal_objval_RTS / size(assortments,1)

			forest_RTS_predict, forest_RTS_predict_by_class = tcm_predictForest(N, K, lambda_RTS, forest_left_RTS, forest_right_RTS, forest_product_RTS, forest_isLeaf_RTS, assortments)
			forest_RTS_train_error = sum( abs.( forest_RTS_predict - choice_probs  )  ) / size(assortments,1)


			forest_RTS_all_predict, forest_RTS_all_predict_by_class = tcm_predictForest(N, K, lambda_RTS, forest_left_RTS, forest_right_RTS, forest_product_RTS, forest_isLeaf_RTS, grand_assortments)
			forest_RTS_all_error = sum( abs.( forest_RTS_all_predict - grand_choice_probs  )  ) / size(grand_assortments,1)

			forest_RTS_test_predict, forest_RTS_test_predict_by_class = tcm_predictForest(N, K, lambda_RTS, forest_left_RTS, forest_right_RTS, forest_product_RTS, forest_isLeaf_RTS, test_assortments)
			forest_RTS_test_error = sum( abs.( forest_RTS_test_predict - test_choice_probs  )  ) / size(test_assortments,1)


			
			row_string = string(N,  ",",
								num_train_assortments, ",",
								d, ",",
								"forest-cold-RTS", ",",
								proc_id, ",",
								attempt, ",",
								num_test_assortments, ",",
								primal_objval_RTS, ",",
								L1_error_RTS, ",",
								forest_RTS_train_error, ",",
								forest_RTS_test_error, ",",
								forest_RTS_all_error, ",",
								K, ",",
								elapsed_time_RTS)
			row_string = string(row_string, "\n")

			# @show row_string

			@spawnat 1 begin
				print(forest_outcsvhandle, row_string)
				flush(forest_outcsvhandle);
			end
		end 
	end

	helper_end_time = time()

	return helper_end_time - helper_start_time;
end