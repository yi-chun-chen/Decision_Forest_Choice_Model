# Script to generate results for the synthetic instances with assortment-based spliting from Chen and Mišić (see Section EC.6.1 of the paper).
#
# This script is configured to be run on Amazon EC2 instances. It is accompanied by two other scripts: a helperfn (helper function) script, synthetic_ass_kfold_helperfn.jl,
# and an include script, synthetic_ass_kfold_include.jl. 
#
# The helper function script contains all the code necessary to load data for a single instance, run all the necessary methods on the instance, 
# and then to record the results. 
#
# The include script contains all the include() statements to make sure the helper function script is equipped with all the necessary functions.
# 
# The present script, synthetic_ass_kfold.jl, is effectively a custom distributed job scheduler. It creates a list of "jobs" (different instances; for this temporal splitting experiment,
# this is just one of the product categories).
# It then loops through these jobs, assigning batches of these jobs to newly created processors. Once the processors 
# finish, the processors are killed and new ones are created for the next batch of jobs. This loop continues until all jobs have been completed. This architecture was necessary due 
# to issues with running Julia v0.6's pmap() on Amazon EC2. 
#
# To run it, navigate to the mdpart_exec_scripts/ directory, and type 
#
# julia synthetic_ass_kfold.jl
#
# in the command line. On Amazon EC2, a more useful way to run it is 
#
# nohup julia synthetic_ass_kfold.jl & 
#  
# The results will be placed in a folder under tcm_testing/synthetic_ass_kfold_test/ . The out_*.csv result files contain the performance of the different methods.
#
# At the end of the script, run(`sudo shutdown -h now`) will initiate the shutdown behavior on the machine. For some (most?) Amazon EC2 instances, this should cause the 
# instance to "stop", to prevent you from accruing more EC2 fees; you can always start the instance later to download the results to your computer. 
# *** Double check to make sure this is the default behavior for you (you default may be that shutdown causes the EC2 instance to terminate!!!). ***


include("synthetic_ass_kfold_include.jl")

machine = 2;
#expdirpath = mdpart_createPath(machine, "exp_dir", "OptionPricing_custompar_test"); 
expdirpath = "../tcm_testing/synthetic_assF_kfold_test/"
if (!isdir(expdirpath))
	mkdir(expdirpath);
end

target_exec_script = string(expdirpath, "exec_script.jl");
if (isfile(target_exec_script))
	rm(target_exec_script)
end
cp(@__FILE__, target_exec_script)

initial_helper_script = string( pwd(), "/synthetic_ass_kfold_helperfn.jl")
target_helper_script = string(expdirpath, "helperfn.jl")
if (isfile(target_helper_script))
	rm(target_helper_script)
end
cp(initial_helper_script, target_helper_script)




forest_outcsvfilepath = string(expdirpath, "out_forest.csv");
forest_outcsvhandle = open(forest_outcsvfilepath, "w")

row_string = string("data_name",  ",",
							"test_fold", ",",
							"num_train_assortments", ",",
							"num_test_assortments", ",",
							"num_train_transactions", ",",
							"num_test_transactions", ",",
							"proc_id", ",", 
							"attempt", ",",
							"method", ",",
							"depth", ",",
							"loglik", ",",
							"K", ",",
							"perTol", ",",
							"test_KL", ",",
							"noinf_count", ",",
							"test_KL_avg", ",",
							"elapsed_time", ",",
							"simple_avg_leaves", ",",
							"weighted_avg_leaves", ",", 
							"max_leaves", ",",
							"simple_avg_depth", ",", 
							"weighted_avg_depth", ",",
							"max_depth")
row_string = string(row_string, "\n");
print(forest_outcsvhandle, row_string);
flush(forest_outcsvhandle);



ranking_outcsvfilepath = string(expdirpath, "out_ranking.csv");
ranking_outcsvhandle = open(ranking_outcsvfilepath, "w")

row_string = string("data_name",  ",",
							"test_fold", ",",
							"num_train_assortments", ",",
							"num_test_assortments", ",",
							"num_train_transactions", ",",
							"num_test_transactions", ",",
							"proc_id", ",", 
							"attempt", ",",
							"method", ",",
							"loglik", ",",
							"K", ",",
							"perTol", ",",
							"test_KL", ",",
							"noinf_count", ",",
							"test_KL_avg", ",",
							"elapsed_time", ",",
							"avg_CS_size", ",", 
							"weighted_avg_CS_size", ",", 
							"max_CS_size")
row_string = string(row_string, "\n");
print(ranking_outcsvhandle, row_string);
flush(ranking_outcsvhandle);



MNL_outcsvfilepath = string(expdirpath, "out_MNL.csv");
MNL_outcsvhandle = open(MNL_outcsvfilepath, "w")

row_string = string("data_name",  ",",
							"test_fold", ",",
							"num_train_assortments", ",",
							"num_test_assortments", ",",
							"num_train_transactions", ",",
							"num_test_transactions", ",",
							"proc_id", ",", 
							"attempt", ",",
							"method", ",",
							"loglik", ",",
							"K", ",",
							"perTol", ",",
							"numInitializations", ",",
							"test_KL", ",",
							"noinf_count", ",",
							"test_KL_avg", ",",
							"elapsed_time")
row_string = string(row_string, "\n");
print(MNL_outcsvhandle, row_string);
flush(MNL_outcsvhandle);




#nfolds = 5
datanames = [ "LCMNL_1", "LCMNL_2", "LCMNL_3", "LCMNL_4", "LCMNL_5",
				"HALOMNL_1", "HALOMNL_2", "HALOMNL_3", "HALOMNL_4", "HALOMNL_5",
				"ranking50_1", "ranking50_2", "ranking50_3", "ranking50_4", "ranking50_5",
				"forest50_1", "forest50_2", "forest50_3", "forest50_4", "forest50_5"]




instances_vec = Any[];

nfolds = 5


for single_data_name in datanames
	# assortments = convert(Array{Int64}, readcsv( string("../tcm_data/IRI_data_assortments/", single_data_name, "_assortments.csv") ) )
	#M = size(assortments,1)
	for fold in 1:nfolds
		push!(instances_vec, Any[single_data_name, fold] )
	end
end





numInstances = length(instances_vec);


par_size = 10;

todo = collect(1:numInstances);

current_jobs = zeros(Int64, 0);
current_jobs_futures = Future[];
current_procs = zeros(Int64, 0)
isfinished = zeros(Bool, numInstances);


results = zeros(Float64, numInstances);
failed_attempts = zeros(Int64,numInstances);


overall_start_time = time()


iter = 0;

while ( !isempty(todo) )
	iter += 1
	numRemaining = length(todo);
	println("Starting iteration $iter ... ($numRemaining left) ")

	for i in 1:par_size
		if (isempty(todo))
			break;
		end
		single_job = pop!(todo)
		push!(current_jobs, single_job)
		newproc = addprocs(1)[1];
		fetch(@spawnat newproc include("synthetic_ass_kfold_include.jl"))

		temp = @spawnat newproc helper_fn(instances_vec[single_job], failed_attempts[single_job])
		push!(current_jobs_futures, temp )
		push!(current_procs, newproc)
	end

	for i in 1:par_size
		if (isempty(current_jobs_futures))
			break;
		end
		single_future = pop!(current_jobs_futures);
		single_job = pop!(current_jobs)
		single_proc = pop!(current_procs)

		try 
			single_result = fetch(single_future)
			@show single_result
			results[single_job] = single_result
			rmprocs(single_proc)
		catch y
			println("\t Error!")
			@show y
			push!(todo, single_job)
			failed_attempts[single_job] += 1;
			rmprocs(single_proc)
		end
	end

end

println("Finished")
@show results
@show failed_attempts

@show nprocs()


@show sum(results)
total_helper_time = sum(results);

overall_end_time = time();

overall_elapsed_time = overall_end_time - overall_start_time

@show overall_elapsed_time

timingcsvfilepath = string(expdirpath, "out_timing.txt");
timingcsvhandle = open(timingcsvfilepath, "w")
print(timingcsvhandle, string("sumhelper_time_vec,",  total_helper_time, "\n") )
print(timingcsvhandle, string("elapsed_time,", overall_elapsed_time, "\n") )
close(timingcsvhandle)



close(forest_outcsvhandle);
close(ranking_outcsvhandle);
close(MNL_outcsvhandle)


run(`sudo shutdown -h now`)