# synthetic_ass_kfold_include.jl

include("../tcm_code/tcm_transactionsToCounts.jl")

include("../tcm_code/tcm_estimateMNL.jl")
include("../tcm_code/tcm_predictMNL.jl")

include("../tcm_code/tcm_estimateMMNL.jl")
include("../tcm_code/tcm_estimateMMNL_kcv.jl")
include("../tcm_code/tcm_predictMMNL.jl")

include("../tcm_code/tcm_estimateHALOMNL.jl")
include("../tcm_code/tcm_predictHALOMNL.jl")

include("../tcm_code/tcm_estimateRanking.jl")
include("../tcm_code/tcm_predictRanking.jl")

include("../tcm_code/tcm_estimateForest.jl")
include("../tcm_code/tcm_estimateForest_kcv.jl")

include("../tcm_code/tcm_predictForest.jl")

include("../tcm_code/tcm_evaluateKL.jl")

include("../tcm_code/tcm_leafStats.jl")
include("../tcm_code/tcm_rankingStats.jl")

include("synthetic_ass_kfold_helperfn.jl")