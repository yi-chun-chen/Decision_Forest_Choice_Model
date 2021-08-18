Decision Forest: A Nonparametric Approach to Modeling Irrational Choice
-------------------------------

This repository contains all the code used in the numerical experiments in the paper:

> Y.-C. Chen and V. V. Mišić (2021). Decision Forest: A Nonparametric Approach to Modeling Irrational Choice. Management Science, to appear.  Available at SSRN: https://ssrn.com/abstract=3376273.

Citation:
---------

If you use the code and/or data in this repository in your own research, please cite the above paper as follows:

```bibtex
@article{chen2021decision,
	title={Decision Forest: A Nonparametric Approach to Modeling Irrational Choice},
	author={Chen, Yi-Chun and Mi\v{s}i\'{c}, Velibor V.},
	journal={Management Science},
	year={2021},
	note={Available at SSRN: \url{https://ssrn.com/abstract=3376273}
}
```

License:
--------

This code is available under the MIT License.

Copyright (C) 2021 Yi-Chun Chen and Velibor Misic

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.


Software Requirements:
----------------------

To run the code, you need to have:
+ Julia version 0.6.*
+ JuMP version 0.18.*
+ Gurobi version 8.0

The code should be compatible with later versions of Gurobi; it is not compatible with newer versions of Julia/JuMP.


Code Structure:
---------------

The code is structured into several directories:

+ `tcm_data/`: Contains the data files used in the numerical experiments. There are three subdirectories
  + `synthetic_data_assortments/`: These are the data files for the assortment splitting experiment using synthetically generated instances (Section EC.6.1 of the ecompanion of the paper).
  + `synthetic_data_models/`: These are the data files for the "known choice probability" experiment using synthetically generated instances (Section EC.6.2 of the ecompanion of the paper).
  + `synthetic_data_ECG_models/`: These are the data files for the comparison of exact column generation with heuristic column generation and randomized tree sampling (Section EC.2 of the ecompanion of the paper).


+ `tcm_code/`: Contains functions needed to estimate the various models, generate synthetic data and evaluate out of sample predictions, as well as helper functions (such as `tcm_transactionsToCounts.jl`, which converts a transaction file to transaction counts, and `tcm_writeTreeToLatex.jl`, which takes a purchase decision tree and writes a Latex file that can be complied into a tikz formatted visualization of that tree).


+ `tcm_exec_scripts/`: Contains scripts that can be executed to run the functions in `tcm_code/` on a large swathe of instances. Each script contains instructions in the comments on how to run it. Note that the are structured for Amazon EC2 and will run best on that platform.


+ `tcm_testing/`: Upon execution of scripts in `tcm_exec_scripts/`, this directory will contain directories with results (on, e.g., estimation time, out-of-sample K-L divergence, etc).
