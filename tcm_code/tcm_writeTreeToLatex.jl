#tcm_writeTreeToLatex

function tcm_writeTreeToLatex(outpath, outdir, tree_left, tree_right, tree_product, tree_isLeaf, var_names, doPDF)

	# temp_fn = x -> x ? "\\textbf{stop}" : "\\textbf{go}";
	# tree_action = map(temp_fn, tree_action)

	numNodes = length(tree_left);
	isclosed = zeros(Bool, numNodes)
	cn = 1;

	outhandle = open(outpath, "w")
	print(outhandle, "\\documentclass{standalone}\n")
	print(outhandle, "\\usepackage{forest}\n\n")

	print(outhandle, "\\begin{document}\n")
	print(outhandle, "\\begin{forest}\n")
	print(outhandle, "for tree={draw=none, l sep=40pt, s sep = 15pt, edge={->,thick}}\n")
	writeNodeToLatex(outhandle, 1, tree_left, tree_right, tree_product, tree_isLeaf, var_names)
	print(outhandle, "\\end{forest}\n")
	print(outhandle, "\\end{document}\n")
	close(outhandle)

	if (doPDF)
		run(`pdflatex -output-directory $outdir $outpath`)
	end


end


function writeNodeToLatex(outhandle, cn, tree_left, tree_right, tree_product, tree_isLeaf, var_names)

	if (tree_isLeaf[cn])
		if (!isempty(var_names))
			single_var_name = var_names[tree_product[cn]];
		else
			single_var_name = string(tree_product[cn]);
		end

		temp = string("[", single_var_name, ", draw")
		if (cn == 2)
			temp = string(temp, ", edge label={node[midway,left, yshift = 0.1cm]{\\emph{yes}}},")
		end

		if (cn == 3)
			temp = string(temp, ", edge label={node[midway,right, yshift = 0.1cm]{\\emph{no}}}")
		end
		temp = string(temp, "] \n");
		print(outhandle, temp);
	else

		if (!isempty(var_names))
			single_var_name = var_names[tree_product[cn]];
		else
			single_var_name = string(tree_product[cn]);
		end



		temp = string("[", single_var_name, ", ")
		if (cn == 2)
			temp = string(temp, "edge label={node[midway,left, yshift = 0.1cm]{\\emph{yes}}},")
		end

		if (cn == 3)
			temp = string(temp, "edge label={node[midway,right, yshift = 0.1cm]{\\emph{no}}}")
		end

		temp = string(temp, "\n");

		print(outhandle, temp)

		left = tree_left[cn]; 
		writeNodeToLatex(outhandle, left, tree_left, tree_right, tree_product, tree_isLeaf,  var_names)

		right = tree_right[cn]; 
		writeNodeToLatex(outhandle, right, tree_left, tree_right, tree_product, tree_isLeaf,  var_names)

		print(outhandle, "]\n")
	end

end