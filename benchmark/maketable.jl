using CUDA, MadNLPGPU, NLPModelsIpopt, JLD2, Artifacts, Logging, HSL_jll, ArgParse, Printf

s = ArgParseSettings()

@add_arg_table s begin
    "class"
    help = "class of benchmark results to be used"
    required = true
end

class = parse_args(s)["class"]

JLD2.@load "results-$(class).jld2" results

maxtime = 900
nnzcut1 = 10000
nnzcut2 = 1000000

check_time_ipopt(status) = status == :first_order 
check_time_madnlp(status) = Int(status) == 1 

total = [(
    name = key,
    nnz = val.meta.nnzj + val.meta.nnzh,
    solved_ipopt = check_time_ipopt(val.sol_ipopt.status),
    t_ipopt = val.t_ipopt,
    solved_madnlp = check_time_madnlp(val.sol_madnlp.status),
    t_madnlp = val.t_madnlp, 
) for (key, val) in results]

total_lowtol = filter(x -> endswith(x.name, "_lowtol"), total)
total_hightol = filter(x -> endswith(x.name, "_hightol"), total)
nsmall = sum(x.nnz < nnzcut1 for x in total_lowtol; init = 0)
nmedium = sum(x.nnz >= nnzcut1 && x.nnz < nnzcut2 for x in total_lowtol; init = 0)
nlarge = sum(x.nnz >= nnzcut2 for x in total_lowtol; init = 0)
ntotal = length(total_lowtol)

function write_row(tol, total)
    small = filter(x -> x.nnz < nnzcut1, total)
    medium= filter(x -> x.nnz >= nnzcut1 && x.nnz < nnzcut2, total)
    large = filter(x -> x.nnz >= nnzcut2, total)

    solved_total_ipopt = sum(x.solved_ipopt for x in total; init = 0)
    solved_total_madnlp = sum(x.solved_ipopt for x in total; init = 0)
    solved_small_ipopt = sum(x.solved_ipopt for x in small; init = 0)
    solved_small_madnlp = sum(x.solved_madnlp for x in small; init = 0)
    solved_medium_ipopt = sum(x.solved_ipopt for x in medium; init = 0)
    solved_medium_madnlp = sum(x.solved_madnlp for x in medium; init = 0)
    solved_large_ipopt = sum(x.solved_ipopt for x in large; init = 0)
    solved_large_madnlp = sum(x.solved_madnlp for x in large; init = 0)

    sgm10_total_ipopt  = @sprintf("%4.4f", prod((x.solved_ipopt ? x.t_ipopt : maxtime ) + 10 for x in total;init = 1)^(1/length(total)) - 10)
    sgm10_total_madnlp = @sprintf("%4.4f", prod((x.solved_madnlp ? x.t_madnlp : maxtime) + 10 for x in total;init = 1)^(1/length(total)) - 10)
    sgm10_small_ipopt  = @sprintf("%4.4f", prod((x.solved_ipopt ? x.t_ipopt : maxtime ) + 10 for x in small;init = 1)^(1/length(small)) - 10)
    sgm10_small_madnlp = @sprintf("%4.4f", prod((x.solved_madnlp ? x.t_madnlp : maxtime) + 10 for x in small;init = 1)^(1/length(small)) - 10)
    sgm10_medium_ipopt  = @sprintf("%4.4f", prod((x.solved_ipopt ? x.t_ipopt : maxtime ) + 10 for x in medium;init = 1)^(1/length(medium)) - 10)
    sgm10_medium_madnlp = @sprintf("%4.4f", prod((x.solved_madnlp ? x.t_madnlp : maxtime) + 10 for x in medium;init = 1)^(1/length(medium)) - 10)
    sgm10_large_ipopt  = @sprintf("%4.4f", prod((x.solved_ipopt ? x.t_ipopt : maxtime ) + 10 for x in large;init = 1)^(1/length(large)) - 10)
    sgm10_large_madnlp = @sprintf("%4.4f", prod((x.solved_madnlp ? x.t_madnlp : maxtime ) + 10 for x in large;init = 1)^(1/length(large)) - 10)
    return """
  \\multirow{2}{*}{\$$(tol)\$} & MadNLP (gpu) & $(solved_small_madnlp) & $(sgm10_small_madnlp) & $(solved_medium_madnlp) & $(sgm10_medium_madnlp) & $(solved_large_madnlp) & $(sgm10_large_madnlp) & $(solved_total_madnlp) & $(sgm10_total_madnlp)  \\\\
                        & Ipopt (cpu) & $(solved_small_ipopt) & $(sgm10_small_ipopt) & $(solved_medium_ipopt) & $(sgm10_medium_ipopt) & $(solved_large_ipopt) & $(sgm10_large_ipopt) & $(solved_total_ipopt) & $(sgm10_total_ipopt)  \\\\
"""
end


write("table-$(class).tex", """
\\begin{tabular}{|c|c|cc|cc|cc|cc|}
  \\hline
  \\multirow{ 3}{*}{Tol} & \\multirow{ 3}{*}{Solver} & \\multicolumn{2}{c|}{\\textbf{Small} ($(nsmall))}& \\multicolumn{2}{c|}{\\textbf{Medium} ($(nmedium))}& \\multicolumn{2}{c|}{\\textbf{Large} ($(nlarge))}& \\multicolumn{2}{c|}{\\multirow{2}{*}{\\textbf{Total} ($(length(total)))}}\\\\
                        && \\multicolumn{2}{c|}{($(maxtime) sec max)}& \\multicolumn{2}{c|}{($(maxtime) sec max)}& \\multicolumn{2}{c|}{($(maxtime) sec max)}&&\\\\
                        &&  Solved & Time &  Solved & Time &  Solved & Time &  Solved & Time \\\\
  \\hline
  $(write_row("10^{-4}", total_hightol))
  \\hline
  $(write_row("10^{-8}", total_lowtol))
  \\hline
\\end{tabular}
""")

