using JLD2, ArgParse, MadNLP, MadIPM, QPSReader, QuadraticModels, SparseArrays, GZip,
    CodecBzip2, NLPModels, CUDA, KernelAbstractions, MadNLPGPU, QuadraticModelsGurobi,
    Gurobi, SolverCore, Printf

import QuadraticModelsGurobi: gurobi
import QuadraticModels: SparseMatrixCOO

JLD2.@load "results/results-miplib.jld2" results

maxtime = 900.
nnzcut1 = 2^18
nnzcut2 = 2^20

check_time_gurobi(status) = status in [:first_order , :acceptable]
check_time_madipm(status) = Int(status) in [1, 2]

total = [(
    name = key,
    nnz = val.meta.nnzj + val.meta.nnzh,
    t_gurobi = min(val.t_gurobi, maxtime),
    t_madipm = min(val.t_madipm, maxtime), 
    solved_madipm = check_time_madipm(val.sol_madipm.status),
    solved_gurobi = val.sol_gurobi == :skipped ? false : check_time_gurobi(val.sol_gurobi.status),
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

    solved_total_gurobi = sum(x.solved_gurobi for x in total; init = 0)
    solved_total_madipm = sum(x.solved_madipm for x in total; init = 0)
    solved_small_gurobi = sum(x.solved_gurobi for x in small; init = 0)
    solved_small_madipm = sum(x.solved_madipm for x in small; init = 0)
    solved_medium_gurobi = sum(x.solved_gurobi for x in medium; init = 0)
    solved_medium_madipm = sum(x.solved_madipm for x in medium; init = 0)
    solved_large_gurobi = sum(x.solved_gurobi for x in large; init = 0)
    solved_large_madipm = sum(x.solved_madipm for x in large; init = 0)

    sgm10_total_gurobi  = @sprintf("%4.4f", prod((x.solved_gurobi ? x.t_gurobi : maxtime ) + 10 for x in total;init = 1)^(1/length(total)) - 10)
    sgm10_total_madipm = @sprintf("%4.4f", prod((x.solved_madipm ? x.t_madipm : maxtime) + 10 for x in total;init = 1)^(1/length(total)) - 10)
    sgm10_small_gurobi  = @sprintf("%4.4f", prod((x.solved_gurobi ? x.t_gurobi : maxtime ) + 10 for x in small;init = 1)^(1/length(small)) - 10)
    sgm10_small_madipm = @sprintf("%4.4f", prod((x.solved_madipm ? x.t_madipm : maxtime) + 10 for x in small;init = 1)^(1/length(small)) - 10)
    sgm10_medium_gurobi  = @sprintf("%4.4f", prod((x.solved_gurobi ? x.t_gurobi : maxtime ) + 10 for x in medium;init = 1)^(1/length(medium)) - 10)
    sgm10_medium_madipm = @sprintf("%4.4f", prod((x.solved_madipm ? x.t_madipm : maxtime) + 10 for x in medium;init = 1)^(1/length(medium)) - 10)
    sgm10_large_gurobi  = @sprintf("%4.4f", prod((x.solved_gurobi ? x.t_gurobi : maxtime ) + 10 for x in large;init = 1)^(1/length(large)) - 10)
    sgm10_large_madipm = @sprintf("%4.4f", prod((x.solved_madipm ? x.t_madipm : maxtime ) + 10 for x in large;init = 1)^(1/length(large)) - 10)
    return """
  \\multirow{2}{*}{\$$(tol)\$} & Madipm (gpu) & $(solved_small_madipm) & $(sgm10_small_madipm) & $(solved_medium_madipm) & $(sgm10_medium_madipm) & $(solved_large_madipm) & $(sgm10_large_madipm) & $(solved_total_madipm) & $(sgm10_total_madipm)  \\\\
                        & Gurobi (cpu) & $(solved_small_gurobi) & $(sgm10_small_gurobi) & $(solved_medium_gurobi) & $(sgm10_medium_gurobi) & $(solved_large_gurobi) & $(sgm10_large_gurobi) & $(solved_total_gurobi) & $(sgm10_total_gurobi)  \\\\
"""
end


write("results/table-miplib.tex", """
\\begin{tabular}{|c|c|cc|cc|cc|cc|}
  \\hline
  \\multirow{ 3}{*}{Tol} & \\multirow{ 3}{*}{Solver} & \\multicolumn{2}{c|}{\\textbf{Small} ($(nsmall))}& \\multicolumn{2}{c|}{\\textbf{Medium} ($(nmedium))}& \\multicolumn{2}{c|}{\\textbf{Large} ($(nlarge))}& \\multicolumn{2}{c|}{\\multirow{2}{*}{\\textbf{Total} ($ntotal)}}\\\\
                        && \\multicolumn{2}{c|}{nnz\$<2^{18}\$}& \\multicolumn{2}{c|}{\$2^{18}\\leq\$nnz\$<2^{20}\$}& \\multicolumn{2}{c|}{\$2^{20}\\leq\$ nnz}&&\\\\
                        &&  Solved & Time &  Solved & Time &  Solved & Time &  Solved & Time \\\\
  \\hline
  $(write_row("10^{-4}", total_hightol))
  \\hline
  $(write_row("10^{-8}", total_lowtol))
  \\hline
\\end{tabular}
""")

