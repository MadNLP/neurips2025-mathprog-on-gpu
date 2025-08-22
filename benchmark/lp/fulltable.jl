using JLD2, ArgParse, MadNLP, MadIPM, QPSReader, QuadraticModels, SparseArrays, GZip,
    CodecBzip2, NLPModels, CUDA, KernelAbstractions, MadNLPGPU, QuadraticModelsGurobi,
    Gurobi, SolverCore, Printf

JLD2.@load "results/results-miplib.jld2" results

res = [
    (
        name = @sprintf("%18s", split(key, ".mps.gz")[1]),
        tol = split(key, "_")[2],
        lognnz = @sprintf("%8.2f", log2(val.meta.nnzj + val.meta.nnzh)),
        t_gurobi = @sprintf("%8.2f", val.t_gurobi),
        t_madipm = @sprintf("%8.2f", val.t_madipm),
        solved_madipm = Int(val.sol_madipm.status) in [1, 2] ? "  1  " : "  0  ",
        solved_gurobi = val.sol_gurobi.status in [:first_order, :acceptable] ? "  1  " : "  0  ",
    )
    for (key, val) in results]

res_high = sort!(filter(x -> x.tol == "hightol", res), by = x -> parse(Float64, x.lognnz))
res_low = sort!(filter(x -> x.tol == "lowtol", res), by = x -> parse(Float64, x.lognnz))



# generate a plain text table with the results

function table(res, tol)
    contents = join([
        join([
            x.name,
            x.lognnz,
            x.solved_madipm,
            x.t_madipm,
            x.solved_gurobi,
            x.t_gurobi,
        ], " | ")
        for x in res], "\n")

    return """
--------------------------------------------------------------------
              MIPLIB benchmark results (tol = $tol)
--------------------------------------------------------------------
           problem | log2(nnz)|      MadIPM      |      Gurobi        
                   |          | solved|     time | solved|     time     
--------------------------------------------------------------------
$contents
--------------------------------------------------------------------
    """
end

write(
    "results/fulltable-miplib-high.txt",
    table(res_high, 1e-4)
)
write(
    "results/fulltable-miplib-low.txt",
    table(res_low, 1e-8)
)

