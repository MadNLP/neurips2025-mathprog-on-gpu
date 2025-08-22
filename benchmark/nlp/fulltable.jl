using CUDA, MadNLPGPU, NLPModelsIpopt, JLD2, Artifacts, Logging, HSL_jll, ArgParse, Printf

s = ArgParseSettings()

@add_arg_table s begin
    "class"
    help = "class of benchmark results to be used"
    required = true
end

function cops(key)
    name, args = split(key, "_model-")
    return replace(name * "-" * split(split(args,"(")[2],")")[1], " " => "")
end
class = parse_args(s)["class"]

JLD2.@load "results/results-$(class).jld2" results

res = [
    (
        name = class == "opf" ?
            @sprintf("%18s", split(split(key, ".m")[1], "pglib_opf_")[2]) :
            @sprintf("%18s", cops(key)),
        tol = split(key, "_")[end],
        lognnz = @sprintf("%8.2f", log2(val.meta.nnzj + val.meta.nnzh)),
        t_ipopt = @sprintf("%8.2f", val.t_ipopt),
        t_madnlp = @sprintf("%8.2f", val.t_madnlp),
        solved_madnlp = Int(val.sol_madnlp.status) in [1, 2] ? "  1  " : "  0  ",
        solved_ipopt = val.sol_ipopt.status in [:first_order, :acceptable] ? "  1  " : "  0  ",
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
            x.solved_madnlp,
            x.t_madnlp,
            x.solved_ipopt,
            x.t_ipopt,
        ], " | ")
        for x in res], "\n")

    return """
--------------------------------------------------------------------
              $class benchmark results (tol = $tol)
--------------------------------------------------------------------
           problem | log2(nnz)|      MadNLP      |      Ipopt        
                   |          | solved|     time | solved|     time     
--------------------------------------------------------------------
$contents
--------------------------------------------------------------------
    """
end

write(
    "results/fulltable-$class-high.txt",
    table(res_high, 1e-4)
)
write(
    "results/fulltable-$class-low.txt",
    table(res_low, 1e-8)
)

