using DelimitedFiles
using MadNLP
using MadIPM
using MadNLPHSL
using QPSReader
using QuadraticModels
using SparseArrays

using GZip
using CodecBzip2
using HSL
using NLPModels
using SparseArrays

import QuadraticModels: SparseMatrixCOO

"""
    import_mps(filename::String)

Import instance from the file whose path is specified in `filename`.

The function parses the file's extension to adapt the import. If the extension
is `.mps`, `.sif` or `.SIF`, it directly reads the file. If the extension
is `.gz` or `.bz2`, it decompresses the file using gzip or bzip2, respectively.

"""
function import_mps(filename)
    ext = match(r"(.*)\.(.*)", filename).captures[2]
    data = if ext ∈ ("mps", "sif", "SIF")
        readqps(filename)
    elseif ext == "gz"
        GZip.open(filename, "r") do gz
            readqps(gz)
        end
    elseif ext == "bz2"
        open(filename, "r") do io
            stream = Bzip2DecompressorStream(io)
            readqps(stream)
        end
    end
    return data
end

function _scale_coo!(A, Dr, Dc)
    k = 1
    for (i, j) in zip(A.rows, A.cols)
        A.vals[k] = A.vals[k] / (Dr[i] * Dc[j])
        k += 1
    end
end

"""
    scale_qp(qp::QuadraticModel)

Scale QP using Ruiz' equilibration method.

The function scales the Jacobian ``A`` as ``As = Dr * A * Dc``, with ``As``
a matrix whose rows and columns have an infinite norm close to 1.

The scaling is computed using `HSL.mc77`, implementing the Ruiz equilibration method.

"""
function scale_qp(qp::QuadraticModel)
    A = qp.data.A
    m, n = size(A)

    if !LIBHSL_isfunctional()
        return qp
    end

    A_csc = sparse(A.rows, A.cols, A.vals, m, n)
    Dr, Dc = HSL.mc77(A_csc, 0)

    Hs = copy(qp.data.H)
    As = copy(qp.data.A)
    _scale_coo!(Hs, Dc, Dc)
    _scale_coo!(As, Dr, Dc)

    data = QuadraticModels.QPData(
        qp.data.c0,
        qp.data.c ./ Dc,
        qp.data.v,
        Hs,
        As,
    )

    return QuadraticModel(
        NLPModelMeta(
            qp.meta.nvar;
            ncon=qp.meta.ncon,
            lvar=qp.meta.lvar .* Dc,
            uvar=qp.meta.uvar .* Dc,
            lcon=qp.meta.lcon ./ Dr,
            ucon=qp.meta.ucon ./ Dr,
            x0=qp.meta.x0 .* Dc,
            y0=qp.meta.y0 ./ Dr,
            nnzj=qp.meta.nnzj,
            lin_nnzj=qp.meta.nnzj,
            lin=qp.meta.lin,
            nnzh=qp.meta.nnzh,
            minimize=qp.meta.minimize,
        ),
        Counters(),
        data,
    )
end

function run_benchmark(src, probs; use_gpu=false, reformulate::Bool=false, test_reader::Bool=false)
    nprobs = length(probs)
    results = zeros(nprobs, 9)
    for (k, prob) in enumerate(probs)
        @info "$prob -- $k / $nprobs"
        qpdat = try
            import_mps(joinpath(src, prob))
        catch e
            @warn "Failed to import $prob: $e"
            continue
        end
        @info "The problem $prob was imported."

        if !test_reader
            qp = QuadraticModel(qpdat)
            presolved_qp, flag = MadIPM.presolve_qp(qp)
            !flag && continue  # problem already solved, unbounded or infeasible
            scaled_qp = scale_qp(presolved_qp)
            qp_cpu = reformulate ? MadIPM.standard_form_qp(scaled_qp) : scaled_qp

            qp_ = if use_gpu
                convert(QuadraticModel{Float64, CuVector{Float64}}, qp_cpu)
            else
                qp_cpu
            end

            try
                solver = MadIPM.MPCSolver(
                    qp_;
                    max_iter=300,
                    linear_solver= use_gpu ? MadNLPGPU.CUDSSSolver : Ma57Solver,
                    cudss_algorithm=MadNLP.LDL,
                    regularization=MadIPM.FixedRegularization(1e-8, -1e-8),
                    print_level=MadNLP.INFO,
                    rethrow_error=true,
                )
                res = MadIPM.solve!(solver)
                results[k, 1] = Int(qp_cpu.meta.nvar)
                results[k, 2] = Int(qp_cpu.meta.ncon)
                results[k, 3] = Int(qp_cpu.meta.nnzj)
                results[k, 4] = Int(qp_cpu.meta.nnzh)
                results[k, 5] = Int(res.status)
                results[k, 6] = res.iter
                results[k, 7] = res.objective
                results[k, 8] = res.counters.total_time
                results[k, 9] = res.counters.linear_solver_time
            catch ex
                results[k, 8] = -1
                @warn "Failed to solve $prob: $ex"
                continue
            end
        end
    end
    return results
end

src = joinpath(@__DIR__, "instances", "miplib2010")
mps_files = readdlm(joinpath(@__DIR__, "miplib_problems.txt"))[:]

reformulate = true
test_reader = false

#=
    Run benchmark on GPU
=#

name_results = "benchmark-miplib-gpu.txt"
results = run_benchmark(src, mps_files; use_gpu=true, reformulate=reformulate, test_reader=test_reader)
path_results = joinpath(@__DIR__, name_results)
writedlm(path_results, [mps_files results])
