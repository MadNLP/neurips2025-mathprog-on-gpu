using Artifacts, JLD2, ArgParse, MadNLP, MadIPM, QPSReader, QuadraticModels, SparseArrays, GZip,
    CodecBzip2, NLPModels, CUDA, KernelAbstractions, MadNLPGPU, QuadraticModelsGurobi,
    Gurobi, SolverCore, DelimitedFiles, HSL

import QuadraticModelsGurobi: gurobi
import QuadraticModels: SparseMatrixCOO


s = ArgParseSettings()

@add_arg_table s begin
    "--device"
    help = "device id to use for GPU computations"
    arg_type = Int
    default = 0
    "--continue"
    help = "continue from previous results"
    action = :store_true
    "--name"
    help = "custom name for the run"
    arg_type = String
    default = "miplib"
end

device = parse_args(s)["device"]
cont = parse_args(s)["continue"]
name = parse_args(s)["name"]

@info "Using device: $device, continue: $cont"
CUDA.device!(device)

if cont && isfile("results/results-$name.jld2")
    @info "Continuing from previous results."
    JLD2.@load "results/results-$name.jld2" results
else
    @info "Starting fresh, no previous results found."
    results = Dict{String,Any}()
end

max_wall_time = 900. # 15 minutes


function madipm(m; kwargs...)
    solver = MadIPM.MPCSolver(m; kwargs...)
    return MadIPM.solve!(solver)
end

gurobi_set_param(env, k, v::Float64) = GRBsetdblparam(env, k, v)
gurobi_set_param(env, k, v::Int) = GRBsetintparam(env, k, v)
gurobi_set_param(env, k, v::String) = GRBsetstrparam(env, k, v)
gurobi_set_param(env, k, v) = error("Unsupported parameter type: $(typeof(v))")

function gurobi(QM::QuadraticModel{T, S, M1, M2}; kwargs...) where {T, S, M1 <: SparseMatrixCOO, M2 <: SparseMatrixCOO}
    env = Gurobi.Env()
    
    for (k, v) in kwargs
        gurobi_set_param(env, k, v)
    end

    model = Ref{Ptr{Cvoid}}()
    GRBnewmodel(env, model, "", QM.meta.nvar, QM.data.c, QM.meta.lvar, QM.meta.uvar, C_NULL, C_NULL)
    GRBsetdblattr(model.x, "ObjCon", QM.data.c0)
    if QM.meta.nnzh > 0
        Hvals = zeros(eltype(QM.data.H.vals), length(QM.data.H.vals))
        for i=1:length(QM.data.H.vals)
            if QM.data.H.rows[i] == QM.data.H.cols[i]
                Hvals[i] = QM.data.H.vals[i] / 2
            else
                Hvals[i] = QM.data.H.vals[i]
            end
        end
        GRBaddqpterms(model.x, length(QM.data.H.cols), convert(Array{Cint,1}, QM.data.H.rows.-1),
                      convert(Array{Cint,1}, QM.data.H.cols.-1), Hvals)
    end

    Acsrrowptr, Acsrcolval, Acsrnzval = QuadraticModelsGurobi.sparse_csr(QM.data.A.rows,QM.data.A.cols,
                                                   QM.data.A.vals, QM.meta.ncon,
                                                   QM.meta.nvar)
    GRBaddrangeconstrs(model.x, QM.meta.ncon, length(Acsrcolval), convert(Array{Cint,1}, Acsrrowptr.-1),
                       convert(Array{Cint,1}, Acsrcolval.-1), Acsrnzval, QM.meta.lcon, QM.meta.ucon, C_NULL)

    GRBoptimize(model.x)

    x = zeros(QM.meta.nvar)
    GRBgetdblattrarray(model.x, "X", 0, QM.meta.nvar, x)
    y = zeros(QM.meta.ncon)
    GRBgetdblattrarray(model.x, "Pi", 0, QM.meta.ncon, y)
    s = zeros(QM.meta.nvar)
    GRBgetdblattrarray(model.x, "RC", 0, QM.meta.nvar, s)
    status = Ref{Cint}()
    GRBgetintattr(model.x, "Status", status)
    baritcnt = Ref{Cint}()
    GRBgetintattr(model.x, "BarIterCount", baritcnt)
    objval = Ref{Float64}()
    GRBgetdblattr(model.x, "ObjVal", objval)
    p_feas = Ref{Float64}()
    GRBgetdblattr(model.x, "ConstrResidual", p_feas)
    d_feas = Ref{Float64}()
    GRBgetdblattr(model.x, "DualResidual", d_feas)
    elapsed_time = Ref{Float64}()
    GRBgetdblattr(model.x, "Runtime", elapsed_time)
    stats = GenericExecutionStats(QM,
                                  status = get(QuadraticModelsGurobi.gurobi_statuses, status[], :unknown),
                                  solution = x,
                                  objective = objval[],
                                  iter = Int64(baritcnt[]),
                                  primal_feas = p_feas[],
                                  dual_feas = d_feas[],
                                  multipliers = y,
                                  elapsed_time = elapsed_time[])
    return stats
end

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

options = [
    :lowtol => 1e-8,
    :hightol => 1e-4,
]

src = joinpath(artifact"MIPLIB2010", "miplib2010")
probs = readdir(src)
nprobs = length(probs)

# precompile
case = probs[1]
qpdat = import_mps(joinpath(src, case))
qp_cpu = QuadraticModel(qpdat)
qp_gpu = convert(QuadraticModel{Float64, CuVector{Float64}}, qp_cpu)
madipm(
    qp_gpu;
    max_wall_time=max_wall_time,
    max_iter = 500,
    linear_solver= MadNLPGPU.CUDSSSolver,
    cudss_algorithm=MadNLP.LDL,
    regularization=MadIPM.FixedRegularization(1e-8, -1e-8),
    print_level=MadNLP.INFO,
    rethrow_error=true,
    tol = 1e-4,
    output_file = "results/madipm_$(case)_lowtol.log",
)

for (k, case) in enumerate(probs)
    for (optname, opt) in options

        if haskey(results, "$(case)_$(optname)")
            @info "Skipping $(case)_$(optname), already processed."
            continue
        else
            @info "$case -- $k / $nprobs"
        end

        qpdat = try
            import_mps(joinpath(src, case))
        catch e
            @warn "Failed to import $prob: $e"
            continue
        end
        @info "The problem $case was imported."

        qp = QuadraticModel(qpdat)
        presolved_qp, flag = MadIPM.presolve_qp(qp)
        !flag && continue  # problem already solved, unbounded or infeasible
        scaled_qp = scale_qp(presolved_qp)
        
        qp_cpu = MadIPM.standard_form_qp(scaled_qp)
        qp_gpu = convert(QuadraticModel{Float64, CuVector{Float64}}, qp_cpu)


        t_madipm = @elapsed begin
            sol_madipm = madipm(
                qp_gpu;
                max_wall_time=max_wall_time,
                max_iter = 500,
                linear_solver= MadNLPGPU.CUDSSSolver,
                cudss_algorithm=MadNLP.LDL,
                regularization=MadIPM.FixedRegularization(1e-8, -1e-8),
                print_level=MadNLP.INFO,
                rethrow_error=true,
                tol = opt,
                output_file = "results/madipm_$(case)_$(optname).log",
            )
        end
        # log_madipm = read("madipm_$(case)_$(optname).log", String)

        
        t_gurobi = @elapsed begin
            sol_gurobi = gurobi(
                qp_cpu;
                TimeLimit=max_wall_time,
                Method=2,  # Barrier method
                Presolve=0,
                FeasibilityTol= opt,
                OptimalityTol= opt,
                Threads = 16,
                Crossover = 0,
                LogFile = "results/gurobi_$(case)_$(optname).log",
            )
        end
        # log_gurobi = read("gurobi_$(case)_$(optname).log", String)

        results["$(case)_$(optname)"] = (
            ;
            t_madipm, t_gurobi,
            sol_madipm = (
                status = sol_madipm.status,
            ),
            sol_gurobi = (
                status = sol_gurobi.status,
            ),
            # log_madipm, log_gurobi,
            meta = (
                nvar = qp_cpu.meta.nvar,
                ncon = qp_cpu.meta.ncon,
                nnzh = qp_cpu.meta.nnzh,
                nnzj = qp_cpu.meta.nnzj
            ),
            tol = opt,
        )

        # resave results to JLD2 file
        JLD2.@save "results/results-$name.jld2" results        
    end
end




