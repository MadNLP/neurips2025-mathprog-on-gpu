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
probs = ["30_70_45_095_100.mps.gz", "30n20b8.mps.gz", "air04.mps.gz", "app1-2.mps.gz", "ash608gpia-3col.mps.gz", "atlanta-ip.mps.gz", "atm20-100.mps.gz", "bab1.mps.gz", "bab3.mps.gz", "bab5.mps.gz", "biella1.mps.gz", "bley_xl1.mps.gz", "blp-ar98.mps.gz", "blp-ic97.mps.gz", "buildingenergy.mps.gz", "circ10-3.mps.gz", "co-100.mps.gz", "core2536-691.mps.gz", "core4872-1529.mps.gz", "d10200.mps.gz", "d20200.mps.gz", "dano3mip.mps.gz", "datt256.mps.gz", "dc1c.mps.gz", "dc1l.mps.gz", "dolom1.mps.gz", "eil33-2.mps.gz", "eilA101-2.mps.gz", "ex10.mps.gz", "ex1010-pi.mps.gz", "ex9.mps.gz", "ger50_17_trans.mps.gz", "germanrr.mps.gz", "gmut-75-50.mps.gz", "hanoi5.mps.gz", "iis-bupa-cov.mps.gz", "iis-pima-cov.mps.gz", "ivu06-big.mps.gz", "lectsched-1-obj.mps.gz", "lectsched-1.mps.gz", "lectsched-2.mps.gz", "lectsched-3.mps.gz", "lectsched-4-obj.mps.gz", "leo1.mps.gz", "leo2.mps.gz", "lrsa120.mps.gz", "map06.mps.gz", "map10.mps.gz", "map14.mps.gz", "map18.mps.gz", "map20.mps.gz", "methanosarcina.mps.gz", "momentum1.mps.gz", "momentum2.mps.gz", "momentum3.mps.gz", "mspp16.mps.gz", "mzzv11.mps.gz", "n15-3.mps.gz", "n3-3.mps.gz", "n3seq24.mps.gz", "neos-1109824.mps.gz", "neos-1140050.mps.gz", "neos-1171737.mps.gz", "neos-1429212.mps.gz", "neos-1605061.mps.gz", "neos-1605075.mps.gz", "neos-476283.mps.gz", "neos-506422.mps.gz", "neos-506428.mps.gz", "neos-520729.mps.gz", "neos-631710.mps.gz", "neos-693347.mps.gz", "neos-738098.mps.gz", "neos-777800.mps.gz", "neos-824661.mps.gz", "neos-826694.mps.gz", "neos-826812.mps.gz", "neos-859770.mps.gz", "neos-885086.mps.gz", "neos-885524.mps.gz", "neos-916792.mps.gz", "neos-932816.mps.gz", "neos-933638.mps.gz", "neos-933966.mps.gz", "neos-934278.mps.gz", "neos-935627.mps.gz", "neos-935769.mps.gz", "neos-937511.mps.gz", "neos-937815.mps.gz", "neos-941262.mps.gz", "neos-941313.mps.gz", "neos-948126.mps.gz", "neos-952987.mps.gz", "neos-957389.mps.gz", "neos-984165.mps.gz", "neos13.mps.gz", "neos6.mps.gz", "neos808444.mps.gz", "net12.mps.gz", "netdiversion.mps.gz", "npmv07.mps.gz", "ns1111636.mps.gz", "ns1116954.mps.gz", "ns1456591.mps.gz", "ns1685374.mps.gz", "ns1758913.mps.gz", "ns1778858.mps.gz", "ns1853823.mps.gz", "ns1854840.mps.gz", "ns1904248.mps.gz", "ns1905797.mps.gz", "ns1952667.mps.gz", "ns2118727.mps.gz", "ns2124243.mps.gz", "ns2137859.mps.gz", "ns930473.mps.gz", "nsr8k.mps.gz", "opm2-z10-s2.mps.gz", "opm2-z11-s8.mps.gz", "opm2-z12-s14.mps.gz", "opm2-z12-s7.mps.gz", "opm2-z7-s2.mps.gz", "pb-simp-nonunif.mps.gz", "queens-30.mps.gz", "rail01.mps.gz", "rail02.mps.gz", "rail03.mps.gz", "rail507.mps.gz", "ramos3.mps.gz", "reblock166.mps.gz", "reblock420.mps.gz", "rmatr200-p10.mps.gz", "rmatr200-p20.mps.gz", "rmatr200-p5.mps.gz", "rmine10.mps.gz", "rmine14.mps.gz", "rmine21.mps.gz", "rocII-4-11.mps.gz", "rocII-7-11.mps.gz", "rocII-9-11.mps.gz", "rococoC12-111000.mps.gz", "rvb-sub.mps.gz", "satellites1-25.mps.gz", "satellites2-60-fs.mps.gz", "satellites2-60.mps.gz", "satellites3-40-fs.mps.gz", "satellites3-40.mps.gz", "sct1.mps.gz", "sct32.mps.gz", "sct5.mps.gz", "shipsched.mps.gz", "sing2.mps.gz", "sing245.mps.gz", "sp97ar.mps.gz", "sp98ic.mps.gz", "sp98ir.mps.gz", "stockholm.mps.gz", "stp3d.mps.gz", "sts405.mps.gz", "sts729.mps.gz", "t1717.mps.gz", "t1722.mps.gz", "tanglegram1.mps.gz", "triptim1.mps.gz", "triptim2.mps.gz", "triptim3.mps.gz", "uc-case11.mps.gz", "uc-case3.mps.gz", "unitcal_7.mps.gz", "van.mps.gz", "vpphard.mps.gz", "vpphard2.mps.gz", "wachplan.mps.gz", "wnq-n100-mw99-14.mps.gz"]
nprobs = length(probs)

# precompile
case = probs[1]
qpdat = import_mps(joinpath(src, case))
qp_cpu = QuadraticModel(qpdat)
qp_gpu = convert(QuadraticModel{Float64, CuVector{Float64}}, qp_cpu)

# to force compilation
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




