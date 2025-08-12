using CUDA, MadNLPGPU, NLPModelsIpopt, JLD2, Artifacts, Logging, HSL_jll, ArgParse

s = ArgParseSettings()

@add_arg_table s begin
    "--device"
    help = "device id to use for GPU computations"
    arg_type = Int
    default = 0
    "--continue"
    help = "continue from previous results"
    action = :store_true
    "class"
    help = "class of problems to be benchmarked"
    arg_type = String
    required = true
end

device = parse_args(s)["device"]
class = parse_args(s)["class"]
cont = parse_args(s)["continue"]

@info "Using device: $device for class: $class, continue: $cont"
CUDA.device!(device)

if cont && isfile("results-$class.jld2")
    @info "Continuing from previous results."
    JLD2.@load "results-$class.jld2" results
else
    @info "Starting fresh, no previous results found."
    results = Dict{String,Any}()
end

if parse_args(s)["class"] == "opf"
    using ExaModelsPower
    cases = [
        (case, (;backend = nothing) -> opf_model(case;form = :polar,T = Float64,backend)[1])
        for case in filter(
            x -> endswith(x, ".m"),
            readdir(joinpath(artifact"PGLib_opf", "pglib-opf-23.07"))
            )]
elseif parse_args(s)["class"] == "cops"
    using ExaModelsExamples
    cases = [
        # Mittelmann instances
        ("bearing_model-(400, 400)", (;backend = nothing)->ExaModelsExamples.bearing_model(400, 400;backend)),
        ("camshape_model-(6400)", (;backend = nothing)->ExaModelsExamples.camshape_model(6400;backend)),
        ("elec_model-(400)", (;backend = nothing)->ExaModelsExamples.elec_model(400;backend)),
        ("gasoil_model-(3200)", (;backend = nothing)->ExaModelsExamples.gasoil_model(3200;backend)),
        ("marine_model-(1600)", (;backend = nothing)->ExaModelsExamples.marine_model(1600;backend)),
        ("pinene_model-(3200)", (;backend = nothing)->ExaModelsExamples.pinene_model(3200;backend)),
        ("robot_model-(1600)", (;backend = nothing)->ExaModelsExamples.robot_model(1600;backend)),
        ("rocket_model-(12800)", (;backend = nothing)->ExaModelsExamples.rocket_model(12800;backend)),
        ("steering_model-(12800)", (;backend = nothing)->ExaModelsExamples.steering_model(12800;backend)),
        # Large-scale instances
        ("bearing_model-(800, 800)", (;backend = nothing)->ExaModelsExamples.bearing_model(800, 800;backend)),
        ("camshape_model-(12800)", (;backend = nothing)->ExaModelsExamples.camshape_model(12800;backend)),
        ("elec_model-(800)", (;backend = nothing)->ExaModelsExamples.elec_model(800;backend)),
        ("gasoil_model-(12800)", (;backend = nothing)->ExaModelsExamples.gasoil_model(12800;backend)),
        ("marine_model-(12800)", (;backend = nothing)->ExaModelsExamples.marine_model(12800;backend)),
        ("pinene_model-(12800)", (;backend = nothing)->ExaModelsExamples.pinene_model(12800;backend)),
        ("robot_model-(12800)", (;backend = nothing)->ExaModelsExamples.robot_model(12800;backend)),
        ("rocket_model-(51200)", (;backend = nothing)->ExaModelsExamples.rocket_model(51200;backend)),
        ("steering_model-(51200)", (;backend = nothing)->ExaModelsExamples.steering_model(51200;backend)),
    ]
else
    error("Unknown class: $(parse_args(s)["class"])")
end

options = [
    :hightol => (
        ipopt_opt = (
            ;
            tol = 1e-4,
            bound_relax_factor = 1e-4,
            max_wall_time = 900.,
            linear_solver="ma27",
            dual_inf_tol = 10000.0,
            constr_viol_tol = 10000.0,
            compl_inf_tol = 10000.0,
            honor_original_bounds = "no",
            print_timing_statistics = "yes",
        ),
        madnlp_opt = (
            ;
            tol = 1e-4,
            max_wall_time = 900.,
        ),
    )
    :lowtol => (
        ipopt_opt = (
            ;
            tol = 1e-8,
            bound_relax_factor = 1e-8,
            max_wall_time = 900.,
            linear_solver="ma27",
            dual_inf_tol = 10000.0,
            constr_viol_tol = 10000.0,
            compl_inf_tol = 10000.0,
            honor_original_bounds = "no",
            print_timing_statistics = "yes",
        ),
        madnlp_opt = (
            ;
            tol = 1e-8,
            max_wall_time = 900.,
        ),
    )
]

for (case, model) in cases
    for (optname, (ipopt_opt, madnlp_opt)) in options
        
        if haskey(results, "$(case)_$(optname)")
            @info "Skipping $(case)_$(optname), already processed."
            continue
        end
        @info "Processing: $(case)_$(optname)"
        
        @info "Solving with madnlp"
        m_gpu = model(backend = CUDABackend())
        madnlp(m_gpu; output_file = "madnlp_$(case)_$(optname).log", madnlp_opt...)
        t_madnlp = @elapsed begin
            sol_madnlp = madnlp(m_gpu; output_file = "madnlp_$(case)_$(optname).log", madnlp_opt...)
        end

        @info "Solving with ipopt"
        m_cpu = model()
        ipopt(m_cpu; output_file = "ipopt_$(case)_$(optname).log", ipopt_opt...)
        t_ipopt  = @elapsed begin
            sol_ipopt = ipopt(m_cpu; output_file = "ipopt_$(case)_$(optname).log", ipopt_opt...)
        end

        results["$(case)_$(optname)"] = (
            ;
            t_madnlp, t_ipopt, sol_madnlp, sol_ipopt,
            meta = m_cpu.meta,
            log_madnlp = read("madnlp_$(case)_$(optname).log", String),
            log_ipopt = read("ipopt_$(case)_$(optname).log", String),
        )

        # resave results to JLD2 file
        JLD2.@save "results-$(class).jld2" results
    end
end

