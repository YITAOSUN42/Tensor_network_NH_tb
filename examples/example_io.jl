using Dates, Pkg, SHA, LinearAlgebra

function new_output(name)
    parent = joinpath(@__DIR__, "..", "outputs")
    mkpath(parent)
    return mktempdir(parent; prefix=name * "_", cleanup=false)
end

function write_rows(path, rows)
    isempty(rows) && throw(ArgumentError("no rows to write"))
    open(path, "w") do io
        println(io, join(string.(keys(first(rows))), ','))
        for row in rows
            keys(row) == keys(first(rows)) || error("inconsistent CSV columns")
            println(io, join(values(row), ','))
        end
    end
end

function spectral_row(omega, site0, result)
    rho = result.physical_density
    return (re=real(omega), im=imag(omega), site0=site0,
        rho_re=real(rho), rho_im=imag(rho), rho_abs=abs(rho),
        scaled_re=real(result.scaled_density), scaled_im=imag(result.scaled_density))
end

function write_run_info(output; kwargs...)
    root = normpath(joinpath(@__DIR__, ".."))
    dependencies = Pkg.dependencies()
    open(joinpath(output, "run_info.txt"), "w") do io
        println(io, "Timestamp: ", now())
        println(io, "Julia: ", VERSION)
        println(io, "Active project: ", Base.active_project())
        println(io, "Julia threads: ", Threads.nthreads())
        println(io, "BLAS threads: ", BLAS.get_num_threads())
        for (key, value) in kwargs
            println(io, key, " = ", value)
        end
        for name in ("ITensors", "ITensorMPS", "Quantics", "QuanticsTCI",
                     "TensorCrossInterpolation", "TCIITensorConversion")
            for info in values(dependencies)
                info.name == name && println(io, name, ": ", info.version)
            end
        end
        for dir in (root, joinpath(root, "src"), joinpath(root, "examples"))
            for file in sort(readdir(dir; join=true))
                isfile(file) && endswith(file, ".jl") || continue
                println(io, "sha256 ", relpath(file, root), " ", bytes2hex(sha256(read(file))))
            end
        end
    end
end
