#!/usr/bin/env julia

using LinearAlgebra
using Random
using SparseArrays
using Statistics
using Printf

const LAPLACIANS_SRC = normpath(joinpath(@__DIR__, "..", "..", "Laplacians.jl", "src"))

module LaplaciansMini
using SparseArrays
using Random
using LinearAlgebra
using Statistics

# approxChol.jl assigns `approxchol_sddm = sddmWrapLap(approxchol_lap)` at top-level.
# We don't use that wrapper in this harness, so a no-op stub is sufficient.
sddmWrapLap(f) = f

include(joinpath(Main.LAPLACIANS_SRC, "approxCholTypes.jl"))
include(joinpath(Main.LAPLACIANS_SRC, "graphUtils.jl"))
include(joinpath(Main.LAPLACIANS_SRC, "approxChol.jl"))
end

struct Options
    out::String
    sizes::Vector{Tuple{Int, Int}}
    seeds::Vector{Int}
    variants::Vector{Symbol}
    factor_warmup::Int
    factor_repeats::Int
    apply_repeats::Int
    tol::Float64
    max_iters::Int
    split::Int
    merge::Int
end

function default_options()
    return Options(
        "target/native_compare/julia.csv",
        [(32, 32), (64, 64), (96, 96)],
        [0, 1, 2, 3, 4],
        [:ac, :ac2],
        1,
        3,
        10,
        1e-6,
        2000,
        2,
        2,
    )
end

function parse_sizes(s::String)
    out = Tuple{Int, Int}[]
    for tok in split(s, ',')
        t = strip(tok)
        isempty(t) && continue
        parts = split(t, 'x')
        length(parts) == 2 || error("invalid size token '$t', expected RxC")
        push!(out, (parse(Int, parts[1]), parse(Int, parts[2])))
    end
    return out
end

function parse_int_list(s::String)
    out = Int[]
    for tok in split(s, ',')
        t = strip(tok)
        isempty(t) && continue
        push!(out, parse(Int, t))
    end
    return out
end

function parse_variants(s::String)
    out = Symbol[]
    for tok in split(s, ',')
        t = strip(tok)
        isempty(t) && continue
        if t == "ac"
            push!(out, :ac)
        elseif t == "ac2"
            push!(out, :ac2)
        else
            error("unknown variant '$t'")
        end
    end
    return out
end

function parse_args(args::Vector{String})
    opts = default_options()

    out = opts.out
    sizes = opts.sizes
    seeds = opts.seeds
    variants = opts.variants
    factor_warmup = opts.factor_warmup
    factor_repeats = opts.factor_repeats
    apply_repeats = opts.apply_repeats
    tol = opts.tol
    max_iters = opts.max_iters
    split_k = opts.split
    merge_k = opts.merge

    i = 1
    while i <= length(args)
        key = args[i]
        i < length(args) || error("missing value for $key")
        val = args[i + 1]
        if key == "--out"
            out = val
        elseif key == "--sizes"
            sizes = parse_sizes(val)
        elseif key == "--seeds"
            seeds = parse_int_list(val)
        elseif key == "--variants"
            variants = parse_variants(val)
        elseif key == "--factor-warmup"
            factor_warmup = parse(Int, val)
        elseif key == "--factor-repeats"
            factor_repeats = parse(Int, val)
        elseif key == "--apply-repeats"
            apply_repeats = parse(Int, val)
        elseif key == "--tol"
            tol = parse(Float64, val)
        elseif key == "--max-iters"
            max_iters = parse(Int, val)
        elseif key == "--split"
            split_k = parse(Int, val)
        elseif key == "--merge"
            merge_k = parse(Int, val)
        else
            error("unknown argument: $key")
        end
        i += 2
    end

    isempty(sizes) && error("sizes list cannot be empty")
    isempty(seeds) && error("seeds list cannot be empty")
    isempty(variants) && error("variants list cannot be empty")
    factor_repeats >= 1 || error("factor-repeats must be >= 1")
    apply_repeats >= 1 || error("apply-repeats must be >= 1")

    return Options(
        out,
        sizes,
        seeds,
        variants,
        factor_warmup,
        factor_repeats,
        apply_repeats,
        tol,
        max_iters,
        split_k,
        merge_k,
    )
end

function make_grid_adjacency(rows::Int, cols::Int)
    n = rows * cols
    I = Int[]
    J = Int[]
    V = Float64[]

    idx(r, c) = (r - 1) * cols + c

    for r in 1:rows, c in 1:cols
        u = idx(r, c)
        if r < rows
            v = idx(r + 1, c)
            push!(I, u); push!(J, v); push!(V, 1.0)
            push!(I, v); push!(J, u); push!(V, 1.0)
        end
        if c < cols
            v = idx(r, c + 1)
            push!(I, u); push!(J, v); push!(V, 1.0)
            push!(I, v); push!(J, u); push!(V, 1.0)
        end
    end

    return sparse(I, J, V, n, n)
end

function lap_from_adjacency(adj::SparseMatrixCSC{Float64, Int})
    d = vec(sum(adj, dims=1))
    return spdiagm(0 => d) - adj
end

function make_rhs(n::Int)
    b = zeros(Float64, n)
    if n >= 2
        b[1] = 1.0
        b[end] = -1.0
    end
    return b
end

function project_zero_mean!(x::Vector{Float64})
    isempty(x) && return
    μ = sum(x) / length(x)
    @inbounds @simd for i in eachindex(x)
        x[i] -= μ
    end
end

function factorize_ldli(adj::SparseMatrixCSC{Float64, Int}, variant::Symbol, seed::Int, split_k::Int, merge_k::Int)
    Random.seed!(seed)
    if variant == :ac
        llmat = LaplaciansMini.LLmatp(adj)
        return LaplaciansMini.approxChol(llmat)
    elseif variant == :ac2
        llmat = LaplaciansMini.LLmatp(adj, split_k)
        return LaplaciansMini.approxChol(llmat, split_k, merge_k)
    else
        error("unsupported variant $variant")
    end
end

function apply_preconditioner!(ldli, rhs::Vector{Float64}, tmp::Vector{Float64}, out::Vector{Float64})
    copyto!(tmp, rhs)
    LaplaciansMini.forward!(ldli, tmp)
    @inbounds for i in eachindex(ldli.d)
        di = ldli.d[i]
        if di != 0.0
            tmp[i] /= di
        end
    end
    LaplaciansMini.backward!(ldli, tmp)

    μ = sum(tmp) / length(tmp)
    @inbounds @simd for i in eachindex(tmp)
        out[i] = tmp[i] - μ
    end
    return out
end

function pcg_with_preconditioner(lap::SparseMatrixCSC{Float64, Int}, b::Vector{Float64}, ldli; tol::Float64, max_iters::Int)
    n = length(b)
    x = zeros(Float64, n)
    r = copy(b)
    project_zero_mean!(r)

    norm_b = max(norm(b), 1e-30)

    z = zeros(Float64, n)
    tmp = zeros(Float64, n)
    apply_preconditioner!(ldli, r, tmp, z)
    project_zero_mean!(z)

    p = copy(z)
    Ap = zeros(Float64, n)

    rz_old = dot(r, z)
    rel_res = norm(r) / norm_b
    if rel_res <= tol
        return (0, rel_res)
    end

    for iter in 1:max_iters
        mul!(Ap, lap, p)
        project_zero_mean!(Ap)

        denom = dot(p, Ap)
        if abs(denom) <= 1e-30
            break
        end

        α = rz_old / denom
        @inbounds @simd for i in eachindex(x)
            x[i] += α * p[i]
            r[i] -= α * Ap[i]
        end
        project_zero_mean!(x)
        project_zero_mean!(r)

        rel_res = norm(r) / norm_b
        if rel_res <= tol
            return (iter, rel_res)
        end

        apply_preconditioner!(ldli, r, tmp, z)
        project_zero_mean!(z)

        rz_new = dot(r, z)
        if abs(rz_old) <= 1e-30
            break
        end

        β = rz_new / rz_old
        @inbounds @simd for i in eachindex(p)
            p[i] = z[i] + β * p[i]
        end
        project_zero_mean!(p)
        rz_old = rz_new
    end

    return (max_iters, rel_res)
end

function run_case(rows::Int, cols::Int, variant::Symbol, seed::Int, opts::Options)
    adj = make_grid_adjacency(rows, cols)
    lap = lap_from_adjacency(adj)
    n = size(adj, 1)
    rhs = make_rhs(n)

    for w in 0:opts.factor_warmup-1
        factorize_ldli(adj, variant, seed + 100_000 + w, opts.split, opts.merge)
    end

    factor_times = Float64[]
    for r in 0:opts.factor_repeats-1
        t0 = time_ns()
        factorize_ldli(adj, variant, seed + r, opts.split, opts.merge)
        push!(factor_times, (time_ns() - t0) / 1e6)
    end

    ldli = factorize_ldli(adj, variant, seed, opts.split, opts.merge)

    tmp = zeros(Float64, n)
    out = zeros(Float64, n)
    apply_times = Float64[]
    for _ in 1:opts.apply_repeats
        t0 = time_ns()
        apply_preconditioner!(ldli, rhs, tmp, out)
        push!(apply_times, (time_ns() - t0) / 1e6)
    end

    residual = rhs - lap * out
    precond_rel_res = norm(residual) / max(norm(rhs), 1e-30)

    t0 = time_ns()
    pcg_iters, pcg_rel_res = pcg_with_preconditioner(lap, rhs, ldli; tol=opts.tol, max_iters=opts.max_iters)
    pcg_ms = (time_ns() - t0) / 1e6

    return (
        factor_ms = median(factor_times),
        apply_ms = median(apply_times),
        pcg_ms = pcg_ms,
        pcg_iters = pcg_iters,
        pcg_rel_res = pcg_rel_res,
        precond_rel_res = precond_rel_res,
        n = n,
    )
end

function main()
    opts = parse_args(ARGS)
    mkpath(dirname(opts.out))

    open(opts.out, "w") do io
        println(io, "impl,variant,rows,cols,n,seed,factor_ms,apply_ms,pcg_ms,pcg_iters,pcg_rel_res,precond_rel_res")
        for (rows, cols) in opts.sizes
            for variant in opts.variants
                for seed in opts.seeds
                    m = run_case(rows, cols, variant, seed, opts)
                    @printf(
                        io,
                        "julia,%s,%d,%d,%d,%d,%.6f,%.6f,%.6f,%d,%.6e,%.6e\n",
                        String(variant),
                        rows,
                        cols,
                        m.n,
                        seed,
                        m.factor_ms,
                        m.apply_ms,
                        m.pcg_ms,
                        m.pcg_iters,
                        m.pcg_rel_res,
                        m.precond_rel_res,
                    )
                end
            end
        end
    end
end

main()
