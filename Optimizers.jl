module Optimizers

using ..SimpleAutoDiff
using Statistics

export Adam, update!, RMSProp, clip_gradients!

mutable struct Adam{T<:Real}
    lr::T
    beta1::T
    beta2::T
    epsilon::T
    params::Vector{<:SimpleAutoDiff.Variable{T}}
    m::Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}
    v::Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}
    t::Int
    function Adam(lr::T, params::Vector{<:SimpleAutoDiff.Variable}; beta1::Real=0.9, beta2::Real=0.999, epsilon::Real=1e-8) where {T<:Real}
        trainable_params = Vector{SimpleAutoDiff.Variable{T}}()
        for p in params; if p.is_param && p isa SimpleAutoDiff.Variable{T}; push!(trainable_params, p); end; end
        if isempty(trainable_params); @warn "Adam initialized with no trainable parameters of type $T."; end
        m_dict = Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}()
        v_dict = Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}()
        for p in trainable_params; m_dict[p] = zero(p.value); v_dict[p] = zero(p.value); end
        new{T}(T(lr), T(beta1), T(beta2), T(epsilon), trainable_params, m_dict, v_dict, 0)
    end
end

function update!(opt::Adam{T}) where {T<:Real}
    opt.t += 1
    bias_correction1 = one(T) - opt.beta1^opt.t + T(1e-9)
    bias_correction2 = one(T) - opt.beta2^opt.t + T(1e-9)
    for p in opt.params
        if p.gradient !== nothing
            g = grad(p)
            is_non_finite = (isa(g, Real) && !isfinite(g)) ||
                            (isa(g, AbstractArray) && !all(isfinite, g))
            if is_non_finite
                 @warn "Adam: Non-finite gradient detected for parameter shape $(size(p.value)). Skipping update."
                 continue
            end
            opt.m[p] .= opt.beta1 .* opt.m[p] .+ (one(T) - opt.beta1) .* g
            opt.v[p] .= opt.beta2 .* opt.v[p] .+ (one(T) - opt.beta2) .* (g .^ 2)
            m_hat = opt.m[p] ./ bias_correction1
            v_hat = opt.v[p] ./ bias_correction2
            update_step = opt.lr .* m_hat ./ (sqrt.(v_hat) .+ opt.epsilon)
            p.value .-= update_step
        end
    end
end

mutable struct RMSProp{T<:Real}
    lr::T
    rho::T
    epsilon::T
    params::Vector{<:SimpleAutoDiff.Variable{T}}
    accumulators::Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}
    function RMSProp(lr::T, params::Vector{<:SimpleAutoDiff.Variable}; rho::Real=0.9, epsilon::Real=1e-6) where {T<:Real}
        trainable_params = Vector{SimpleAutoDiff.Variable{T}}()
        for p in params; if p.is_param && p isa SimpleAutoDiff.Variable{T}; push!(trainable_params, p); end; end
        if isempty(trainable_params); @warn "RMSProp initialized with no trainable parameters of type $T."; end
        acc_dict = Dict{SimpleAutoDiff.Variable{T}, Union{T, AbstractArray{T}}}()
        for p in trainable_params; acc_dict[p] = zero(p.value); end
        new{T}(T(lr), T(rho), T(epsilon), trainable_params, acc_dict)
    end
end

function update!(opt::RMSProp{T}) where {T<:Real}
    for p in opt.params
        if haskey(opt.accumulators, p) && p.gradient !== nothing
            g = grad(p)
            is_non_finite = (isa(g, Real) && !isfinite(g)) ||
                            (isa(g, AbstractArray) && !all(isfinite, g))
            if is_non_finite
                 @warn "RMSProp: Non-finite gradient detected for parameter shape $(size(p.value)). Skipping update."
                 continue
            end
            acc = opt.accumulators[p]
            acc .= opt.rho .* acc .+ (one(T) - opt.rho) .* (g .^ 2)
            update_step = (opt.lr ./ (sqrt.(acc) .+ opt.epsilon)) .* g
            p.value .-= update_step
        end
    end
end

function clip_gradients!(params::Vector{<:Variable}, threshold::Real)
    if threshold <= 0; error("Gradient clipping threshold must be positive."); end
    global_norm_sq::Float64 = 0.0
    for p in params
        if p.is_param && p.gradient !== nothing
            grad_val = p.gradient
            global_norm_sq += sum(abs2, grad_val)
        end
    end
    global_norm = sqrt(global_norm_sq)
    clip_coef = Float32(threshold) / (Float32(global_norm) + eps(Float32))
    if clip_coef < 1.0
        for p in params
            if p.is_param && p.gradient !== nothing
                p.gradient .*= clip_coef
            end
        end
    end
    return global_norm
end

end
