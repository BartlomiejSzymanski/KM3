
module Optimizers

using ..AutoDiff
using Statistics # For sqrt

export Adam, update!

mutable struct Adam{T<:Real}
    lr::T
    beta1::T
    beta2::T
    epsilon::T
    params::Vector{<:AutoDiff.Variable{T}}
    m::Dict{AutoDiff.Variable{T}, Union{T, AbstractArray{T}}}
    v::Dict{AutoDiff.Variable{T}, Union{T, AbstractArray{T}}}
    t::Int
    function Adam(lr::T, params::Vector{<:AutoDiff.Variable}; beta1::Real=0.9, beta2::Real=0.999, epsilon::Real=1e-8) where {T<:Real}
        trainable_params = Vector{AutoDiff.Variable{T}}()
        for p in params; if p.is_param && p isa AutoDiff.Variable{T}; push!(trainable_params, p); end; end
        if isempty(trainable_params); @warn "Adam initialized with no trainable parameters of type $T."; end
        m_dict = Dict{AutoDiff.Variable{T}, Union{T, AbstractArray{T}}}()
        v_dict = Dict{AutoDiff.Variable{T}, Union{T, AbstractArray{T}}}()
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


end # module Optimizers