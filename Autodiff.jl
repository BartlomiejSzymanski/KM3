
module AutoDiff

using Statistics, Random, LinearAlgebra # Add Statistics

export Variable, value, grad, backward!, zero_grad!, parameters, matmul, accumulate_gradient!, mean, tanh # Add mean and tanh

mutable struct Variable{T<:Real}
    value::Union{T, AbstractArray{T}}
    gradient::Union{Nothing, T, AbstractArray{T}}
    children::Vector{<:Variable} # Field type: Vector of any kind of Variable{S}
    backward_fn::Function
    is_param::Bool

    function Variable(val::Union{S, AbstractArray{S}}; is_param::Bool=false) where {S<:Real}
        empty_children = Vector{Variable{S}}() # Or Vector{<:Variable}()
        new{S}(val, nothing, empty_children, () -> nothing, is_param)
    end

    function Variable(val::Union{S, AbstractArray{S}}, children_input::Vector{<:Variable}, bwd_fn::Function) where {S<:Real}
        new{S}(val, nothing, children_input, bwd_fn, false)
    end
end

value(v::Variable) = v.value
value(x::Real) = x # Allow non-Variables in ops

grad(v::Variable) = v.gradient # Get the gradient value

_eltype(v::Variable{T}) where T = T # Get inner type T
_eltype(v::AbstractArray{T}) where T = T
_eltype(v::T) where T<:Real = T

function grad!(v::Variable{T}, g) where T
    g_converted = try # Convert safely
        if isa(g, AbstractArray)
             if eltype(g) == T; g; else convert(AbstractArray{T}, g); end
        elseif isa(g, Real)
             convert(T, g)
        else
             g # Pass through if not Real/AbstractArray
        end
    catch e
         @error "grad! failed to convert gradient" typeof_v=T typeof_g=typeof(g) exception=(e, catch_backtrace())
         return # Don't assign if conversion fails
    end

    if v.gradient === nothing
        if isa(v.value, AbstractArray) && isa(g_converted, Real)
            v.gradient = fill(g_converted, size(value(v)))
        else
            v.gradient = deepcopy(g_converted)
             if isa(v.gradient, AbstractArray) && size(v.gradient) != size(v.value) && !(isa(v.value, Real) && isa(g_converted,Real))
                 @warn "grad! shape mismatch after deepcopy init" size_v=size(v.value) size_g=size(g_converted) size_grad=size(v.gradient)
             end
        end
    else
        accumulate_gradient!(v, g_converted) # Pass converted grad
    end
end

function parameters(v::Variable)
    params = Set{Variable}()
    visited = Set{Variable}()
    nodes_to_visit = [v]
    while !isempty(nodes_to_visit)
        current = pop!(nodes_to_visit)
        if !(current in visited)
            push!(visited, current)
            if current.is_param; push!(params, current); end
            for child in current.children; push!(nodes_to_visit, child); end
        end
    end
    return collect(params) # Return as Vector{Variable}
end

Base.show(io::IO, v::Variable) = print(io, "Variable{$(eltype(v.value))}(grad=$(v.gradient !== nothing))")


"""
    accumulate_gradient!(v::Variable{T}, g) where {T<:Real}

Accumulates a (finite) gradient `g` into `v.gradient`, initializing if needed.
Skips entirely on NaN/Inf inputs.
"""
function accumulate_gradient!(v::Variable{T}, g) where {T<:Real}
    # 1) Reject non-finite gradients immediately
    if isnan_or_inf(g)
        warn_nan_inf!(v, g)
        return
    end

    # 2) Convert incoming gradient to the right type/shape
    g2 = try
        convert_gradient(g, T, size(v.value))
    catch e
        @error "Gradient conversion failed" exception=(e, catch_backtrace())
        return
    end

    # 3) Optional logging for large parameters
    log_large_param!(v, g2, stage=:converted)

    # 4) Initialize or accumulate
    if v.gradient === nothing
        init_gradient!(v, g2)
        log_large_param!(v, g2, stage=:initialized)
    else
        try
            accumulate_into!(v, g2)
            log_large_param!(v, g2, stage=:accumulated)
        catch e
            @warn "Gradient accumulation failed" exception=(e, catch_backtrace())
        end
    end
end

# — Helpers — #

# Check for NaN/Inf in scalars or arrays
isnan_or_inf(g) = g === nothing ||
                  (isa(g, Real) && !isfinite(g)) ||
                  (isa(g, AbstractArray) && !all(isfinite, g))

function warn_nan_inf!(v, g)
    @warn "Skipping NaN/Inf gradient" var_type=eltype(v.value) grad_type=typeof(g)
    if v.gradient === nothing
        v.gradient = zero(v.value)
        @warn "Initialized gradient to zero because first incoming was NaN/Inf."
    end
end

# Convert g into Array{T} or scalar T matching v
function convert_gradient(g, ::Type{T}, target_size) where {T}
    if g === nothing
        error("Unexpected `nothing` gradient after NaN/Inf check")
    elseif isa(g, AbstractArray)
        # If it's already the right element type, trust it; else convert
        return eltype(g) === T ? g : convert(Array{T}, g)
    else
        # Scalar case
        return convert(T, g)
    end
end

# Initialize v.gradient from the first gradient g2
function init_gradient!(v::Variable{T}, g2) where {T}
    if isa(v.value, AbstractArray)
        v.gradient = isa(g2, Real) ? fill(g2, size(v.value)) : deepcopy(g2)
    else
        v.gradient = isa(g2, Real) ? g2 : sum(g2)
    end
end

# Do the in-place accumulation
function accumulate_into!(v::Variable{T}, g2) where {T}
    if isa(v.gradient, AbstractArray) && isa(g2, AbstractArray) && size(v.gradient)==size(g2)
        @. v.gradient += g2
    elseif isa(v.gradient, AbstractArray) && isa(g2, Real)
        @. v.gradient += g2
    elseif isa(v.gradient, Real) && isa(g2, AbstractArray)
        v.gradient += sum(g2)
    else
        # fallback broadcast
        v.gradient .+= g2
    end
end

# Log occasional diagnostics for very large parameters
function log_large_param!(v, g2; stage::Symbol)
    if v.is_param && isa(v.value, AbstractMatrix) && prod(size(v.value)) > 10_000 && rand() < 0.05
        norm_val = isa(g2, AbstractArray) ? norm(g2) : abs(g2)
        println(" [dbg:$stage] size=$(size(v.value)), grad_norm=$(round(norm_val, sigdigits=4))")
    end
end


function backward!(v::Variable{T}) where {T<:Real}
    if v.gradient === nothing
        if isa(v.value, Real) || length(v.value) == 1
             grad!(v, one(T))
        else
             error("Backward! started on non-scalar, multi-element Variable without initial gradient. Shape: $(size(v.value)), Type: $T")
        end
    end

    topo_stack = Variable[]
    visited_topo = Set{Variable}()
    function build_topo_stack(node)
        push!(visited_topo, node)
        for child in node.children
             if !(child in visited_topo); build_topo_stack(child); end
        end
        push!(topo_stack, node)
    end
    build_topo_stack(v)

    visited_in_pass = Set{Variable}()
    processed_count = 0
    while !isempty(topo_stack)
        current_node = pop!(topo_stack)
        if current_node.gradient !== nothing && !(current_node in visited_in_pass)
             processed_count += 1
            current_node.backward_fn() # Execute the node's specific backward logic
            push!(visited_in_pass, current_node)
        end
    end
end


function zero_grad!(params::AbstractVector{<:Variable})
    for p in params
        if p.is_param # Only zero gradients of actual parameters
            T = _eltype(p.value)
            if p.gradient !== nothing
                 if isa(p.gradient, AbstractArray)
                     fill!(p.gradient, zero(T))
                 else
                     p.gradient = zero(T)
                 end
            end
        end
    end
end


function Base.:+(a::Variable{T}, b::Variable{T}) where {T<:Real}
    val = value(a) .+ value(b)
    children = Variable[a, b]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        if output_grad !== nothing
            grad_a = output_grad; grad_b = output_grad
            if size(value(a)) != size(output_grad); grad_a = sum_to(output_grad, size(value(a))); end
            if size(value(b)) != size(output_grad); grad_b = sum_to(output_grad, size(value(b))); end
            accumulate_gradient!(a, grad_a)
            accumulate_gradient!(b, grad_b)
        end
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end
Base.:+(a::Variable{T}, b::Real) where T = a + Variable(fill(T(b), size(value(a))))
Base.:+(a::Real, b::Variable{T}) where T = Variable(fill(T(a), size(value(b)))) + b

function Base.:-(a::Variable{T}, b::Variable{T}) where {T<:Real}
    val = value(a) .- value(b)
    children = Variable[a, b]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        if output_grad !== nothing
            grad_a = output_grad
            grad_b = -output_grad
            if size(value(a)) != size(output_grad); grad_a = sum_to(output_grad, size(value(a))); end
            if size(value(b)) != size(output_grad); grad_b = sum_to(-output_grad, size(value(b))); end
            accumulate_gradient!(a, grad_a)
            accumulate_gradient!(b, grad_b)
        end
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end
Base.:-(a::Variable{T}, b::Real) where T = a - Variable(fill(T(b), size(value(a))))
Base.:-(a::Real, b::Variable{T}) where T = Variable(fill(T(a), size(value(b)))) - b
function Base.:-(a::Variable{T}) where {T<:Real} # Unary minus
    zero_var = Variable(zeros(T, size(value(a))), is_param=false)
    return zero_var - a # Relies on binary minus backward
end

function Base.:*(a::Variable{T}, b::Variable{T}) where {T<:Real}
    val_a = value(a); val_b = value(b); size_a = size(val_a); size_b = size(val_b)
    
    if isa(val_a, AbstractMatrix) && isa(val_b, AbstractMatrix) && length(size_a) == 2 && length(size_b) == 2 && size_a[2] == size_b[1]
        return matmul(a, b)
    end

    can_broadcast = false
    local val
    try
        val = val_a .* val_b # Attempt the operation
        can_broadcast = true
    catch e
        error("Incompatible shapes for multiplication or broadcasting: $(size_a) and $(size_b). Error: $e")
    end

    if can_broadcast
        children = Variable[a, b]
        local new_var # Define new_var once for this block
        function backward_fn_multiply() # Give it a unique name or ensure it's only defined once
            output_grad = grad(new_var)
            if output_grad !== nothing
                grad_a_unshaped = output_grad .* val_b # Use val_b captured from outer scope
                grad_b_unshaped = output_grad .* val_a # Use val_a captured from outer scope
                
                grad_a = sum_to(grad_a_unshaped, size_a)
                grad_b = sum_to(grad_b_unshaped, size_b)
                
                accumulate_gradient!(a, grad_a)
                accumulate_gradient!(b, grad_b)
            end
        end
        new_var = Variable(val, children, backward_fn_multiply)
        return new_var
    else
        error("Unhandled multiplication case for shapes: $(size_a) and $(size_b)")
    end
end
function Base.:*(a::Variable{T}, b_scalar::Real) where T
    val_a = value(a)
    val = val_a .* b_scalar # Use the scalar directly
    children = Variable[a]
    local new_var
    function backward_fn_scalar_multiply()
        output_grad = grad(new_var)
        if output_grad !== nothing
            grad_a_unshaped = output_grad .* b_scalar # Use b_scalar captured
            grad_a = sum_to(grad_a_unshaped, size(val_a))
            accumulate_gradient!(a, grad_a)
        end
    end
    new_var = Variable(val, children, backward_fn_scalar_multiply)
    return new_var
end
function Base.:*(a_scalar::Real, b::Variable{T}) where T; return b * a_scalar; end

function matmul(a::Variable{T}, b::Variable{T}) where T
     val_a = value(a); val_b = value(b); size_a = size(val_a); size_b = size(val_b)
     if length(size_a)!=2 || length(size_b)!=2 || size_a[2]!=size_b[1]; error("Incompatible matrix dimensions for matmul: $(size_a) and $(size_b)"); end
     val = val_a * val_b
     children = Variable[a, b]
     local new_var
     function backward_fn()
         output_grad = grad(new_var)
         if output_grad !== nothing
             grad_a = output_grad * transpose(val_b)
             grad_b = transpose(val_a) * output_grad
             accumulate_gradient!(a, grad_a)
             accumulate_gradient!(b, grad_b)
         end
     end
     new_var = Variable(val, children, backward_fn)
     return new_var
end

function Base.:/(a::Variable{T}, b::Variable{T}) where {T<:Real}
    eps_T = Base.eps(T)
    val = value(a) ./ (value(b) .+ eps_T)
    children = Variable[a, b]
    local new_var
    function backward_fn()
        output_grad = grad(new_var)
        if output_grad !== nothing
            denom_stable = value(b) .+ eps_T
            grad_a_u = output_grad ./ denom_stable
            grad_b_u = -output_grad .* value(a) ./ (denom_stable .^ 2)
            grad_a = sum_to(grad_a_u, size(value(a)))
            grad_b = sum_to(grad_b_u, size(value(b)))
            accumulate_gradient!(a, grad_a)
            accumulate_gradient!(b, grad_b)
        end
    end
    new_var = Variable(val, children, backward_fn)
    return new_var
end
Base.:/(a::Variable{T}, b::Real) where T = a / Variable(fill(T(b), size(value(a))))
Base.:/(a::Real, b::Variable{T}) where T = Variable(fill(T(a), size(value(b)))) / b

function Base.:^(a::Variable{T}, n::Real) where {T<:Real}
    n_T = T(n); eps_T = Base.eps(T); base_stable = value(a) .+ T(sign(value(a)) * eps_T); val = base_stable .^ n_T; children = Variable[a]; local new_var
    function backward_fn(); output_grad = grad(new_var); if output_grad !== nothing; grad_a_u = output_grad .* n_T .* (base_stable .^ (n_T - one(T))); grad_a = sum_to(grad_a_u, size(value(a))); accumulate_gradient!(a, grad_a); end; end
    new_var = Variable(val, children, backward_fn); return new_var
end

function Base.exp(a::Variable{T}) where {T<:Real}
    val = exp.(value(a)); children = Variable[a]; local new_var
    function backward_fn(); output_grad = grad(new_var); if output_grad !== nothing; grad_a_u = output_grad .* val; grad_a = sum_to(grad_a_u, size(value(a))); accumulate_gradient!(a, grad_a); end; end
    new_var = Variable(val, children, backward_fn); return new_var
end

function Base.log(a::Variable{T}; ϵ::Union{Nothing,Real}=nothing) where {T<:Real}
     eps_T = ϵ === nothing ? Base.eps(T) : T(ϵ); val_stable = max.(value(a), eps_T); val = log.(val_stable); children = Variable[a]; local new_var
    function backward_fn(); output_grad = grad(new_var); if output_grad !== nothing; grad_a_u = output_grad ./ val_stable; grad_a = sum_to(grad_a_u, size(value(a))); accumulate_gradient!(a, grad_a); end; end
    new_var = Variable(val, children, backward_fn); return new_var
end

function Base.max(a::Variable{T}, val::Real) where {T<:Real}
    val_T = T(val); res_val = max.(value(a), val_T); children = Variable[a]; local new_var
    function backward_fn(); output_grad = grad(new_var); if output_grad !== nothing; mask = T.(value(a) .> val_T); grad_a_u = output_grad .* mask; grad_a = sum_to(grad_a_u, size(value(a))); accumulate_gradient!(a, grad_a); end; end
    new_var = Variable(res_val, children, backward_fn); return new_var
end
Base.max(val::Real, a::Variable{T}) where T = max(a, val)


function sigmoid(x::Variable{T}; ϵ=1e-8) where T<:Real
     one_T = Variable(fill(T(1.0), size(value(x))), is_param=false)
     exp_neg_x = exp(-x) # relies on AD of exp and unary -
     denom = one_T + exp_neg_x # relies on AD of +
     sig_val = one_T / (denom + Variable(fill(eps(T), size(value(denom))), is_param=false)) # relies on AD of /
     return sig_val
end


function Base.sum(a::Variable{T}) where {T<:Real}
    val = sum(value(a)); children = Variable[a]; local new_var
    function backward_fn(); output_grad = grad(new_var); if output_grad !== nothing; grad_a = fill(output_grad, size(value(a))); accumulate_gradient!(a, grad_a); end; end
    new_var = Variable(val, children, backward_fn); return new_var
end

function Statistics.mean(v::Variable{T}; dims) where T
    val = mean(value(v); dims=dims); original_shape = size(value(v)); output_shape = size(val)
    num_elements_pooled = prod(size(value(v), d) for d in dims); N = T(num_elements_pooled)
    children = Variable[v]; local new_var
    function backward_fn(); output_grad = grad(new_var); if output_grad !== nothing && !all(iszero, output_grad); input_grad = similar(value(v)); input_grad .= output_grad ./ N; AutoDiff.accumulate_gradient!(v, input_grad); end; end # Qualify accumulate_gradient!
    new_var = Variable(val, children, backward_fn); return new_var
end


function sum_to(x::AbstractArray{T}, target_size::Tuple) where T
    if size(x) == target_size; return x; end; if isempty(target_size) || target_size == (1,); return sum(x)::T; end
    ndims_x = ndims(x); ndims_target = length(target_size); dims_to_sum = Int[]; for d = 1:ndims_x; if d > ndims_target || (target_size[d] == 1 && size(x, d) > 1); push!(dims_to_sum, d); elseif d <= ndims_target && target_size[d] != 1 && size(x, d) != target_size[d] && size(x, d) != 1; error("..."); end; end
    result = isempty(dims_to_sum) ? x : sum(x, dims=tuple(dims_to_sum...)); return size(result) == target_size ? result : reshape(result, target_size);
end
function sum_to(x::T, target_size::Tuple) where T<:Real
    if isempty(target_size) || target_size == (1,); return x::T; end; return fill(x, target_size)::AbstractArray{T};
end


end # module AutoDiff