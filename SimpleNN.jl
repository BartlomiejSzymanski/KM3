# SimpleNN.jl (formerly MLP.jl)
module SimpleNN

using ..SimpleAutoDiff # Assumes it's findable via LOAD_PATH

using Random, LinearAlgebra, Statistics, InteractiveUtils
# Export new layers as well
export Dense, EmbeddingLayer, FlattenLayer, MeanPoolLayer, DropoutLayer,
       PermuteLayer, TransposeLayer, FlattenToFeaturesBatch, # New general layers
       relu, sigmoid, sigmoid_val, tanh_approx, # Added sigmoid_val
       MLPModel, forward, get_params, load_embeddings!,
       train_mode!, eval_mode!

# --- Activation Functions ---
relu(x::Variable{T}) where T<:Real = SimpleAutoDiff.max(x, T(0.0)) # Use AD max

function sigmoid(x::Variable{T}; epsilon=1e-8) where T<:Real # Changed ϵ to epsilon
    one_val = ones(T, size(value(x)))
    one_var = Variable(one_val, is_param=false)
    exp_neg_x = exp(-x)
    denominator = one_var + exp_neg_x + Variable(fill(T(epsilon), size(value(exp_neg_x))), is_param=false)
    return one_var / denominator
end

function sigmoid_val(v::T; epsilon=T(1e-8)) where T<:Real # Changed ϵ to epsilon
    return T(1) / (T(1) + exp(-v + epsilon))
end
sigmoid_val(A::AbstractArray{T}; epsilon=T(1e-8)) where T<:Real = sigmoid_val.(A; epsilon=epsilon) # Changed ϵ to epsilon


function tanh_approx(x::Variable{T}) where T<:Real
    two = Variable(fill(T(2.0),size(value(x))),is_param=false)
    one = Variable(fill(T(1.0),size(value(x))),is_param=false)
    return two*SimpleNN.sigmoid(two*x)-one
end


# --- Dense Layer ---
struct Dense{F<:Function}
    W::Variable
    b::Variable
    activation::F
    function Dense(input_features::Int, output_features::Int, activation::F=identity; dtype=Float32) where F
        limit = sqrt(dtype(6) / dtype(input_features + output_features))
        W_val = (rand(dtype, input_features, output_features) .* dtype(2) .- limit) .* limit
        b_val = zeros(dtype, 1, output_features)
        W = Variable(W_val, is_param=true)
        b = Variable(b_val, is_param=true)
        new{F}(W, b, activation)
    end
end

function forward(layer::Dense, x::Variable)
    linear_out = matmul(x, layer.W) + layer.b
    return layer.activation(linear_out)
end
get_params(layer::Dense) = [layer.W, layer.b]


# --- Embedding Layer ---
struct EmbeddingLayer
    weight::Variable
    vocab_size::Int
    embedding_dim::Int
    pad_idx::Int

    function EmbeddingLayer(vocab_size::Int, embedding_dim::Int; pad_idx::Int=0, dtype=Float32)
        limit = dtype(0.05)
        W_val = (rand(dtype, vocab_size, embedding_dim) .* dtype(2) .- dtype(1)) .* limit
        weight = Variable(W_val, is_param=true)
        new(weight, vocab_size, embedding_dim, pad_idx)
    end
end

function forward(layer::EmbeddingLayer, x_indices_var::Variable{<:Integer})
    x_indices = value(x_indices_var)
    T_emb = SimpleAutoDiff._eltype(layer.weight.value)
    batch_size, seq_len = size(x_indices)
    output_shape = (batch_size, seq_len, layer.embedding_dim)
    output_val = zeros(T_emb, output_shape)

    for b in 1:batch_size
        for t in 1:seq_len
            idx = x_indices[b, t]
            if idx != layer.pad_idx && 1 <= idx <= layer.vocab_size
                output_val[b, t, :] .= view(layer.weight.value, idx, :)
            end
        end
    end
    children = [layer.weight]
    local new_var
    function backward_fn_embedding()
        grad_output = grad(new_var)
        if grad_output === nothing; return; end
        T_w = SimpleAutoDiff._eltype(layer.weight.value)
        dL_dW = zeros(T_w, size(layer.weight.value))
        for b in 1:batch_size
            for t in 1:seq_len
                idx = x_indices[b, t]
                if idx != layer.pad_idx && 1 <= idx <= layer.vocab_size
                    for emb_d_idx in 1:layer.embedding_dim
                        dL_dW[idx, emb_d_idx] += grad_output[b,t,emb_d_idx]
                    end
                end
            end
        end
        accumulate_gradient!(layer.weight, dL_dW)
    end
    new_var = Variable(output_val, children, backward_fn_embedding)
    return new_var
end
get_params(layer::EmbeddingLayer) = [layer.weight]
function load_embeddings!(layer::EmbeddingLayer, embeddings_matrix::AbstractMatrix)
    expected_size = (layer.vocab_size, layer.embedding_dim)
    if size(embeddings_matrix) != expected_size
        error("Embedding matrix size mismatch. Expected $(expected_size), got $(size(embeddings_matrix)).")
    end
    layer.weight.value .= convert(typeof(layer.weight.value), embeddings_matrix)
    println("Embeddings loaded into EmbeddingLayer.")
end

# --- Flatten Layer ---
struct FlattenLayer
    input_shape_ref::Ref{Tuple}
    FlattenLayer() = new(Ref{Tuple}(()))
end
function forward(layer::FlattenLayer, x::Variable)
    x_val = value(x)
    input_s = size(x_val)
    layer.input_shape_ref[] = input_s
    batch_size = input_s[1]
    num_features = prod(input_s[2:end])
    output_val = reshape(x_val, batch_size, num_features)
    children = [x]
    local new_var
    function backward_fn_flatten()
        grad_output = grad(new_var)
        if grad_output !== nothing
            grad_input = reshape(grad_output, layer.input_shape_ref[])
            accumulate_gradient!(x, grad_input)
        end
    end
    new_var = Variable(output_val, children, backward_fn_flatten)
    return new_var
end
get_params(layer::FlattenLayer) = []

# --- MeanPoolLayer ---
struct MeanPoolLayer; dims; MeanPoolLayer(dims)=new(dims); end
function forward(layer::MeanPoolLayer, x::Variable) # Removed 'where T'
    return Statistics.mean(x; dims=layer.dims)
end
get_params(layer::MeanPoolLayer)=[];

# --- Dropout Layer ---
mutable struct DropoutLayer{T<:Real}
    p::T
    is_training::Ref{Bool}
    mask::Ref{Union{Nothing, AbstractArray{T}}}
    function DropoutLayer(p::T; is_training_ref::Base.RefValue{Bool}=Ref(true)) where {T<:Real}
        if !(0 <= p < 1); error("Dropout probability p must be in [0, 1)"); end
        new{T}(p, is_training_ref, Ref{Union{Nothing, AbstractArray{T}}}(nothing))
    end
end
function forward(layer::DropoutLayer{T}, x::Variable{T}) where {T<:Real}
    if !layer.is_training[]; return x; end
    val = value(x); p = layer.p;
    mask_val = rand!(similar(val)) .> p;
    layer.mask[] = convert(AbstractArray{T}, mask_val);
    scale_factor = T(1.0 / (1.0 - p));
    output_val = (val .* layer.mask[]) .* scale_factor;
    children = Variable[x]; local new_var;
    function backward_fn_dropout()
        output_grad = grad(new_var);
        if output_grad !== nothing && layer.mask[] !== nothing
            input_grad = (output_grad .* layer.mask[]) .* scale_factor;
            accumulate_gradient!(x, input_grad);
        end
    end
    new_var = Variable(output_val, children, backward_fn_dropout);
    return new_var;
end
get_params(layer::DropoutLayer) = []

# --- PermuteLayer ---
struct PermuteLayer
    dims_tuple::Tuple{Vararg{Int}} # Input permutation
    inv_dims_tuple::Tuple{Vararg{Int}} # Store the inverse permutation directly, not as a Ref

    function PermuteLayer(dims::Tuple{Vararg{Int}})
        if isempty(dims)
            error("Permutation dimensions cannot be empty.")
        end
        # Validate dims (e.g., ensure it's a permutation of 1:length(dims))
        n = length(dims)
        if sort(collect(dims)) != 1:n
            error("Permutation dimensions $dims must be a permutation of 1:$n.")
        end

        inv_dims_val_vec = Vector{Int}(undef, n)
        for (i, d_val) in enumerate(dims)
            inv_dims_val_vec[d_val] = i
        end
        inv_dims_actual_tuple = Tuple(inv_dims_val_vec)
        
        # new takes arguments in the order of fields
        new(dims, inv_dims_actual_tuple)
    end
end

# Convenience constructor: dims... collects arguments into a tuple
PermuteLayer(dims::Int...) = PermuteLayer(dims) # 'dims' here is already a tuple

function forward(layer::PermuteLayer, x::Variable)
    x_val = value(x)
    # Ensure the permutation length matches the number of dimensions to permute
    # typically ndims(x_val) if permuting all, or a subset if that's the design.
    # For now, assume layer.dims_tuple is for permuting ndims(x_val) dimensions.
    if length(layer.dims_tuple) != ndims(x_val) && ndims(x_val) > 1 # Allow scalar pass-through if ndims=0 or 1
        # This check depends on how you intend PermuteLayer to be used.
        # Flux.permutedims(x, (2,1,3)) permutes the first 3 dims.
        # Let's assume the tuple must match ndims for now.
        if ndims(x_val) > 0 # Don't error on scalars which have ndims 0
             @warn "PermuteLayer: length of dims_tuple ($(length(layer.dims_tuple))) does not match ndims of input ($(ndims(x_val))). Behavior might be unexpected."
        end
        # If it's a scalar or 1D array and dims_tuple is (1,), it's fine.
        # If dims_tuple is longer than ndims(x_val), permutedims will error.
        # If shorter, it permutes the first length(dims_tuple) dimensions.
    end

    output_val = permutedims(x_val, layer.dims_tuple)
    
    children = [x]
    local new_var
    function backward_fn_permute()
        grad_output = grad(new_var)
        if grad_output !== nothing
            grad_input = permutedims(grad_output, layer.inv_dims_tuple) # Use stored inv_dims_tuple
            accumulate_gradient!(x, grad_input)
        end
    end
    new_var = Variable(output_val, children, backward_fn_permute)
    return new_var
end
get_params(layer::PermuteLayer) = []

# --- TransposeLayer ---
struct TransposeLayer end
function forward(layer::TransposeLayer, x::Variable)
    x_val = value(x)
    if ndims(x_val) != 2
        error("TransposeLayer expects a 2D input Variable. Got $(ndims(x_val))D.")
    end
    output_val = permutedims(x_val, (2,1))
    children = [x]
    local new_var
    function backward_fn_transpose()
        grad_output = grad(new_var)
        if grad_output !== nothing
            grad_input = permutedims(grad_output, (2,1))
            accumulate_gradient!(x, grad_input)
        end
    end
    new_var = Variable(output_val, children, backward_fn_transpose)
    return new_var
end
get_params(layer::TransposeLayer) = []

# --- FlattenToFeaturesBatch Layer ---
struct FlattenToFeaturesBatch
    input_shape_ref::Ref{Tuple}
    FlattenToFeaturesBatch() = new(Ref{Tuple}(()))
end
function forward(layer::FlattenToFeaturesBatch, x::Variable)
    x_val = value(x)
    s = size(x_val)
    layer.input_shape_ref[] = s
    batch_size = s[end]
    num_features = prod(s[1:end-1])
    output_val = reshape(x_val, num_features, batch_size)
    children = [x]
    local new_var
    function backward_fn_flatten_feat_batch()
        grad_output = grad(new_var)
        if grad_output !== nothing
            grad_input = reshape(grad_output, layer.input_shape_ref[])
            accumulate_gradient!(x, grad_input)
        end
    end
    new_var = Variable(output_val, children, backward_fn_flatten_feat_batch)
    return new_var
end
get_params(layer::FlattenToFeaturesBatch) = []

# --- MLPModel (Chain Abstraction) ---
struct MLPModel
    layers::Vector{Any}
    is_training_ref::Ref{Bool}
    function MLPModel(layers_arg...)
        model_layers = [l for l in layers_arg]
        is_training_ref = Ref(true)
        for layer_item in model_layers
            if hasfield(typeof(layer_item), :is_training) && isa(layer_item.is_training, Ref{Bool})
                layer_item.is_training = is_training_ref
            end
        end
        new(model_layers, is_training_ref)
    end
end
function forward(model::MLPModel, x_var::Variable)
    current_var = x_var
    #println("\n--- MLPModel Forward Pass ---")
    for (i, layer_item) in enumerate(model.layers)
        #println("Layer $i: $(layer_item) (Type=$(typeof(layer_item))), Input Var Type=$(typeof(current_var))")

        if layer_item isa EmbeddingLayer && !(current_var.value isa AbstractMatrix{<:Integer})
             error("Input to EmbeddingLayer must be Variable containing integer indices.")
        end

        # Explicitly check for and call forward methods from Main.CNNLayers
        if layer_item isa Main.CNNLayers.Conv1DLayer
            #println("Dispatching to Main.CNNLayers.forward for Conv1DLayer")
            arg_types = Tuple{typeof(layer_item), typeof(current_var)}
            if !hasmethod(Main.CNNLayers.forward, arg_types)
                #println("FAILURE: Method Main.CNNLayers.forward NOT found for Conv1DLayer with types: $arg_types")
                InteractiveUtils.display(methods(Main.CNNLayers.forward))
                error("Stopping due to missing Conv1DLayer forward method during dispatch check.")
            end
            current_var = Main.CNNLayers.forward(layer_item, current_var)
        elseif layer_item isa Main.CNNLayers.MaxPool1DLayer # ADDED THIS BLOCK
            #println("Dispatching to Main.CNNLayers.forward for MaxPool1DLayer")
            arg_types = Tuple{typeof(layer_item), typeof(current_var)}
            if !hasmethod(Main.CNNLayers.forward, arg_types)
                #println("FAILURE: Method Main.CNNLayers.forward NOT found for MaxPool1DLayer with types: $arg_types")
                InteractiveUtils.display(methods(Main.CNNLayers.forward))
                error("Stopping due to missing MaxPool1DLayer forward method during dispatch check.")
            end
            current_var = Main.CNNLayers.forward(layer_item, current_var)
        else
            # Fallback to generic dispatch for other layers (from SimpleNN itself)
            # This should find methods like SimpleNN.forward(::Dense, ...)
            #println("Dispatching with invokelatest for layer type: $(typeof(layer_item))")
            current_var = Base.invokelatest(forward, layer_item, current_var)
        end
    end
    #println("--- MLPModel Forward Pass END ---")
    return current_var
end
(model::MLPModel)(x_var::Variable) = forward(model, x_var)
function get_params(model::MLPModel)
    all_params = Variable[]
    for layer_item in model.layers
        if hasmethod(get_params, (typeof(layer_item),))
            append!(all_params, get_params(layer_item))
        end
    end
    return unique(all_params)
end
function train_mode!(model::MLPModel)
    model.is_training_ref[] = true
    for l_item in model.layers # changed l to l_item
        if l_item isa DropoutLayer
            l_item.is_training[] = true
        end
    end
end
function eval_mode!(model::MLPModel)
    model.is_training_ref[] = false
    for l_item in model.layers # changed l to l_item
        if l_item isa DropoutLayer
            l_item.is_training[] = false
        end
    end
end

end # module SimpleNN