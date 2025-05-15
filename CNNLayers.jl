# CNN.jl
module CNNLayers

using ..SimpleAutoDiff
using ..SimpleNN

using Statistics, Random, LinearAlgebra

export Conv1DLayer, MaxPool1DLayer, CNNModel, forward, get_params

# --- Conv1D Layer ---
struct Conv1DLayer{T<:Real, F<:Function}
    W::Variable{T}
    b::Variable{T}
    activation::F
    stride::Int
    padding::Int

    function Conv1DLayer(kernel_width::Int, in_channels::Int, out_channels::Int, activation_fn::F=SimpleNN.relu;
                         stride::Int=1, padding::Int=0, dtype=Float32) where F
        limit = sqrt(dtype(6) / dtype(kernel_width * in_channels + out_channels))
        W_val = (rand(dtype, kernel_width, in_channels, out_channels) .* dtype(2) .- dtype(1)) .* limit
        b_val = zeros(dtype, out_channels)
        new{dtype, F}(Variable(W_val, is_param=true),
                      Variable(b_val, is_param=true),
                      activation_fn, stride, padding)
    end
end

# MODIFIED SIGNATURE HERE
function forward(layer::Conv1DLayer{T, F}, x_var::Variable{T}) where {T<:Real, F<:Function}
    #println("--- INSIDE CNNLayers.forward for Conv1DLayer ---") # DEBUG PRINT
    x = value(x_var)
    W_val = value(layer.W)
    b_val = value(layer.b)
    in_width, in_channels, batch_size = size(x)
    kernel_w, _, out_channels = size(W_val) # Assuming W_val is (kernel_w, in_channels, out_channels)
    out_width = (in_width - kernel_w) ÷ layer.stride + 1
    out_val_raw = zeros(T, out_width, out_channels, batch_size)

    for b_idx in 1:batch_size
        for c_out in 1:out_channels
            for w_out in 1:out_width
                w_in_start = (w_out - 1) * layer.stride + 1
                # Ensure receptive field slicing is correct for W dimensions
                receptive_field = x[w_in_start : w_in_start + kernel_w - 1, :, b_idx] # (kernel_w, in_channels)
                kernel_slice = W_val[:, :, c_out] # (kernel_w, in_channels) for specific out_channel
                out_val_raw[w_out, c_out, b_idx] = sum(receptive_field .* kernel_slice) + b_val[c_out]
            end
        end
    end

    activated_out_val = if layer.activation == SimpleNN.relu
        max.(out_val_raw, T(0))
    elseif layer.activation == SimpleNN.sigmoid
        SimpleNN.sigmoid_val.(out_val_raw)
    elseif layer.activation == identity
         out_val_raw
    else
        # This case might be for other custom Function-like objects passed.
        # If layer.activation is a struct or something not directly callable on array, this will fail.
        # Assuming it's an element-wise function if not recognized explicitly.
        @warn "Conv1DLayer applying unknown activation $(typeof(layer.activation)) element-wise. Ensure this is intended."
        layer.activation.(out_val_raw)
    end

    children = [x_var, layer.W, layer.b]
    local new_var
    function backward_fn_conv1d()
        dL_dy_activated = grad(new_var)
        if dL_dy_activated === nothing; return; end

        dL_dy_raw = activation_gradient_manual(dL_dy_activated, out_val_raw, layer.activation, T)

        dL_dx = zeros(T, size(x))
        dL_dW = zeros(T, size(W_val))
        dL_db = zeros(T, size(b_val))

        for b_idx_bw in 1:batch_size
            for c_out in 1:out_channels
                for w_out in 1:out_width
                    w_in_start = (w_out - 1) * layer.stride + 1
                    dL_db[c_out] += dL_dy_raw[w_out, c_out, b_idx_bw]
                    receptive_field_bw = x[w_in_start : w_in_start + kernel_w - 1, :, b_idx_bw] # Renamed
                    kernel_slice_bw = W_val[:, :, c_out] # Renamed

                    dL_dW[:, :, c_out] .+= receptive_field_bw .* dL_dy_raw[w_out, c_out, b_idx_bw]
                    dL_dx[w_in_start : w_in_start + kernel_w - 1, :, b_idx_bw] .+= kernel_slice_bw .* dL_dy_raw[w_out, c_out, b_idx_bw]
                end
            end
        end
        accumulate_gradient!(x_var, dL_dx)
        accumulate_gradient!(layer.W, dL_dW)
        accumulate_gradient!(layer.b, dL_db)
    end

    new_var = Variable(activated_out_val, children, backward_fn_conv1d)
    return new_var
end

# Manual gradient calculation for common activations (applied to values)
function activation_gradient_manual(dL_dy_activated, y_raw, activation_fn_original, T_type::Type)
    if activation_fn_original == SimpleNN.relu
        return dL_dy_activated .* T_type.(y_raw .> 0)
    elseif activation_fn_original == identity # Base.identity
        return dL_dy_activated
    elseif activation_fn_original == SimpleNN.sigmoid
         sig_y_raw_values = SimpleNN.sigmoid_val.(y_raw)
         return dL_dy_activated .* sig_y_raw_values .* (T_type(1) .- sig_y_raw_values)
    else
        @warn "Trying to compute gradient for unknown activation $(typeof(activation_fn_original)) in activation_gradient_manual. Assuming derivative is 1 (like identity)."
        # Fallback or error:
        # error("Unsupported activation in Conv1D backward (activation_gradient_manual): $activation_fn_original")
        return dL_dy_activated # Fallback: treat as identity if unknown (potentially wrong)
    end
end

get_params(layer::Conv1DLayer) = [layer.W, layer.b]

# --- MaxPool1D Layer ---
struct MaxPool1DLayer{T<:Real}
    pool_size::Int
    stride::Int
    switches::Ref{Array{CartesianIndex{3}, 3}}
    function MaxPool1DLayer(pool_size::Int; stride::Int = pool_size, dtype=Float32)
        new{dtype}(pool_size, stride, Ref(Array{CartesianIndex{3},3}(undef,0,0,0)))
    end
end

function forward(layer::MaxPool1DLayer{T}, x_var::Variable{T}) where T
    x = value(x_var)
    in_width, channels, batch_size = size(x)
    out_width = (in_width - layer.pool_size) ÷ layer.stride + 1
    out_val = zeros(T, out_width, channels, batch_size)
    switches_val = Array{CartesianIndex{3}, 3}(undef, out_width, channels, batch_size)

    for b_idx in 1:batch_size # Renamed b to b_idx
        for c in 1:channels
            for w_out in 1:out_width
                w_in_start = (w_out - 1) * layer.stride + 1
                w_in_end = w_in_start + layer.pool_size - 1
                window_data = x[w_in_start:w_in_end, c, b_idx] # Renamed window to window_data
                max_val, rel_idx = findmax(window_data)
                out_val[w_out, c, b_idx] = max_val
                switches_val[w_out, c, b_idx] = CartesianIndex(w_in_start + rel_idx[1] - 1, c, b_idx)
            end
        end
    end
    layer.switches[] = switches_val
    children = [x_var]
    local new_var
    function backward_fn_maxpool() # Renamed
        dL_dy = grad(new_var)
        if dL_dy === nothing; return; end
        dL_dx = zeros(T, size(x))
        sw = layer.switches[]
        if size(dL_dy) != size(sw)
            @warn "MaxPool1D backward: dL_dy shape $(size(dL_dy)) mismatch with switches shape $(size(sw))"
            if length(dL_dy) == 1 && isa(dL_dy, Real)
                dL_dy_scalar_val = dL_dy[] # Renamed dL_dy_val
                for i_idx in eachindex(sw) # Renamed i to i_idx
                    dL_dx[sw[i_idx]] += dL_dy_scalar_val
                end
            else
                for i_idx in eachindex(dL_dy)
                    dL_dx[sw[i_idx]] += dL_dy[i_idx]
                end
            end
        else
            for i_idx in eachindex(dL_dy)
                dL_dx[sw[i_idx]] += dL_dy[i_idx]
            end
        end
        accumulate_gradient!(x_var, dL_dx)
    end
    new_var = Variable(out_val, children, backward_fn_maxpool)
    return new_var
end
get_params(layer::MaxPool1DLayer) = []

# --- CNNModel ---
struct CNNModel
    layers::Vector{Any}
    parameters::Vector{Variable} # Should be collected by get_params
    is_training_ref::Ref{Bool}
    function CNNModel(layers_arg...)
        model_layers = [l for l in layers_arg]
        # params will be collected by get_params(model) later
        # params = Variable[] # Not strictly needed to store here if get_params rebuilds it
        is_training_ref = Ref(true)
        for layer_item in model_layers # Renamed layer to layer_item
            if hasfield(typeof(layer_item), :is_training_ref) && hasfield(typeof(layer_item), :is_training)
                 # This logic was for DropoutLayer specifically.
                 # A DropoutLayer has `is_training::Ref{Bool}`.
                 # The model's `is_training_ref` can be assigned to it.
                 if layer_item.is_training isa Ref{Bool}
                    layer_item.is_training = is_training_ref
                 end
            end
            # if hasmethod(get_params, (typeof(layer_item),))
            #     append!(params, get_params(layer_item))
            # end
        end
        # new(model_layers, unique(params), is_training_ref)
        # Let get_params handle parameter collection dynamically
        new(model_layers, Variable[], is_training_ref) # Initialize with empty params
    end
end

function forward(model::CNNModel, x_var::Variable)
    current_var = x_var
    for layer_item in model.layers # Renamed layer to layer_item
        current_var = forward(layer_item, current_var)
    end
    return current_var
end
(model::CNNModel)(x_var::Variable) = forward(model, x_var)

function get_params(model::CNNModel) # Rebuild params list each time or store in constructor
    all_params = Variable[]
    for layer_item in model.layers
        if hasmethod(get_params, (typeof(layer_item),))
            append!(all_params, get_params(layer_item))
        end
    end
    return unique(all_params)
end

function train_mode!(model::CNNModel)
    model.is_training_ref[] = true
    for layer_item in model.layers
        if layer_item isa SimpleNN.DropoutLayer # Check specific type
            layer_item.is_training[] = true
        end
    end
end
function eval_mode!(model::CNNModel)
    model.is_training_ref[] = false
     for layer_item in model.layers
        if layer_item isa SimpleNN.DropoutLayer
            layer_item.is_training[] = false
        end
    end
end

end # module CNNLayers