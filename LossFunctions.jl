# Loss.jl (Revised for Batch Input)
module LossFunctions

using ..SimpleAutoDiff # Depends on the updated SimpleAutoDiff
using Statistics

export binary_cross_entropy

# Binary Cross Entropy Loss - Handles Batches
# y_pred: Variable{T} containing batch predictions, e.g., shape (batch_size, 1)
# y_true: AbstractMatrix{T} or AbstractVector{T} containing batch labels, e.g., shape (batch_size, 1) or (batch_size,)
function binary_cross_entropy(y_pred::Variable{T}, y_true::Union{AbstractMatrix{T}, AbstractVector{T}}; ϵ=1e-9) where T<:Real
    # Ensure y_true has a compatible shape for broadcasting with y_pred's value
    # Typically want y_true to be (batch_size, 1) if y_pred is (batch_size, 1)
    y_true_reshaped = reshape(y_true, size(value(y_pred))) # Ensure same shape as prediction value

    # --- Create constants matching type T and shape ---
    one_val = ones(T, size(value(y_pred)))
    one_var = Variable(one_val, is_param=false)

    # Create a Variable for y_true (needed for AD graph if y_true itself required gradients, but not here)
    # We need it element-wise, so wrap the value directly for calculations.
    # For BCE, y_true doesn't need to be a Variable unless you were calculating gradients w.r.t labels.
    # Instead, use the raw y_true_reshaped matrix directly in the formula where applicable.

    # --- Use type-aware log ---
    # log function handles stability internally
    eps_T = T(ϵ)
    log_ypred = log(y_pred; ϵ=eps_T)
    log_one_minus_ypred = log(one_var - y_pred; ϵ=eps_T) # log(1 - y_pred)

    # --- Calculate BCE element-wise using broadcasting ---
    # BCE = -[y * log(p) + (1 - y) * log(1 - p)]
    # y_true_reshaped is Matrix{T}, log_ypred is Variable{T}
    # We need element-wise Variable * Matrix -> This needs careful handling in AD or simplification.

    # Simplification: Assume y_true is constant w.r.t. gradients.
    # Perform calculations using the *values* for the y_true part, but keep Variables for y_pred part.
    # This is standard practice for BCE loss gradient w.r.t y_pred.

    # term1 = y_true_reshaped .* value(log_ypred) # Element-wise mult
    # term2 = (T(1.0) .- y_true_reshaped) .* value(log_one_minus_ypred) # Element-wise mult
    # loss_value = -(term1 .+ term2) # This would be the value, not the Variable

    # Correct AD approach: Overload element-wise ops or use existing ones carefully
    # We need y_true_var * log_ypred
    y_true_var = Variable(y_true_reshaped, is_param=false) # Wrap y_true as non-parameter Variable

    term1 = y_true_var * log_ypred # Element-wise Variable * Variable
    term2 = (one_var - y_true_var) * log_one_minus_ypred # Element-wise Variable * Variable

    # loss is now a Variable containing element-wise losses for the batch, shape (batch_size, 1)
    loss_elements = -(term1 + term2)

    # --- Average the loss across the batch ---
    # Calculate the mean loss over the batch elements
    num_samples = T(length(loss_elements.value)) # Number of elements in the batch loss
    sum_loss = sum(loss_elements) # sum() should return a scalar Variable{T}

    # Divide by scalar Variable containing num_samples
    mean_loss = sum_loss / Variable(num_samples, is_param=false)

    return mean_loss # Return the scalar mean loss Variable{T}
end


end # module LossFunctions