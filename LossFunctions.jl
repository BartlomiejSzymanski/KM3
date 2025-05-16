module LossFunctions

using ..SimpleAutoDiff
using Statistics

export binary_cross_entropy

function binary_cross_entropy(y_pred::Variable{T}, y_true::Union{AbstractMatrix{T}, AbstractVector{T}}; ϵ=1e-9) where T<:Real
    y_true_reshaped = reshape(y_true, size(value(y_pred)))
    one_val = ones(T, size(value(y_pred)))
    one_var = Variable(one_val, is_param=false)

    eps_T = T(ϵ)
    log_ypred = log(y_pred; ϵ=eps_T)
    log_one_minus_ypred = log(one_var - y_pred; ϵ=eps_T)

    y_true_var = Variable(y_true_reshaped, is_param=false)

    term1 = y_true_var * log_ypred
    term2 = (one_var - y_true_var) * log_one_minus_ypred

    loss_elements = -(term1 + term2)

    num_samples = T(length(loss_elements.value))
    sum_loss = sum(loss_elements)

    mean_loss = sum_loss / Variable(num_samples, is_param=false)

    return mean_loss
end

end
