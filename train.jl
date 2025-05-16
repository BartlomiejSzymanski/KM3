# train_cnn.jl

# --- Include local modules directly ---
# This will define Main.SimpleAutoDiff, Main.SimpleNN, etc.
println("Including local modules...")
include("SimpleAutoDiff.jl") # Defines Main.SimpleAutoDiff
include("NeuralNet.jl")       # Defines Main.SimpleNN (uses Main.SimpleAutoDiff)
include("LossFunctions.jl")
include("Optimizers.jl")
println("Local modules included.")

# --- Use the modules now defined in Main ---
# The '.' means "from the current module" (which is Main here)
using .SimpleAutoDiff
using .LossFunctions  # Module defined in LossFunctions.jl
using .Optimizers      # Expects Optimizers.jl to define module Optimizers
using .NeuralNet
using JLD2, Random, Printf, Statistics, LinearAlgebra, InteractiveUtils # Added InteractiveUtils for debug

# --- Data Loading and Preparation ---
println("Loading prepared dataset...")
# Path to data folder is now directly relative
data_dir = joinpath(@__DIR__, "data") # @__DIR__ is MyCNNProject/
prepared_data_path = joinpath(data_dir, "imdb_dataset_prepared.jld2")

if !isfile(prepared_data_path)
    println("Prepared data not found: $(prepared_data_path)")
    println("Please run data_prep.jl manually first from the project root directory (MyCNNProject/):")
    # To run data_prep.jl, it also needs to be in MyCNNProject/
    println("  julia --project data_prep.jl")
    exit()
end

X_train_loaded = load(prepared_data_path, "X_train")
y_train_loaded = load(prepared_data_path, "y_train")
X_test_loaded = load(prepared_data_path, "X_test")
y_test_loaded = load(prepared_data_path, "y_test")
embeddings_matrix = load(prepared_data_path, "embeddings")
vocab = load(prepared_data_path, "vocab")

embedding_dim = size(embeddings_matrix, 1)
vocab_size = length(vocab)
max_len = size(X_train_loaded, 1)

embeddings_for_layer = permutedims(embeddings_matrix, (2,1))
println("Dataset loaded. Vocab size: $vocab_size, Embedding dim: $embedding_dim, Max len: $max_len")

# --- Model Definition ---
cnn_embedding_dim = embedding_dim
cnn_vocab_size = vocab_size
cnn_kernel_width = 3
cnn_out_channels = 8
cnn_pool_size = 8
# After conv (kernel 3, stride 1, pad 0): max_len - 3 + 1 = 130 - 3 + 1 = 128
# After maxpool (pool 8, stride 8): 128 / 8 = 16
cnn_dense_in_features = 16 * cnn_out_channels # 16 * 8 = 128
cnn_dense_out_features = 1

padding_idx_val = findfirst(x->x=="<pad>", vocab)
if padding_idx_val === nothing; error("<pad> token not found in vocab."); end

embedding_layer = NeuralNet.EmbeddingLayer(cnn_vocab_size, cnn_embedding_dim, pad_idx=padding_idx_val)
NeuralNet.load_embeddings!(embedding_layer, embeddings_for_layer)

# Permute output of Embedding (bs, sl, ed) -> (sl, ed, bs) for Conv1D
permute_to_conv_format = NeuralNet.PermuteLayer((2,3,1))

# Conv1DLayer expects activation like NeuralNet.relu (Variable -> Variable)
# or a value-based one if its internal logic is value-based.
# The corrected NeuralNet.Conv1DLayer uses the passed function in activation_gradient_manual
conv_layer = NeuralNet.Conv1DLayer(cnn_kernel_width, cnn_embedding_dim, cnn_out_channels, NeuralNet.relu)

maxpool_layer = NeuralNet.MaxPool1DLayer(cnn_pool_size, stride=cnn_pool_size)

# FlattenToFeaturesBatch: (W, C, BS) -> (W*C, BS)
flatten_layer = NeuralNet.FlattenToFeaturesBatch()

# Transpose for Dense: (Features, BS) -> (BS, Features)
transpose_for_dense = NeuralNet.TransposeLayer()

dense_layer = NeuralNet.Dense(cnn_dense_in_features, cnn_dense_out_features, NeuralNet.sigmoid)

# Using NeuralNet.MLPModel as a generic chainer
model = NeuralNet.MLPModel(
    embedding_layer,
    permute_to_conv_format,
    conv_layer,
    maxpool_layer,
    flatten_layer,
    transpose_for_dense,
    dense_layer
)

println("Model created:")
# ... (model printing logic can remain) ...

# --- Training Setup ---
learning_rate = 0.001f0
epochs = 5 # Increase for better results if needed
batch_size = 64

# Get parameters AFTER model construction, as CNNModel might not store them directly
model_params = NeuralNet.get_params(model) # Or NeuralNet.get_params if using CNNModel type
optimizer = Optimizers.Adam(learning_rate, model_params)

# --- Training Loop ---
num_samples_train = size(X_train_loaded, 2)
num_batches = ceil(Int, num_samples_train / batch_size)

println("\nStarting training...")
NeuralNet.train_mode!(model)

for epoch in 1:epochs
    epoch_loss = 0.0
    epoch_acc = 0.0
    shuffled_indices = shuffle(1:num_samples_train)
    t_epoch = @elapsed begin
        for i in 1:num_batches
            start_idx = (i-1) * batch_size + 1
            end_idx = min(i * batch_size, num_samples_train)
            current_batch_indices = shuffled_indices[start_idx:end_idx]
            
            X_batch_raw = X_train_loaded[:, current_batch_indices]
            y_batch_raw = y_train_loaded[:, current_batch_indices]
            X_batch_for_embedding = permutedims(X_batch_raw, (2,1))
            y_true_for_loss = permutedims(convert(Matrix{Float32}, y_batch_raw), (2,1))

            SimpleAutoDiff.zero_grad!(model_params)
            
            x_input_var = SimpleAutoDiff.Variable(X_batch_for_embedding)
            y_pred_var = model(x_input_var)
            
            loss_var = LossFunctions.binary_cross_entropy(y_pred_var, y_true_for_loss) # Renamed loss_val to loss_var
            
            SimpleAutoDiff.backward!(loss_var)
            Optimizers.update!(optimizer)
            
            epoch_loss += SimpleAutoDiff.value(loss_var)
            preds = SimpleAutoDiff.value(y_pred_var) .> 0.5f0
            true_labels = y_true_for_loss .> 0.5f0
            epoch_acc += Statistics.mean(preds .== true_labels)
            
            if i % 50 == 0 || i == num_batches
                 @printf("  Epoch %d, Batch %d/%d: Avg Batch Loss: %.4f, Avg Batch Acc: %.4f\n",
                        epoch, i, num_batches, epoch_loss/i, epoch_acc/i)
            end
        end
    end

    avg_epoch_loss = epoch_loss / num_batches
    avg_epoch_acc = epoch_acc / num_batches
    
    NeuralNet.eval_mode!(model)
    num_samples_test = size(X_test_loaded, 2)
    X_test_for_embedding = permutedims(X_test_loaded, (2,1))
    y_test_for_loss = permutedims(convert(Matrix{Float32}, y_test_loaded), (2,1))

    x_test_input_var = SimpleAutoDiff.Variable(X_test_for_embedding)
    y_test_pred_var = model(x_test_input_var)
    
    test_loss_var = LossFunctions.binary_cross_entropy(y_test_pred_var, y_test_for_loss) # Renamed test_loss_val
    test_preds = SimpleAutoDiff.value(y_test_pred_var) .> 0.5f0
    test_true_labels = y_test_for_loss .> 0.5f0
    test_acc = Statistics.mean(test_preds .== test_true_labels)
    
    NeuralNet.train_mode!(model)

    @printf("Epoch %d (%.2fs): Train Loss: %.4f, Train Acc: %.4f | Test Loss: %.4f, Test Acc: %.4f\n",
            epoch, t_epoch, avg_epoch_loss, avg_epoch_acc, SimpleAutoDiff.value(test_loss_var), test_acc)
    
    if test_acc >= 0.80 && epoch >=2
        println("Target accuracy potentially achieved!")
        # break # Optionally uncomment to stop early
    end
end

println("Training finished.")
NeuralNet.eval_mode!(model)
X_test_for_embedding_final = permutedims(X_test_loaded, (2,1))
y_test_for_loss_final = permutedims(convert(Matrix{Float32}, y_test_loaded), (2,1))
x_test_input_var_final = SimpleAutoDiff.Variable(X_test_for_embedding_final)
y_test_pred_var_final = model(x_test_input_var_final)
final_test_preds = SimpleAutoDiff.value(y_test_pred_var_final) .> 0.5f0
final_test_true_labels = y_test_for_loss_final .> 0.5f0
final_test_acc = Statistics.mean(final_test_preds .== final_test_true_labels)
println("Final Test Accuracy: $(round(final_test_acc*100, digits=2))%")