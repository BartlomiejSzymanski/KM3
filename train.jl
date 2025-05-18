

println("Including local modules...")
include("Autodiff.jl") # Defines Main.AutoDiff
include("NeuralNet.jl")            # Defines Main.NeuralNet (uses Main.AutoDiff, Main.NeuralNet)
include("LossFunctions.jl")
include("Optimizers.jl")
println("Local modules included.")

using .AutoDiff
using .NeuralNet      # Module defined in CNN.jl
using .LossFunctions  # Module defined in LossFunctions.jl
using .Optimizers      # Expects Optimizers.jl to define module Optimizers

using JLD2, Random, Printf, Statistics, LinearAlgebra, InteractiveUtils # Added InteractiveUtils for debug

println("Loading prepared dataset...")
data_dir = joinpath(@__DIR__, "data") # @__DIR__ is MyCNNProject/
prepared_data_path = joinpath(data_dir, "imdb_dataset_prepared.jld2")


X_train_loaded = load(prepared_data_path, "X_train")
y_train_loaded = load(prepared_data_path, "y_train")
X_test_loaded = load(prepared_data_path, "X_test")
y_test_loaded = load(prepared_data_path, "y_test")
embeddings = load(prepared_data_path, "embeddings")
vocab = load(prepared_data_path, "vocab")

embedding_dim = size(embeddings, 1)
vocab_size = length(vocab)
max_len = size(X_train_loaded, 1)

embeddings_for_layer = permutedims(embeddings, (2,1))
println("Dataset loaded. Vocab size: $vocab_size, Embedding dim: $embedding_dim, Max len: $max_len")

NeuralNet.load_embeddings!(NeuralNet.EmbeddingLayer(vocab_size, embedding_dim, pad_idx=findfirst(x->x=="<pad>", vocab)), embeddings_for_layer)

model = NeuralNet.Chain(
    NeuralNet.EmbeddingLayer(vocab_size, embedding_dim, pad_idx=findfirst(x->x=="<pad>", vocab)),
    NeuralNet.PermuteLayer((2,3,1)),
    NeuralNet.Conv1DLayer(3, embedding_dim, 8, NeuralNet.relu),
    NeuralNet.MaxPool1DLayer(8, stride=8),
    NeuralNet.FlattenLayer(),
    NeuralNet.TransposeLayer(),
    NeuralNet.Dense(128, 1, NeuralNet.sigmoid)
)

println("Model created:")

learning_rate = 0.001f0
epochs = 5 # Increase for better results if needed
batch_size = 64

model_params = NeuralNet.get_params(model) # Or NeuralNet.get_params if using CNNModel type
optimizer = Optimizers.Adam(learning_rate, model_params)

num_samples_train = size(X_train_loaded, 2)
num_batches = ceil(Int, num_samples_train / batch_size)

println("\nStarting training...")
model.is_training_ref[] = true

for epoch in 1:epochs
    total_train_loss = 0.0
    total_train_accuracy = 0.0
    shuffled = shuffle(1:num_samples_train)

    epoch_time = @elapsed begin
        for batch_id in 1:num_batches
            batch_start = (batch_id - 1) * batch_size + 1
            batch_end = min(batch_id * batch_size, num_samples_train)
            batch_indices = shuffled[batch_start:batch_end]

            X_batch = permutedims(X_train_loaded[:, batch_indices], (2, 1))
            y_batch = permutedims(convert(Matrix{Float32}, y_train_loaded[:, batch_indices]), (2, 1))

            AutoDiff.zero_grad!(model_params)

            input_var = AutoDiff.Variable(X_batch)
            prediction = model(input_var)

            loss = LossFunctions.binary_cross_entropy(prediction, y_batch)
            AutoDiff.backward!(loss)
            Optimizers.update!(optimizer)

            total_train_loss += AutoDiff.value(loss)

            predicted_labels = AutoDiff.value(prediction) .> 0.5f0
            correct_labels = y_batch .> 0.5f0
            total_train_accuracy += Statistics.mean(predicted_labels .== correct_labels)
        end
    end

    avg_train_loss = total_train_loss / num_batches
    avg_train_accuracy = total_train_accuracy / num_batches

    # --- Evaluation ---
    model.is_training_ref[] = false
    X_test_input = permutedims(X_test_loaded, (2, 1))
    y_test = permutedims(convert(Matrix{Float32}, y_test_loaded), (2, 1))
    test_input_var = AutoDiff.Variable(X_test_input)

    test_prediction = model(test_input_var)
    test_loss = LossFunctions.binary_cross_entropy(test_prediction, y_test)

    test_pred_labels = AutoDiff.value(test_prediction) .> 0.5f0
    test_true_labels = y_test .> 0.5f0
    test_accuracy = Statistics.mean(test_pred_labels .== test_true_labels)
    model.is_training_ref[] = true

    @printf("Epoch %d (%.2fs): Train Loss: %.4f, Train Acc: %.4f | Test Loss: %.4f, Test Acc: %.4f\n",
            epoch, epoch_time, avg_train_loss, avg_train_accuracy,
            AutoDiff.value(test_loss), test_accuracy)

    if test_accuracy ≥ 0.80 && epoch ≥ 2
        println("Target accuracy potentially achieved!")
    end
end
