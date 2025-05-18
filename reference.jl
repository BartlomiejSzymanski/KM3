using JLD2
using Flux
using Printf, Statistics
X_train = load("./data/imdb_dataset_prepared.jld2", "X_train")
y_train = load("./data/imdb_dataset_prepared.jld2", "y_train")
X_test = load("./data/imdb_dataset_prepared.jld2", "X_test")
y_test = load("./data/imdb_dataset_prepared.jld2", "y_test")
embeddings = load("./data/imdb_dataset_prepared.jld2", "embeddings")
vocab = load("./data/imdb_dataset_prepared.jld2", "vocab")
nothing

embedding_dim = size(embeddings,1)


model = Chain(
    Flux.Embedding(length(vocab), embedding_dim),
    x->permutedims(x, (2,1,3)),
    Conv((3,), embedding_dim => 8, relu),
    MaxPool((8,)),
    Flux.flatten,
    Dense(128, 1, σ)
)


dataset = Flux.DataLoader((X_train, y_train), batchsize=64, shuffle=true)

loss(m, x, y) = Flux.Losses.binarycrossentropy(m(x), y)
accuracy(m, x, y) = mean((m(x) .> 0.5) .== (y .> 0.5))

opt = Optimisers.setup(Adam(), model)

epochs = 5
for epoch in 1:epochs
    total_loss = 0.0
    total_acc = 0.0
    num_samples = 0

    t = @elapsed begin
        for (x, y) in dataset
            grads = Flux.gradient(model) do m
                loss(m, x, y)
            end
            Optimisers.update!(opt, model, grads[1])
            total_loss += loss(model, x, y)
            total_acc += accuracy(model, x, y)
            num_samples += 1
        end

        train_loss = total_loss / num_samples
        train_acc = total_acc / num_samples

        test_acc = accuracy(model, X_test, y_test)
        test_loss = loss(model, X_test, y_test)
    end

    println(@sprintf("Epoch: %d (%.2fs) \tTrain: (l: %.2f, a: %.2f) \tTest: (l: %.2f, a: %.2f)", 
        epoch, t, train_loss, train_acc, test_loss, test_acc))
end