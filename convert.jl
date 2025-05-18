using JLD2
using NPZ

# Załaduj dane z JLD2
jld_data = jldopen("./data/imdb_dataset_prepared.jld2", "r") do file
    Dict(
        "X_train" => read(file, "X_train"),
        "y_train" => read(file, "y_train"),
        "X_test" => read(file, "X_test"),
        "y_test" => read(file, "y_test"),
        "embeddings" => read(file, "embeddings"),
        "vocab" => read(file, "vocab"),
    )
end

# Zapisz jako .npz (do Pythona)
npzwrite("./data/imdb_dataset_prepared.npz", jld_data)
