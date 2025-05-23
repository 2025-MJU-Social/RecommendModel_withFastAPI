def add_genre_embeddings(contents, model):
    contents["genre_vector"] = contents["detail_genre"].apply(lambda x: model.encode(x))
    return contents
