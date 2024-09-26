import faiss
from sentence_transformers import SentenceTransformer


def init_rag_vector_database(raw_txt_data_path:str, \
                             sentence_embedding_encoder='distilbert-base-nli-stsb-mean-tokens'):
    assert raw_txt_data_path is not None
    with open(raw_txt_data_path, 'r', encoding='utf-8') as f:
        documents = f.readlines()
    # load sentence embedding sentence encoder
    sentence_encoder = SentenceTransformer(sentence_embedding_encoder)
    # create sentence embedding vectors
    embeddings = sentence_encoder.encode(documents, convert_to_numpy=True)
    # initialize vector indices
    dimension = embeddings.shape[1]  # embedding vector dimension
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    # update vector indices
    index.add(embeddings.astype('float32'))
    # save database
    faiss.write_index(index, 'vector_index.faiss')
    return index, documents, sentence_encoder

def restore_rag_vector_database(database_path:str, raw_txt_data_path:str, \
                                sentence_embedding_encoder='distilbert-base-nli-stsb-mean-tokens'):
    assert raw_txt_data_path is not None
    with open(raw_txt_data_path, 'r', encoding='utf-8') as f:
        documents = f.readlines()
    index = faiss.read_index(database_path)
    sentence_encoder = SentenceTransformer(sentence_embedding_encoder)
    return index, documents, sentence_encoder

def load_rag_vector_database(database_path:str, raw_txt_data_path:str, \
                             sentence_embedding_encoder='distilbert-base-nli-stsb-mean-tokens'):
    if database_path is None:
        index, documents, sentence_encoder = init_rag_vector_database(
            raw_txt_data_path, sentence_embedding_encoder
        )
    else:
        index, documents, sentence_encoder = restore_rag_vector_database(
            database_path, raw_txt_data_path, sentence_embedding_encoder
        )
    return index, documents, sentence_encoder

def query_database(query:str, database_path:str, raw_txt_data_path:str, knn=3, verbose=False):
    index, documents, sentence_encoder = load_rag_vector_database(database_path, raw_txt_data_path)
    return_info_list = []
    assert isinstance(query, str)
    query_embedding = sentence_encoder.encode([query], convert_to_numpy=True)
    # search similar text
    D, I = index.search(query_embedding.astype('float32'), knn)  # distances, ids
    # verbose corresponding results
    for i in range(knn):
        document = documents[I[0][i]]
        distance = D[0][i]
        return_info_list.append(document)
        if verbose:
            print(f'similar text: {document}, distance: {distance}')
    return return_info_list