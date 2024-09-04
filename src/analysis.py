import numpy as np
from langchain.embeddings import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

# from analysis import compare_texts


def calculate_similarities(a, b):
    """
    Calculate cosine similarity and euclidean distance between two vectors.

    Parameters:
    - a (array-like): First input vector.
    - b (array-like): Second input vector.

    Returns:
    - dict: A dictionary containing the 'cosine_similarity' and 'euclidean_distance' between vectors a and b.
    """
    # shape a and b to be 2d arrays
    a = np.reshape(a, (1, -1))
    b = np.reshape(b, (1, -1))

    cos_sim = cosine_similarity(a, b)

    return {
        "cosine_similarity": float(cos_sim),
        "euclidean_distance": float(euclidean_distances(a, b)),
    }


def create_embedding(model_name="all-MiniLM-L6-v2"):
    """
    Create an embedding object using HuggingFaceEmbeddings with the specified model.

    Parameters:
    - model_name (str, optional): The name of the Hugging Face model to use for embeddings. Defaults to 'all-MiniLM-L6-v2'.

    Returns:
    - HuggingFaceEmbeddings: An embeddings object initialized with the specified model.
    """
    return HuggingFaceEmbeddings(model_name=model_name)


def compare_texts(text1, text2, embeddings_model="all-MiniLM-L6-v2"):
    """
    Compare two texts by computing their embeddings and then calculating cosine similarity and euclidean distance.

    Parameters:
    - text1 (str or array-like): First text or its embedding.
    - text2 (str or array-like): Second text or its embedding.
    - embeddings_model (str or HuggingFaceEmbeddings, optional): The embeddings model or its name to use for text embedding.
      Defaults to 'all-MiniLM-L6-v2'.

    Returns:
    - dict: A dictionary containing the 'cosine_similarity' and 'euclidean_distance' between the embeddings of text1 and text2.
    """
    if isinstance(embeddings_model, str):
        embeddings_model = HuggingFaceEmbeddings(model_name=embeddings_model)
    if isinstance(text1, str):
        text1 = embeddings_model.embed_query(text1)
    if isinstance(text2, str):
        text2 = embeddings_model.embed_query(text2)
    return calculate_similarities(text1, text2)


def calculate_similarity_for_cluster_part(item):
    """
    Calculate the pairwise cosine similarities within a cluster of text embeddings.

    Parameters:
    - item (tuple): A tuple containing the cluster identifier and a list of text embeddings or strings representing the cluster.

    Returns:
    - dict: A dictionary containing the 'essay_id', 'similarities' (a list of similarity measures for each text pair), and
            'average_cosine_similarity' across all pairs in the cluster.
    """
    essay_id, cluster = item
    similarities = []
    for i in range(len(cluster)):
        for j in range(i + 1, len(cluster)):
            similarity = compare_texts(cluster[i], cluster[j])
            similarities.append(similarity)
    return {
        'essay_id': essay_id,
        'similarities': similarities,
        'average_cosine_similarity': np.mean([entry['cosine_similarity'] for entry in similarities])
    }
