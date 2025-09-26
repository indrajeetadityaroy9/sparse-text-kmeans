import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class DatasetBundle:
    def __init__(self, data, labels, target_names, raw_documents):
        self.data = data
        self.labels = labels
        self.target_names = target_names
        self.raw_documents = raw_documents


_VECTORIZER_BUILDERS = {
    "tfidf": TfidfVectorizer,
    "count": CountVectorizer,
}


def load_20newsgroups(
    subset="all",
    remove_metadata=True,
    categories=None,
    random_state=42,
):
    remove = ("headers", "footers", "quotes") if remove_metadata else ()
    dataset = fetch_20newsgroups(
        subset=subset,
        categories=categories,
        shuffle=True,
        remove=remove,
        random_state=random_state,
    )
    return dataset


def build_vectorizer(config):
    config = config.copy()
    vectorizer_type = config.pop("type", "tfidf").lower()
    if vectorizer_type not in _VECTORIZER_BUILDERS:
        raise ValueError(f"Unsupported vectorizer type: {vectorizer_type}")
    builder = _VECTORIZER_BUILDERS[vectorizer_type]
    return builder(**config)


def reduce_dimensions(matrix, config, random_state):
    if not config:
        return matrix

    method = config.get("method", "svd").lower()
    n_components = config.get("n_components")
    if n_components is None:
        raise ValueError("n_components must be specified for dimensionality reduction")

    if method == "svd":
        reducer = TruncatedSVD(n_components=n_components, random_state=random_state)
        transformed = reducer.fit_transform(matrix)
        return transformed  # keep dense
    if method == "pca":
        dense = matrix.toarray()
        reducer = PCA(n_components=n_components, random_state=random_state)
        transformed = reducer.fit_transform(dense)
        return transformed  # keep dense
    raise ValueError(f"Unsupported reduction method: {method}")


def prepare_dataset(
    vectorizer_config,
    subset="all",
    remove_metadata=True,
    categories=None,
    random_state=42,
    dimensionality_reduction=None,
):
    dataset = load_20newsgroups(
        subset=subset,
        remove_metadata=remove_metadata,
        categories=categories,
        random_state=random_state,
    )
    vectorizer = build_vectorizer(vectorizer_config)
    matrix = vectorizer.fit_transform(dataset.data)
    matrix = reduce_dimensions(matrix, dimensionality_reduction, random_state)
    return DatasetBundle(
        data=matrix,
        labels=np.asarray(dataset.target),
        target_names=list(dataset.target_names),
        raw_documents=list(dataset.data),
    )
