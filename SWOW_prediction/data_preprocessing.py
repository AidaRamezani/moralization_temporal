import os
import pickle
from typing import Any

import numpy as np
import pandas as pd
import torch
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torch_geometric.utils import add_self_loops, negative_sampling, remove_self_loops

from SWOW_prediction.custom_tokenizers import MyTokenizer
from SWOW_prediction.data_preprocessing_utils import (
    get_data,
    get_sentence_encodings,
    get_swow_data,
    get_two_grams,
    get_word_embedding,
    get_word_position,
)
from SWOW_prediction.utils import *


MIN_FRQUENCY = 50
MIN_LENGTH = 3

class CueDataset(Dataset):
    def __init__(self, cue_data):
        super(CueDataset, self).__init__()
        self.cue_data = cue_data

    def __len__(self):
        return len(self.cue_data)

    def __getitem__(self, index):

        ids, mask, token_ids, _, _ = self.cue_data[index]

        positions = ids.index(token_ids[0])
        positions = torch.arange(positions, positions + len(token_ids))

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            positions,
        )


def get_edge_index_from_edges(
    edges: dict[str, dict[str, float]],
    vocab_mapping: dict[str, int] | None = None,
    fill_value: str = "add",
    add_self_loop: bool = True,
    nodes_total: dict[str, float] | None = None,
    normalize: bool = False,
    sample_negatives: bool = False,
    negative_sample_num: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int], int]:
    """Convert edge dictionary to PyTorch Geometric edge index format.

    Args:
        edges: Dictionary mapping source words to target words with weights
        vocab_mapping: Dictionary mapping node names to indices
        fill_value: Method to calculate self-loop weights ('add', 'mean', 'max')
        add_self_loop: Whether to add self-loops
        nodes_total: Total weight for each node (for normalization)
        normalize: Whether to normalize edge weights
        sample_negatives: Whether to sample negative edges
        negative_sample_num: Number of negative samples to generate

    Returns:
        edge_index: Tensor of shape [2, num_edges] containing edge indices
        edge_weight: Tensor of shape [num_edges] containing edge weights
        vocab_mapping: Updated vocabulary mapping
        n: Number of edges

    """
    edge_index = []
    edge_weight = []

    if vocab_mapping is None:
        vocab_mapping = {}

    for n1, node_links in edges.items():
        total = 1 if not normalize else nodes_total[n1]
        node_edges = []
        node_weights = []

        for n2 in node_links:
            if n1 not in vocab_mapping:
                vocab_mapping[n1] = len(vocab_mapping)
            if n2 not in vocab_mapping:
                vocab_mapping[n2] = len(vocab_mapping)

            i1 = vocab_mapping[n1]
            i2 = vocab_mapping[n2]
            node_edges.append([i1, i2])
            node_weights.append(node_links[n2])

        if len(node_weights) == 0:
            continue

        if add_self_loop:
            i1 = vocab_mapping[n1]
            node_edges.append([i1, i1])

            if fill_value == "add":
                node_weights.append(sum(node_weights))
            elif fill_value == "mean":
                node_weights.append(sum(node_weights) / len(node_weights))
            elif fill_value == "max":
                node_weights.append(max(node_weights))
            else:
                raise ValueError("fill_value must be add, mean or max")

        node_weights = [w / total for w in node_weights]
        edge_index += node_edges
        edge_weight += node_weights

    edge_index = torch.tensor(edge_index).T
    edge_weight = torch.tensor(edge_weight)

    if sample_negatives:
        negative_edge_index = negative_sampling(
            edge_index, num_neg_samples=negative_sample_num,
        )
        edge_index = torch.cat([edge_index, negative_edge_index], dim=1)
        edge_weight = torch.cat(
            [edge_weight, torch.zeros(negative_edge_index.shape[1])],
        )

        if not add_self_loop:
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    n = len(edge_weight)
    return edge_index, edge_weight, vocab_mapping, n


@time_function
def sentensize_articles(articles_text: list[str]) -> list[str]:
    """Split articles into sentences.

    Args:
        articles_text: List of article texts

    Returns:
        List of sentences

    """
    all_sentences = []
    for article in articles_text:
        if pd.isna(article) or len(str(article)) < 10:
            continue
        sentences = sent_tokenize(str(article))
        all_sentences.extend(sentences)

    return all_sentences


@time_function
def get_word_count(
    tokenized_data: list[list[str]],
    sentences: list[str],
    swow_cues: list[str] | None = None,
    add_two_gram: bool = True,
    strategy: str = "frequency",
    k: int = 10000,
) -> tuple[dict[str, int], list[str]]:
    """Calculate word counts and select cues based on strategy.

    Args:
        tokenized_data: List of tokenized sentences
        sentences: Original sentences
        swow_cues: List of cues from SWOW dataset
        add_two_gram: Whether to add two-grams
        strategy: Strategy for selecting cues ('frequency' or other)
        k: Number of top cues to select

    Returns:
        word_count: Dictionary of word counts
        cues: List of selected cues

    """
    from nltk.corpus import stopwords

    word_count = {}
    stop = set(stopwords.words("english"))

    for i, tokenized_sentence in enumerate(tokenized_data):
        for j, token in enumerate(tokenized_sentence):

            if token in stop or len(token) < MIN_LENGTH:
                continue

            word_count[token] = word_count.get(token, 0) + 1

            if add_two_gram and j < len(tokenized_sentence) - 1:
                next_token = tokenized_sentence[j + 1]
                two_gram = f"{token} {next_token}"

                # only add if the two-gram appears in the original sentence (this happens because of the punctuation and lemmatization)
                if two_gram in sentences[i]:
                    word_count[two_gram] = word_count.get(two_gram, 0) + 1

    if strategy == "frequency":
        cues = sorted(word_count.keys(), key=lambda x: -word_count[x])[
            :k
        ]  # We will only select the top k frequent words

        if swow_cues:
            cues.extend([w for w in swow_cues if w not in cues])
    else:
        # Implement other strategies here, but I have not tested anything other than frequency
        cues = []

    return word_count, cues


@time_function
def co_occurrence(
    articles: list[list[str]],
    cues: list[str],
    two_gram: bool = False,
    two_grams: list[str] | None = None,
    window_size: int = 1,
) -> tuple[pd.DataFrame, list[str], int, dict[str, int]]:
    """Calculate co-occurrence matrix for cues.

    Args:
        articles: List of tokenized articles
        cues: List of cues
        two_gram: Whether to include two-grams
        two_grams: List of two-grams to include
        window_size: Size of the window for co-occurrence

    Returns:
        df: Co-occurrence matrix
        vocab: List of vocabulary items
        total: Total number of co-occurrences
        col_totals: Column totals

    """
    total = 0
    col_totals = {}
    vocab = sorted(cues)

    df = pd.DataFrame(
        data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
        index=vocab,
        columns=vocab,
    )

    for tokenized_data in articles:
        n = len(tokenized_data)

        # Handle two-grams if specified
        if two_gram and two_grams:
            tokens = []
            i = 0
            while i < n:
                token = tokenized_data[i]
                potential_two_gram = (
                    f"{token} {tokenized_data[i + 1]}" if i < n - 1 else ""
                )

                if i < n - 1 and potential_two_gram in two_grams:
                    tokens.append(potential_two_gram)
                    i += 2
                else:
                    tokens.append(token)
                    i += 1
        else:
            tokens = tokenized_data.copy()

        for i, token in enumerate(tokens):
            if token == ".":  # Because sentence ends
                continue

            # Get next tokens within window, I always use a window of 1.
            next_token = tokens[i + 1 : min(i + 1 + window_size, len(tokens))]

            for t in next_token:
                if t == ".":
                    continue

                if t in cues and token in cues:
                    key = tuple(sorted([t, token]))
                    df.loc[key[0], key[1]] += 1
                    df.loc[key[1], key[0]] += 1

                col_totals[t] = col_totals.get(t, 0) + 1
                col_totals[token] = col_totals.get(token, 0) + 1
                total += 1

    return df, vocab, total, col_totals


def get_data_graph(
    tokenized_data: list[list[str]],
    cues: list[str],
    strategy: str,
    graph_version: int = 1,
    vocab_mapping: dict[str, int] | None = None,
    fill: str = "add",
    add_self_loop: bool = True,
    n: int = 25,
    two_gram: bool = False,
    two_grams: list[str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    """Generate graph data from tokenized data.

    Args:
        tokenized_data: List of tokenized sentences
        cues: List of cues
        strategy: Strategy for building graph ('ppmi' or other)
        graph_version: Version of graph construction
        vocab_mapping: Dictionary mapping words to indices
        fill: Method for filling self-loops
        add_self_loop: Whether to add self-loops
        n: Number of neighbors to consider
        two_gram: Whether to include two-grams
        two_grams: List of two-grams to include

    Returns:
        edge_index: Edge indices tensor
        edge_weight: Edge weights tensor
        vocab_mapping: Dictionary mapping words to indices

    """

    def get_pmi(
        word: str,
        df: pd.DataFrame,
        col_totals: dict[str, int],
        total: int,
        cues: list[str],
        n: int = 25,
    ) -> dict[str, float]:
        """Calculate PMI for a word with other words."""
        words = np.array(df.columns)
        word_occurrence = df.loc[word, :].values

        # Calculate PMI
        denominator = np.array(
            [col_totals.get(k, 0) * col_totals.get(word, 0) / total for k in words],
        )

        valid_indices = denominator > 0
        word_occurrence_valid = np.zeros_like(word_occurrence, dtype=float)
        word_occurrence_valid[valid_indices] = (
            word_occurrence[valid_indices] / denominator[valid_indices]
        )

        word_occurrence = np.log(np.maximum(word_occurrence_valid, 1e-10))

        indices = np.where(word_occurrence > 0)[0]

        if len(indices) == 0:
            return {}

        sorted_occurrences = -np.sort(-word_occurrence[indices])[
            : min(n, len(indices))
        ]  # Selecting top n neighbors
        total_value = np.sum(sorted_occurrences)

        if total_value == 0:
            return {}

        sorted_indices = np.argsort(-word_occurrence[indices])[: min(n, len(indices))]
        sorted_occurrences = sorted_occurrences / total_value
        wanted_words = words[indices][sorted_indices]

        return dict(zip(wanted_words, sorted_occurrences, strict=False))

    edges = {}
    nodes_total = {}

    if strategy != "ppmi":
        raise ValueError("Only ppmi strategy is supported")

    df, vocab, total, col_totals = co_occurrence(
        tokenized_data, cues, two_gram, two_grams,
    )

    # Initialize edges and nodes_total for all cues
    for w in cues:
        if w not in col_totals:
            col_totals[w] = 0
            edges[w] = {}
            nodes_total[w] = 0

    # Calculate PMI for each cue
    for w in cues:
        if col_totals.get(w, 0) == 0:
            continue

        edges[w] = get_pmi(w, df, col_totals, total, cues, n)
        nodes_total[w] = sum(edges[w].values())

    edge_index, edge_weight, vocab_mapping, _ = get_edge_index_from_edges(
        edges,
        vocab_mapping,
        fill,
        add_self_loop,
        nodes_total,
        normalize=True,
        sample_negatives=False,
    )

    return edge_index, edge_weight, vocab_mapping


def get_two_grams_from_swow(version: int, lemmatizer: WordNetLemmatizer) -> list[str]:
    """Get two-grams from SWOW data.

    Args:
        version: SWOW data version
        lemmatizer: WordNetLemmatizer instance

    Returns:
        List of two-grams

    """
    df = get_swow_data(version)
    if len(df) == 0:
        return []
    df["response"] = [lemmatizer.lemmatize(str(r)) for r in df.response]
    df["cue"] = [lemmatizer.lemmatize(str(c)) for c in df.cue]
    two_grams = [w for w in df["cue"] if len(str(w).split()) == 2]

    return two_grams


def get_swow_words(version: int, lemmatizer: WordNetLemmatizer) -> list[str]:
    """Get words from SWOW data.

    Args:
        version: SWOW data version
        lemmatizer: WordNetLemmatizer instance

    Returns:
        List of words

    """
    df = get_swow_data(version)
    if len(df) == 0:
        return []
    df["cue"] = [lemmatizer.lemmatize(str(c).lower()) for c in df.cue]
    words = list(df.cue.unique())

    return words


@time_function
def get_sentiments(
    tokenized_data: list[list[str]], sentences: list[str], cues: list[str],
) -> dict[str, float]:
    """Calculate sentiment scores for cues.

    Args:
        tokenized_data: List of tokenized sentences
        token_list: List of token lists
        lemmatized_tokens: List of lemmatized token lists
        sentences: Original sentences
        cues: List of cues
        two_grams: List of two-grams
        tokenizer: Tokenizer instance
        model_name: Name of the model
        max_length: Maximum sequence length

    Returns:
        Dictionary mapping cues to sentiment scores

    """
    from collections import defaultdict

    cue_sentiments = defaultdict(list)
    sid = SentimentIntensityAnalyzer()

    for i, tokenized_sentence in enumerate(tokenized_data):
        sentiment = sid.polarity_scores(sentences[i])

        for token in tokenized_sentence:
            if token in cues:
                cue_sentiments[token].append(sentiment["compound"])

    cue_sentiments = {k: np.mean(v) if v else 0.0 for k, v in cue_sentiments.items()}

    return cue_sentiments


def store_sentiments(
    year: int = 2000, data_name: str = "coha", **kwargs,
) -> dict[str, float]:
    """Store sentiment scores for cues.

    Args:
        model_name: Name of the model
        year: Year to process
        data_name: Name of the dataset
        **kwargs: Additional arguments

    Returns:
        Dictionary mapping cues to sentiment scores

    """
    data_path = kwargs["data_path"]
    swow_version = kwargs["swow_version"]
    token_strategy = kwargs["token_strategy"]
    two_gram = kwargs["two_gram"]

    lemmatizer = WordNetLemmatizer()
    two_grams = get_two_grams_from_swow(swow_version, lemmatizer) if two_gram else []
    swow_cues = get_swow_words(swow_version, lemmatizer)
    new_one_grams, new_two_grams = get_two_grams()
    two_grams.extend(new_two_grams)
    two_grams = list(set(two_grams))

    data_config = {"data_features": {data_name: {"year": [year]}}}

    data = get_data(data_name, data_path, data_features=data_config["data_features"])
    sentences = sentensize_articles(data)

    tokenizer = MyTokenizer()
    token_list, lemmatized_tokens, tokenized_data = tokenizer.tokenize_batch(
        sentences, True, "simple",
    )

    word_count, cues = get_word_count(
        tokenized_data, sentences, swow_cues, two_gram, token_strategy, k=10000,
    )

    cues = [
        w
        for w in cues
        if len(w) >= MIN_LENGTH and w != "nt" and w in word_count and word_count[w] >= MIN_FRQUENCY
    ]
    cues.extend([w for w in swow_cues if len(w) == 2 or w not in word_count])
    cues.extend(new_one_grams + two_grams)
    cues = list(set(cues))

    print(f"Got all cues: {len(cues)}")

    sentiments = get_sentiments(tokenized_data, sentences, cues)

    return sentiments


def get_encodings(
    tokenized_data: list[list[str]],
    token_list: list[list[str]],
    lemmatized_tokens: list[list[str]],
    sentences: list[str],
    cues: list[str],
    two_grams: list[str],
    tokenizer,
    model_name: str,
    max_length: int = 200,
) -> tuple[list[list[Any]], list[Any]]:
    """Get encodings for sentences containing cues.

    Args:
        tokenized_data: List of tokenized sentences
        token_list: List of token lists
        lemmatized_tokens: List of lemmatized token lists
        sentences: Original sentences
        cues: List of cues
        two_grams: List of two-grams
        tokenizer: Tokenizer instance
        model_name: Name of the model
        max_length: Maximum sequence length

    Returns:
        sentence_positions: List of positions for cues in sentences
        all_encodings: List of encoded sentences

    """
    special_token = tokenizer.special_tokens_map["sep_token"]
    special_token_id = tokenizer.encode(special_token, add_special_tokens=False)[0]

    sentence_positions = []
    all_encodings = []

    for i, tokenized_sentence in enumerate(tokenized_data):
        sentence_encodings = None
        curr_positions = []
        has_cue = False

        for j, token in enumerate(tokenized_sentence):
            if token in cues:
                has_cue = True

                # Encode sentence if not already done
                if sentence_encodings is None:
                    sentence_encodings = get_sentence_encodings(
                        tokenizer, sentences[i], max_length=max_length,
                    )

                try:
                    i_th_positions = get_word_position(
                        token,
                        sentence_encodings,
                        sentences[i],
                        lemmatized_tokens[i],
                        token_list[i],
                        special_token_id,
                        tokenizer,
                        model_name=model_name,
                    )
                    curr_positions.append(i_th_positions)
                except Exception as e:
                    print(f"Position not retrieved for token '{token}': {e}")

            if j < len(tokenized_sentence) - 1:
                two_gram_text = f"{token} {tokenized_sentence[j + 1]}"
                if two_gram_text in two_grams and two_gram_text in sentences[i]:
                    has_cue = True

                    if sentence_encodings is None:
                        sentence_encodings = get_sentence_encodings(
                            tokenizer, sentences[i], max_length=max_length,
                        )

                    try:
                        i_th_positions = get_word_position(
                            two_gram_text,
                            sentence_encodings,
                            sentences[i],
                            lemmatized_tokens[i],
                            token_list[i],
                            special_token_id,
                            tokenizer,
                            model_name=model_name,
                        )
                        curr_positions.append(i_th_positions)
                    except Exception as e:
                        print(
                            f"Position not retrieved for two-gram '{two_gram_text}': {e}",
                        )

        if has_cue and curr_positions:
            sentence_positions.append(curr_positions)
            all_encodings.append(sentence_encodings)

    return sentence_positions, all_encodings


def store_encoding_data(
    model_name: str,
    year: int = 2000,
    max_length: int = 200,
    data_name: str = "coha",
    **kwargs,
) -> tuple[list[list[Any]], list[Any]]:
    """Store encoding data for sentences containing cues.

    Args:
        model_name: Name of the model
        year: Year to process
        max_length: Maximum sequence length
        data_name: Name of the dataset
        **kwargs: Additional arguments

    Returns:
        sentence_positions: List of positions for cues in sentences
        all_encodings: List of encoded sentences

    """
    data_path = kwargs["data_path"]
    swow_version = kwargs["swow_version"]
    token_strategy = kwargs["token_strategy"]
    two_gram = kwargs["two_gram"]

    lemmatizer = WordNetLemmatizer()
    two_grams = get_two_grams_from_swow(swow_version, lemmatizer) if two_gram else []
    swow_cues = get_swow_words(swow_version, lemmatizer)
    new_one_grams, new_two_grams = get_two_grams()
    two_grams.extend(new_two_grams)
    two_grams = list(set(two_grams))

    data_config = {"data_features": {data_name: {"year": [year]}}}

    data = get_data(data_name, data_path, data_features=data_config["data_features"])
    sentences = sentensize_articles(data)

    tokenizer = MyTokenizer()
    token_list, lemmatized_tokens, tokenized_data = tokenizer.tokenize_batch(
        sentences, True, "simple",
    )

    word_count, cues = get_word_count(
        tokenized_data, sentences, swow_cues, two_gram, token_strategy, k=10000,
    )

    # Filtering cues
    cues = [
        w
        for w in cues
        if len(w) >= MIN_LENGTH and w != "nt" and w in word_count and word_count[w] >= MIN_FRQUENCY
    ]
    cues.extend([w for w in swow_cues if len(w) == 2 or w not in word_count])
    cues.extend(new_one_grams + two_grams)
    cues = list(set(cues))

    tokenizer = get_tokenizer(model_name)
    sentence_positions, all_encodings = get_encodings(
        tokenized_data,
        token_list,
        lemmatized_tokens,
        sentences,
        cues,
        two_grams,
        tokenizer,
        model_name,
        max_length,
    )

    return sentence_positions, all_encodings


def store_embedding_data(
    model_name, year=2000, max_length=200, data_name="coha", **kwargs,
):
    """Generate and store word embeddings from contextual data.

    Args:
        model_name (str): Name of the pretrained model to use
        year (int): Year of the data to process
        max_length (int): Maximum sequence length for tokenization
        data_name (str): Name of the dataset ('coha' or 'nyt')
        kwargs: Additional parameters including data_path, swow_version, etc.

    Returns:
        tuple: (word_embeddings, word_counts, normalized_embeddings)

    """
    data_path = kwargs["data_path"]
    swow_version = kwargs["swow_version"]
    token_strategy = kwargs["token_strategy"]
    two_gram = kwargs["two_gram"]
    device_name = kwargs["device_name"]

    lemmatizer = WordNetLemmatizer()

    two_grams = get_two_grams_from_swow(swow_version, lemmatizer) if two_gram else []
    swow_cues = get_swow_words(swow_version, lemmatizer)
    new_one_grams, new_two_grams = get_two_grams()

    two_grams = list(set(two_grams + new_two_grams))

    data_features = {"data_features": {data_name: {"year": [year]}}}

    data = get_data(data_name, data_path, data_features=data_features["data_features"])
    sentences = sentensize_articles(data)

    tokenizer = MyTokenizer()
    _, _, tokenized_data = tokenizer.tokenize_batch(sentences, True, "simple")

    word_count, cues = get_word_count(
        tokenized_data, sentences, swow_cues, two_gram, token_strategy, k=10000,
    )

    cues = [
        w
        for w in cues
        if len(w) >= MIN_LENGTH and w != "nt" and w in word_count and word_count[w] >= MIN_FRQUENCY
    ] + [w for w in swow_cues if len(w) == 2 or w not in word_count]
    cues += new_one_grams + two_grams
    cues = list(set(cues))  # Remove duplicates

    tokenizer = get_tokenizer(model_name)

    encoding_file = (
        f"data/SWOW_prediction/{data_name}_{year}_encodings_{model_name}.pkl"
    )
    try:
        sentence_positions, all_encodings = pickle.load(open(encoding_file, "rb"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Encoding file {encoding_file} not found. Run the encoding step first.",
        )

    final_cue_embeddings, final_word_counts = get_word_embedding(
        all_encodings,
        sentence_positions,
        model_name,
        cues,
        device=device_name,
        word_embeddings={},
        word_counts={},
    )

    embedding_data = {
        w: final_cue_embeddings[w] / final_word_counts[w]
        for w in final_cue_embeddings
        if final_word_counts[w] > 0
    }

    return final_cue_embeddings, final_word_counts, embedding_data


def store_textual_data(
    model_name: str,
    year: int = 2000,
    max_length: int = 200,
    data_name: str = "coha",
    **kwargs,
) -> dict[str, Any]:
    """Generate and store textual data for graph construction.

    Args:
        model_name: Name of the model
        year: Year to process
        max_length: Maximum sequence length
        data_name: Name of the dataset
        **kwargs: Additional arguments

    Returns:
        Dictionary containing textual data components

    """
    token_strategy = kwargs["token_strategy"]
    two_gram = kwargs["two_gram"]
    swow_version = kwargs.get("swow_version", 1)
    data_path = kwargs.get("data_path", "")
    graph_strategy = kwargs.get("graph_strategy", "ppmi")
    graph_version = kwargs.get("graph_version", 1)
    fill = kwargs.get("fill", "add")
    add_self_loops = kwargs.get("add_self_loops", True)
    node_neighbors = kwargs.get("node_neighbors", 25)

    # years = year_range.get(data_name, [])
    # i = years.index(year) if year in years else 0

    lemmatizer = WordNetLemmatizer()
    two_grams = get_two_grams_from_swow(swow_version, lemmatizer) if two_gram else []
    swow_cues = get_swow_words(swow_version, lemmatizer)
    new_one_grams, new_two_grams = get_two_grams()
    two_grams.extend(new_two_grams)
    two_grams = list(set(two_grams))

    data_features = {"data_features": {data_name: {"year": [year]}}}
    data = get_data(data_name, data_path, data_features=data_features["data_features"])
    sentences = sentensize_articles(data)

    tokenizer = MyTokenizer()
    _, _, tokenized_data = tokenizer.tokenize_batch(sentences, True, "simple")
    word_count, cues = get_word_count(
        tokenized_data, sentences, swow_cues, two_gram, token_strategy, k=10000,
    )

    cues = [
        w
        for w in cues
        if len(w) >= MIN_LENGTH and w != "nt" and w in word_count and word_count[w] >= MIN_FRQUENCY
    ]
    cues.extend([w for w in swow_cues if len(w) == 2 or w not in word_count])
    cues.extend(new_one_grams + two_grams)
    cues = list(set(cues))

    tokenizer = get_tokenizer(model_name)
    vocab_mapping = None
    try:
        _, _, embedding_data = pickle.load(
            open(f"data/SWOW_prediction/{year}_{data_name}_emb_{model_name}.pkl", "rb"),
        )
        cues = [c for c in cues if c in embedding_data]
    except (FileNotFoundError, EOFError):
        print(
            "Warning: Embedding file not found or corrupted. Processing may be incomplete.",
        )
        embedding_data = {}

    edge_index, edge_weight, vocab_mapping = get_data_graph(
        tokenized_data,
        cues,
        graph_strategy,
        graph_version,
        vocab_mapping=vocab_mapping,
        fill=fill,
        add_self_loop=add_self_loops,
        n=node_neighbors,
        two_gram=two_gram,
        two_grams=two_grams,
    )

    cues = [c for c in cues if c in vocab_mapping]
    embedding_data = {w: e for w, e in embedding_data.items() if w in cues}

    return {
        "embedding_data": embedding_data,
        "vocab_mapping": vocab_mapping,
        "edge_index": edge_index,
        "edge_weight": edge_weight,
        "cues": cues,
    }


def save_word_count(
    data_name: str = "coha",
    model_name: str = "bert-base-uncased",
    storing_dir="./data/SWOW_prediction",
) -> None:
    """Save word count data to CSV file.

    Args:
        data_name: Name of the dataset
        model_name: Name of the model

    """
    word_count_df = pd.DataFrame()
    words = []
    counts = []
    years = []

    year_range = range(1850, 2010, 10) if data_name == "coha" else range(1987, 2008)

    for year in year_range:
        try:
            final_cue_embeddings, final_word_counts, _ = pickle.load(
                open(
                    os.path.join(
                        storing_dir, f"{year}_{data_name}_emb_{model_name}.pkl",
                    ),
                    "rb",
                ),
            )
            words += list(final_word_counts.keys())
            counts += list(final_word_counts.values())
            years += [year] * len(final_word_counts)
        except (FileNotFoundError, EOFError):
            print(f"Warning: Could not load data for year {year}. Skipping.")
            continue

    word_count_df["word"] = words
    word_count_df["count"] = counts
    word_count_df["year"] = years
    word_count_df.to_csv(
        os.path.join(storing_dir, f"{data_name}_word_count.csv"), index=False,
    )


def get_textual_data_input_with_sections(
    model_name: str, max_length: int = 200, data_name: str = "", **kwargs,
) -> dict[int, list]:
    """Load textual data for all years in specified dataset.

    Args:
        model_name: Name of the model
        max_length: Maximum sequence length
        data_name: Name of the dataset
        **kwargs: Additional arguments

    Returns:
        Dictionary mapping indices to data lists

    """
    years = kwargs.get("data_features", {}).get(data_name, {}).get("year", [])
    all_data = {}
    storing_dir = kwargs.get("storing_dir", "./data/SWOW_prediction")

    for i, year in enumerate(years):
        store_dir = os.path.join(
            storing_dir, f"data_{data_name}_{year}_{model_name}.pkl",
        )

        if not os.path.exists(store_dir):
            print(f"Warning: {store_dir} does not exist. Run data preprocessing first.")
            continue

        try:
            d = pickle.load(open(store_dir, "rb"))
            embedding_data = d["embedding_data"]
            vocab_mapping = d["vocab_mapping"]
            edge_index = d["edge_index"]
            edge_weight = d["edge_weight"]
            cues = d["cues"]
            all_data[i] = [
                embedding_data,
                edge_index,
                edge_weight,
                year,
                vocab_mapping,
                cues,
            ]
        except (EOFError, KeyError):
            print(f"Error loading data from {store_dir}. File may be corrupted.")
            continue

    return all_data


def construct_property_dataset(
    model_name: str,
    max_length: int = 200,
    data_name_store: str = "",
    data_path: str = "",
    token_strategy: str = "frequency",
    graph_strategy: str = "ppmi",
    device_name: str = "cpu",
    graph_version: int = 2,
    swow_version: int = 1,
    fill: str = "add",
    add_self_loops_cmd: bool = True,
    node_neighbors: int = 25,
    normalize: bool = True,
    take_log: bool = False,
    sample_negative: bool = False,
    negative_sample_num: int | None = None,
    feature: str = "link",
    two_gram: bool = True,
    **kwargs,
) -> tuple[list, dict, dict]:
    """Construct dataset for property prediction.

    Args:
        model_name: Name of the model
        max_length: Maximum sequence length
        data_name_store: Name to use for storing data
        data_path: Path to data
        token_strategy: Strategy for tokenization
        graph_strategy: Strategy for graph construction
        device_name: Device to use for computation
        graph_version: Version of graph construction
        swow_version: Version of SWOW data
        fill: Method for filling self-loops
        add_self_loops_cmd: Whether to add self-loops
        node_neighbors: Number of neighbors to consider
        normalize: Whether to normalize edge weights
        take_log: Whether to apply log transformation
        sample_negative: Whether to sample negative examples
        negative_sample_num: Number of negative samples
        feature: Feature to predict ('previous_link' or 'polarity')
        two_gram: Whether to include two-grams
        **kwargs: Additional arguments

    Returns:
        Tuple containing training data components

    """
    if feature not in ["previous_link", "polarity"]:
        raise ValueError(
            "Feature must be either 'previous_link' (moral relevance) or 'polarity' (moral polarity)",
        )

    data_name = data_name_store
    year = kwargs.get("data_features", {}).get(data_name, {}).get("year", [0])[0]
    storing_dir = kwargs.get("store_dir", "./data/SWOW_prediction")

    stored_data_dir = (
        f"property_{feature}_{data_name_store}_{year}_{model_name}_"
        f"{token_strategy}_{graph_strategy}_{graph_version}_{swow_version}_{fill}_"
        f"{add_self_loops_cmd}_{two_gram}_{node_neighbors}_{max_length}_{take_log}.pkl"
    )
    stored_data_dir = os.path.join(storing_dir, stored_data_dir)

    # Load response data for moral foundations
    try:
        response_mfd = pd.read_csv(f"data/SWOWEN/moralized_v{swow_version}_mfd.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Moral foundations data file not found at data/SWOWEN/moralized_v{swow_version}_mfd.csv",
        )

    # Process property data based on feature
    if feature == "previous_link":  # Moral relevance
        property_data = response_mfd[["previous_link", "cue"]]
        property_data = property_data.loc[~pd.isna(property_data["previous_link"])]
        zeros = property_data.loc[property_data["previous_link"] == 0]
        property_data = property_data.loc[property_data["previous_link"] != 0]

        if take_log:
            property_data["previous_link"] = np.log(
                property_data["previous_link"] * 100 + 1,
            )

        if sample_negative and negative_sample_num is not None:
            zeros = zeros.sample(negative_sample_num, random_state=42)
            property_data = pd.concat([property_data, zeros], ignore_index=True)

    elif feature == "polarity":  # Moral polarity
        property_data = response_mfd[["polarity", "cue", "pos_score", "neg_score"]]
        property_data = property_data.loc[~pd.isna(property_data["polarity"])]
        zeros = property_data.loc[property_data["polarity"] == 0.5]  # pos == neg
        property_data = property_data.loc[
            property_data["polarity"] != 0.5
        ]  # pos != neg

        if take_log:
            property_data["polarity"] = np.log(property_data["polarity"] * 100 + 1)
            zeros["polarity"] = np.log(zeros["polarity"] * 100 + 1)

        if sample_negative and negative_sample_num is not None:
            zeros = zeros.sample(negative_sample_num, random_state=42)
            property_data = pd.concat([property_data, zeros], ignore_index=True)

    cues = list(property_data.cue)

    # Get textual data
    all_data = get_textual_data_input_with_sections(
        model_name,
        max_length,
        data_name,
        data_path=data_path,
        token_strategy=token_strategy,
        graph_strategy=graph_strategy,
        device_name=device_name,
        graph_version=graph_version,
        fill=fill,
        add_self_loops_cmd=add_self_loops_cmd,
        node_neighbors=node_neighbors,
        swow_version=swow_version,
        two_gram=two_gram,
        data_features=kwargs.get("data_features", {}),
        storing_dir=storing_dir,
    )

    if not all_data:
        raise ValueError("No textual data found. Please run data preprocessing first.")

    vocab_mapping_train = all_data[0][-2]  # Modern time point
    cues = [c for c in cues if c in vocab_mapping_train]
    vocab_mapping_reverse = {v: k for k, v in vocab_mapping_train.items()}

    targets = {
        w: property_data.loc[property_data.cue == w][feature].iloc[0] for w in cues
    }
    target_tensor = torch.tensor(
        [property_data.loc[property_data.cue == w][feature].iloc[0] for w in cues],
    )
    target_node_index = torch.tensor([vocab_mapping_train[w] for w in cues])

    indices = np.arange(len(targets))
    random_states = [42, 1231, 523, 432, 21]
    all_train, test = train_test_split(
        indices, test_size=0.2, random_state=42, shuffle=True,
    )

    swow_data_sets_all = []
    for random_state in random_states:
        train, dev = train_test_split(
            all_train, test_size=0.25, random_state=random_state, shuffle=True,
        )

        all_words = np.array(cues)
        train_words = all_words[train]
        dev_words = all_words[dev]
        test_words = all_words[test]
        train_target = target_tensor[train]
        dev_target = target_tensor[dev]
        test_target = target_tensor[test]

        swow_data_sets = {
            "train": [target_node_index[train], train_target, train_words],
            "dev": [target_node_index[dev], dev_target, dev_words],
            "test": [target_node_index[test], test_target, test_words],
        }
        swow_data_sets_all.append(swow_data_sets)

    (
        embedding_data,
        text_edge_index,
        text_edge_weight,
        year,
        vocab_mapping,
        data_cues,
    ) = all_data[0]

    try:
        dimension = next(iter(embedding_data.values())).shape[0]
    except (StopIteration, AttributeError):
        raise ValueError("No valid embeddings found in data.")

    embedding_tensor = torch.zeros((len(vocab_mapping_reverse), dimension))
    for i in range(len(vocab_mapping_reverse)):
        word = vocab_mapping_reverse.get(i)
        if word and word in embedding_data:
            embedding_tensor[i] = torch.tensor(embedding_data[word])

    edge_indices = [
        k
        for k in range(text_edge_index.shape[1])
        if (
            vocab_mapping_reverse.get(int(text_edge_index[0, k])) in cues
            and vocab_mapping_reverse.get(int(text_edge_index[1, k])) in cues
        )
    ]

    text_edge_index = text_edge_index[:, edge_indices]
    text_edge_weight = text_edge_weight[edge_indices]

    textual_data = {
        "embedding": embedding_tensor,
        "train": [text_edge_index, text_edge_weight],
    }

    pickle.dump(
        [swow_data_sets_all, textual_data, vocab_mapping], open(stored_data_dir, "wb"),
    )

    return swow_data_sets_all, textual_data, vocab_mapping


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Process textual data for word embeddings and graph construction",
    )
    parser.add_argument("--year", type=int, default=2000, help="Year to process")
    parser.add_argument(
        "--model", type=str, default="bert-base-uncased", help="Model to use",
    )
    parser.add_argument("--data", type=str, default="coha", help="Dataset name")
    parser.add_argument(
        "--function",
        type=str,
        default="encoding",
        choices=["encoding", "embedding", "graph", "sentiment"],
        help="Function to execute",
    )
    parser.add_argument(
        "--length", type=int, default=200, help="Maximum sequence length",
    )
    parser.add_argument(
        "--node_neighbors",
        type=int,
        default=100,
        help="Number of neighbors for graph construction",
    )
    parser.add_argument(
        "--data_path", type=str, default="./data/COHA.zip", help="Path to data file",
    )
    parser.add_argument(
        "--store_dir",
        type=str,
        default="./data/SWOW_prediction",
        help="Directory to store results",
    )

    args = parser.parse_args()

    # Check if data path exists
    if not os.path.exists(args.data_path):
        raise argparse.ArgumentTypeError(
            f"Data path: {args.data_path} is not a valid path",
        )

    # Create storage directory
    os.makedirs(args.store_dir, exist_ok=True)

    # Determine device
    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    # Set year ranges by dataset
    year_range = {
        "coha": list(np.arange(2000, 1840, -10)),
        "nyt": list(np.arange(2007, 1986, -1)),
    }

    # Common hyperparameters
    token_strategy = "frequency"
    graph_strategy = "ppmi"
    graph_version = 2
    swow_version = 1
    fill = "add"
    add_self_loops = True

    # Execute requested function
    if args.function == "encoding":
        sentence_positions, all_encodings = store_encoding_data(
            args.model,
            year=args.year,
            max_length=args.length,
            data_name=args.data,
            data_path=args.data_path,
            token_strategy=token_strategy,
            graph_strategy=graph_strategy,
            device_name=device_name,
            graph_version=graph_version,
            fill=fill,
            add_self_loops=add_self_loops,
            node_neighbors=args.node_neighbors,
            swow_version=swow_version,
            two_gram=True,
        )

        output_path = os.path.join(
            args.store_dir, f"{args.data}_{args.year}_encodings_{args.model}.pkl",
        )
        with open(output_path, "wb") as f:
            pickle.dump((sentence_positions, all_encodings), f)
        print(f"Saved encodings to {output_path}")

    elif args.function == "embedding":
        final_cue_embeddings, final_word_counts, embedding_data = store_embedding_data(
            args.model,
            year=args.year,
            max_length=args.length,
            data_name=args.data,
            data_path=args.data_path,
            token_strategy=token_strategy,
            graph_strategy=graph_strategy,
            device_name=device_name,
            graph_version=graph_version,
            fill=fill,
            add_self_loops=add_self_loops,
            node_neighbors=args.node_neighbors,
            swow_version=swow_version,
            two_gram=True,
        )

        output_path = os.path.join(
            args.store_dir, f"{args.year}_{args.data}_emb_{args.model}.pkl",
        )
        with open(output_path, "wb") as f:
            pickle.dump((final_cue_embeddings, final_word_counts, embedding_data), f)
        print(f"Saved embeddings to {output_path}")

    elif args.function == "graph":
        textual_data = store_textual_data(
            args.model,
            year=args.year,
            max_length=args.length,
            data_name=args.data,
            data_path=args.data_path,
            token_strategy=token_strategy,
            graph_strategy=graph_strategy,
            device_name=device_name,
            graph_version=graph_version,
            fill=fill,
            add_self_loops=add_self_loops,
            node_neighbors=args.node_neighbors,
            swow_version=swow_version,
            two_gram=True,
        )

        # i = year_range[args.data].index(args.year) if args.year in year_range[args.data] else 0
        output_path = os.path.join(
            args.store_dir, f"data_{args.data}_{args.year}_{args.model}.pkl",
        )
        with open(output_path, "wb") as f:
            pickle.dump(textual_data, f)
        print(f"Saved graph data to {output_path}")

    elif args.function == "sentiment":
        sentiments = store_sentiments(
            year=args.year,
            data_name=args.data,
            data_path=args.data_path,
            token_strategy=token_strategy,
            graph_strategy=graph_strategy,
            device_name=device_name,
            graph_version=graph_version,
            fill=fill,
            add_self_loops=add_self_loops,
            node_neighbors=args.node_neighbors,
            swow_version=swow_version,
            two_gram=True,
        )

        output_path = os.path.join(
            args.store_dir, f"sentiments_{args.data}_{args.year}_{args.model}.pkl",
        )
        with open(output_path, "wb") as f:
            pickle.dump(sentiments, f)
        print(f"Saved sentiment data to {output_path}")
