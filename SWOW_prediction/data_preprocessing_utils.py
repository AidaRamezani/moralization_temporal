import gc
import io
import os
import pickle
import warnings
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


def get_swow_data(version: int = 1) -> pd.DataFrame:
    """Load SWOW (Small World of Words) data from CSV file.

    Args:
        version: The version number of the SWOW dataset to use

    Returns:
        DataFrame containing cue-response associations with mean total counts

    """
    if not os.path.exists("data/SWOWEN/responses_R1.csv"):
        return pd.DataFrame()
    df = pd.read_csv(f"data/SWOWEN/responses_R{version}.csv")
    return df.groupby(["cue", "response"])["total"].mean().reset_index()


def get_coha_path() -> str:
    """Get the file path to the COHA corpus."""
    return "data/COHA.zip"


def get_coha_articles(year: int) -> tuple[list[list[str]], list[str], list[str]]:
    """Extract articles from the COHA corpus for a specific year.

    Args:
        year: The year to extract articles from

    Returns:
        Tuple containing (articles, article_texts, article_genres)

    """
    coha_path = get_coha_path()
    encoding = "ISO-8859-1"
    articles = []
    article_texts = []
    article_genres = []

    with ZipFile(coha_path) as cf_zip:
        files = cf_zip.namelist()
        file = next(f for f in files if str(year) in f)

        with ZipFile(io.BytesIO(cf_zip.read(file))) as fzip:
            for article_name in fzip.namelist():
                with fzip.open(article_name) as f:
                    article = f.readlines()[1:]
                    article = [s.decode(encoding).split("\t")[1] for s in article]
                    articles.append(article)
                    article_texts.append(" ".join(article))
                    genre = article_name[: article_name.index("_")]
                    article_genres.append(genre)

    return articles, article_texts, article_genres


def get_coha_data(year: int) -> list[str]:
    """Get article texts from COHA for a specific year.

    Args:
        year: The year to get data for

    Returns:
        List of article texts

    """
    _, article_texts, _ = get_coha_articles(year)
    return article_texts


def get_nyt_data(year: int, data_path: str) -> list[str]:
    """Load New York Times data for a specific year.

    Args:
        year: The year to load data for
        data_path: Path to the directory containing NYT data

    Returns:
        List of article texts

    """
    year_data_path = os.path.join(data_path, f"{year}-lemmatized.pkl")
    with open(year_data_path, "rb") as f:
        data = pickle.load(f)
    return [item["article"] for item in data]


def get_custom_data(time_point, data_path: str) -> list[str]:
    """General function to load custom data at a specific time point.

    Args:
        time_point: The time point to load data for. Your dataset should be diachronic. We train the model on the
        last time point and test on all other time points.

        data_path: Path to the directory containing your custom dataset.

    Returns:
        List of article texts

    Note:
        - This is a sample function that reads all .txt files from a specific directory, and returns them.
        If you have a different format, please modify this function accordingly.

    """
    time_point_directory = os.path.join(data_path, str(time_point))
    if not os.path.exists(time_point_directory):
        raise FileNotFoundError(f"Directory {time_point_directory} does not exist.")
    texts = []
    for filename in os.listdir(time_point_directory):
        if filename.endswith(".txt"):
            with open(
                os.path.join(time_point_directory, filename), encoding="utf-8",
            ) as file:
                text = file.read()
                texts.append(text)
    return texts


def get_data(data_name: str, data_path: str, **kwargs) -> list[str]:
    """Load data from the specified source.

    Args:
        data_name: The name of the dataset ('coha' or 'nyt')
        data_path: Path to the data directory
        **kwargs: Additional arguments including data_features

    Returns:
        List of texts from the requested dataset

    """
    data_features = kwargs["data_features"][data_name]
    years = data_features.get("year", [])

    if not years:
        raise ValueError(f"No years specified for {data_name} data")

    year = years[0]  # Use the first year as the training year

    if data_name == "coha":
        return get_coha_data(year)
    if data_name == "nyt":
        return get_nyt_data(year, data_path)
    warnings.warn(f"Unknown data source: {data_name}. Using custom data.")
    return get_custom_data(year, data_path)


def get_two_grams() -> tuple[list[str], list[str]]:
    """Get curated sets of words and bigrams from various sources.

    Returns:
        Tuple containing (one_grams, two_grams)

    """
    term_sources = {
        "tech_cues": (
            "data/moralization_terms/technologies_inventions_brands.csv",
            "Terms",
        ),
        "civil_terms": ("data/moralization_terms/civil_unrest.csv", None),
        "disease_cues": ("data/moralization_terms/diseases.csv", None),
        "epidemic_cues": ("data/moralization_terms/epidemics.csv", None),
        "political_cues": ("data/moralization_terms/political_figures.csv", None),
        "event_cues": ("data/moralization_terms/world_event.csv", None),
        "wars_cues": ("data/moralization_terms/wars_conflicts.csv", None),
    }

    all_term_collections = {}

    try:
        tech_df = pd.read_csv(term_sources["tech_cues"][0])
        all_term_collections["tech_cues"] = list(set(tech_df["Terms"]))
    except FileNotFoundError:
        print(f"Warning: File not found: {term_sources['tech_cues'][0]}")
        all_term_collections["tech_cues"] = []

    # Process president data (special case)
    try:
        president_df = pd.read_csv("data/president.csv")
        president_df["pre_name"] = [
            n.split()[-1].lower() for n in president_df["PRESIDENT"]
        ]
        president_df["vice_name"] = [
            n.split()[-1].lower() if not pd.isna(n) else n
            for n in president_df["VICE PRESIDENT"]
        ]
        president_df["first_name"] = [
            n.split()[0].lower() for n in president_df["PRESIDENT"]
        ]
        president_df["year1"] = [
            y.split("-")[0] if "-" in y else y for y in president_df["YEAR"]
        ]
        president_df["year2"] = [
            y.split("-")[1] if "-" in y else y for y in president_df["YEAR"]
        ]

        president_two_grams = [
            f"{first_name} {last_name}"
            for first_name, last_name in zip(
                president_df.first_name, president_df.pre_name, strict=False,
            )
        ]
        president_two_grams += [
            f"president {last_name}" for last_name in president_df.pre_name
        ]
        all_term_collections["president_cues"] = list(set(president_two_grams))
    except FileNotFoundError:
        print("Warning: President data file not found")
        all_term_collections["president_cues"] = []

    # Process the remaining term sources
    for term_name, (file_path, _) in term_sources.items():
        if term_name in all_term_collections:
            continue  # Skip already processed sources

        try:
            df = pd.read_csv(file_path)
            terms = []
            for _, row in df.iterrows():
                if pd.isna(row.get("Terms", "")):
                    continue
                row_terms = row["Terms"].split(",")
                terms.extend([s.lower().strip() for s in row_terms])
            all_term_collections[term_name] = terms
        except FileNotFoundError:
            print(f"Warning: Terms file not found: {file_path}")
            all_term_collections[term_name] = []

    all_cues = []
    for term_collection in all_term_collections.values():
        all_cues.extend(term_collection)

    all_cues = list(set(all_cues))

    one_grams = [x for x in all_cues if len(x.split()) == 1]
    two_grams = [x for x in all_cues if len(x.split()) == 2]

    try:
        with open("data/moralization_terms/cues.pkl", "wb") as f:
            pickle.dump(all_term_collections, f)
    except OSError:
        print("Warning: Could not save cues dictionary")

    return one_grams, two_grams


def get_tokenizer(model_name: str):
    """Get the appropriate tokenizer for the specified model.

    Args:
        model_name: Name of the model to get tokenizer for

    Returns:
        The initialized tokenizer

    """
    if "roberta" in model_name:
        from transformers import RobertaTokenizer

        return RobertaTokenizer.from_pretrained(model_name)
    # Default to BERT tokenizer
    from transformers import BertTokenizer

    return BertTokenizer.from_pretrained(model_name)


def get_model(model_name: str):
    """Get the appropriate model based on the model name.

    Args:
        model_name: Name of the model to load

    Returns:
        The initialized model

    """
    if "roberta" in model_name:
        from transformers import RobertaModel

        return RobertaModel.from_pretrained(model_name)
    if "clip" in model_name:
        from transformers import CLIPModel

        return CLIPModel.from_pretrained(model_name)
    # Default to BERT model
    from transformers import BertModel

    return BertModel.from_pretrained(model_name)


def get_sentence_encodings(
    tokenizer, context: str, max_length: int = 200,
) -> dict[str, Any]:
    """Encode a sentence using the provided tokenizer.

    Args:
        tokenizer: The tokenizer to use
        context: The text to encode
        max_length: Maximum sequence length

    Returns:
        Dictionary containing tokenized input features

    """
    return tokenizer(
        context,
        None,
        padding="max_length",
        truncation="longest_first",
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )


def get_word_position(
    word: str,
    encodings: dict[str, Any],
    context: str,
    lemmas: list[str],
    tokens: list[str],
    special_token_id: int,
    tokenizer,
    model_name: str = "bert-base-uncased",
) -> dict[str, Any]:
    """Get the position of a word or two-gram in the tokenized input.

    Args:
        word: The word or two-gram to locate
        encodings: The tokenized input
        context: The original text
        lemmas: List of lemmatized words
        tokens: List of tokens
        special_token_id: ID of the special token
        tokenizer: The tokenizer used
        model_name: Name of the model

    Returns:
        Dictionary containing position information

    """
    if len(word.split()) == 1:
        if word not in lemmas:
            return {"has_data": False}
        word_index = lemmas.index(word)
        word_token = tokens[word_index]

    else:
        word_indices = [lemmas.index(w) for w in word.split() if w in lemmas]
        if not word_indices:
            return {"has_data": False}
        word_token = " ".join([tokens[i] for i in word_indices])

    # Special handling for RoBERTa models
    if model_name == "roberta-base":
        token_id = []
        try:
            begin_index = context.index(word_token)
            end_index = begin_index + len(word_token)
            offsets_mapping = encodings["offset_mapping"]

            for index, positions in enumerate(offsets_mapping):
                if positions[0] >= begin_index and positions[1] <= end_index:
                    token_id.append(encodings["input_ids"][index])

            if not token_id:
                return {"has_data": False}
        except ValueError:
            return {"has_data": False}
    # Handling for other models
    else:
        token_inputs = tokenizer(word_token)["input_ids"]
        # Extract token IDs excluding special tokens
        try:
            token_id = token_inputs[1 : token_inputs.index(special_token_id)]
        except ValueError:
            token_id = token_inputs[1:]

        if not token_id:
            return {"has_data": False}

    return {"has_data": True, "data": [token_id, word]}


class CueDataset(Dataset):
    """Dataset for cue-related data processing."""

    def __init__(
        self, encoding_data: list[dict[str, Any]], position_data: list[list[list[Any]]],
    ):
        """Initialize the CueDataset.

        Args:
            encoding_data: List of tokenized inputs
            position_data: List of position information for each encoding

        """
        super(CueDataset, self).__init__()
        self.encoding_data = encoding_data
        self.position_data = position_data

    def __len__(self) -> int:
        """Return the number of items in the dataset."""
        return len(self.encoding_data)

    def __getitem__(
        self, index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[Any]]]:
        """Get a dataset item.

        Args:
            index: The index of the item to retrieve

        Returns:
            Tuple containing (input_ids, attention_mask, positions)

        """
        ids = self.encoding_data[index]["input_ids"]
        mask = self.encoding_data[index]["attention_mask"]
        positions = self.position_data[index]

        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(mask, dtype=torch.long),
            positions,
        )


def get_word_embedding(
    all_encodings: list[dict[str, Any]],
    position_results: list[list[dict[str, Any]]],
    model_name: str,
    og_cues: list[str],
    device: str,
    word_embeddings: dict[str, np.ndarray] = None,
    word_counts: dict[str, int] = None,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Get word embeddings for cues in the given encodings.

    Args:
        all_encodings: List of encodings for each sentence
        position_results: List of position information for each encoding
        model_name: Name of the model to use
        og_cues: List of original cues to extract embeddings for
        device: Device to run the model on ('cpu' or 'cuda')
        word_embeddings: Existing word embeddings dictionary to update
        word_counts: Existing word counts dictionary to update

    Returns:
        Tuple containing (updated_word_embeddings, updated_word_counts)

    """
    if word_embeddings is None:
        word_embeddings = {}
    if word_counts is None:
        word_counts = {}

    def collate_fn(data):
        return tuple(zip(*data, strict=False))

    model = get_model(model_name)
    if device != "cpu" and torch.cuda.is_available():
        model = model.to(device)

    all_cue_positions = []
    all_encodings_final = []

    for i, encoding in enumerate(all_encodings):
        cue_positions = [
            x["data"]
            for x in position_results[i]
            if x["has_data"] == True
            and x["data"][1] in og_cues
            and len(x["data"][0]) > 0
            and all(t in encoding["input_ids"] for t in x["data"][0])
        ]

        if cue_positions:
            all_cue_positions.append(cue_positions)
            all_encodings_final.append(encoding)

    cue_dataset = CueDataset(all_encodings_final, all_cue_positions)
    cue_dataloader = DataLoader(cue_dataset, batch_size=16, collate_fn=collate_fn)

    for batch in cue_dataloader:
        ids, mask, sentence_positions = batch

        ids = torch.stack(list(ids)).to(device)
        mask = torch.stack(list(mask)).to(device)

        outputs = model(ids, mask)[0]

        ids_np = ids.detach().cpu().numpy()
        outputs_np = outputs.detach().cpu().numpy()

        for b, embeddings in enumerate(outputs_np):
            positions_at_b = sentence_positions[b]

            # Handle CLIP models differently (use sentence embeddings)
            if "clip" in model_name:
                for positions in positions_at_b:
                    word = positions[1]

                    if word not in word_counts:
                        word_counts[word] = 1
                        word_embeddings[word] = np.copy(embeddings)
                    else:
                        word_counts[word] += 1
                        word_embeddings[word] += np.copy(embeddings)

            # Handle other models (use token embeddings)
            else:
                for positions in positions_at_b:
                    token_ids = positions[0]
                    word = positions[1]

                    try:

                        start_pos = list(ids_np[b]).index(token_ids[0])
                        token_positions = np.arange(
                            start_pos, start_pos + len(token_ids),
                        )

                        if np.max(token_positions) >= embeddings.shape[0]:
                            continue

                        word_embedding = embeddings[token_positions].mean(axis=0)

                        if word not in word_counts:
                            word_counts[word] = 1
                            word_embeddings[word] = np.copy(word_embedding)
                        else:
                            word_counts[word] += 1
                            word_embeddings[word] += np.copy(word_embedding)
                    except ValueError:
                        continue

    model.cpu()
    del model
    gc.collect()

    return word_embeddings, word_counts
