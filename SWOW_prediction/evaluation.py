import gc
import os
import pickle

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import r2_score

from SWOW_prediction.data_preprocessing import construct_property_dataset
from SWOW_prediction.loss import get_l1_regularization, get_likelihood_loss
from SWOW_prediction.models import BasicPropertyPredictor, ParameterPropertyPredictor


def evaluate(config_file, section="dev"):
    """Evaluate the model on specified section."""
    # Handle historical sections
    if section not in ["test", "dev"]:
        return evaluate_on_new(config_file, section)

    # Extract configuration parameters
    model_name = config_file["model_name"]
    data_name = config_file["data_name"]
    data_path = config_file["data_path"][data_name]
    device_name = config_file["device_name"]
    max_length = config_file["max_length"]
    token_strategy = config_file["token_strategy"]
    graph_strategy = config_file["graph_strategy"]
    graph_version = config_file["graph_version"]
    node_neighbors = config_file["node_neighbors"]
    swow_version = config_file["swow_version"]
    fill = config_file["fill"]
    add_self_loops = config_file["add_self_loops"]
    normalize_swow = config_file["normalize_swow"]
    negative_sampling = config_file["negative_sampling"]
    negative_sample_num = config_file["negative_sample_num"]
    property_name = config_file["property"]
    k = config_file["k"]
    take_log = config_file["take_log"]
    walking_type = config_file["walking_type"]
    reduce = config_file["reduce"]
    two_gram = config_file["two_gram"]
    train_section = config_file["train_section"]

    negative_sample_num = negative_sample_num if negative_sample_num > 0 else None

    device = torch.device(device_name)

    swow_data_sets, textual_data, vocab_mapping = construct_property_dataset(
        model_name,
        max_length,
        data_name,
        data_path,
        token_strategy,
        graph_strategy,
        device_name,
        graph_version,
        swow_version,
        fill,
        add_self_loops,
        node_neighbors,
        normalize_swow,
        take_log,
        negative_sampling,
        negative_sample_num,
        property_name,
        two_gram,
        k=k,
        walking_type=walking_type,
        data_features=config_file["data_features"],
        store_dir=config_file["store_dir"],
    )

    swow_data_sets = swow_data_sets[train_section]

    n = len(vocab_mapping)
    in_channels = textual_data["embedding"].shape[1]
    hidden_channels = config_file["graph_encoder_out_size"]
    out_channels = config_file["graph_encoder_out_size"]
    num_layers = config_file["graph_encoder_num_layers"]
    num_heads = config_file["graph_encoder_num_heads"]
    dropout = config_file["graph_encoder_dropout"]
    encoder_model_name = config_file["graph_encoder"]
    l1_lambda = config_file["l1_regularization"]
    loss_function_name = config_file["loss_function"]
    add_linear = config_file["add_linear"]

    if config_file["baseline"]:
        model = BasicPropertyPredictor(in_channels, dropout)
    else:
        model = ParameterPropertyPredictor(
            n,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            num_heads,
            dropout,
            encoder_model_name,
            reduce,
            add_linear,
        )

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        model.cuda(device)

    torch.autograd.set_detect_anomaly(True)
    loss_function = get_likelihood_loss(reduction="sum", device=device)

    train_index, train_target, train_words = swow_data_sets["train"]
    train_index = train_index.long().to(device)
    train_target = train_target.float().to(device)

    dev_index, dev_target, dev_words = swow_data_sets[section]
    dev_index = dev_index.long().to(device)
    dev_target = dev_target.float().to(device)

    embeddings = textual_data["embedding"].float().to(device)
    text_edge_index, text_edge_weight = textual_data["train"]
    text_edge_index = text_edge_index.long().to(device)
    text_edge_weight = text_edge_weight.float().to(device)

    # Loading pretrained model
    if not config_file["baseline"]:
        model_path = os.path.join(
            config_file["model_saving_dir"],
            f"{property_name}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_"
            f"graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_"
            f"fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{take_log}.pt",
        )
    else:
        model_path = os.path.join(
            config_file["model_saving_dir"],
            f"{property_name}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_"
            f"graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_"
            f"fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}.pt",
        )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    with torch.no_grad():
        if not config_file["baseline"]:
            output = model(embeddings, text_edge_index, text_edge_weight)
        else:
            output = model(embeddings)

        output_dev = output[dev_index]

        loss = loss_function(output_dev, dev_target) + get_l1_regularization(
            model, l1_lambda,
        )
        outputs = output_dev.cpu().numpy()
        targets = dev_target.cpu().numpy()

        spearman_corr = spearmanr(targets.reshape(-1), outputs.reshape(-1))
        r2 = r2_score(targets.reshape(-1), outputs.reshape(-1))

        print(f"Evaluation results on {section} set:")
        print(f"Spearman correlation: {spearman_corr[0]:.4f}")
        print(f"p-value: {spearman_corr[1]:.6f}")
        print(f"R2: {r2:.4f}")
        print(f"Loss: {loss.item():.6f}")

    # Clean up
    model.cpu()
    del model
    gc.collect()
    with torch.cuda.device(device_name):
        torch.cuda.empty_cache()

    # Save results
    df = pd.DataFrame({"targets": targets, "outputs": outputs, "words": dev_words})

    if not config_file["baseline"]:
        df_dir = os.path.join(
            config_file["test_results_path"],
            f"{property_name}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_"
            f"graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_"
            f"fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{section}.csv",
        )
    else:
        df_dir = os.path.join(
            config_file["test_results_path"],
            f"{property_name}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_"
            f"graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_"
            f"fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{section}.csv",
        )

    df.to_csv(df_dir, index=False)

    # Final correlation check
    spearman_corr = spearmanr(df["targets"], df["outputs"])
    print(f"Final Spearman correlation: {spearman_corr[0]:.4f}")
    print(f"Final p-value: {spearman_corr[1]:.6f}")

    return df


def evaluate_on_new(config_file, section=1):
    """Evaluate model on new historical data."""
    model_name = config_file["model_name"]
    data_name = config_file["data_name"]
    device_name = config_file["device_name"]
    token_strategy = config_file["token_strategy"]
    graph_strategy = config_file["graph_strategy"]
    graph_version = config_file["graph_version"]
    swow_version = config_file["swow_version"]
    fill = config_file["fill"]
    add_self_loops = config_file["add_self_loops"]
    property_name = config_file["property"]
    take_log = config_file["take_log"]
    reduce = config_file["reduce"]
    train_section = config_file["train_section"]
    store_dir = config_file["store_dir"]

    device = torch.device(device_name)
    data_file = os.path.join(store_dir, f"data_{data_name}_{section}_{model_name}.pkl")

    with open(data_file, "rb") as f:
        d = pickle.load(f)

    embedding_data = d["embedding_data"]
    vocab_mapping = d["vocab_mapping"]
    edge_index = d["edge_index"]
    edge_weight = d["edge_weight"]
    year = d["year"]

    vocab_mapping_reverse = {v: k for k, v in vocab_mapping.items()}
    embedding_tensor = np.array(
        [embedding_data[vocab_mapping_reverse[i]] for i in range(len(vocab_mapping))],
    )
    embedding_tensor = torch.tensor(embedding_tensor)

    torch.cuda.empty_cache()

    n = len(vocab_mapping)
    in_channels = embedding_tensor.shape[1]
    hidden_channels = config_file["graph_encoder_out_size"]
    out_channels = config_file["graph_encoder_out_size"]
    num_layers = config_file["graph_encoder_num_layers"]
    num_heads = config_file["graph_encoder_num_heads"]
    dropout = config_file["graph_encoder_dropout"]
    encoder_model_name = config_file["graph_encoder"]
    loss_function_name = config_file["loss_function"]
    add_linear = config_file["add_linear"]

    if config_file["baseline"]:
        model = BasicPropertyPredictor(in_channels, dropout)
    else:
        model = ParameterPropertyPredictor(
            n,
            in_channels,
            hidden_channels,
            out_channels,
            num_layers,
            num_heads,
            dropout,
            encoder_model_name,
            reduce,
            add_linear,
        )

    if torch.cuda.is_available():
        model.cuda(device)

    torch.autograd.set_detect_anomaly(True)

    embeddings = embedding_tensor.float().to(device)
    text_edge_index = edge_index.long().to(device)
    text_edge_weight = edge_weight.float().to(device)

    if not config_file["baseline"]:
        model_path = os.path.join(
            config_file["model_saving_dir"],
            f"{property_name}_{model_name}_{reduce}_{data_name}_{train_section}_{loss_function_name}_"
            f"graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_"
            f"fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}_{take_log}.pt",
        )
    else:
        model_path = os.path.join(
            config_file["model_saving_dir"],
            f"{property_name}_basic_{model_name}_{data_name}_{train_section}_{loss_function_name}_"
            f"graph_{graph_strategy}_graph_version_{graph_version}_swow_version_{swow_version}_"
            f"fill_{fill}_add_self_loops_{add_self_loops}_token_strategy_{token_strategy}.pt",
        )

    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to(device)

    with torch.no_grad():
        if not config_file["baseline"]:
            output = model(embeddings, text_edge_index, text_edge_weight)
        else:
            output = model(embeddings)

        outputs = output.cpu().numpy()

    model.cpu()
    del model
    gc.collect()
    with torch.cuda.device(device_name):
        torch.cuda.empty_cache()

    df = pd.DataFrame(
        {
            "outputs": outputs,
            "words": [vocab_mapping_reverse[i] for i in range(len(embedding_tensor))],
            "year": [year] * len(outputs),
        },
    )

    if not config_file["baseline"]:
        df_dir = os.path.join(
            config_file["test_results_path"],
            "time_series",
            f"testing_data_{data_name}_{model_name}_{section}_{property_name}_train_section_{train_section}.csv",
        )
    else:
        df_dir = os.path.join(
            config_file["test_results_path"],
            "time_series",
            f"testing_data_{data_name}_{model_name}_{section}_{property_name}_baseline_train_section_{train_section}.csv",
        )

    df.to_csv(df_dir, index=False)
