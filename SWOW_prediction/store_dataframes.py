import argparse
import os
import pickle

import pandas as pd
import yaml


def get_config(config_path="src/SWOW_prediction/config_features.yml"):
    """Load configuration from YAML file."""
    assert os.path.exists(config_path), f"Config file {config_path} does not exist."
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Store dataframes")

    parser.add_argument("--data", type=str, default="coha", help="Dataset name")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs",
        help="Directory to store the config files",
    )

    args = parser.parse_args()

    config = get_config(args.config_path)

    assert (
        "data_features" in config
    ), f"Key 'data_features' not found in config file {args.config_dir}"
    assert (
        "year" in config["data_features"][args.data]
    ), f"Key 'year' not found in config file {args.config_dir}"

    year_range = config["data_features"][args.data]["year"]
    model_name = config["model_name"]
    test_result_path = config["test_results_path"]
    data_name = args.data
    train_sections = range(5)
    loss_function_name = config["loss_function"]
    properties = ["polarity", "previous_link"]  # Moral polarity, Moral relevance
    ts_df = pd.DataFrame()
    for property_name in properties:

        for train_section in train_sections:
            for section in range(len(year_range)):

                # Load the data
                df_path = os.path.join(
                    test_result_path,
                    "time_series",
                    f"testing_data_{data_name}_{model_name}_{section}_{property_name}_train_section_{train_section}.csv",
                )
                assert os.path.exists(df_path), f"Data file {df_path} does not exist."
                df = pd.read_csv(df_path)
                df["train_section"] = train_section
                df["property"] = property_name
                ts_df = pd.concat([ts_df, df], ignore_index=True)

    ts_df = ts_df.reset_index(drop=True)
    word_count = pd.read_csv(
        os.path.join(config["store_dir"], f"{data_name}_word_count.csv"),
    )

    word_count = word_count.groupby(["word", "year"])["count"].mean().to_dict()
    ts_df["word_count"] = ts_df.apply(
        lambda x: word_count.get((x["words"], x["year"]), 0), axis=1,
    )

    # Normalizing moral association scores
    mean_values = (
        ts_df.groupby(["train_section", "property"])["outputs"].mean().to_dict()
    )
    std_values = ts_df.groupby(["train_section", "property"])["outputs"].std().to_dict()
    ts_df["outputs_z"] = [
        (o - mean_values[s, p]) / std_values[s, p]
        for o, w, y, p, s in zip(
            ts_df.outputs, ts_df.words, ts_df.year, ts_df.property, ts_df.train_section, strict=False,
        )
    ]

    if data_name == "coha":
        save_dir = os.path.join(test_result_path, "time_series", "ts_df.csv")
    else:
        save_dir = os.path.join(
            test_result_path, "time_series", f"{data_name}_ts_df.csv",
        )
    ts_df.to_csv(save_dir, index=False)

    # storing sentiments
    sentiment_df = pd.DataFrame()
    for year in year_range:
        d = pickle.load(
            open(
                os.path.join(
                    config["store_dir"],
                    f"sentiments_{data_name}_{year}_{model_name}.pkl",
                ),
                "rb",
            ),
        )
        df = pd.DataFrame(d.items(), columns=["words", "sentiments"])
        df["year"] = year
        sentiment_df = pd.concat([sentiment_df, df], ignore_index=True)

    sentiment_df.to_csv(
        f"data/SWOW_prediction/eval/{data_name}_sentiments.csv", index=False,
    )
