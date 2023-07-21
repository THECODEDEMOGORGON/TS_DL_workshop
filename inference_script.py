import argparse
import json
import pickle
from typing import Any, Dict
from tqdm import tqdm

import pandas as pd
import torch
import pytorch_lightning as pl
from sklearn.preprocessing import LabelEncoder
from pytorch_forecasting import Baseline, DeepAR, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import RMSE
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.metrics import SMAPE
from pytorch_forecasting.metrics import MultivariateNormalDistributionLoss

def load_data(file_path: str, data_col: str, id_col: str, target_column: str) -> pd.DataFrame:
    """
    Load data from a file and return as a DataFrame.

    Args:
        file_path (str): Path to the data file.
        data_col (str): Name of the column with dates.
        id_col (str): Name of the identifier column.
        target_column (str): Name of the target column.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """
    df = pd.read_csv(file_path, parse_dates=True)[[data_col, id_col, target_column]]
    df[id_col] = df[id_col].astype(str)

    dates_transformer = LabelEncoder()
    df['time_idx'] = dates_transformer.fit_transform(df[data_col])
    df['time_idx'] += 1

    return df


def process_data(df: pd.DataFrame, config: dict) -> tuple:
    """
    Process the data and create datasets and data loaders.

    Args:
        df (pd.DataFrame): Input data as a DataFrame.
        config (dict): Configuration parameters.

    Returns:
        tuple: A tuple containing the dataset, training data loader, and validation data loader.
    """

    dataset = TimeSeriesDataSet(
        df,
        time_idx="time_idx",
        target=config["target_column"],
        categorical_encoders={config["id_column"]: NaNLabelEncoder().fit(df[config["id_column"]])},
        group_ids=[config["id_column"]],
        static_categoricals=[config["id_column"]],
        time_varying_unknown_reals=[config["target_column"]],
        max_encoder_length=config["context_length"],
        max_prediction_length=config["prediction_length"],
        allow_missing_timesteps=False
    )

    train_dataloader = dataset.to_dataloader(
        train=True,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        drop_last=True,
        batch_sampler="synchronized"
    )

    val_dataloader = dataset.to_dataloader(
        train=False,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        drop_last=True,
        batch_sampler="synchronized"
    )

    return dataset, train_dataloader, val_dataloader


def fit_model(train_dataloader: torch.utils.data.dataloader.DataLoader,
              val_dataloader: torch.utils.data.dataloader.DataLoader,
              subset_data_loader: torch.utils.data.dataloader.DataLoader,
              config: Dict[str, Any]) -> DeepAR:
    """
    Train the DeepAR model.

    Args:
        train_dataloader (torch.utils.data.dataloader.DataLoader): Data loader for training.
        val_dataloader (torch.utils.data.dataloader.DataLoader): Data loader for validation.
        subset_data_loader (torch.utils.data.dataloader.DataLoader): Data loader for subset data.
        config (Dict[str, Any]): Configuration parameters.

    Returns:
        DeepAR: Trained DeepAR model.
    """
    trainer = pl.Trainer(
        max_epochs=config["max_epochs"],
        accelerator=config["accelerator"],
        enable_model_summary=True,
        gradient_clip_val=config["gradient_clip_val"],
        callbacks=[pl.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
        )],
        limit_train_batches=300,
        limit_val_batches=100,
        enable_checkpointing=True,
        logger=config["logger"]
    )

    net = DeepAR.from_dataset(
        train_dataloader.dataset,
        learning_rate=config["learning_rate"],
        hidden_size=config["hidden_size"],
        rnn_layers=config["rnn_layers"],
        optimizer=config["optimizer"]
    )

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader
    )

    ar_predictions = net.predict(subset_data_loader,
                                 trainer_kwargs=dict(accelerator=config["accelerator"]),
                                 return_y=False)

    rmse = RMSE()(ar_predictions, val_dataloader.dataset[config["target_column"]])
    print(f"RMSE: {rmse}")

    return net


def load_train_dataset(config: Dict[str, Any]) -> TimeSeriesDataSet:
    """
    Load the training dataset from the specified path.

    Args:
        config (Dict[str, Any]): Configuration parameters.

    Returns:
        TimeSeriesDataSet: Loaded training dataset.
    """
    path = config['training_dataset_path']
    with open(path, 'rb') as f:
        dataset = pickle.load(f)

    return dataset

def make_infer_loader(df: pd.DataFrame, dataset: TimeSeriesDataSet, config: Dict[str, Any]) -> torch.utils.data.dataloader.DataLoader:
    """
    Create a data loader for inference.

    Args:
        df (pd.DataFrame): Input data as a DataFrame.
        dataset (TimeSeriesDataSet): Data set.
        config (Dict[str, Any]): Configuration parameters.

    Returns:
        torch.utils.data.dataloader.DataLoader: Data loader for inference.
    """
    test_df = df[df.time_idx > df.time_idx.max() - config['context_length']]
    start_idx = test_df.time_idx.max() + 1
    for square_id in test_df.square_id.unique():
        data = []
        for d in range(config['prediction_length']):
            data.append({
                "square_id": square_id,
                "time_idx": start_idx + d,
                "internet": 0
            })
        data = pd.DataFrame(data)
        test_df = pd.concat([test_df, data])

    test_df = test_df.reset_index(drop=True)

    infer_dataset = TimeSeriesDataSet.from_dataset(dataset, test_df, predict_mode=True)
    infer_loader = infer_dataset.to_dataloader(batch_size=test_df[config['id_column']].nunique())

    return infer_loader, test_df


def make_forecasts(net: DeepAR,
                   infer_dataset,
                   test_df,
                   infer_loader: torch.utils.data.dataloader.DataLoader) -> torch.Tensor:
    """
    Create forecasts using the trained model.

    Args:
        net (DeepAR): Trained DeepAR model.
        infer_loader (torch.utils.data.dataloader.DataLoader): Data loader for inference.

    Returns:
        torch.Tensor: Model forecasts.
    """
    preds = net.predict(infer_loader,
                        trainer_kwargs=dict(accelerator="cpu"),
                        return_y=False)

    x, y = next(iter(infer_loader))
    encoder = infer_dataset.categorical_encoders['square_id']
    ids = encoder.inverse_transform(x['decoder_cat'])
    time_steps = list(range(1, infer_dataset.max_prediction_length+1))

    result = pd.DataFrame(columns=['square_id', 'internet', 'time_idx'])
    for base in tqdm(range(test_df.square_id.nunique())):
        base_id = ids[base]
        forecast = pd.DataFrame({'internet': preds[base].numpy().tolist(),
                                 'square_id': base_id.tolist(),
                                 'time_idx': time_steps})
        result = pd.concat([result, forecast], axis=0)

    return result


def main(config: Dict[str, Any]):
    """
    Main program function for training and making forecasts using the DeepAR model.

    Args:
        config (Dict[str, Any]): Configuration parameters.
    """
    data = load_data(config["data_file"],
                     config['date_column'],
                     config['id_column'],
                     config['target_column'])
    print(data.shape)
    print(data.info())

    if config["fit_model"]:
        training, train_dataloader, val_dataloader = process_data(data, config)
        net = fit_model(train_dataloader, val_dataloader, subset_data_loader, config)
        net.save(config["model_path"])
    else:
        training = load_train_dataset(config)
        net = DeepAR.load_from_checkpoint(config["model_path"])

    infer_loader, test_df = make_infer_loader(data, training, config)
    forecast = make_forecasts(net, training, test_df, infer_loader)
    print(forecast)

    # Save forecasts to Excel file
    forecast.to_csv(config['forecasts_path'], index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Program for training and making forecasts using the DeepAR model")
    parser.add_argument("config", help="Path to the configuration file")

    args = parser.parse_args()
    config_path = args.config

    with open(config_path, "r") as file:
        config = json.load(file)

    main(config)
