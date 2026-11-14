import json
import os
from pathlib import Path

import numpy as np
import torch


def _dataset_root():
    """Return the dataset root, independent of the process working directory."""
    override = os.environ.get("FEDOMG_DATA_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    # data_utils.py is <repo>/system/utils/data_utils.py.
    return (Path(__file__).resolve().parents[2] / "dataset").resolve()


def _validate_dataset_config(config_path, num_clients, niid, balance, alpha):
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Dataset config file is missing: {config_path}. "
            "Regenerate the dataset with the repository's dataset generator."
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = json.load(file)

    expected = {
        "num_clients": int(num_clients),
        "non_iid": bool(niid),
        "balance": bool(balance),
    }
    mismatches = {
        key: (config.get(key), value)
        for key, value in expected.items()
        if config.get(key) != value
    }

    try:
        stored_alpha = float(config["alpha"])
        requested_alpha = float(alpha)
        if not np.isclose(stored_alpha, requested_alpha, rtol=0.0, atol=1e-12):
            mismatches["alpha"] = (stored_alpha, requested_alpha)
    except (KeyError, TypeError, ValueError):
        mismatches["alpha"] = (config.get("alpha"), alpha)

    if mismatches:
        details = ", ".join(
            f"{key}: stored={stored}, requested={requested}"
            for key, (stored, requested) in mismatches.items()
        )
        raise ValueError(
            "The selected dataset directory does not match the current "
            f"experiment arguments ({details}). Check --num_clients, "
            "--noniid, --balance, and --alpha_dirich, or regenerate the data."
        )


def _resolve_dataset_dir(
    dataset,
    num_clients=None,
    niid=None,
    balance=None,
    alpha=None,
):
    """
    Resolve the generated dataset directory.

    The standard generated layout is:
        dataset/<dataset>/<num_clients>/<balance>_<niid>_<alpha>/

    A small number of legacy datasets use:
        dataset/<dataset>/{train,test}/
    """
    root = _dataset_root()
    dataset_root = root / str(dataset)

    supplied = [
        num_clients is not None,
        niid is not None,
        balance is not None,
        alpha is not None,
    ]

    if any(supplied) and not all(supplied):
        raise ValueError(
            "Dataset configuration must provide all of: num_clients, "
            "niid, balance, and alpha."
        )

    if all(supplied):
        setting = f"{bool(balance)}_{bool(niid)}_{float(alpha)}"
        dataset_dir = dataset_root / str(int(num_clients)) / setting

        if dataset_dir.is_dir():
            _validate_dataset_config(
                dataset_dir / "config.json",
                num_clients=num_clients,
                niid=niid,
                balance=balance,
                alpha=alpha,
            )
            return dataset_dir

        # Legacy generators such as Fashion-MNIST use a flat train/test
        # directory and do not encode the partition in the directory name.
        if (dataset_root / "train").is_dir() and (dataset_root / "test").is_dir():
            return dataset_root

        raise FileNotFoundError(
            "Requested dataset partition was not found.\n"
            f"Expected directory: {dataset_dir}\n"
            "For CIFAR-10, generate it from the dataset directory, for example:\n"
            f"  cd {_dataset_root()}\n"
            f"  python generate_cifar10.py noniid balance dir "
            f"{int(num_clients)} {float(alpha)}"
        )

    # Backward compatibility for callers that do not pass partition
    # arguments: infer the setting only when exactly one is available.
    if (dataset_root / "train").is_dir() and (dataset_root / "test").is_dir():
        return dataset_root

    candidates = sorted(dataset_root.glob("*/*/config.json"))
    if len(candidates) == 1:
        return candidates[0].parent
    if len(candidates) > 1:
        raise ValueError(
            f"Multiple generated partitions exist under {dataset_root}. "
            "Pass num_clients, niid, balance, and alpha explicitly."
        )

    raise FileNotFoundError(
        f"No generated dataset found for {dataset!r} under {root}. "
        "Pass the experiment's dataset configuration and generate the data first."
    )


def _load_npz_record(file_path):
    if not file_path.is_file():
        raise FileNotFoundError(f"Dataset file is missing: {file_path}")
    with np.load(file_path, allow_pickle=True) as archive:
        if "data" not in archive:
            raise KeyError(f"NPZ file does not contain a 'data' entry: {file_path}")
        return archive["data"].tolist()


def read_data(
    dataset,
    idx,
    is_train=True,
    *,
    num_clients=None,
    niid=None,
    balance=None,
    alpha=None,
):
    dataset_dir = _resolve_dataset_dir(
        dataset,
        num_clients=num_clients,
        niid=niid,
        balance=balance,
        alpha=alpha,
    )
    split = "train" if is_train else "test"
    file_path = dataset_dir / split / f"{int(idx)}.npz"
    return _load_npz_record(file_path)


def read_test_data(
    dataset,
    *,
    num_clients=None,
    niid=None,
    balance=None,
    alpha=None,
):
    dataset_dir = _resolve_dataset_dir(
        dataset,
        num_clients=num_clients,
        niid=niid,
        balance=balance,
        alpha=alpha,
    )
    return _load_npz_record(dataset_dir / "test" / "test.npz")


def client_test_data(
    dataset,
    *,
    num_clients=None,
    niid=None,
    balance=None,
    alpha=None,
):
    test_data = read_test_data(
        dataset,
        num_clients=num_clients,
        niid=niid,
        balance=balance,
        alpha=alpha,
    )
    x_test = torch.as_tensor(test_data["x"], dtype=torch.float32)
    y_test = torch.as_tensor(test_data["y"], dtype=torch.int64)
    return list(zip(x_test, y_test))


def read_client_data(
    dataset,
    idx,
    is_train=True,
    *,
    num_clients=None,
    niid=None,
    balance=None,
    alpha=None,
):
    data = read_data(
        dataset,
        idx,
        is_train=is_train,
        num_clients=num_clients,
        niid=niid,
        balance=balance,
        alpha=alpha,
    )
    x = torch.as_tensor(data["x"], dtype=torch.float32)
    y = torch.as_tensor(data["y"], dtype=torch.int64)
    return list(zip(x, y))


def read_client_data_text(
    dataset,
    idx,
    niid,
    balance,
    alpha,
    is_train=True,
    num_clients=None,
):
    data = read_data(
        dataset,
        idx,
        is_train=is_train,
        num_clients=num_clients,
        niid=niid,
        balance=balance,
        alpha=alpha,
    )

    x_values, x_lengths = zip(*data["x"])
    x_tensor = torch.as_tensor(x_values, dtype=torch.int64)
    length_tensor = torch.as_tensor(x_lengths, dtype=torch.int64)
    y_tensor = torch.as_tensor(data["y"], dtype=torch.int64)

    return [
        ((x, length), label)
        for x, length, label in zip(x_tensor, length_tensor, y_tensor)
    ]


def read_client_data_shakespeare(
    dataset,
    idx,
    niid,
    balance,
    alpha,
    is_train=True,
    num_clients=None,
):
    data = read_data(
        dataset,
        idx,
        is_train=is_train,
        num_clients=num_clients,
        niid=niid,
        balance=balance,
        alpha=alpha,
    )

    x_tensor = torch.as_tensor(data["x"], dtype=torch.int64)
    y_tensor = torch.as_tensor(data["y"], dtype=torch.int64)
    return list(zip(x_tensor, y_tensor))
