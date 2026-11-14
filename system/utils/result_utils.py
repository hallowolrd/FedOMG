import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10):
    if times <= 0:
        raise ValueError(f"times must be positive, got {times}")

    curves = get_all_results_for_one_algo(
        algorithm=algorithm,
        dataset=dataset,
        goal=goal,
        times=times,
    )

    if len(curves) != times:
        raise RuntimeError(
            f"Expected {times} result curves, but loaded {len(curves)}."
        )

    for run_index, curve in enumerate(curves):
        if curve.size == 0:
            raise ValueError(f"Run {run_index} has an empty accuracy curve.")

    best_accuracy = np.asarray([curve.max() for curve in curves], dtype=float)
    final_accuracy = np.asarray([curve[-1] for curve in curves], dtype=float)

    best_mean = float(np.mean(best_accuracy))
    best_std = float(np.std(best_accuracy))
    final_mean = float(np.mean(final_accuracy))
    final_std = float(np.std(final_accuracy))

    print(
        "Best accuracy across runs: "
        f"{best_mean * 100:.2f}% ± {best_std * 100:.2f}%"
    )
    print(
        "Final accuracy across runs: "
        f"{final_mean * 100:.2f}% ± {final_std * 100:.2f}%"
    )

    return {
        "best_mean": best_mean,
        "best_std": best_std,
        "final_mean": final_mean,
        "final_std": final_std,
        "best_per_run": best_accuracy,
        "final_per_run": final_accuracy,
    }


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    curves = []
    for run_index in range(times):
        file_name = f"{dataset}_{algorithm}_{goal}_{run_index}"
        curves.append(np.asarray(read_data_then_delete(file_name, delete=False)))

    return curves


def read_data_then_delete(file_name, delete=False):
    project_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
    file_path = os.path.join(project_root, "results", file_name + ".h5")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(
            f"Experiment result file not found: {file_path}"
        )

    with h5py.File(file_path, "r") as hf:
        if "rs_test_acc" not in hf:
            raise KeyError(
                f"Result file does not contain 'rs_test_acc': {file_path}"
            )
        test_acc = np.asarray(hf["rs_test_acc"])

    if delete:
        os.remove(file_path)

    print(f"Loaded {file_name}: {len(test_acc)} evaluation points")
    return test_acc
