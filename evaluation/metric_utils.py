"""
Methods adapted from the following codebases:
https://github.com/Y-debug-sys/Diffusion-TS/blob/main/Utils/metric_utils.py
https://github.com/jsyoon0823/TimeGAN/blob/master/metrics/visualization_metrics.py
https://github.com/issaz/sigker-nsdes/blob/3d40b2922db4a7446917bdfd225c9252dd7d01a2/src/evaluation/evaluation_functions.py#L4
"""

## Necessary Packages
import numpy as np
import scipy
from scipy.stats import ks_2samp
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def display_scores(results):
    """
    Calculate and display the mean and standard deviation of the given results.

    Parameters:
    results (list or array-like): A list or array of numerical results.

    Returns:
    None

    Prints:
    The mean and the standard deviation of the results.
    """
    mean = np.mean(results)
    std_dev = np.std(results)
    print("Final Score: ", f"{mean} ± {std_dev}")


def train_test_divide(data_x, data_x_hat, data_t, data_t_hat, train_rate=0.8):
    """Divide train and test data for both original and synthetic data.

    Args:
      - data_x: original data
      - data_x_hat: generated data
      - data_t: original time
      - data_t_hat: generated time
      - train_rate: ratio of training data from the original data
    """
    # Divide train/test index (original data)
    no = len(data_x)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x = [data_x[i] for i in train_idx]
    test_x = [data_x[i] for i in test_idx]
    train_t = [data_t[i] for i in train_idx]
    test_t = [data_t[i] for i in test_idx]

    # Divide train/test index (synthetic data)
    no = len(data_x_hat)
    idx = np.random.permutation(no)
    train_idx = idx[: int(no * train_rate)]
    test_idx = idx[int(no * train_rate) :]

    train_x_hat = [data_x_hat[i] for i in train_idx]
    test_x_hat = [data_x_hat[i] for i in test_idx]
    train_t_hat = [data_t_hat[i] for i in train_idx]
    test_t_hat = [data_t_hat[i] for i in test_idx]

    return (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    )


def extract_time(data):
    """Returns Maximum sequence length and each sequence length.

    Args:
      - data: original data

    Returns:
      - time: extracted time information
      - max_seq_len: maximum sequence length
    """
    time = list()
    max_seq_len = 0
    for i in range(len(data)):
        max_seq_len = max(max_seq_len, len(data[i][:, 0]))
        time.append(len(data[i][:, 0]))

    return time, max_seq_len


def visualization(ori_data, generated_data, analysis):
    """Using PCA or tSNE for generated and original data visualization.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - analysis: tsne or pca
    """
    anal_sample_no = min([1000, len(ori_data)])
    idx = np.random.permutation(len(ori_data))[:anal_sample_no]

    ori_data = ori_data[idx]
    generated_data = generated_data[idx]

    no, seq_len, dim = ori_data.shape

    for i in range(anal_sample_no):
        if i == 0:
            prep_data = np.reshape(np.mean(ori_data[0, :, :], 1), [1, seq_len])
            prep_data_hat = np.reshape(
                np.mean(generated_data[0, :, :], 1), [1, seq_len]
            )
        else:
            prep_data = np.concatenate(
                (prep_data, np.reshape(np.mean(ori_data[i, :, :], 1), [1, seq_len]))
            )
            prep_data_hat = np.concatenate(
                (
                    prep_data_hat,
                    np.reshape(np.mean(generated_data[i, :, :], 1), [1, seq_len]),
                )
            )

    # Visualization parameter
    colors = ["red" for i in range(anal_sample_no)] + [
        "blue" for i in range(anal_sample_no)
    ]

    if analysis == "tsne":

        # Do t-SNE Analysis together
        prep_data_final = np.concatenate((prep_data, prep_data_hat), axis=0)

        # TSNE anlaysis
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, max_iter=300)
        tsne_results = tsne.fit_transform(prep_data_final)

        # Plotting
        f, ax = plt.subplots(1)

        plt.scatter(
            tsne_results[:anal_sample_no, 0],
            tsne_results[:anal_sample_no, 1],
            c=colors[:anal_sample_no],
            alpha=0.2,
            label="Original",
        )
        plt.scatter(
            tsne_results[anal_sample_no:, 0],
            tsne_results[anal_sample_no:, 1],
            c=colors[anal_sample_no:],
            alpha=0.2,
            label="Synthetic",
        )

        ax.legend()

        # plt.title("t-SNE plot")
        # plt.xlabel("x-tsne")
        # plt.ylabel("y_tsne")
        plt.show()


def get_ks_scores(real_paths, generated_paths, marginals, dim=1):
    """
    Calculate the Kolmogorov-Smirnov (KS) statistic and p-value for the given real and generated paths.

    Parameters:
    real_paths (np.ndarray): Array of real paths with shape (num_samples, path_length, num_dimensions).
    generated_paths (np.ndarray): Array of generated paths with shape (num_samples, path_length, num_dimensions).
    marginals (list or np.ndarray): List or array of marginal indices to evaluate.
    dim (int, optional): Dimension of the path we are currently evaluating.

    Returns:
    np.ndarray: Array of KS statistics and p-values with shape (len(marginals), 2).
    """
    _, path_length, _ = real_paths.shape
    scores = np.zeros((len(marginals), 2))

    for i, m in enumerate(marginals):
        ind_ = int(m * path_length)

        real_marginals = real_paths[:, ind_, dim]
        gen_marginals = generated_paths[:, ind_, dim]

        ks_stat, ks_p_value = ks_2samp(
            real_marginals, gen_marginals, alternative="two_sided"
        )

        scores[i, 0] = ks_stat
        scores[i, 1] = ks_p_value

    return scores


def generate_ks_results(
    real_dataloader, generated_dataloader, marginals, n_runs, dims=1
):
    """
    Generate Kolmogorov-Smirnov (KS) results for multiple runs and dimensions.

    Parameters:
    real_dataloader (DataLoader): DataLoader for real data samples.
    generated_dataloader (DataLoader): DataLoader for generated data samples.
    marginals (list or np.ndarray): List or array of marginal indices to evaluate.
    n_runs (int): Number of runs to perform.
    dims (int, optional): Dimension of the paths we are evaluating.

    Returns:
    np.ndarray: Array of KS statistics and p-values with shape (n_runs, dims, len(marginals), 2).
    """
    total_scores = np.zeros((n_runs, dims, len(marginals), 2))

    for i in range(n_runs):
        with torch.no_grad():
            (real_samples,) = next(iter(real_dataloader))
            (generated_samples,) = next(iter(generated_dataloader))
            for k in range(dims):
                total_scores[i, k] = get_ks_scores(
                    real_samples, generated_samples, marginals, dim=k
                )

    return total_scores
