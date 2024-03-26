"""Reimplementation of predictive score from the TimeGAN Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com)

"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import mean_absolute_error
from tqdm.auto import tqdm


class Predictor(nn.Module):
    """
    A simple RNN-based predictor model using GRU and a fully connected layer.
    """

    def __init__(self, input_dim, hidden_dim):
        super(Predictor, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        outputs, _ = self.rnn(x)
        y_hat_logit = self.fc(outputs)
        y_hat = torch.sigmoid(y_hat_logit)
        return y_hat


def predictive_score_metrics(ori_data, generated_data, window_size=20):
    """Report the performance of Post-hoc RNN one-step ahead prediction.

    Args:
      - ori_data: original data
      - generated_data: generated synthetic data
      - window_size: number of steps ahead to predict in the univariate case

    Returns:
      - predictive_score: MAE of the predictions on the original data
    """
    # Basic Parameters
    no, seq_len, dim = ori_data.shape

    # Network parameters
    hidden_dim = max(dim // 2, 2)
    iterations = 5000
    batch_size = 128

    model = Predictor(input_dim=(dim - 1) if dim > 1 else 1, hidden_dim=hidden_dim).to(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters())

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def prepare_batch(data, batch_indices):
        if dim > 1:
            # Multivariate case
            X_mb = [
                torch.tensor(data[i][:-1, : dim - 1], dtype=torch.float32)
                for i in batch_indices
            ]
            Y_mb = [
                torch.tensor(data[i][1:, dim - 1 :], dtype=torch.float32)
                for i in batch_indices
            ]
        else:
            # Univariate case
            X_mb = [
                torch.tensor(data[i][:-window_size], dtype=torch.float32).unsqueeze(-1)
                for i in batch_indices
            ]
            Y_mb = [
                torch.tensor(data[i][window_size:], dtype=torch.float32).unsqueeze(-1)
                for i in batch_indices
            ]

        # Use `pad_sequence` for consistent batching in the multivariate case
        if dim > 1:
            X_tensor = nn.utils.rnn.pad_sequence(X_mb, batch_first=True).to(device)
            Y_tensor = nn.utils.rnn.pad_sequence(Y_mb, batch_first=True).to(device)
        else:
            # Stack directly in the univariate case since lengths are constant
            X_tensor = torch.stack(X_mb).squeeze(-1).to(device)
            Y_tensor = torch.stack(Y_mb).squeeze(-1).to(device)

        return X_tensor, Y_tensor

    # Training loop
    model.train()
    for _ in tqdm(range(iterations), desc="training", total=iterations):
        batch_indices = np.random.choice(len(generated_data), batch_size, replace=False)
        X_mb, Y_mb = prepare_batch(generated_data, batch_indices)

        optimizer.zero_grad()
        y_pred = model(X_mb)
        loss = criterion(y_pred, Y_mb)
        loss.backward()
        optimizer.step()

    # Testing loop
    model.eval()
    batch_indices = np.arange(no)
    X_mb, Y_mb = prepare_batch(ori_data, batch_indices)

    with torch.no_grad():
        pred_Y_curr = model(X_mb)

    # Compute the performance in terms of MAE
    MAE_temp = 0
    for i in range(no):
        pred = pred_Y_curr[i].cpu().numpy()
        target = Y_mb[i].cpu().numpy()
        MAE_temp += mean_absolute_error(target, pred)

    predictive_score = MAE_temp / no

    return predictive_score
