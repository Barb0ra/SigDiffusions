"""Reimplementation of discrimintive score from the TimeGAN Codebase.

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
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm
from metric_utils import extract_time, train_test_divide


def batch_generator(data, time, batch_size):
    """Mini-batch generator.

    Args:
        - data: time-series data
        - time: time information
        - batch_size: the number of samples in each batch

    Returns:
        - X_mb: time-series data in each batch
        - T_mb: time information in each batch
    """
    no = len(data)
    idx = np.random.permutation(no)
    train_idx = idx[:batch_size]

    X_mb = [data[i] for i in train_idx]
    T_mb = [time[i] for i in train_idx]

    return X_mb, T_mb


class Discriminator(nn.Module):
    """
    Discriminator model for distinguishing between real and synthetic time-series data.

    Args:
        input_dim (int): Dimensionality of the input data.
        hidden_dim (int): Dimensionality of the hidden state in the GRU.

    Methods:
        forward(x, seq_lengths):
            Forward pass of the Discriminator model.
    """

    def __init__(self, input_dim, hidden_dim):
        super(Discriminator, self).__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, seq_lengths):
        packed_input = nn.utils.rnn.pack_padded_sequence(
            x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed_input)
        logits = self.fc(hidden[-1])
        return logits


def discriminative_score_metrics(ori_data, generated_data):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
        - ori_data: original data
        - generated_data: generated synthetic data

    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    no, seq_len, dim = np.asarray(ori_data).shape
    hidden_dim = max(dim // 2, 2)
    iterations = 2000
    batch_size = 128

    # Prepare the data
    ori_time, _ = extract_time(ori_data)
    generated_time, _ = extract_time(generated_data)
    (
        train_x,
        train_x_hat,
        test_x,
        test_x_hat,
        train_t,
        train_t_hat,
        test_t,
        test_t_hat,
    ) = train_test_divide(ori_data, generated_data, ori_time, generated_time)

    # Initialize the discriminator
    discriminator = Discriminator(dim, hidden_dim).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(discriminator.parameters())

    # Training loop
    discriminator.train()
    for itt in tqdm(range(iterations), desc="training", total=iterations):
        X_mb, T_mb = batch_generator(train_x, train_t, batch_size)
        X_hat_mb, T_hat_mb = batch_generator(train_x_hat, train_t_hat, batch_size)

        X_mb = torch.tensor(np.array(X_mb), dtype=torch.float32).to(device)
        T_mb = torch.tensor(np.array(T_mb), dtype=torch.long).to(device)
        X_hat_mb = torch.tensor(np.array(X_hat_mb), dtype=torch.float32).to(device)
        T_hat_mb = torch.tensor(np.array(T_hat_mb), dtype=torch.long).to(device)

        optimizer.zero_grad()

        logits_real = discriminator(X_mb, T_mb)
        logits_fake = discriminator(X_hat_mb, T_hat_mb)

        loss_real = criterion(logits_real, torch.ones_like(logits_real))
        loss_fake = criterion(logits_fake, torch.zeros_like(logits_fake))
        loss = loss_real + loss_fake

        loss.backward()
        optimizer.step()

    # Evaluation
    discriminator.eval()
    test_x = torch.tensor(np.array(test_x), dtype=torch.float32).to(device)
    test_t = torch.tensor(np.array(test_t), dtype=torch.long).to(device)
    test_x_hat = torch.tensor(np.array(test_x_hat), dtype=torch.float32).to(device)
    test_t_hat = torch.tensor(np.array(test_t_hat), dtype=torch.long).to(device)

    with torch.no_grad():
        y_pred_real = torch.sigmoid(discriminator(test_x, test_t)).cpu().numpy()
        y_pred_fake = torch.sigmoid(discriminator(test_x_hat, test_t_hat)).cpu().numpy()

    y_pred_final = np.concatenate((y_pred_real, y_pred_fake), axis=0).squeeze()
    y_label_final = np.concatenate(
        [
            np.ones(len(y_pred_real)),
            np.zeros(len(y_pred_fake)),
        ]
    )

    acc = accuracy_score(y_label_final, y_pred_final > 0.5)
    fake_acc = accuracy_score(np.zeros(len(y_pred_fake)), y_pred_fake > 0.5)
    real_acc = accuracy_score(np.ones(len(y_pred_real)), y_pred_real > 0.5)

    discriminative_score = np.abs(0.5 - acc)
    return discriminative_score, fake_acc, real_acc
