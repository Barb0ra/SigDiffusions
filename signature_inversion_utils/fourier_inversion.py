from math import comb
import numpy as np

from signature_inversion_utils.free_lie_algebra import (
    Word,
    shuffleProductMany,
    rightHalfShuffleProduct,
    word2Elt,
)

pi = np.pi
elt1 = word2Elt(Word("1"))
elt2 = word2Elt(Word("2"))
elt3 = word2Elt(Word("3"))
eltd = word2Elt(Word("4"))


def sig_vector_to_tensors(sig, path_dim, max_level):
    """Reshape a signature vector to a list of tensors of shape (path_dim,) * i for i = 1, ..., level."""
    tensors = []
    start = 0
    for i in range(1, max_level + 1):
        end = start + path_dim**i
        tensors.append(sig[start:end].reshape((path_dim,) * i))
        start = end
    return tensors


def get_signature_at_index(signature_tensors, level, index):
    level_tensor = signature_tensors[level - 1]
    return level_tensor[tuple(index)]


def evaluate_linear_functional(signature_tensors, elt, d):
    total = 0
    for elt_dict in elt.data:
        for word, count in elt_dict.items():
            letters = word.letters
            # 4 is a placeholder for d, which is the index of the dimension we are currently inverting
            letters = [
                d + 2 if int(letter) == 4 else int(letter) - 1 for letter in letters
            ]
            element = get_signature_at_index(
                signature_tensors,
                len(letters),
                letters,
            )
            total += count * element
    return total


def get_bn(signature_tensors, n, d):
    result = 0
    for k in range(0, n + 1):
        for q in range(0, k + 1):
            big_shuffle = [eltd]
            if n - k != 0:
                big_shuffle.append(shuffleProductMany([elt2] * (n - k)))
            if q != 0:
                big_shuffle.append(shuffleProductMany([elt3] * q))
            linear_functional = rightHalfShuffleProduct(
                shuffleProductMany(big_shuffle), elt1
            )
            coefficient = evaluate_linear_functional(
                signature_tensors, linear_functional, d
            )
            coefficient *= comb(n, k) * comb(k, q) * np.sin((n - k) * pi / 2)
            result += coefficient
    return result


def get_an(signature_tensors, n, d):
    result = 0
    for k in range(0, n + 1):
        for q in range(0, k + 1):
            big_shuffle = [eltd]
            if n - k != 0:
                big_shuffle.append(shuffleProductMany([elt2] * (n - k)))
            if q != 0:
                big_shuffle.append(shuffleProductMany([elt3] * q))
            linear_functional = rightHalfShuffleProduct(
                shuffleProductMany(big_shuffle), elt1
            )
            coefficient = evaluate_linear_functional(
                signature_tensors, linear_functional, d
            )
            coefficient *= comb(n, k) * comb(k, q) * np.cos((n - k) * pi / 2)
            result += coefficient
    return result


def get_ans_bns(signature_tensors, n_coeffs, d):
    a_n = np.zeros(n_coeffs + 1)
    b_n = np.zeros(n_coeffs + 1)
    for i in range(0, n_coeffs + 1):
        a_n[i] = get_an(signature_tensors, i, d) / pi
        b_n[i] = get_bn(signature_tensors, i, d) / pi
    a_n[0] /= 2
    return a_n, b_n


def get_fourier_coeffs_from_sig(sig, dim, sig_depth, n_coeffs):
    signature_tensors = sig_vector_to_tensors(sig, dim + 3, sig_depth)
    a_n_sig = []
    b_n_sig = []
    for d in range(1, dim + 1):
        a_n_sig_d, b_n_sig_d = get_ans_bns(signature_tensors, n_coeffs, d)
        a_n_sig.append(a_n_sig_d)
        b_n_sig.append(b_n_sig_d)
    return a_n_sig, b_n_sig


# reconstruct function from an and bn
def reconstruct_from_fourier_coeffs(t, a_n, b_n):
    dim = len(a_n)
    n = len(a_n[0]) - 1
    all_dims = []
    for d in range(dim):
        result = np.full(t.shape[0], a_n[d][0])
        for i in range(1, n + 1):
            result += a_n[d][i] * np.cos(i * t) + b_n[d][i] * np.sin(i * t)
        all_dims.append(result)

    return np.stack(all_dims, axis=1)
