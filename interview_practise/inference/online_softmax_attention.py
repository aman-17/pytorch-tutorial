import numpy as np

logits = np.array([2.0, 1.0, 0.1, 3.0])

def standard_softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

def online_softmax(x_stream):
    max_so_far = -np.inf
    sum_exp_so_far = 0.0

    for x_i in x_stream:
        old_max = max_so_far
        max_so_far = max(max_so_far, x_i)
        sum_exp_so_far = sum_exp_so_far * np.exp(old_max - max_so_far) + np.exp(x_i - max_so_far)

    final_probs = []
    for x_i in x_stream:
        prob = np.exp(x_i - max_so_far) / sum_exp_so_far
        final_probs.append(prob)

    return np.array(final_probs)

print("Standard Softmax:", standard_softmax(logits))
print("Online Softmax:", online_softmax(logits))
print("Standard Softmax Sum:", np.sum(standard_softmax(logits)))
print("Online Softmax Sum:", np.sum(online_softmax(logits)))