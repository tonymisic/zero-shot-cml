import torch

def temporal_accuracy(pred, y, threshold, device):
    assert pred.size() == y.size()
    count, total = 0, 0
    for i in range(pred.size(0)):
        current_pred = torch.zeros(pred.size(1)).to(device)
        for j in range(pred.size(1)):
            if pred[i][j] > threshold:
                current_pred[j] = 1
        if torch.equal(current_pred, y[i]):
            count += 1
        total += 1
    return count / total