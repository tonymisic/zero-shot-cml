import matplotlib.pyplot as plt, seaborn as sns, numpy as np, torch
from matplotlib.lines import Line2D

def histogram(gt, pred_v2a, pred_a2v, file="histogram.jpg", time='start'):
    gt = torch.bincount(gt).numpy()
    pred_v2a = torch.bincount(pred_v2a).numpy()
    pred_a2v = torch.bincount(pred_a2v).numpy()
    print(np.sum(gt))
    print(np.sum(pred_v2a))
    print(np.sum(pred_a2v))
    if time == 'start':
        x_label = np.array([0,1,2,3,4,5,6,7,8])
        ax = plt.subplots()
        ax = sns.barplot(y=gt, x=x_label, facecolor=(0, 1, 0, 1), edgecolor=(0,1,0))
        ax = sns.barplot(y=pred_v2a, x=x_label, facecolor=(1, 1, 1, 0), edgecolor=(0,0,1))
        ax = sns.barplot(y=pred_a2v, x=x_label, facecolor=(1, 1, 1, 0), edgecolor=(0,0,0))
        ax.set(xlabel="Start Times (s)", ylabel="Count")
        figure = ax.get_figure()
        custom_lines = [Line2D([0], [0], color=(0, 1, 0), lw=5), Line2D([0], [0], color=(0,0,1), lw=2), Line2D([0], [0], color=(0,0,0), lw=2)]
        plt.legend(custom_lines, ['GT', 'V2A', 'A2V'])
        figure.savefig(file)
    elif time == 'end':
        x_label = np.array([2,3,4,5,6,7,8,9,10])
        ax = plt.subplots()
        ax = sns.barplot(y=gt[-9:], x=x_label, facecolor=(0, 1, 0, 1), edgecolor=(0,1,0))
        ax = sns.barplot(y=pred_v2a[-9:], x=x_label, facecolor=(1, 1, 1, 0), edgecolor=(0,0,1))
        ax = sns.barplot(y=pred_a2v[-9:], x=x_label, facecolor=(1, 1, 1, 0), edgecolor=(0,0,0))
        ax.set(xlabel="End Times (s)", ylabel="Count")
        figure = ax.get_figure()
        custom_lines = [Line2D([0], [0], color=(0, 1, 0), lw=5), Line2D([0], [0], color=(0,0,1), lw=2), Line2D([0], [0], color=(0,0,0), lw=2)]
        plt.legend(custom_lines, ['GT', 'V2A', 'A2V'])
        figure.savefig(file)