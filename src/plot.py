import numpy as np
from sklearn.calibration import calibration_curve
NUM_BINS = 20

def _plot(ax, w, mean_predicted_value, fraction_of_positives, num_bins, color, label, plot_hist=True):
    
    # ax1.grid(True)
    ax.plot(mean_predicted_value * num_bins, fraction_of_positives, c=color, marker='*', label=label)
    ax.plot(np.linspace(0, 1, num_bins + 1), c="gray")

    ax.set_xlabel("Score(Model Confidence)")
    ax.set_xlim((0.0, num_bins))
    ax.set_xticks(range(num_bins + 1))
    ax.set_xticklabels(["%.2f" % i for i in np.linspace(0, 1, num_bins + 1)])
    ax.set_ylabel("Accuracy")
    ax.set_ylim((0.0, 1.0))
    ax.set_yticks(np.linspace(0, 1, num_bins + 1))
    ax.set_yticklabels(["%.2f" % i for i in np.linspace(0, 1, num_bins + 1)])

    if plot_hist:
        # add histogram to the reliability diagram
        ax2 = ax.twinx() 
        ax2.bar([i + 0.5 for i in range(num_bins)], w, color=color, alpha=0.3, label="Histogram", width=1.0)
        #ax2.set_ylabel('Histogram', color=color)
        ax2.set_ylim((0.0, 2.0))
        ax2.set_yticks([0, 0.5, 1.0])
        ax2.set_yticklabels([0, 0.5, 1.0], color=color, fontsize=3)
        ax2.yaxis.set_label_coords(1.01, 0.25)
    
    return ax

def plot_reliability_diagram(ax, s, y):
    
    num_bins = NUM_BINS
    y_pred = np.argmax(s, axis=1)
    s_max = np.max(s, axis=1)
    correctness = y==y_pred
    fraction_of_positives, mean_predicted_value = calibration_curve(correctness, s_max, \
                                                                    n_bins=num_bins, strategy='quantile')
    hist = np.histogram(s_max, bins=num_bins, range=(0, 1), density=False)[0]
    w = hist * 1.0 / hist.sum()
    
    ax = _plot(ax, w, mean_predicted_value, fraction_of_positives, num_bins, color='r', label='hard label')
    ax.annotate('accuracy = %.1f%%\nconfidence=%.2f' % (100*(correctness).mean(), s_max.mean()), 
                            xy=(0.4, 0.2), xycoords='axes fraction', fontsize=12,
                            ha='left', va='top', color='r')
    
    return ax


def plot_reliability_diagram_with_soft_labels(ax, s, y_dist):
    
    num_bins = NUM_BINS
    num_samples = s.shape[0]
    acc = (y_dist[np.arange(num_samples), np.argmax(s, axis=1)]).mean()
    l1 = np.abs(s - y_dist).sum() / s.shape[0]
    l2 = np.sqrt(((s - y_dist)**2).sum() / s.shape[0])
    s = s.flatten()
    correctness = y_dist.flatten()
    
    # compute fraction_of_positives, mean_predicted_value 
    fraction_of_positives = np.zeros((num_bins,))
    mean_predicted_value = np.zeros((num_bins,))
    strategy = 'quantile'
    if strategy == 'uniform':
        bins = np.linspace(0, 1, num_bins + 1)
        categories = np.digitize(s, bins[1:-1]).astype(int)
    elif strategy == 'quantile':
        bin_size = s.shape[0] // num_bins
        categories = np.zeros((s.shape[0],)).astype(int)
        ranked = np.argsort(s)
        for idx in range(num_bins):
            start = idx * bin_size
            if idx == num_bins - 1:
                end = -1
            else:
                end = (idx+1) * bin_size
            categories[ranked[start : end]] = idx
    for i in range(num_bins):
        mean_predicted_value[i] = s[categories==i].mean()
        fraction_of_positives[i] = correctness[categories==i].mean()
    
    hist = np.histogram(s, bins=num_bins, range=(0, 1), density=False)[0]
    w = hist * 1.0 / hist.sum()
    
    ax = _plot(ax, w, mean_predicted_value, fraction_of_positives, num_bins, color='b', 
               label='soft label', plot_hist=False)
    ax.annotate('accuracy(max)= %.1f%%\nl1 error=%.2f\nl2 error=%.2f' % (100*acc, l1, l2), 
                            xy=(0.05, 0.8), xycoords='axes fraction', fontsize=12,
                            ha='left', va='top', color='b')
    
    return ax