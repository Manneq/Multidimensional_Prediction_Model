"""
    File 'plotting.py' has functions for plotting different data.
"""
import matplotlib.pyplot as plt
import seaborn as sns


def plot_heatmap(data, title,
                 width=1920, height=1080, dpi=96):
    """
        Method to plot missing data as binary heatmap.
        param:
            1. data - numpy array of data that should be plotted
            2. title - string name of plot
            3. width - int value of plot width in pixels (1920 as default)
            4. height - int value of plot height in pixels (1080 as default)
            5. dpi - int value of plot dpi (96 as default)
    """
    plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    sns.heatmap(data.isnull(), cbar=False, yticklabels=False).set_title(title)
    plt.savefig("plots/missing_data/" + title + ".png", dpi=dpi)
    plt.close()

    return
