import seaborn as sns
import matplotlib.pyplot as plt

def plot_graph(x, title, xlabel, ylabel):
    plt.figure(figsize=(12, 12))
    ax = sns.barplot(x=x.index, y=x.values, alpha=0.8)
    plt.title(title)
    plt.ylabel(ylabel, fontsize=12)
    plt.xlabel(xlabel, fontsize=12)

    # Adding labels to the bars
    rects = ax.patches
    labels = x.values
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')

    plt.show()

    return