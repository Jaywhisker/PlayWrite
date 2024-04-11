import pandas as pd
import matplotlib.pyplot as plt

def analyze_class_distribution(df):
    """
    Analyze the class distribution of the DataFrame.

    Args:
        df (DataFrame): Input DataFrame containing the data.

    Returns:
        Series: Class distribution.
        int: Total number of images.
    """
    class_distribution = df['classified_label'].value_counts()
    total_images = class_distribution.sum()
    return class_distribution, total_images

def plot_class_distribution(class_distribution, total_images):
    """
    Plot a pie chart showing the distribution of image classes.

    Args:
        class_distribution (Series): Class distribution.
        total_images (int): Total number of images.
    """
    labels = class_distribution.index
    colors = ['#8EA4D2', '#F7B2AD', '#FFD6A5', '#BEE1E6', '#C3ACEE']

    def func(pct, allvals):
        absolute = int(pct/100.*total_images)
        return "{:d}\n({:.1f}%)".format(absolute, pct)

    plt.figure(figsize=(6, 6))
    plt.pie(class_distribution, labels=labels, colors=colors, startangle=140,
            autopct=lambda pct: func(pct, class_distribution))
    plt.title('Distribution of Image Classes', fontsize=16)
    plt.tight_layout() 
    plt.show()

class_distribution, total_images = analyze_class_distribution(flickr30k)
plot_class_distribution(class_distribution, total_images)