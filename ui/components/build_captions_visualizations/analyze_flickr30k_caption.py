import pandas as pd
import matplotlib.pyplot as plt

def analyze_caption_counts(df):
    """
    Analyze the caption counts of the DataFrame.

    Args:
        df (DataFrame): Input DataFrame containing the data.

    Returns:
        Series: Categorized caption counts.
        int: Total number of images.
    """
    caption_counts = df.groupby('image_filename')['caption'].count()
    caption_counts_categorized = caption_counts.value_counts().reindex(range(1, 6), fill_value=0)
    total_images = caption_counts_categorized.sum()
    return caption_counts_categorized, total_images

def plot_pie_chart(caption_counts_categorized, total_images):
    """
    Plot a pie chart showing the distribution of images by the number of captions.

    Args:
        caption_counts_categorized (Series): Categorized caption counts.
        total_images (int): Total number of images.
    """
    labels = [f'{i} caption' for i in caption_counts_categorized.index]
    colors = ['#8EA4D2', '#F7B2AD', '#FFD6A5', '#BEE1E6', '#C3ACEE']

    def func(pct, allvals):
        absolute = int(pct/100.*total_images)
        return "{:d}\n({:.1f}%)".format(absolute, pct)

    plt.figure(figsize=(6, 6))
    plt.pie(caption_counts_categorized, labels=labels, colors=colors, startangle=140,
            autopct=lambda pct: func(pct, caption_counts_categorized))
    plt.title('Distribution of Images by Number of Captions', fontsize=16) 
    plt.tight_layout() 
    plt.show()

caption_counts_categorized, total_images = analyze_caption_counts(flickr30k)
plot_pie_chart(caption_counts_categorized, total_images)