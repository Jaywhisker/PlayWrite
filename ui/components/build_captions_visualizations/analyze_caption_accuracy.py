# IMPORTS
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def analyze_caption_accuracy(file_path):
    """
    Analyze captions within the provided DataFrame for repetitions and plot the results.

    Args:
        file_path (string): The file_path to the csv containing the data.
    """
    df = pd.read_csv(file_path)

    caption_col = df.columns[df.columns.str.contains('caption', case=False)][0]

    def has_repetition(caption):
        words = caption.split()
        seen = set()
        for i in range(len(words) - 1):
            phrase = ' '.join(words[i:i+2])
            if phrase in seen:
                return True
            seen.add(phrase)
        return False

    df['is_inaccurate'] = df[caption_col].apply(has_repetition)

    accurate_count = (df['is_inaccurate'] == False).sum()
    inaccurate_count = (df['is_inaccurate'] == True).sum()

    sizes = [accurate_count, inaccurate_count]
    labels = ['Accurate', 'Inaccurate']
    colors = ['#FF8C00', '#4682B4'] 
    total = sum(sizes)
    percentages = [f'{100 * (size / total):.2f}%' for size in sizes]
    labels_with_counts = [f'{label}: {count},\n{percentage}' for label, count, percentage in zip(labels, sizes, percentages)]

    fig, ax = plt.subplots()
    textprops = {'fontsize': 9, 'weight': 'bold'}
    wedges, texts = ax.pie(sizes, labels=labels_with_counts, colors=colors, startangle=90, counterclock=False, textprops=textprops)
    plt.legend(wedges, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, frameon=False)
    plt.title('Caption Accuracy', pad=25, fontsize=14)
    plt.axis('equal')
    st.pyplot(plt)
