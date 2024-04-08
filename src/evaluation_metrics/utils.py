from src.data.dataloader import *
"""
Utils function for all evaluation metric
1. decode_predictions()
2. visualise_graph()
"""


def decode_predictions(outputs, batch_first:bool, vocabulary:Vocabulary):
    """
    Function to convert model tensor outputs to sentences

    Args:
        outputs (torch tensor object): Model's output to be decoded, either in size (seq len, batch, vocab_size) or (batch, seq len, vocab_size)
        batch_first (bool): Boolean of if dataloader was configured to batch_first
        vocabulary (Vocabulary): dataset Vocabulary Class for decoding

    Returns:
        list of predicted sentences each corresponding to 1 sample in the batch
            - will be of length (batch_size)
            eg. ['predicted sentence 1 for sample 1', ...'predicted sentence N for sample N']
    
    """

    all_prediction = []
    predicted_tokens = outputs.argmax(-1) #flatten vocab size dimensions
    if not batch_first:
        predicted_tokens = predicted_tokens.T 
    
    for sentence_tokens in predicted_tokens:
        sentence_tokens = sentence_tokens.tolist()

        try:
            #cropping predicted sentence to first EOS
            eos_index = sentence_tokens.index(vocabulary.stoi['<EOS>']) #get first instance of <EOS> to crop sentence accordingly
            predicted_sentence = sentence_tokens[:eos_index]
        except:
            predicted_sentence = sentence_tokens

        try:
            #getting predicted_sentence by remove <SOS>
            predicted_sentence.remove(vocabulary.stoi['<SOS>'])
        except:
            pass
    
        all_prediction.append(" ".join([vocabulary.itos[idx] for idx in predicted_sentence]))

    return all_prediction


def visualise_graph(training_data:list, validation_data:list, y_label:str, x_label:str, ylim:list=None):
    """
    Plot a line graph against training and validation data

    Args:
        training_data (list): Training data to be plotted
        validation_data (list): Validation data to be plotted
        y_label (str): Label of y axis
        x_label (str): Label of x axis
        ylim (list, optional): Range of y axis, defaults to None
    """
    #plotting line graph of training data and validation data
    plt.plot(training_data, label='Train')
    plt.plot(validation_data, label='Validation')

    #labels and title
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'{y_label} against {x_label}')
    
    if ylim != None:
        plt.ylim(ylim)
        
    #show legend
    plt.legend()
    plt.show()
    


if __name__ == "__main__":
    pass