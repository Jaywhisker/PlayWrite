import torch
from torchvision import models
from src.data.dataloader import *


"""
InceptionV3 CNN and LSTM decoder
1. CNNtoRNN()
"""


# CNN encoder (using inceptionV3)
class EncoderCNN(torch.nn.Module):
  """
  InceptionV3 CNN Encoder Model, last layer is always trainable

  Args:
      finetuned_model: Finetuned InceptionV3 model, else None
      embed_size (int): Embedding dimension to convert features size to
      train_CNN (bool, optional): Determines if the entire CNN model will be unfreeze and trained during the training
  """
  def __init__(self, finetuned_model, embed_size:int, train_CNN:bool=False):
    super(EncoderCNN, self).__init__()
    #load pre-trained model
    if finetuned_model != None:
        self.inception = list(finetuned_model.children())[0]
    else:
        self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
    self.inception.aux_logits = False

    #converting the last layer of inception to linear layer [inception last layer input, embed size]
    self.inception.fc = torch.nn.Linear(self.inception.fc.in_features, embed_size) 
    #Train the feature map, the rest depends on train_CNN
    for name, param in self.inception.named_parameters():
      if "fc.weight" in name or "fc.bias" in name:
        param.requires_grad = True #finetuning the last layer
      else:
        param.requires_grad = train_CNN

    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=0.5)


  def forward(self, images):
    features = self.inception(images)
    return self.dropout(self.relu(features))


class DecoderRNN(torch.nn.Module):
  """
  LSTM decoder model

  Args:
      embed_size (int): Embedding dimension to embed words
      hidden_size (int): Hidden state dimension for LSTM
      vocab_size (int): Total number of unique vocab
      num_layers (int): Total number of LSTM layers
  """
  def __init__(self, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int):
    super(DecoderRNN, self).__init__()
    self.embed = torch.nn.Embedding(vocab_size, embed_size) #embed / tokenize the word
    self.lstm = torch.nn.LSTM(embed_size, hidden_size, num_layers)
    self.linear = torch.nn.Linear(hidden_size, vocab_size)  #classification layer
    self.dropout = torch.nn.Dropout(0.5)

  def forward(self, features, captions):
    embeddings = self.dropout(self.embed(captions))
    embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)

    hiddens, _ = self.lstm(embeddings)
    outputs = self.linear(hiddens)
    return outputs
  
class CNNtoRNN(torch.nn.Module):
  """
  Model that merges Encoder CNN to Decoder RNN

  Args:
      embed_size (int): Embedding dimension to embed words
      hidden_size (int): Hidden state dimension for LSTM
      vocab_size (int): Total number of unique vocab
      num_layers (int): Total number of LSTM layers
      train_cnn (bool, optional): Determines if unfreezing entire InceptionV3 model, defaults to False
  """
  def __init__(self, finetuned_model, embed_size:int, hidden_size:int, vocab_size:int, num_layers:int, train_cnn:bool=False):
    super(CNNtoRNN, self).__init__()
    self.encoderCNN = EncoderCNN(finetuned_model, embed_size, train_cnn)
    self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

  #for training with a caption
  def forward(self, images, captions):
    features = self.encoderCNN(images)
    outputs = self.decoderRNN(features, captions)
    return outputs

  #for prediction where there is a semi caption and they have to continue the caption
  def caption_image(self, image, vocabulary, device, max_length=50):
    result_caption = []

    with torch.no_grad(): #no training
      x = self.encoderCNN(image.to(device)).unsqueeze(0) #image as input
      states = None

      for _ in range(max_length):
        hiddens, states = self.decoderRNN.lstm(x, states) #image is the initial state for the LSTM
        output = self.decoderRNN.linear(hiddens.squeeze(0)) #get the output from the LSTM for the first word
        predicted = output.argmax(1) #first word

        result_caption.append(predicted.item())
        x = self.decoderRNN.embed(predicted).unsqueeze(0) #update the input to the lstm to now be the first word

        if vocabulary.itos[predicted.item()] == "<EOS>":
          break

    return [vocabulary.itos[idx] for idx in result_caption], None  #get the string from the vocabulary instead of the indices, no attention


if __name__ == '__main__':
    pass