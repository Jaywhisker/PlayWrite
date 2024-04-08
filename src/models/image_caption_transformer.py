import timm
import math
import torch
from torchvision import models
from src.data.dataloader import *
from src.models.finetuned_cnn import PreTrainedCNNModels

"""
Maxvit encoder and transformer Decoder
1. TransformerEncoderToDecoder()
"""

class EncoderMaxVit(torch.nn.Module):
  """
    An encoder module based on the MaxVit architecture, adapted for image captioning by converting images into dense feature embeddings.

    The final fully connected (fc) layer of the MaxVit model is replaced to map the extracted features to the desired embedding size.
    Feature extraction layer (classifier.5 for pretrained and head for finetuned) is always trainable.

    Args:
      finetuned_model: Finetuned MaxVit model, else None
      embedding_size (int): Embedding Dimension for size of feature embedding
      train_CNN (bool, optional): Determines if the entire MaxVit model will be unfreeze and trained during the training. Defaults to False.
      drop_p (float, optional): Dropout probability to use in the dropout layer for regularization. Defaults to 0.5.
  """
  
  def __init__(self, finetuned_model:PreTrainedCNNModels, embedding_size:int, train_CNN:bool=False, drop_p:float=0.5):
    super(EncoderMaxVit, self).__init__()
    self.embed_size = embedding_size

    if finetuned_model is not None:
      # Initialize custom finetuned model and make the last linear layer trainable
      self.model = list(finetuned_model.children())[0]
      self.model.classifier[-1] = torch.nn.Linear(in_features=self.model.classifier[-1].in_features, out_features=self.embed_size, bias=True)
      #Remove creating feature extraction layer and make it trainable, the rest depends on train_cnn
      for name, param in self.model.named_parameters():
        if "classifier.5.weight" in name or "classifier.5.bias" in name:
          param.requires_grad = True
        else:
          param.requires_grad = train_CNN

    else:
      # Initialize pretrained model and make last linear layer trainable
      model_used = 'maxvit_rmlp_tiny_rw_256.sw_in1k'
      self.model = timm.create_model(model_used, pretrained=True)
      self.model.head.fc = torch.nn.Linear(in_features=self.model.head.fc.in_features, out_features=self.embed_size, bias=True)

      for name , param in self.model.named_parameters():
        if "head.fc.weight" in name or "head.fc.bias" in name:
          param.requires_grad = True
        else:
          param.requires_grad = train_CNN

    self.relu = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(drop_p)
    
              
  def forward(self, image):
    output = self.model(image)
    output = self.dropout(self.relu(output))
    
    return output
  

class PositionalEncoding(torch.nn.Module):
  """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    Embeddings do not encode the relative position of tokens in a sentence. 
    With positional encoding, words will be closer to each other based on the similarity of their meaning and their position in the sequence, in the d-dimensional space.

    Args:
      d_model (int): The dimension of the embeddings.
      dropout (float, optional): The dropout probability.
      max_len (int, optional): The maximum length of the input sequences.
  """
  
  def __init__(self, d_model: int, dropout: float=0.1, max_len: int=5000):
    super().__init__()

    self.dropout = torch.nn.Dropout(p=dropout)
    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer('pe', pe)

  def forward(self, x):
    """
      Args:
        x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
    """
    x = x + self.pe[:x.size(0)]
    return self.dropout(x)
  


class TransformersDecoder(torch.nn.Module):
  """
    A Transformer-based decoder that generates captions from image features. 
    Uses the features from the encoder as the memory input to the Transformer layers,
    along with the target tokens (captions) which are shifted right.
    It applies positional encoding to the target token embeddings to maintain their sequence information.

    Args:
      embedding_size (int): The size of the embedding vector for each token.
      trg_vocab_size (int): The size of the target vocabulary.
      num_heads (int): The number of heads in the multihead-attention models.
      num_decoder_layers (int): The number of sub-decoder-layers in the decoder.
      dropout (float): The dropout probability.

    The decoder generates predictions for the next token in the sequence, given the current sequence of tokens and the image features.
  """
   
  def __init__(self, embedding_size:int, trg_vocab_size:int, num_heads:int, num_decoder_layers:int, dropout:float):
    super(TransformersDecoder, self).__init__()
    self.num_heads = num_heads

    self.embedding = torch.nn.Embedding(trg_vocab_size, embedding_size)
    self.pos = PositionalEncoding(d_model=embedding_size)
    self.decoder_layer = torch.nn.TransformerDecoderLayer(d_model=embedding_size, nhead=num_heads)
    self.decoder = torch.nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
    self.linear = torch.nn.Linear(embedding_size, trg_vocab_size)
    self.drop = torch.nn.Dropout(dropout)
      
  def make_mask(self, sz):
    """
      Generate a square attention mask of size (sz, sz),
      with upper triangular filled with float('-inf').
    """

    mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
    return mask
  
  def forward(self, features, caption, device):
    embed = self.drop(self.embedding(caption))
    embed = self.pos(embed)
    trg_mask = self.make_mask(caption.size(0)).to(device)
    decoder = self.decoder(tgt = embed , memory = features.unsqueeze(0), tgt_mask = trg_mask )
    output = self.linear(decoder)

    return output
  

class TransformerEncoderToDecoder(torch.nn.Module):
  """
    MaxVit Encoder with Transformer Decoder

    Args:
        finetuned_model: Finetuned MaxVit model, else None
        embedding_size (int): The size of the embedding vector for the image features and the tokens.
        trg_vocab_size (int): The size of the target vocabulary for the decoder.
        num_heads (int, optional): The number of attention heads in the Transformer decoder. Defaults to 8
        num_decoder_layers (int, optional): The number of layers in the Transformer decoder. Defaults to 6
        dropout (float, optional): The dropout probability used in the Transformer decoder. Defaults to 0.2
        train_cnn (bool, optional): Determines if MaxVit Encoder should be unfreezed. Defaults to False
        device (str, optional): cuda or cpu. Defaults to cuda
  """
  
  def __init__(self, finetuned_model:PreTrainedCNNModels, embedding_size:int, trg_vocab_size:int, num_heads:int=8, num_decoder_layers:int=6, dropout:float=0.2, train_cnn:bool=False, device:str='cuda'):
    super(TransformerEncoderToDecoder,self).__init__()
    self.device = device
    self.encoder = EncoderMaxVit(finetuned_model, embedding_size, train_cnn)
    self.decoder = TransformersDecoder(embedding_size=embedding_size,
                                        trg_vocab_size=trg_vocab_size,
                                        num_heads=num_heads,
                                        num_decoder_layers=num_decoder_layers,
                                        dropout=dropout)
      
  def forward(self, image, caption):
    features = self.encoder(image)
    output = self.decoder(features, caption, self.device)
    return output
  
  #for inference
  def caption_image(self, image, vocabulary, device, max_length=50):
    """
    Generate caption using a greedy algorithm based on image input

    Args:
        image: image input
        vocabulary (Vocabulary): Vocabulary to decode predictions
        device (str): cuda or cpu
        max_length (int, optional): Max length of generated captions. Defaults to 50.

    Returns:
        captions: string caption in a list
        atten_weights: None for transformer
    """
    outputs=[vocabulary.stoi["<SOS>"]]

    for i in range(max_length):
      trg_tensor = torch.LongTensor(outputs).unsqueeze(1).to(device)
      image = image.to(device)
      
      with torch.no_grad():
        output = self.forward(image, trg_tensor)
          
      best_guess = output.argmax(2)[-1, :].item()
      outputs.append(best_guess)
      
      if best_guess == vocabulary.stoi["<EOS>"]:
        break

    caption = [vocabulary.itos[idx] for idx in outputs]
    
    return caption[:-1], None
  


if __name__ == '__main__':
    pass