import torch
from torchvision import models
from src.data.dataloader import *
from src.models.finetuned_cnn import PreTrainedCNNModels

"""
InceptionV3 and ResNet50 CNN and LSTM decoder with Adaptive Attention
1. InceptV3EncoderAttentionDecoder()
2. ResNetEncoderAttentionDecoder()
"""


################################################################
#                        InceptionV3 Encoder
################################################################
class InceptionV3EncoderCNN(torch.nn.Module):
    """
    InceptionV3 CNN Encoder Model, feature extraction layer (mixed_7c) is always trainable

    Args:
        finetuned_model: Finetuned inceptionV3 model, else None
        train_cnn (bool, optional): Determines if the entire CNN model will be unfreeze and trained during the training. Defaults to False.
    """
    def __init__(self, finetuned_model:PreTrainedCNNModels, train_cnn:bool=False):
        super(InceptionV3EncoderCNN, self).__init__()
        if finetuned_model != None:
            self.inception = list(finetuned_model.children())[0]
        
        else:
            self.inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT)
        self.inception.aux_logits = False
        
        #Remove last classification layer
        self.inception.fc = torch.nn.Identity()

        #Variable that will hold the features
        self.features = None
        
        #Register the hook to capture features at output of last CNN layer
        self.inception.Mixed_7c.register_forward_hook(self.capture_features_hook)

        #Train the feature map, the rest depends on train_CNN
        for name, param in self.inception.named_parameters():
            if 'Mixed_7c' in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(train_cnn)


    def capture_features_hook(self, module, input, output):
        self.features = output #update feature 


    def forward(self, images):
        """
        Take images and return feature maps of size (batch, height*width)
        """
        _ = self.inception(images)  #Pass through the inception network
        batch, feature_maps, size_1, size_2 = self.features.size()  #self.features contain the feature map of size (batch size, 2048, 8,8)
        features = self.features.permute(0, 2, 3, 1)
        features = features.view(batch, size_1*size_2, feature_maps) #resize to (batch size, h*w, feature_maps)

        return features


################################################################
#                        Resnet50 Encoder
################################################################
class ResNetEncoderCNN(torch.nn.Module):
    """
    ResNet50 CNN Encoder Model, feature extraction layer (fc) is always trainable

    Args:
        finetuned_model: Finetuned inceptionV3 model, else None
        train_cnn (bool, optional): Determines if the entire CNN model will be unfreeze and trained during the training
    """
    def __init__(self, finetuned_model:PreTrainedCNNModels, train_cnn:bool=False):
        super(ResNetEncoderCNN, self).__init__()

        if finetuned_model != None:
            resnet = list(finetuned_model.children())[0]
        else:
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        #Train the feature map, the rest depends on train_CNN
        for name, param in resnet.named_parameters(): 
            if "layer4.2.conv3.weight" in name:
                param.requires_grad_(True)
            else:
                param.requires_grad_(train_cnn)
        
        modules = list(resnet.children())[:-2]
        self.resnet = torch.nn.Sequential(*modules)
        
    def forward(self, images):
        """
        Take images and return feature maps of size (batch, height*width)
        """
        features = self.resnet(images)
        # first, we need to resize the tensor to be 
        # (batch, h*w, feature_maps)
        batch, feature_maps, size_1, size_2 = features.size()       
        features = features.permute(0, 2, 3, 1)
        features = features.view(batch, size_1*size_2, feature_maps)
       
        return features
    

################################################################
#                        Adaptive Attention
################################################################
class BahdanauAttention(torch.nn.Module):
    """
    Adaptive Attention Module

    Args:
        feature_dim (int): Dimension of feature maps (h*w)
        hidden_dim (int): Dimension of hidden states
        output_dim (int, optional): Dimension of output, default to 1
    """
    def __init__(self, feature_dim:int, hidden_dim:int, output_dim:int = 1):
        super(BahdanauAttention, self).__init__()
         # fully-connected layer to learn first weight matrix Wa
        self.W_a = torch.nn.Linear(feature_dim, hidden_dim)
        # fully-connected layer to learn the second weight matrix Ua
        self.U_a = torch.nn.Linear(hidden_dim, hidden_dim)
        # fully-connected layer to produce score (output), learning weight matrix va
        self.v_a = torch.nn.Linear(hidden_dim, output_dim)


    def forward(self, features, hidden_state):
        """
        Args:
            features: image features from Encoder
            hidden_state: hidden state output for Decoder

        Returns:
            context: context vector with size (1,2048)
            atten_weights: probabilities of feature relevance 
        """
        #add additional dimension to a hidden (required for summation) 
        hidden_state = hidden_state.unsqueeze(1) #(batch size, 1, seq length)

        atten_1 = self.W_a(features) #(batch size, h*w, hidden_dim)
        atten_2 = self.U_a(hidden_state) #(batch size, 1, hidden_dim)

        #apply tangent to combine result from 2 fc layers
        atten_tan = torch.tanh(atten_1+atten_2)
        atten_score = self.v_a(atten_tan) #(batch size, hidden_dim)
        atten_weight = torch.nn.functional.softmax(atten_score, dim = 1) #get softmax probablilities

        #multiply each vector with its softmax score and sum to get attention context vector
        context = torch.sum(atten_weight * features,  dim = 1) #size of context equals to a number of feature maps
        atten_weight = atten_weight.squeeze(dim=2)
        
        return context, atten_weight
    

################################################################
#                        LSTM Decoder
################################################################
class DecoderRNN(torch.nn.Module):
     """
     LSTM decoder model

     Args:
          feature_dim (int): Feature Map dimension (h*w)
          embed_size (int): Embedding dimension to embed words
          hidden_size (int): Hidden state dimension for LSTM
          vocab_size (int): Total number of unique vocab
          drop_prob (float, optional): Dropout layer probability, deafults to 0.5
          sample_temp (float, optional): Scale outputs before softmax to allow the model to be more picky as the differences are exaggerated. Defaults to 0.5
     """
     def __init__(self, feature_dim:int, embedding_dim:int, hidden_dim:int, vocab_size:int, drop_prob:float=0.5, sample_temp:float=0.5):
          super(DecoderRNN, self).__init__()
          
          self.feature_dim = feature_dim
          self.embedding_dim = embedding_dim
          self.hidden_dim = hidden_dim
          self.vocab_size = vocab_size
          self.sample_temp = sample_temp #scale the outputs b4 softmax

          #layers

          #embedding layer that turns words into index 
          self.embeddings = torch.nn.Embedding(vocab_size, embedding_dim)
          #lstm layer that takes in feature + embedding (image + caption) and output hidden_dim
          self.lstm = torch.nn.LSTMCell(embedding_dim + feature_dim, hidden_dim)
          #fc linear layer that predicts next word
          self.fc = torch.nn.Linear(hidden_dim, vocab_size)
          #attention layer
          self.attention = BahdanauAttention(feature_dim, hidden_dim)
          #dropout layer
          self.drop = torch.nn.Dropout(p=drop_prob)
          #initialisation of fully-connected layers
          self.init_h = torch.nn.Linear(feature_dim, hidden_dim) #initiialising hidden state and cell memory using avg of feature
          self.init_c = torch.nn.Linear(feature_dim, hidden_dim)

     def init_hidden(self, features):
          """
          Initializes hidden state and cell memory using average feature vector
          Args:
               features: feature map of the image
          Returns:
               h0: initial hidden state (short-term memory)
               c0: initial cell state (long-term memory)
          """
          mean_annotations = torch.mean(features, dim = 1) #getting average of the features
          h0 = self.init_h(mean_annotations)
          c0 = self.init_c(mean_annotations)
          return h0, c0

     def forward(self, features, captions, device:str, sample_prob:float=0.2):
          """
          Args:
               features: feature map of image
               captions: true caption of image
               device (str): cuda or cpu
               sample_prob (float, optional): Probability for auto-regressive RNN where they train on RNN output rather than true layer, defaults to 0.2

          """
          embed = self.embeddings(captions)
          h,c = self.init_hidden(features)
          batch_size = captions.size(0) #captions: (batch size, seq length)
          seq_len = captions.size(1) 
          feature_size = features.size(1) #features: (batch size, size, 2048)

          #storage of outputs and attention weights of lstm
          outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(device)
          atten_weights = torch.zeros(batch_size, seq_len, feature_size).to(device)

          #scheduled sampling for training, using the models output to train instead of using the true output
          #autoregressive RNN training, only when length of seq > 1 (cannot be first word)
          for t in range(seq_len):
               s_prob = 0.0 if t==0 else sample_prob
               use_sampling = np.random.random() < s_prob

               if not use_sampling: #no sampling
                    word_embeddings = embed[:, t, :] #embedding until word t
               
               context, atten_weight = self.attention(features,h)
               inputs = torch.cat([word_embeddings, context], 1) #embed captions and features for next lstm state
               h, c = self.lstm(inputs, (h,c)) #pass through lstm
               output = self.fc(self.drop(h))
               
               if use_sampling: #using predicted word instead of true output
                    scaled_output = output/self.sample_temp #using scaling temp to amplify the values
                    #this way softmax will have a larger difference in values
                    #makes the model more selective of whats its picking 
                    
                    scoring = torch.nn.functional.log_softmax(scaled_output, dim=1)
                    top_idx = scoring.topk(1)[1]
                    word_embeddings = self.embeddings(top_idx).squeeze(1) #update word embeddings with predicted instead of actual
               
               #update results
               outputs[:,t,:] = output
               atten_weights[:, t, :] = atten_weight

          return outputs, atten_weights


################################################################
#                   InceptV3 Image Captioning
################################################################
class InceptV3EncoderAttentionDecoder(torch.nn.Module):
    """
    InceptionV3 Encoder with Attention and LSTM Decoder

    Args:
        finetuned_model: Finetuned InceptionV3 model, else None
        feature_dim (int): Feature Map dimension (h*w)
        embedding_dim (int): Embedding dimension to embed words
        hidden_dim (int): Hidden state dimension for LSTM
        vocab_size (int): Total number of unique vocab
        device (str): cuda or cpu
        train_cnn (boolean, optional): Determines if inceptionCNN model is unfreezed. Defaults to False.
        drop_prob (float, optional): Dropout layer probability. Defaults to 0.5.
        sample_temp (float, optional): Scale outputs before softmax to allow the model to be more picky as the differences are exaggerated. Defaults to 0.5
    """
    def __init__(self, finetuned_model:PreTrainedCNNModels, feature_dim:int, embedding_dim:int, hidden_dim:int, vocab_size:int, device:str, train_cnn:bool=False, drop_prob:float=0.5, sample_temp:float=0.5):
        super(InceptV3EncoderAttentionDecoder, self).__init__()
        self.encoder = InceptionV3EncoderCNN(finetuned_model, train_cnn)
        self.decoder= DecoderRNN(feature_dim, embedding_dim, hidden_dim, vocab_size, drop_prob, sample_temp)
        self.sample_temp = sample_temp
        self.device = device

    def forward(self, image, captions):
        features = self.encoder(image)
        outputs, atten_weights = self.decoder(features, captions, self.device, self.sample_temp)
        return outputs, atten_weights
    

    #for inference
    def caption_image(self, image, vocabulary:Vocabulary, device:str, max_length:int=50):
        """
        Generate caption using a greedy algorithm based on image input

        Args:
            image: image input
            vocabulary (Vocabulary): Vocabulary to decode predictions
            device (str): cuda or cpu
            max_length (int, optional): Max length of generated captions. Defaults to 50.

        Returns:
            captions: string caption in a list
            atten_weights: probabilities of feature relevance 
        """
        self.encoder.eval()

        result_caption = []
        result_weights = []

        with torch.no_grad(): #no training
            input_word = torch.tensor(1).unsqueeze(0).to(device)
            result_caption.append(1)
            features = self.encoder(image)
            h, c = self.decoder.init_hidden(features)

            for _ in range(max_length):
                embedded_word = self.decoder.embeddings(input_word)
                context, atten_weight = self.decoder.attention(features, h)
                # input_concat shape at time step t = (batch, embedding_dim + context size)
                input_concat = torch.cat([embedded_word, context],  dim = 1)
                h, c = self.decoder.lstm(input_concat, (h,c))
                h = self.decoder.drop(h)
                output = self.decoder.fc(h) 
                scoring = torch.nn.functional.log_softmax(output, dim=1)
                top_idx = scoring[0].topk(1)[1]
                result_caption.append(top_idx.item())
                result_weights.append(atten_weight)
                input_word = top_idx

                if (len(result_caption) >= max_length or vocabulary.itos[input_word.item()] == "<EOS>"):
                    break

            return [vocabulary.itos[idx] for idx in result_caption], result_weights


################################################################
#                   ResNet Image Captioning
################################################################
class ResNetEncoderAttentionDecoder(torch.nn.Module):
    """
    Resnet50 Encoder with Attention and LSTM Decoder

    Args:
        finetuned_model: Finetuned InceptionV3 model, else None
        feature_dim (int): Feature Map dimension (h*w)
        embedding_dim (int): Embedding dimension to embed words
        hidden_dim (int): Hidden state dimension for LSTM
        vocab_size (int): Total number of unique vocab
        device (str): cuda or cpu
        train_cnn (boolean, optional): Determines if inceptionCNN model is unfreezed. Defaults to False.
        drop_prob (float, optional): Dropout layer probability. Defaults to 0.5.
        sample_temp (float, optional): Scale outputs before softmax to allow the model to be more picky as the differences are exaggerated. Defaults to 0.5
    """
    def __init__(self, finetuned_model:PreTrainedCNNModels, feature_dim:int, embedding_dim:int, hidden_dim:int, vocab_size:int, device:str, train_cnn:bool=False, drop_prob:float=0.5, sample_temp:float=0.5):
        super(ResNetEncoderAttentionDecoder, self).__init__()
        self.encoder = ResNetEncoderCNN(finetuned_model, train_cnn)
        self.decoder= DecoderRNN(feature_dim, embedding_dim, hidden_dim, vocab_size, drop_prob, sample_temp)
        self.sample_temp = sample_temp
        self.device = device

    def forward(self, image, captions):
        features = self.encoder(image)
        outputs, atten_weights = self.decoder(features, captions, self.device, self.sample_temp)
        return outputs, atten_weights
    

    #for inference
    def caption_image(self, image, vocabulary:Vocabulary, device:str, max_length:int=50):
        """
        Generate caption using a greedy algorithm based on image input

        Args:
            image: image input
            vocabulary (Vocabulary): Vocabulary to decode predictions
            device (str): cuda or cpu
            max_length (int, optional): Max length of generated captions. Defaults to 50.

        Returns:
            captions: string caption in a list
            atten_weights: probabilities of feature relevance 
        """
        self.encoder.eval()

        result_caption = []
        result_weights = []

        with torch.no_grad(): #no training
            input_word = torch.tensor(1).unsqueeze(0).to(device)
            result_caption.append(1)
            features = self.encoder(image)
            h, c = self.decoder.init_hidden(features)

            for _ in range(max_length):
                embedded_word = self.decoder.embeddings(input_word)
                context, atten_weight = self.decoder.attention(features, h)
                # input_concat shape at time step t = (batch, embedding_dim + context size)
                input_concat = torch.cat([embedded_word, context],  dim = 1)
                h, c = self.decoder.lstm(input_concat, (h,c))
                h = self.decoder.drop(h)
                output = self.decoder.fc(h) 
                scoring = torch.nn.functional.log_softmax(output, dim=1)
                top_idx = scoring[0].topk(1)[1]
                result_caption.append(top_idx.item())
                result_weights.append(atten_weight)
                input_word = top_idx

                if (len(result_caption) >= max_length or vocabulary.itos[input_word.item()] == "<EOS>"):
                    break

            return [vocabulary.itos[idx] for idx in result_caption], result_weights


if __name__ == '__main__':
    pass