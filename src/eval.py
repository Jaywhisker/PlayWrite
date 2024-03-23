import torch
from src.data.dataloader import *
from src.evaluation_metrics.bleu import get_bleu_score
from src.evaluation_metrics.rouge import get_rouge_score
from src.evaluation_metrics.utils import decode_predictions



"""
Eval code for image caption model or finetune CNN
1. eval() -> Image caption
2. finetuning_eval() -> finetune CNN
"""


def eval(model, 
         criterion, 
         dataloader, 
         image_size:tuple,
         transformer:bool,
         batch_first:bool, 
         vocabulary:Vocabulary, 
         device:str):
    """
    Function to evaluate model performance

    Args:
        model: The model that is to be evaluated
        criterion: Loss criterion of the model
        dataloader: validation / test dataset
        image_size (tuple): image size of model
        transformer (bool): boolean if decoder is a transformer
        batch_first (bool): boolean if dataloader samples tensor are (batch, seq len) or (seq len, batch)
        vocabulary (Vocabulary): dataset vocabulary class
        device (str): cpu or cuda

    Returns:
        avg_val_loss: average validation loss
        Bleu_score: dictionary of BLEU 1-4 score
        Rouge_score: dictionary of Rouge  1,2,L,LSum score
    """
    
    model.eval()

    total_val_loss = 0

    #BLEU predictions container
    predictions = []
    references = []
    
    with torch.no_grad():
        for idx, (imgs, annotations, all_annotations) in enumerate(dataloader):
            #getting img and annotations
            imgs = torch.nn.functional.interpolate(imgs, size=image_size, mode='bilinear') #resize image for model, using same as transforms.resize()
            imgs = imgs.to(device)
            annotations = annotations.to(device)
            
            if transformer:
                #running model prediction
                outputs = model(imgs, annotations[:-1]) #training model to guess the last word
                targets = annotations[1:].reshape(-1)
                #updating model parameters
                loss = criterion(outputs.view(-1, len(vocabulary)), targets)
            
            else:
                if not batch_first:
                    #running model prediction
                    outputs = model(imgs, annotations[:-1]) #training model to guess the last word
                    
                    #updating model parameters
                    loss = criterion(outputs.reshape(-1, outputs.shape[2]), annotations.reshape(-1)) #reshape output (seq_len, N, vocabulary_size) to (N, vocabulary_size)
                
                if batch_first:
                    #running model prediction
                    outputs, atten_weights = model(imgs, annotations[:, :-1]) #training model to guess the last word
                    targets = annotations[:, 1:]
                    #updating model parameters
                    loss = criterion(outputs.view(-1, len(vocabulary)), targets.reshape(-1)) #reshape output (seq_len, N, vocabulary_size) to (N, vocabulary_size)
            
            total_val_loss += loss.item()

            #get model predictions and update
            predictions.extend(decode_predictions(outputs, batch_first, vocabulary))

            #update references
            references.extend(all_annotations)

        Bleu_score = get_bleu_score(predictions, references)
        Rouge_score = get_rouge_score(predictions, references)

        return total_val_loss/(idx+1), Bleu_score, Rouge_score



def finetuning_eval(
    model, 
    criterion, 
    val_dataloader, 
    device:str='cuda'):

  """
  Evaluation function for finetuning CNN models

  Args:
    model: CNN model
    criterion: Loss function
    val_dataloader: Validation / Test dataloader
    device (str, optional): 'cpu' or 'cuda', defaults to cuda

  Returns:
      accuracy: float of the accuracy
      avg_val_loss: float of the average val loss
  """

  #set model to eval mode
  model.eval()

  #variables 
  val_loss = 0.0
  correct = 0
  total = 0

  with torch.no_grad():
      for image, label in val_dataloader:
        image = image.to(device)
        label = label.to(device)

        outputs = model(image) #predict label 
        loss = criterion(outputs, label) #calculate loss
        val_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item() #check if predicted is the same as the label

  #calculate accuracy and loss
  accuracy = (correct / total) * 100
  avg_val_loss = val_loss / len(val_dataloader)

  return accuracy, avg_val_loss


if __name__ == "__main__":
    pass