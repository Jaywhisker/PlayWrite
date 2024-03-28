import torch
from src.data.dataloader import *
from src.eval import eval, finetuning_eval
from src.evaluation_metrics.bleu import get_bleu_score
from src.evaluation_metrics.rouge import get_rouge_score
from src.evaluation_metrics.utils import decode_predictions


"""
Training code for image caption model or finetune CNN
1. train() -> Image caption
2. finetuning_train() -> finetune CNN
"""


def train(model, 
          criterion, 
          optimiser, 
          train_dataloader, 
          val_dataloader, 
          image_size: tuple,
          batch_first:bool, 
          transformer:bool,
          vocabulary:Vocabulary, 
          device:str, 
          num_epochs:int, 
          early_stopping_threshold: float=0.003,
          show_train_metrics:bool=None, 
          save_every:int=None,
          model_name:str=None, 
          overwrite:bool=False):
    
    """
    Function to train the model
    If change in val loss is less than 0.01 for 2 epochs in a row, stop training

    Args:
        model: The model that is to be evaluated
        criterion: Loss criterion of the model
        optimiser: Optimiser function of the model
        train_dataloader: Train dataset
        val_dataloader: Validation dataset, use None if no Validation dataset
        image_size (tuple): image size of model
        batch_first (bool): Boolean if dataloader samples tensor are (batch, seq len) or (seq len, batch)
        transformer (bool): Boolean if decoder is a transformer
        vocabulary (Vocabulary): Dataset vocabulary class
        device (str): cpu or cuda
        num_epochs (int): Number of epochs for training
        early_stopping_threshold (float, optional): Threshold for early stopping, defaults to 0.003
        show_train_metrics (bool, optional): Booleon on should calculate BLEU & Rouge score during training, defaults to False
        save_every (int, optional): Save model after every ___ epochs, defaults to None (no saving)
        model_name (str, optional): Model Name to be saved after, required if save_every != None, model will be saved as (model_name)_epoch or just model_name
        overwrite (bool, optional): Boolean on overwriting model saves or saving each specific epoch as a new model, defaults to False
    
    Returns
        train_loss: list of average training loss per epoch
        train_bleu: list of dictionary of training BLEU score per epoch, [] if show_train_metric = False
        train_rouge: list of dictionary of training Rouge score per epoch, [] if show_train_metric = False
        val_loss: list of average validation loss per epoch, [] if val_dataloader = None
        val_bleu: list of dictionary of validation BLEU score per epoch, [] if val_dataloader = None
        val_rouge: list of dictionary of validation Rouge score per epoch, [] if val_dataloader = None
    
    """
    
    #initialise results container
    train_loss = []
    train_bleu = []
    train_rouge = []

    val_loss = []
    val_bleu = []
    val_rouge = []

    for epoch in range(num_epochs):

        total_train_loss = 0

        #BLEU predictions container
        predictions = []
        references = []

        #start model training
        model.train()
        for idx, (imgs, annotations, all_annotations) in enumerate(train_dataloader):
            
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
                
            optimiser.zero_grad() #remove optimiser gradient
            loss.backward()
            optimiser.step()
            
            #calculate loss and update it for each batch
            total_train_loss += loss.item()

            if show_train_metrics:
                #get model predictions and update
                predictions.extend(decode_predictions(outputs, batch_first, vocabulary))

                #update references
                references.extend(all_annotations)

        if show_train_metrics:   
            #calculating bleu and rouge score
            Bleu_score = get_bleu_score(predictions, references)
            Rouge_score = get_rouge_score(predictions, references)
            train_bleu.append(Bleu_score)
            train_rouge.append(Rouge_score)

        #updating values
        train_loss.append(total_train_loss/(idx+1))

        if val_dataloader != None:
            #validation
            avg_val_loss, val_bleu_score, val_rouge_score = eval(
                                                                model=model,
                                                                criterion=criterion,
                                                                dataloader=val_dataloader,
                                                                image_size=image_size,
                                                                transformer=transformer,
                                                                batch_first=batch_first,
                                                                vocabulary=vocabulary,
                                                                device=device
                                                            )
            
            val_loss.append(avg_val_loss)
            val_bleu.append(val_bleu_score)
            val_rouge.append(val_rouge_score)

        #printing progress
        if num_epochs <= 10 or (num_epochs >10 and (epoch+1)%5 == 0):
            print(f"Epoch {epoch+1} completed\navg training loss per batch: {total_train_loss/(idx+1)}")
            
            if show_train_metrics:
                print(f"train bleu score:{Bleu_score}\ntrain rouge score: {Rouge_score}\n")

            if val_dataloader != None:
                print(f"avg validation loss per batch: {avg_val_loss}\nval bleu score: {val_bleu_score}\nval rouge score: {val_rouge_score}")

            print("------------------------------------------------------------------")
            
        #saving model every x
        if save_every != None and (epoch+1)%save_every == 0:
            try:
                if overwrite:
                    torch.save(model.state_dict(), f"../models/image_captioning/{model_name}.pt")
                else:
                    torch.save(model.state_dict(), f"../models/image_captioning/{model_name}_{epoch+1}.pt")
            except:
                print(f"Unable to save model at epoch {epoch+1}")

        
        #saving best model
        if (len(val_loss) > 1) and val_loss[-1] < min(val_loss[:-1]):
            try:
                torch.save(model.state_dict(), f"../models/image_captioning/{model_name}_best.pt")
            except:
                print(f"Unable to save best model")
        

        #early stopping
        if (len(val_loss) >= 3) and abs(val_loss[-2] - val_loss[-1]) < early_stopping_threshold and abs(val_loss[-3] - val_loss[-2]) < early_stopping_threshold:
            print(f"validation loss did not decrease, stopping training at epoch {epoch +1}")
            try:
                if overwrite:
                    torch.save(model.state_dict(), f"../models/image_captioning/{model_name}.pt")
                else:
                    torch.save(model.state_dict(), f"../models/image_captioning/{model_name}_{epoch+1}.pt")
            except:
                print(f"Unable to save model at epoch {epoch+1}")
            break


    return train_loss, train_bleu, train_rouge, val_loss, val_bleu, val_rouge



def finetuning_train(model, 
                  criterion, 
                  optimiser, 
                  train_dataloader, 
                  val_dataloader=None, 
                  num_epochs:int=25, 
                  device:str='cuda', 
                  model_name:str=None):
  
  """
  Training function for CNN models

  Args:
    model: CNN model to be finetuned
    criterion: Loss function
    optimiser: Optimiser function of the model
    train_dataloader: Train Dataset
    val_dataloader (optional): Validation Dataset, use None if no Validation dataset
    num_epochs (int, optional): Number of training epochs, defaults to 25
    device (str, optional): cpu or cuda, defaults to cuda
    model_name (str, optional): Model Name for saving model, None if no saving to be done. Defaults to None

  Returns:
      train_loss: list of the average train loss per epoch
      val_loss: list of the average val loss per epoch
  """

  #initialising results container
  train_loss = []
  val_loss = []

  for epoch in range(num_epochs):

    #model in train mode
    model.train()
    training_loss = 0.0

    #training model
    for image, label in tqdm(train_dataloader):
      image = image.to(device)
      label = label.to(device)

      optimiser.zero_grad()
      outputs = model(image)
      loss = criterion(outputs, label)
      loss.backward()
      optimiser.step()
      training_loss += loss.item()

    epoch_loss = training_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} completed, training loss: {epoch_loss}")

    #train if there is a validation dataset
    if val_dataloader != None:
      accuracy, avg_val_loss = finetuning_eval(model, criterion, val_dataloader, device)
      print(f"Validation loss: {avg_val_loss}, Validation Accuracy: {accuracy:.2f}")
      val_loss.append(avg_val_loss)

    train_loss.append(epoch_loss)

    #saving best model based on lowest validation loss (no validation dataset = no best model saving)
    #only runs if there is a model name to save the model on
    if len(val_loss) > 1 and val_loss[-1] < min(val_loss[:-1]) and model_name!=None:
      torch.save(model, f'../models/cnn/{model_name}_best.pt')
      print("Best model saved")

  #saving the last model if there is a model name
  if model_name != None:
    torch.save(model, f'../models/cnn/{model_name}_{num_epochs}.pt')
  
  return train_loss, val_loss


if __name__ == "__main__":
    pass