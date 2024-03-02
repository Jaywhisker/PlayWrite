import torch
from src.data.dataloader import *
from src.evaluation_metrics.bleu import get_bleu_score
from src.evaluation_metrics.rouge import get_rouge_score
from src.evaluation_metrics.decode_prediction import decode_predictions

def train(model, 
          criterion, 
          optimiser, 
          train_dataloader, 
          val_dataloader, 
          batch_first:bool, 
          vocabulary:Vocabulary, 
          device:str, 
          num_epochs:int, 
          show_train_metrics:bool=None, 
          save_every:int=None,
          model_name:str=None, 
          overwrite:bool=False):
    
    """
    Function to train the model

    Args:
        model: The model that is to be evaluated
        criterion: Loss criterion of the model
        optimiser: Optimiser function of the model
        train_dataloader: Train dataset
        val_dataloader: Validation dataset, use None if no Validation dataset
        batch_first (bool): Boolean if dataloader samples tensor are (batch, seq len) or (seq len, batch)
        vocabulary (Vocabulary): Dataset vocabulary class
        device (str): cpu or cuda
        num_epochs (int): Number of epochs for training
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
            imgs = imgs.to(device)
            annotations = annotations.to(device)
            #running model prediction
            outputs = model(imgs, annotations[:-1]) #training model to guess the last word
            
            #updating model parameters
            loss = criterion(outputs.reshape(-1, outputs.shape[2]), annotations.reshape(-1)) #reshape output (seq_len, N, vocabulary_size) to (N, vocabulary_size)
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
            
        #saving model
        if save_every != None and (epoch+1)%save_every == 0:
            try:
                if overwrite:
                    torch.save(model.state_dict(), f"../models/image_captioning/{model_name}.pt")
                else:
                    torch.save(model.state_dict(), f"../models/image_captioning/{model_name}_{epoch+1}.pt")
            except:
                print(f"Unable to save model at epoch {epoch+1}")

    return train_loss, train_bleu, train_rouge, val_loss, val_bleu, val_rouge


if __name__ == "__main__":
    pass