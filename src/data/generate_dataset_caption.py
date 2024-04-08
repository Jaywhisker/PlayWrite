from transformers import AutoProcessor, AutoModelForVision2Seq #If we using kosmos-2
from transformers import Blip2Processor, Blip2ForConditionalGeneration #If we using BLIP (JIC we want more data)
from PIL import Image
import os
import pandas as pd
from tqdm import tqdm

"""
Create Image Captions for a set of images using Kosmos2 and Blip2 Models
    - Aids with generation of datasets
"""

def kosmos2_image_text_pair(model:AutoModelForVision2Seq, processor:AutoProcessor, source_folder_path=str, dest_folder_path=str):
    """
    Function to generate image caption pair using kosmos_2 model. 
    
    Args:
        model (AutoModelForVision2Seq): Kosmos-2 model,
        processor (AutoProcessor): Kosmos-2 model tokenizer
        source_folder_path (str): Path to image folder containing all images
        dest_folder_path (str): Path to labels folder where the csv will be stored as label.csv
    
    Output:
        CSV file that container the image name and the respective image caption generated
        CSV file will auto save every 100 images as a checkpoint    
    """
    all_images_names = os.listdir(source_folder_path)
    text_input = "<grounding>An image of"

    image_dataframe = pd.DataFrame(columns=['image_filename', 'image_caption'])

    with tqdm(total=len(all_images_names)) as pbar:
        for index, image_name in enumerate(all_images_names):
            #read image
            image_path = f"{source_folder_path}/{image_name}"
            image_input = Image.open(image_path)

            #process image and text input into tensors
            inputs = processor(text=text_input, images=image_input, return_tensors="pt").to("cuda")

            #run data through the model
            generated_ids = model.generate(
                pixel_values=inputs["pixel_values"],
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_embeds=None,
                image_embeds_position_mask=inputs["image_embeds_position_mask"],
                use_cache=True,
                max_new_tokens=128,
            )

            #decode output into readable text
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            processed_text, entities = processor.post_process_generation(generated_text)
            image_caption = processed_text.replace("An image of ", "")
            image_dataframe.loc[len(image_dataframe)] = [image_name, image_caption.strip()]

            #Auto checkpoint saving at every 100
            if index % 100 == 0:
                image_dataframe.to_csv(f"{dest_folder_path}/Kosmos_Label.csv")

            pbar.update(1)
        #Save final data
        image_dataframe.to_csv(f"{dest_folder_path}/Kosmos_Label.csv")

        pbar.close()



def BLIP2_image_text_pair(model:Blip2ForConditionalGeneration, processor:Blip2Processor, source_folder_path=str, dest_folder_path=str):
    """
    Function to generate image caption pair using BLIP2 model. 
    
    Args:
        model (Blip2ForConditionalGeneration): Kosmos-2 model,
        processor (Blip2Processor): Kosmos-2 model tokenizer
        source_folder_path (str): Path to image folder containing all images
        dest_folder_path (str): Path to labels folder where the csv will be stored as label.csv
    
    Output:
        CSV file that container the image name and the respective image caption generated
        CSV file will auto save every 100 images as a checkpoint    
    """
    all_images_names = os.listdir(source_folder_path)

    image_dataframe = pd.DataFrame(columns=['image_filename', 'image_caption'])

    with tqdm(total=len(all_images_names)) as pbar:
        for index, image_name in enumerate(all_images_names):
            #read image
            image_path = f"{source_folder_path}/{image_name}"
            image_input = Image.open(image_path)

            #process image and text input into tensors
            inputs = processor(image_input, return_tensors="pt").to("cuda")

            #run data through the model
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=20,
            )

            #decode output into readable text
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            image_dataframe.loc[len(image_dataframe)] = [image_name, generated_text.strip()]

            #Auto checkpoint saving at every 100
            if index % 100 == 0:
                image_dataframe.to_csv(f"{dest_folder_path}/Blip_Label.csv")

            pbar.update(1)
        #Save entire dataset
        image_dataframe.to_csv(f"{dest_folder_path}/Blip_Label.csv")
        pbar.close()


if __name__ == "__main__":
    train_file_path = "../input/Image Captions/Train/Images"
    train_dest_path = "../input/Image Captions/Train/Labels"

    test_file_path = "../input/Image Captions/Test/Images"
    test_dest_path = "../input/Image Captions/Test/Labels"

    val_file_path = "../input/Image Captions/Validation/Images"
    val_dest_path = "../input/Image Captions/Validation/Labels"

    source_paths = [test_file_path, train_file_path, val_file_path]
    dest_paths = [test_dest_path, train_dest_path, val_dest_path]

    try:
        print("Creating Kosmos-2 Image Captions")
        kosmos_model = AutoModelForVision2Seq.from_pretrained("microsoft/kosmos-2-patch14-224").to("cuda")
        kosmos_processor = AutoProcessor.from_pretrained("microsoft/kosmos-2-patch14-224")
        
        for index, source_path in enumerate(source_paths):
            kosmos2_image_text_pair(kosmos_model, kosmos_processor, source_path, dest_paths[index])

        print("Kosmos-2 captions created successfully.\nCreating Blip2 Image Captions")

        blip_model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", device_map="auto")
        blip_processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

        for index, source_path in enumerate(source_paths):
            BLIP2_image_text_pair(blip_model, blip_processor, source_path, dest_paths[index])

        print("Blip2 captions created successfully.")

    except Exception as e:
        print(f"Failed to create captions due to {e} error")