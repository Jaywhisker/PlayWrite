#Imports
import pickle
import io
import os
import sys
import gc
import soundfile as sf

from PIL import Image
import numpy as np
import torch
from torchvision import transforms, models
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


# Resolving importing errors 
"""
Main challenge: Mustango requires to be in its root directory to be imported

main.py expects itself to be called from the root folder but may be called by /ui by streamlit

Checks which path is calling it and revert itself back to the root folder as necessary
"""
original_path = sys.path
print(os.getcwd())

def is_current_directory(target):
    current_dir_name = os.path.basename(os.getcwd())
    return current_dir_name == target

#UI folder is calling function, return to root folder
if not is_current_directory('PlayWrite'):
    os.chdir('../') 
    sys.path.insert(1, os.getcwd())


#Imports from root folder
from src.data.dataloader import Vocabulary
from src.models.finetuned_cnn import PreTrainedCNNModels
from src.models.image_caption_attention import *
from src.utils.llama_prompt_template import PromptTemplate

print(os.getcwd())

#updating the mustango path to allow for import
os.chdir('models/mustango')
sys.path = original_path #revert to original before updating again
sys.path.insert(1, os.getcwd())
print(os.getcwd())

from mustango import Mustango

#revert system path to original
sys.path = original_path

if not is_current_directory('PlayWrite'):
    os.chdir('../../ui') #head to ui folder
else:
    os.chdir('../..')



"""
PlayWrite Code to Generate music based on image and text input
1. PlayWrite() Class
2. Generate() function that creates the music based on image, text input
"""


class playWrite():
    def __init__(self, 
                 device:str,
                 vocab_path:str,
                 image_caption_path:str,
                 hg_access_token:str=None,
                 llama_model_path: str=None,
                 llama_tokenizer_path:str=None
                 ):
        self.vocab = self._load_vocab(vocab_path)
        self.image_caption = self._load_image_caption(image_caption_path, device)
        self.llama_model, self.llama_tokenizer = self._load_llama(hg_access_token, llama_model_path, llama_tokenizer_path)
        self.mustango = self._load_mustango()
        self.device = device

    def _load_vocab(self, filepath:str):
        file = open(filepath, 'rb')
        vocab = pickle.load(file)
        if isinstance(vocab, Vocabulary):
            print("Vocabulary Loaded Successfully")
            return vocab
        else:
            raise Exception("Invalid Vocabulary")
    
    def _load_image_caption(self, model_path, device):
        try:
            model = torch.load(model_path).to(device)
            print("Image Captioning Model Loaded Successfully")
            return model
        
        except Exception as e:
            raise Exception(f"Unable to load torch model, reasons: {e}")

    def _load_llama(self, hg_access, llama_model_path, llama_tokenizer_path):
        if hg_access != None:
            try:
                print("Loading Llama Models")
                llama_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', token=hg_access, torch_dtype=torch.bfloat16, device_map="auto")
                llama_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', token=hg_access)
                print("Llama Loaded Successfully")
                return llama_model, llama_tokenizer
            
            except Exception as e:
                raise Exception(f"Unable to load Llama model from hugging face, reasons: {e}")

        elif llama_model_path != None and llama_tokenizer_path != None:
            try:
                print("Loading Llama Models")
                llama_model = AutoModelForCausalLM.from_pretrained(llama_model_path, torch_dtype=torch.bfloat16, device_map="auto")
                llama_tokenizer = AutoTokenizer.from_pretrained(llama_tokenizer_path)
                print("Llama Loaded Successfully")
                return llama_model, llama_tokenizer
            
            except Exception as e:
                raise Exception(f"Unable to load Llama model from directory, reasons: {e}")
        
        else:
            raise Exception("No Llama resources provided")

    def _load_mustango(self):
        try:
            mustango = Mustango("declare-lab/mustango")
            print("Mustango Loaded Successfully")
            return mustango
        
        except Exception as e:
            raise Exception(f"Unable to load mustango, reasons: {e}")



    def caption_image(self, image:bytes, model:InceptV3EncoderAttentionDecoder, vocab: Vocabulary, device:str, max_length:int=50):
        """
        Function to caption uploaded image from streamlit

        Args:
            image (bytes): uploaded image by users in bytes based on streamlit file reading format
            model: image captioning model 
            vocab (Vocabulary): image captioning model vocabulary
            device (str): cuda or cpu
            max_length (int, optiona;): max length of generated captions, default to 50

        Returns:
            generated captions: string of image caption
        """
        
        pil_image = Image.open(io.BytesIO(image)).convert('RGB') #convert bytes to image

        #setup transform to convert image to readable tensor for the model
        transform = transforms.Compose([
            transforms.Resize((299,299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
        ])

        transformed_image = transform(pil_image).unsqueeze(0) #unsqueeze to add a batch size of 1
        generated_captions, attention = model.caption_image(transformed_image.to(device), vocab, device, max_length) 

        return " ".join(generated_captions[1:-1])



    def generate_music_prompt(self, caption:str, text_prompt:str, llama_model:AutoModelForCausalLM, llama_tokenizer:AutoTokenizer, device:str):
        """
        Function to generate music prompt by merging image captions and text prompts from user with llama2-7b

        Args:
            caption (str): generated image caption
            text_prompt (str): text prompt from user
            llama_model (AutoModelForCausalLM): llama model
            llama_tokenizer (AutoTokenizer): llama tokenizer
            device (str): cuda or cpu

        Returns:
            music_prompt: string of llama results
        """
        prompt = f"""=============context================
        {caption},
        {text_prompt},
        =========================================

        Take the following 2 context and merge them to create a textual prompt for music generation. Your prompt should be a single line. Do not give prompts that suggest increasing intensity.
        The prompt should contain the atmosphere of the song, where the song would fit environment wise and chord progression you have come up with. I have given you some example prompts, format your prompt similarly to them but do not copy their content.
        Example prompts: This is a live performance of a classical music piece. There is an orchestra performing the piece with a violin lead playing the main melody. The atmosphere is sentimental and heart-touching. This piece could be playing in the background at a classy restaurant.
        The song is an instrumental. The song is in medium tempo with a classical guitar playing a lilting melody in accompaniment style. The song is emotional and romantic. The song is a romantic instrumental song.
        This is a new age piece. There is a flute playing the main melody with a lot of staccato notes. The rhythmic background consists of a medium tempo electronic drum beat with percussive elements all over the spectrum. There is a playful atmosphere to the piece.

    """
        promptGenerator =  PromptTemplate(system_prompt=prompt)
        llama_prompt = promptGenerator.build_prompt()
        config = GenerationConfig(max_new_tokens=1024,
                                do_sample=True,
                                top_k = 10,
                                num_return_sequences = 1,
                                return_full_text = False,
                                temperature = 0.1,
                                )
            
        encoded_input = llama_tokenizer.encode(llama_prompt, return_tensors='pt', add_special_tokens=False).to(device)
        results = llama_model.generate(encoded_input, generation_config=config)
        decoded_output = llama_tokenizer.decode(results[0], skip_special_tokens=True)
        response = decoded_output.split("[/INST]")[-1].strip()
        
        #cleaning up the response to remove additional prompts
        quote_index = response.find('"')
        last_quote_index = response.rfind('"')
        if quote_index != -1 and last_quote_index != -1: #if the result is in quotation marks
            music_prompt = response[quote_index+1:last_quote_index]

        else:
            colon_index = response.rfind(":") #getting text in the format of prompt:\n{actual prompt}
            music_prompt = response[colon_index+3:] #remove the \n as well
        return music_prompt



    def generate_music(self, music_prompt:str, model:Mustango, steps:int, guidance:int):
        """
        Function to generate music from music prompt with mustango

        Args:
            music_prompt (str): text prompt to generate music with mustango
            model (Mustango): mustango model
            steps (int): Number of epochs the music generation model iterates through
            guidance (int): How much guidance needed for the model

        Returns:
            generated music
        """
        music = model.generate(music_prompt, steps, guidance)
        return music



def generate(playwrite:playWrite, byte_image:bytes, text_prompt:str, max_length:int=50, steps:int=100, guidance:int=3, delete_model:bool=True):
    """
    Overall Function to generate music from image and textual prompts

    Args:
        playwrite (playWrite): Class instance with all models initiated
        byte_image (bytes): Image prompt
        text_prompt (str): Textural prompt
        max_length (int, optional): Maximum caption length, defaults to 50
        steps (int, optional): Number of epochs the music generation model iterates through, defaults to 100
        guidance (int, optional): How much guidance needed for the model, defaults to 3
        delete_mode (bool, optional): Delete models after they are used, mainly used for a memory situation as it requires models to be re-initialiased. Defaults too True

    Returns:
        generated music
    """

    image_caption = playwrite.caption_image(image=byte_image,
                                            model=playwrite.image_caption,
                                            vocab=playwrite.vocab,
                                            device=playwrite.device,
                                            max_length=max_length
                                            )
    print(f"Image Caption: {image_caption}")
    if delete_model:
        del playwrite.image_caption
        torch.cuda.empty_cache()
        gc.collect()
    music_prompt = playwrite.generate_music_prompt(caption=image_caption,
                                                   text_prompt=text_prompt,
                                                   llama_model=playwrite.llama_model,
                                                   llama_tokenizer=playwrite.llama_tokenizer,
                                                   device=playwrite.device)
    print(f"Music Prompt: {music_prompt}")
    if delete_model:
        del playwrite.llama_model
        torch.cuda.empty_cache()
        gc.collect()
    music = playwrite.generate_music(music_prompt=music_prompt,
                                     model=playwrite.mustango,
                                     steps=steps,
                                     guidance=guidance)
    
    return music




if __name__ == "__main__":
    print("Import completed!")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # PlayWrite = playWrite(device=device,
    #                       vocab_path='resources/Vocabulary.pkl',
    #                       image_caption_path='models/image_captioning/model.pt',
    #                       hg_access_token=None,
    #                       llama_model_path='models/llama/model',
    #                       llama_tokenizer_path='models/llama/tokenizer')
    
    # print("All models loaded")
    # with open("input/Landscape/Test/Images/image_00001.jpg", "rb") as uploaded_image:
    #     f = uploaded_image.read()
    #     b = bytearray(f)
    # text_prompt = "racing game among 12 players with power ups scattered around the map"

    # print("Generating music")
    # generated_music = generate(playwrite=PlayWrite,
    #                             byte_image=b,
    #                             text_prompt=text_prompt,
    #                             max_length=50,
    #                             steps=150,
    #                             guidance=3,
    #                             delete_model=True)
    # print("Generating music completed.")
    # sf.write(f"resources/music_results/test.wav", generated_music, samplerate=16000) #no space allowed

