import os
from sys import platform
from transformers import AutoTokenizer, AutoModelForCausalLM

"""
Function to download models that are used in this project
    - LLama2 7b hf model
    - Mustango model

LLama7b model path = 
Mustango model path = 

Note: 
To download LLama2-7b from hugging face, you require access from meta.
Once you have gotten access, please retrieve an authentication key from your own hugging face account.

Get access here: https://huggingface.co/meta-llama/Llama-2-7b
"""

def setup_llama(model_directory:str="models/llama/model" , tokenizer_directory:str="models/llama/tokenizer", hg_access_token:str=""):
    """
    Function to download llama2 7b model

    Args:
        model_directory (str): directory to the path with llama model, defaults to ../../models/llama/model'
        tokenizer_directory (str): directory to the path with llama tokenizer, defaults to ../../models/llama/tokenizer'
        hg_access_token (str): hugging face access token

    Returns:
        0 for success and 1 for failure
    """
    if len(hg_access_token) <=0:
        print("Not a valid access token")
        return 1

    if not os.path.isdir(model_directory):
        os.makedirs(model_directory) 
        print("Llama Model directory not found, directory created")

    if not os.path.isdir(tokenizer_directory):
        os.makedirs(tokenizer_directory) 
        print("Llama tokenizer directory not found, directory created")

    print("Preparing to download LLama")

    try:
        llm_model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-chat-hf', token=hg_access_token)
        llm_model.save_pretrained(model_directory)
        print("LLama Model downloaded")

        llm_tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf', token=hg_access_token)
        llm_tokenizer.save_pretrained(tokenizer_directory)
        print("LLama Tokenizer downloaded")
        return 0

    except Exception as e:
        print(f"Failed to download Llama, error: {e}")
        return 1




def setup_mustango(model_directory:str="models"):
    """
    Function to download mustango model. This command makes use of os to fun terminal commands.
    Alternatively, please follow the installation process here: https://huggingface.co/declare-lab/mustango

    Args:
        model_directory (str): directory to the mustango model, defaults to models'

    Returns:
        0 for success and 1 for failure
    """

    if not os.path.isdir(model_directory):
        os.makedirs(model_directory) 
        print("Mustango Model directory not found, directory created")

    try:
        os.chdir(model_directory)
        os.system("git clone https://github.com/AMAAI-Lab/mustango")
        os.chdir("mustango")
        #edit requirements file. 
        #scikit-image, scipy has issues with installment because of version issues with matplotlib, numpy and pandas
        #torch, torchaudio and torchvision does not have cuda support in requirements file
        #solution is to remove the specific requirements
        remove_version_constraints('requirements.txt', {'torch': 'torch==2.2.1+cu121', 'torchaudio': 'torchaudio==2.2.1+cu121', 'torchvision':'torchvision==0.17.1+cu121'}, ['matplotlib', 'numpy', 'pandas', 'scikit_image', 'scipy'])
        os.system("pip install -r requirements.txt")
        os.chdir("diffusers")
        os.system("pip install -e .")
        print("Installed mustango successfully")
        return 0
    
    except Exception as e:
        print(f"Failed to download mustango, error: {e}")
        return 1


def remove_version_constraints(input_file, cuda_package, packages):
    mac_system = (platform == "darwin")
    print(f"{platform} OS detected, installing pytorch with {'cuda' if platform != 'darwin' else 'cpu'}")
    with open(input_file, 'r+') as f:
        lines = f.readlines()
        f.seek(0)
        f.truncate()
        f.write("--find-links https://download.pytorch.org/whl/torch_stable.html".rstrip('\r\n') + '\n')
        for line in lines:                
            package_name = line.split('==')[0].strip()
            if package_name in cuda_package.keys() and not mac_system:
                #not a mac sytem
        
                f.write(f"{cuda_package[package_name]}\n")
            
            elif package_name in packages:
                f.write(f"{package_name}\n")
            else:
                f.write(line)


if __name__ == "__main__":
    status = setup_llama(hg_access_token=insert_hg_token_here)
    print(f"Llama download exited with process {status}")
    status = setup_mustango()
    print(f"mustango download exited with process {status}")
