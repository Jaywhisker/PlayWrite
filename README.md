# PlayWrite
This is the repository for PlayWrite. PlayWrite is a multimodal generative ambient game music model. It takes in landscape images and text prompts to generate 10s samples of ambient game musics.
You may read on our work here: (TBD)

## Setup
### Library Dependcies
Run the following code to download all required libraries for the first setup
```
git clone https://github.com/Jaywhisker/PlayWrite.git
pip install -r requirements.txt
```
___
### Dataset
Please download the dataset here: https://drive.google.com/drive/folders/1xwBZWMfBc-11yA93oS3OHRBwTSvr5UDH <br/> and place it inside the input folder
___
### PlayWrite Model Setup
The following models are required for setup:
- Llama2-7b
- Mustango
- inaSpeechSegmenter (If you are running the text-to-game music pipeline)

The setup code has already been created, please run the following to setup Llama2-7b and Mustango locally:
Edit the following python file (src/utils) to include your hugging face access keys and run the following
```
python -m src.utils.setup.py
```
Else you may just want to run the notebook:
```
notebooks/utils/setup.ipynb
```
___
### Text-to-Game Music Dataset Setup
To run the text-to-game music dataset, you will require a **youtubeV3 API key** and either **a hugging face key or a locally setup Llama2-7b**.

Run the setup in the ```notebooks/data/text_music_dataset_generation.ipynb``` before using the pipeline in the same notebook.
___
## Streamlit
Due to the current compute constraint, the UI only runs llama2-7b and does not run mustango. To run both, simply uncomment the code in ```ui/components/build_show_generation_process_section/index.py```

To start the streamlit, please run the following:
```
#be in root
cd ui
streamlit run app.py
```

## File Structure
There are many components of the repository, please look at the following file structure to better understand the use case
```{ssh}
PlayWrite
├── input                                           <- folder containing all datasets (Blip2, Kosmos2, FilteredFlickr, text-to-music dataset)
├── models                                          <- folder containing all models (llama and mustango will be set up here)
│   └── image_captioning 
│       ├── encoder.pt                              <- fine-tuned InceptionV3 model
│       └── model.pt                                <- image captioning model
│
├── notebooks
│   ├── cnn_finetuning.ipynb                        <- notebook to fine-tune cnn models
│   ├── image_caption_attention.ipynb               <- notebook to train attention-based cnn-lstm image captioning models
│   ├── image_caption_base.ipynb                    <- notebook to train cnn-lstm image captioning models
│   ├── image_caption_transformers.ipynb            <- notebook to train transformer-based image captioning models
│   ├── playwrite_main.ipynb                        <- notebook to run entire PlayWrite pipeline
|   |
│   ├── data                                        <- data creation folder
│   │   ├── filter_flickr.ipynb                     <- notebook to filter flickr30k
│   │   ├── flickr_split.ipynb                      <- notebook to create flickr30k dataset
│   │   ├── image_caption_dataset_generation.ipynb  <- notebook to caption images with BLIP-2 and Komos-2
│   │   └── text_music_dataset_generation.ipynb     <- notebook for text-to-game music dataset pipeline
|   |
│   ├── data_visualisation                          <- data visualisation folder
│   │   ├── dataset_visualisation.ipynb             <- notebook for dataset visualisation (caption length, unique captions, caption repitition, clustering)
│   │   └── results_visualisation.ipynb             <- notebook to visualise training results
|   |
│   └── utils                                       <- utils folder
│       └── llama_mustango_setup.ipynb              <- setup llama and mustango locally
│
├── resources                                       <- resources folder for results 
│   ├── Vocabulary.pkl                              <- image captioning model's vocabulary
│   ├── data_visualisation                          <- data visualisation results from UI
│   ├── music_results                               <- generated music from main.py / notebook / UI
│   └── training_results                            <- csv training results
│
├── src                                             <- folder containing all the python files of the code in the notebooks
│   ├── eval.py
│   ├── inference.py
│   ├── main.py                                     <- PlayWrite pipeline code
│   ├── train.py
│   ├── __init__.py
│   ├── data                                        <- folder containing image captioning dataset and dataloader code
│   ├── data_visualisation                          <- folder containing data visualisation code 
│   ├── evaluation_metrics                          <- folder containing eval metrics (ROUGE, BLEU, METEOR)
│   ├── models                                      <- folder containing pytorch model architecture classes
│   └── utils                                       <- folder containing setup code and llama prompt template
│
├── ui                                              <- folder containing ui features
│    ├── app.py                                     <- main ui python file
│    ├── __init__.py
│    ├── components                                 <- folder containing ui components
│    ├── images                                     <- folder containing ui assets
│    └── utils                                      <- folder containing ui utils
│
├── README.md
└── requirements.txt                                <- library dependecies
```
