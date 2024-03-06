# BERTModel


This project aims to train a Bi-Directiional Reinforcement Transformer model to predict a product group from a text description. I employ HuggingFace for my base models and train them on Amazon prouduct data. 

## Data Cleanup

This dataset required pretraining to be fed correctly to the tokenizer and model, such as encoding the labels and reducing the possible product groups to only those with 500+ listings. I split the data 80/10/10 and used the HuggingFace DataSet class to load the data efficeintly to the model

## Training

To start I loaded the distilbert-base-uncased model and its respective tokenizer, and then mapped through the dataset with a tokenizer function. HuggingFace has built in methods 'TrainingArguments' and 'Trainer' which allow you easy training. 

## Quantization

Using dynamic quantization, I reduce the datatype from int32 to int8, resulting in lower memory use and faster latency.

Original Model:  Size (MB):  267.942302  
Quantized Model: Size (MB):  138.729314

Moving from eager compilitaion to a traced model using JIT, 

From here the model and its configuration is bundled up and saved.

## Deployment 

This model is deployed using TorchServe, which places the model onto a serve with endpoints that can be queried for inferences. 

