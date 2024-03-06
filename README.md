# BERTModel


This project aims to train a Bi-Directiional Reinforcement Transformer model to predict a product group from a text description. I employ HuggingFace for my base models and train them on Amazon prouduct data. 

## Data Cleanup

This dataset required pretraining to be fed correctly to the tokenizer and model, such as encoding the labels and reducing the possible product groups to only those with 500+ listings. I split the data 80/10/10 and used the HuggingFace DataSet class to load the data efficeintly to the model

## Training

To start I loaded the distilbert-base-uncased model and its respective tokenizer, and then mapped through the dataset with a tokenizer function. 