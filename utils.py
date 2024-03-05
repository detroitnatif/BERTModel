import datasets
import tqdm 
import torch
import numpy as np
from ts.utils.util  import map_class_to_label

def prediction_batch(model, dataset, device='cpu', batchsize=32):
    metric_accuracy = datasets.load_metric('accuracy', trust_remote_code=True)
    
    l = len(dataset)
    all_y_preds = []
    
    model.eval()
    model = model.to(device)
    
    for ndx in tqdm.trange(0, l, batchsize):
        
        input_ids = dataset['input_ids'][ndx: ndx + batchsize].to(device)
        attention_mask = dataset['attention_mask'][ndx: ndx + batchsize].to(device)
        
        
        with torch.no_grad():
            res = model( input_ids = input_ids, attention_mask = attention_mask )
            
            # output of torchscript model doesn't have logits property 
            #logits = res.logits.detach().cpu().numpy()
            
            logits = res[0].detach().cpu().numpy()
            
            y_preds = np.argmax(logits, axis=1)
            
            all_y_preds.extend(y_preds)
            
            
    accuracy = metric_accuracy.compute(predictions = all_y_preds, references = dataset['label'])
    
    return accuracy
            
        
        
        