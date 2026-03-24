import torch
from transformers import BertTokenizer, BertModel

# This is the "Beast" backbone
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("Rostlab/prot_bert")

# Test sequence (a flexible IDP-like string)
sequence = "M A S K A S T S G R" # Note the spaces! ProtBERT needs spaces.
inputs = tokenizer(sequence, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    # If this prints a shape like [1, length, 1024], you have successfully 
    # harnessed the power of Protein BERT.
    print("Success! Embedding Shape:", outputs.last_hidden_state.shape)