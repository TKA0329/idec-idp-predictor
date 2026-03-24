import torch
import yaml
import importlib.util
import os
from transformers import BertTokenizer, BertModel
from transformers import BertConfig
import warnings
import streamlit as st
import random
from huggingface_hub import hf_hub_download

import numpy as np

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False    # Disables auto-optimisation for stability

seed_everything(42)

# 1. Setup paths
MODELS = ["cv", "rog", "tau"]
BASE_DIR = "best_models"

@st.cache_resource
def load_models():
    models = {}
    for model_dir in MODELS:
        local_path = os.path.join(BASE_DIR, model_dir, "model.pt")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        if not os.path.exists(local_path):
            hf_hub_download(
                repo_id="dsadasiv/IDP-BERT",
                filename=f"best_models/{model_dir}/model.pt",
                local_dir=".",
                local_dir_use_symlinks=False
            )

        config_path = os.path.join(model_dir, "config.yaml")
        weights_path = os.path.join(BASE_DIR, model_dir, "model.pt")
        network_path = os.path.join(model_dir, "network.py")

        spec = importlib.util.spec_from_file_location(f"network_{model_dir}", network_path)
        net_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(net_module)

        with open(config_path, 'r') as f:
            idp_config = yaml.safe_load(f)

        bert_config = BertConfig.from_pretrained("Rostlab/prot_bert_bfd")
        model = net_module.IDPBERT(bert_config, idp_config, get_embeddings=False)

        state_dict = torch.load(weights_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        models[model_dir] = model
    
    return models

models = load_models()
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

# Testing + Streamlit 
def predict(sequence, model):
    processed_seq = " ".join(list(sequence.replace(" ", "")))
    inputs = tokenizer(processed_seq, return_tensors="pt")
    
    with torch.no_grad():
        prediction = model(inputs['input_ids'], inputs['attention_mask'])
    return prediction.item()

sequence = st.text_input("Protein Sequence:", placeholder="e.g. MKVIFLALVLSLA")

if sequence:
    for name, model in models.items():
        result = predict(sequence, model)
        st.write(f"{name}: {result}")