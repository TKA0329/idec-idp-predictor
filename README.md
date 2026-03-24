# IDP Property Predictor

A Streamlit app for predicting biophysical properties of Intrinsically Disordered Proteins (IDPs) using IDP-BERT.

Built on top of [IDP-BERT](https://github.com/DanushSadasivam/IDP-BERT) by DanushSadasivam et al.

## Properties Predicted
- Cv (Heat Capacity)
- Rog (Radius of Gyration)
- Tau (end to end decorrelation time)

## Usage
pip install -r requirements.txt
streamlit run my_inference.py