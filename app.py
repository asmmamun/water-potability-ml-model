#gradio app 

import gradio as gr
import pandas as pd
import pickle
import numpy as np

# 1. Load the Model
with open("rf_potability.pkl", "rb") as f:
    model = pickle.load(f)

# 2. The Logic Function
def predict_potability(ph, Hardness, Solids, Chloramines, 
                       Sulfate, Conductivity, Organic_carbon, 
                       Trihalomethanes, Turbidity):
    # Pack inputs into a DataFrame
    # The column names must match your CSV file exactly
    input_df = pd.DataFrame([[
        ph, Hardness, Solids, Chloramines, 
        Sulfate, Conductivity, Organic_carbon, 
        Trihalomethanes, Turbidity]],
      columns=[
          'ph', 'Hardness', 'Solids', 'Chloramines', 
          'Sulfate', 'Conductivity', 'Organic_carbon', 
          'Trihalomethanes', 'Turbidity'
      ])
    # Predict
    prediction = model.predict(input_df)[0]
    
    # Return formatted result (Clipped 0-5)
    return f"Predicted Potability: {np.clip(prediction, 0, 1)}"


# 3. The App Interface
# Defining inputs in a list to keep it clean
inputs = [
    gr.Number(label="ph"),
    gr.Number(label="Hardness"),
    gr.Number(label="Solids"),
    gr.Number(label="Chloramines"),
    gr.Number(label="Sulfate"),
    gr.Number(label="Conductivity"),
    gr.Number(label="Organic_carbon"),
    gr.Number(label="Trihalomethanes"),
    gr.Number(label="Turbidity")
]

app = gr.Interface(
    fn=predict_potability,
      inputs=inputs,
        outputs="text", 
        title="Water Potability Predictor")

app.launch(share=True)