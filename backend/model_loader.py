import os
import joblib
import streamlit as st

BASE_DIR=os.path.abspath(os.path.join(os.path.dirname(__file__),".."))
MODELS_DIR=os.path.join(BASE_DIR,"models")

@st.cache_resource
def load_model():

    pipeline=joblib.load(
        os.path.join(
            MODELS_DIR,
            "final_pipeline.pkl"
        )
    )

    scaler=pipeline['scaler']
    iso_forest=pipeline['iso_forest']
    feature_cols=pipeline['feature_cols']

    return pipeline,scaler,iso_forest,feature_cols