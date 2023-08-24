import shap
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import joblib

st.markdown("# Page 3 🎉")
st.sidebar.markdown("# Page 3 🎉")

if "data" in st.session_state:
    st.write("la valeur de la page 2 a été récupéré")
    st.table(st.session_state)
# If page1 already executed, this should write True


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.title("SHAP in Streamlit")

# train XGBoost model
model = joblib.load("data/modeles sauvegardés/lightGBMClasiifer_model.sav")

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn and spark models)
explainer = shap.Explainer(model[-1])
shap_values = explainer.shap_values( model[:-1].transform(st.session_state.data) )
#shap_values = explainer.shap_values( model[:-1].transform( data) )

# visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
shap.force_plot(shap.force_plot(explainer.expected_value[0], np.array(shap_values[0]), st.session_state.data.loc[st.session_state.chosen_radio]))
#st_shap(shap.force_plot(explainer.expected_value, shap_values[0,:], data.loc[data.SK_ID_CURR==216093]))

# visualize the training set predictions
shap.force_plot(shap.force_plot(explainer.expected_value, shap_values, st.session_state.data), 400)