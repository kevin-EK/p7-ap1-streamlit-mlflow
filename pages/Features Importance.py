import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import joblib
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance


st.markdown("# Page 3 🎉")
st.sidebar.markdown("# Page 3 🎉")

if "data" in st.session_state:
    st.write("la valeur de la page 2 a été récupéré")
    #st.table(st.session_state)
# If page1 already executed, this should write True


def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)

st.title("Feature Importance by permutation_importance")

# train XGBoost model
model = joblib.load("data/model-save/HistGradientBoostingClassifier_model.sav")

#from sklearn.model_selection import train_test_split

X = train.drop(columns=['TARGET','SK_ID_CURR'])
y = train['TARGET']

feature_names = [f"feature {i}" for i in range(X.shape[1])]

## inf
X[X==np.inf] = np.nan

#X = X[X['CODE_GENDER'] != 'XNA']

for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
    X[bin_feature], uniques = pd.factorize(X[bin_feature])
X['INCOME_CREDIT_PERC'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
X['time_to_repay'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
X['ANNUITY_INCOME_PERC'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']

random_seed = 971

# Le dataset est grand on peut diviser le données en train test valiation 
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=random_seed,stratify=y) #~~100000 samples
X_test, X_validation, y_test, y_validation = train_test_split( X_test, y_test, test_size=0.5, random_state=random_seed,stratify=y_test) 


result = permutation_importance(
    model, X_test, y_test, n_repeats=5, random_state=42, n_jobs=-1
)
#print(f"Elapsed time to compute the importances: {elapsed_time:.3f} seconds")
forest_importances = pd.Series(result.importances_mean, index=X.columns)


import matplotlib.pyplot as plt
fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()