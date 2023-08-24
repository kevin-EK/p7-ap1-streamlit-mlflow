import pandas as pd
import numpy as np
import streamlit as st
#import datetime
import requests
import joblib


MLFLOW_URI = 'http://127.0.0.1:5001/invocations'

def clean_data1(data):
    # Optional: Remove 4 applications with XNA CODE_GENDER (train set)
    data = data[data['CODE_GENDER'] != 'XNA']
    for bin_feature in ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY']:
        data[bin_feature], uniques = pd.factorize(data[bin_feature])

    data['DAYS_EMPLOYED'].replace(365243, np.nan, inplace= True)
    return data

def request_prediction(model_uri, data):
    headers = {"Content-Type": "application/json"}

    data_json = {'dataframe_split': data.to_dict(orient='split')}
    response = requests.request(
        method='POST', headers=headers, url=model_uri, json=data_json)

    if response.status_code != 200:
        raise Exception(
            "Request failed with status {}, {}".format(response.status_code, response.text))

    return response.json()


@st.cache_data
def load_data():
    important_features = joblib.load('data/cleaned/list_col_to_keep_from_train_application_final.joblib')
    data1 = pd.read_csv("data/source/application_train_clean.csv", usecols=important_features+['TARGET','SK_ID_CURR'])
    data2 = pd.read_csv("data/source/application_test_clean.csv",  usecols= important_features+['SK_ID_CURR'] )
    
    # Données numériques
    number_data = data1.select_dtypes(include = np.number).groupby(['TARGET']).agg(pd.Series.median).T
    
    # Données catégorielles
    categ_data_data = data1.groupby(['TARGET']).agg(pd.Series.mode).select_dtypes(exclude = np.number).T
    
    data_agg = pd.concat([number_data, categ_data_data],axis=0)
    data_agg.reset_index(inplace=True)
    data_agg.rename(columns={'index':'info', 0:'Bon', 1:'Mauvais'},inplace = True)
    
    data = pd.concat([data1.drop(columns = ['TARGET']), data2],axis=0).drop_duplicates()
    # Some simple new features (percentages)
    data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
    data['time_to_repay'] = data['AMT_CREDIT'] / data['AMT_ANNUITY']
    data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
    data = clean_data1(data)
    return data, data_agg

chosen_radio = st.sidebar.radio(
'Comment souhaitez vous entrer les valeurs du client?',("Index", "Valeur"))
if 'chosen_radio' not in st.session_state:
    st.session_state['chosen_radio'] = chosen_radio

if chosen_radio == "Index":
    #Chargement des données
    identifiant, data_agg = load_data()
    
    if 'data' not in st.session_state:
        st.session_state['data'] = identifiant

    latest_iteration = st.empty()
    latest_iteration.write(
        "Vous identifierez le client grâce à son numéro d'identification !")
    
    #time.sleep(3)
    
    latest_iteration.markdown("## :blue[Choix de l'identifiant]")
    chosen_customer = st.selectbox(
    "Quel est le numéro d'identifiant du client?", identifiant['SK_ID_CURR'])


    tab_comparaison = data_agg[['info',	'Bon', 'Mauvais']].merge(
        identifiant[identifiant.SK_ID_CURR==chosen_customer].T\
        .reset_index().rename(columns = {'index':'info', 0:'Client selectionné'}),
        on = 'info',how='inner'
                )

    
    # Bouton Score
    predict_btn = st.sidebar.button('Obtenir le score')

    if predict_btn:
        st.balloons()
        pred = request_prediction(
            MLFLOW_URI, 
            identifiant.drop(columns=['SK_ID_CURR']).loc[identifiant.SK_ID_CURR==chosen_customer]
            )
        
        pred = pred['predictions'][0] 
        if pred == 0:
            st.markdown("## :green[Prédit comme bon client]")
        else:
            st.markdown("## :red[Prédit comme client à risque]")
    
else:
    st.write("## :blue[Vous entrerez une à une les informations du client!]")
    
    #Initialisation du DataFrame
    DFManu = dict() 
    
    # Input CODEGENDER
    if st.radio('De quel genre est le client', ("Homme", "Femme")) == "Homme": 
        DFManu['CODE_GENDER'] = 0 
    else: 
        DFManu['CODE_GENDER'] = 1
    
    # Input FLAG_OWN_CAR
    if st.radio('Le client possède-t-il unu voiture?', ("Oui", "Non")) == "Oui": 
        DFManu['FLAG_OWN_CAR'] = 1 
    else: 
        DFManu['FLAG_OWN_CAR'] = 0
        
    # Input FLAG_OWN_REALTY : Indicateur si le client est propriétaire d'une maison ou d'un appartement        
    if st.radio("Le client est-il propriétaire d'une maison ou d'un appartement?", ("Oui", "Non")) == "Oui": 
        DFManu['FLAG_OWN_REALTY'] = 0 
    else: 
        DFManu['FLAG_OWN_REALTY'] = 1
        
    # CNT_CHILDREN : Nombre d'enfants du client            
    DFManu['CNT_CHILDREN'] = st.number_input("entrer le nombre d'enfants du client",    min_value=0,    max_value=70) # record du monde du nombre d'enfant 69
    
    # AMT_INCOME_TOTAL : Revenu du client           
    DFManu['AMT_INCOME_TOTAL'] = st.number_input("entrer le montant du Revenu du client:", min_value=0, max_value=200000000)
    
    # AMT_CREDIT : Montant du crédit
    DFManu['AMT_CREDIT'] = st.number_input("entrer le montant du crédit:", min_value=0.0, value = 500211.5)
    
    # AMT_ANNUITY: Annuité du prêt
    DFManu['AMT_ANNUITY'] = st.number_input("entrer l'annuité du prêt':", min_value=0.1, value = 25078.500000)
    
    # NAME_INCOME_TYPE: Type de revenu du client (homme d'affaires  travailleur  congé de maternité ...)     
    DFManu['NAME_INCOME_TYPE'] = st.selectbox("Type de revenu du client (homme d'affaires  travailleur  congé de maternité ...) ",
                                              ['Working', 'State servant', 'Commercial associate', 
                                               'Pensioner','Unemployed', 'Student', 'Businessman', 
                                               'Maternity leave'])
    
    # NAME_EDUCATION_TYPE: Niveau d'études le plus élevé atteint par le client  
    DFManu['NAME_EDUCATION_TYPE'] = st.selectbox("Niveau d'études le plus élevé atteint par le client",
                                                 ['Secondary / secondary special', 'Higher education',
                                                 'Incomplete higher', 'Lower secondary', 'Academic degree'])
    
    # NAME_FAMILY_STATUS: Situation familiale du client      
    DFManu['NAME_FAMILY_STATUS'] = st.selectbox("Situation familiale du client  ",
                                                 ['Single / not married', 'Married', 'Civil marriage', 'Widow',
                                                  'Separated', 'Unknown'])
    
    # DAYS_BIRTH : âge du client          
    DFManu['DAYS_BIRTH'] = st.number_input("entrer l'âge du client en jours au moment de la demande:", min_value=-43800,value=-15755, max_value=-6570) 
    
    
    #today = datetime.datetime.now()
    #year_adulte = today.year - 18
    #month_adulte = today.month    
    #d = st.date_input("Entrer la date du client",value=datetime.date(year_adulte, month_adulte, 1),
    #                  min_value=datetime.date(1900, 1, 1), # record du monde de longévité
    #                  max_value = datetime.date(year_adulte, month_adulte, 1)  )
    #st.write('Your birthday is:', d, (d-datetime.date.today()).days)
    
    # 'DAYS_EMPLOYED', Combien de jours avant la demande la personne a-t-elle commencé à travailler ?        
    DFManu['DAYS_EMPLOYED'] = st.number_input("Combien de jours avant la demande la personne a-t-elle commencé à travailler ?", min_value=-17912,value=-1663, max_value=0)
    
    # OCCUPATION_TYPE: Quelle est la profession du client ?  
    DFManu['OCCUPATION_TYPE'] = st.selectbox("Quelle est la profession du client ?",
                                                 ['Laborers', 'Core staff', 'Accountants', 'Managers', np.nan,
                                                'Drivers', 'Sales staff', 'Cleaning staff', 'Cooking staff',
                                                'Private service staff', 'Medicine staff', 'Security staff',
                                                'High skill tech staff', 'Waiters/barmen staff',
                                                'Low-skill Laborers', 'Realty agents', 'Secretaries', 'IT staff',
                                                'HR staff'])

    # REG_REGION_NOT_LIVE_REGION: Indicateur si l'adresse permanente du client ne correspond pas à l'adresse de contact (1=différente  0=même  au niveau de la région)
    if st.radio("L'adresse permanente du client ne correspond pas à l'adresse de contact?", ("L'adresse permanente du client ne correspond pas à l'adresse de contact", "L'adresse permanente du client correspond à l'adresse de contact")) == "L'adresse permanente du client correspond à l'adresse de contact": 
        DFManu['REG_REGION_NOT_LIVE_REGION'] = 0 
    else: 
        DFManu['REG_REGION_NOT_LIVE_REGION'] = 1     
    
    # REG_REGION_NOT_WORK_REGION Indicateur si l'adresse permanente du client ne correspond pas à l'adresse du lieu de travail (1=différent  0=même  au niveau régional)      
    if st.radio("L'adresse permanente du client ne correspond pas à l'adresse du lieu de travail?", ("L'adresse permanente du client ne correspond pas à l'adresse du lieu de travail?", "L'adresse permanente du client correspond à l'adresse du lieu de travail?")) == "L'adresse permanente du client correspond à l'adresse du lieu de travail?": 
        DFManu['REG_REGION_NOT_WORK_REGION'] = 0 
    else: 
        DFManu['REG_REGION_NOT_WORK_REGION'] = 1 
    
    # REG_CITY_NOT_LIVE_CITY Indicateur si l'adresse permanente du client ne correspond pas à l'adresse de contact (1=différent  0=même  au niveau de la ville)      
    if st.radio("L'adresse permanente du client ne correspond pas à l'adresse de contact?", ("L'adresse permanente du client ne correspond pas à l'adresse de contact?", "L'adresse permanente du client correspond à l'adresse de contact?")) == "L'adresse permanente du client correspond à l'adresse de contact?": 
        DFManu['REG_CITY_NOT_LIVE_CITY'] = 0 
    else: 
        DFManu['REG_CITY_NOT_LIVE_CITY'] = 1 
    
    # REG_CITY_NOT_WORK_CITY Indicateur si l'adresse permanente du client ne correspond pas à l'adresse du lieu de travail (1=différent  0=même  au niveau de la ville)      
    if st.radio("L'adresse permanente du client ne correspond pas à l'adresse du lieu de travail?", ("La ville permanente du client ne correspond pas à la ville du lieu de travail?", "La ville permanente du client correspond à la ville du lieu de travail?")) == "La ville permanente du client correspond à la ville du lieu de travail?": 
        DFManu['REG_CITY_NOT_WORK_CITY'] = 0 
    else: 
        DFManu['REG_CITY_NOT_WORK_CITY'] = 1 
        
    # ORGANIZATION_TYPE : Type d'organisation où le client travaille 
    DFManu['ORGANIZATION_TYPE'] = st.selectbox("Quel est le type d'organisation où le client travaille?",
                                                 ['Business Entity Type 3', 'School', 'Government', 'Religion',
                                                        'Other', 'XNA', 'Electricity', 'Medicine',
                                                        'Business Entity Type 2', 'Self-employed', 'Transport: type 2',
                                                        'Construction', 'Housing', 'Kindergarten', 'Trade: type 7',
                                                        'Industry: type 11', 'Military', 'Services', 'Security Ministries',
                                                        'Transport: type 4', 'Industry: type 1', 'Emergency', 'Security',
                                                        'Trade: type 2', 'University', 'Transport: type 3', 'Police',
                                                        'Business Entity Type 1', 'Postal', 'Industry: type 4',
                                                        'Agriculture', 'Restaurant', 'Culture', 'Hotel',
                                                        'Industry: type 7', 'Trade: type 3', 'Industry: type 3', 'Bank',
                                                        'Industry: type 9', 'Insurance', 'Trade: type 6',
                                                        'Industry: type 2', 'Transport: type 1', 'Industry: type 12',
                                                        'Mobile', 'Trade: type 1', 'Industry: type 5', 'Industry: type 10',
                                                        'Legal Services', 'Advertising', 'Trade: type 5', 'Cleaning',
                                                        'Industry: type 13', 'Trade: type 4', 'Telecom',
                                                        'Industry: type 8', 'Realtor', 'Industry: type 6'])
    # EXT_SOURCE_2 : Score normalisé à partir d'une source de données externe 
    DFManu['EXT_SOURCE_2'] = st.number_input("Score normalisé à partir d'une source de données externe 2:", min_value=0.0001,value=0.5000, max_value=1.0000) 
    
    # EXT_SOURCE_3 : Score normalisé à partir d'une source de données externe
    DFManu['EXT_SOURCE_3'] = st.number_input("Score normalisé à partir d'une source de données externe 3:", min_value=0.0001,value=0.5000, max_value=1.0000)  
    
    if DFManu['AMT_CREDIT'] <= 0.0: 
        DFManu['INCOME_CREDIT_PERC'] = 0.0 
    else: 
        DFManu['INCOME_CREDIT_PERC'] = DFManu['AMT_INCOME_TOTAL'] / DFManu['AMT_CREDIT']
    
    if DFManu['AMT_ANNUITY'] <= 0.0: 
        DFManu['time_to_repay'] = 0
    else: 
        DFManu['time_to_repay'] = DFManu['AMT_CREDIT'] / DFManu['AMT_ANNUITY']
    
    if DFManu['AMT_INCOME_TOTAL'] <= 0.0:
        DFManu['ANNUITY_INCOME_PERC'] = 0
    else:
        DFManu['ANNUITY_INCOME_PERC'] = DFManu['AMT_ANNUITY'] / DFManu['AMT_INCOME_TOTAL']
        
           
    
    #st.write( DFManu )
    
    # Bouton Score
    predict_btn_manu = st.sidebar.button('Obtenir le score')

    if predict_btn_manu:
        predManu = request_prediction( MLFLOW_URI, pd.DataFrame( DFManu, index=[0]) )
        
        predManu = predManu['predictions'][0] 
        if predManu == 0:
            st.balloons()
            st.markdown("## :green[Prédit comme bon client]")
        else:
            st.markdown("## :red[Prédit comme client à risque]")
            st.snow()
    
     
