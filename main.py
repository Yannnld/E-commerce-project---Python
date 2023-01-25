import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import io
import shap

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score,precision_score, recall_score
from sklearn.utils import resample
from joblib import load


events = pd.read_csv('events.csv')
parentid = pd.read_csv('events.csv')
properties1= pd.read_csv('item_properties_part1.csv')
properties2= pd.read_csv('item_properties_part2.csv')
# On rassemble les deux dataframe - properties1 et properties2
properties = pd.concat([properties1, properties2], ignore_index=True)
# Dataframe créée
events_enhanced = pd.read_csv("events_enhanced.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

pages = ["Présentation", "Pre-processing", "Visualisation", "Modélisation"]
st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers:", pages)
datasets = ["events", "parentid", "properties"]

# présentation
if page == pages[0]:

    st.title("Suivi des utilisateurs de e-commerce sur une période de 4,5 mois")
    st.header("Un projet de pre-processing et de Machine Learning")
    st.write(
        "Vous pouvez retrouver notre projet [ici](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)   ")

    st.subheader("Par Artur Caiano, Yann Ladouceur et Hugo Martinez")

    st.write("A notre disposition nous avons eu les jeux de données suivants :\n events, \n parentid, \n properties.")

    st.header("Première exploration : ")
    st.write("Utilisez les boutons suivants pour naviguer entre les datasets et lire les informations les concernant")
    dataset = st.selectbox("Choisissez votre jeu de données", datasets)

    if dataset == datasets[0]:
        st.subheader("Dataset Events")
        st.image("events.jpg")

        st.write("Les principales informations à retenir sont les suivantes :")

        st.markdown(" - Pas ou peu de visiteurs récurrents")
        st.markdown("- Les visites ne se transforment pas en achat")
        st.markdown("- Répartition déséquilibrée des données")

    if dataset == datasets[1]:
        st.subheader("Dataset Parentid")
        st.image("parentid.jpg")

        st.write("Les principales informations à retenir sont les suivantes :")

        st.markdown(" - Nombre de categoryid : 1 669 ")
        st.markdown("- Nombre de parentid : 362 ")

    if dataset == datasets[2]:
        st.subheader("Dataset Properties")
        st.image(properties.jpg)

        st.write("Les principales informations à retenir sont les suivantes :")

        st.markdown(" - visitorid est l'utilisateur unique qui a navigué sur le site web ")
        st.markdown(
            " - La variable « event » décrit ce que l'utilisateur a fait durant sa visite sur le site au travers de 3 classes ")
        st.markdown(
            " - Les codes « itemid » correspondent à l'ensemble des produits qui ont eu une interaction avec l'utilisateur, c'est-à-dire les produits qui ont été vus, ajoutés ou achetés  ")
        st.markdown(
            " - Les codes « transactionid » n'auront des valeurs que si l'utilisateur a effectué un achat, c'est pour cela qu'on observe à l'œil nu beaucoup de NaN puisqu'en proportion, un utilisateur ne réalise une transaction que dans 1% des cas environ  ")

    st.header("Les difficultés identifiées : ")
    st.markdown("- Les données sont anonymisées")
    st.markdown("- Certaines données indiquant des caractéristiques évoluent dans le temps ")
    st.markdown("- Les datasets sont volumineux")
    st.markdown("- Les variables des datasets ne correspondent pas")

# Page - Pre-processing
if page == pages[1]:
    st.header("Première exploration : ")
    st.subheader("Dataset Events")

    st.write("Utilisez les boutons suivants pour naviguer entre les datasets et lire les informations les concernant")

    st.write("Nous avons réalisé plusieurs étapes de pre-processing pour obtenir le DataFrame suivant :")
    st.dataframe(events_enhanced.head(10))

# Page Visualisation
if page == pages[2]:

    st.set_option('deprecation.showPyplotGlobalUse', False)

    st.header("Visualisation des données ")
    st.subheader("La répartition de la variable cible 'event' ")
    events_count = events["event"].value_counts()
    events_count
    sns.barplot(x=events_count.index, y=events_count.values)
    st.pyplot()

    explode = (0, 0.25, 0.55)
    plt.pie(events_count.values, explode=explode, labels=events_count.index,autopct='%1.2f%%')
    st.pyplot()

# Traitement de données
    events['timestamp_bis'] = pd.to_datetime(events['timestamp'], unit='ms')
    events['date'] = events['timestamp_bis'].dt.date
    events['hour'] = events['timestamp_bis'].dt.hour
    events['month'] = events['timestamp_bis'].dt.month
    events['weekday'] = events['timestamp_bis'].dt.weekday
    sept = events[events['month'] == 9]
    sept_duplicated = sept.copy()
    events_sept = pd.concat([events, sept_duplicated], ignore_index=True)

    st.header("Evolution chronologique des ventes")
    st.subheader("CA en volume / mois")
    sns.countplot(x='month', data=events_sept[events_sept['event'] == 'transaction'], palette='crest')
    plt.xticks(np.arange(5), ['Mai', 'Juin', 'Juillet', 'Août', 'Septembre'])
    plt.xticks(rotation=40)
    st.pyplot()

    st.subheader("CA en volume / semaine")
    sns.countplot(x='weekday', data=events_sept[events_sept['event'] == 'transaction'], palette='flare')
    plt.xticks(np.arange(7), ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    plt.xticks(rotation=40)
    st.pyplot()

    st.subheader("Visualisation de l'activité journalière")
    sns.countplot(x='hour', data=events[events['event'] == 'transaction'])
    st.pyplot()

    st.header("Analyse du caractère récurrent des transactions ")
    st.subheader("Nombre de transaction pour un même produit, par tranche")
    st.image("Nb transac for same product.png")

# Traitement de données
    st.header("Analyse de la distribution des évènements par produit")
    most_viewed_items = events[events['event'] == 'view'].itemid.value_counts()
    x = most_viewed_items[:20].index
    y = most_viewed_items[:20].values

    st.subheader("Les produits les plus vues")
    sns.barplot(x=x,
                y=y,
                order=x,
                palette="icefire")
    plt.xticks(rotation=40)
    st.pyplot()

# Traitement de données
    most_added_to_cart_items = events[events['event'] == 'addtocart'].itemid.value_counts()
    x = most_added_to_cart_items[:20].index
    y = most_added_to_cart_items[:20].values

    st.subheader("Les produits les plus rajoutés au panier")
    sns.barplot(x=x,
                y=y,
                order=x,
                palette="icefire")
    plt.xticks(rotation=40)
    st.pyplot()

# Traitement de données
    most_purchased_items = events[events['event'] == 'transaction'].itemid.value_counts()
    x = most_purchased_items[:20].index
    y = most_purchased_items[:20].values

    st.subheader("Les produits les plus vendus")
    sns.barplot(x=x,
                y=y,
                order=x,
                palette="icefire")
    plt.xticks(rotation=40)
    st.pyplot()

    st.header("Analyse de la distribution des évènements par client")
    st.subheader("Nombre de client pour un même produit par tranche")
    st.image("Nb client for a same product by range.png")

# Traitement de données
    most_purchased_clients = events[events['event'] == 'transaction'].visitorid.value_counts()
    x = most_purchased_clients[:20].index
    y = most_purchased_clients[:20].values

    st.subheader("Les clients qui ont le plus achetés")
    sns.barplot(x=x,
                y=y,
                order=x,
                palette="muted")
    plt.xticks(rotation=40)
    st.pyplot()

# Page Modélisation
if page == pages[3]:
    st.header("Classification du problème")
    st.write("Rappelons que la variable cible identifiée est la variable « event »."
             " Cette variable présente trois classes possibles. Il s’agit ainsi d’une variable discrète puisque ne "
             "présentant pas un nombre infini d’issues possibles.De ce fait nous abordons notre problème "
             "comme une problématique de classification dans un cas d’apprentissage supervisé.")

    st.header("La séparation des données")
    st.subheader(" Test_size = 0,25 - Random_state = 42 ")
    st.write("Ce choix s’appuie sur notre expérience et correspond à ce qui est généralement retenu pour ce genre de modèle.")

    st.header("Encodage")
    st.subheader("LabelEncoder")
    st.write("Notre modèle RandomForest et GradientBoosting n’appréhende pas les variables cibles sous forme de chaîne de caractères. Notre variable cible étant précisément sous ce format,"
             " nous devons procéder à un encodage. ")

    st.header("Transformation des données")
    st.write("Il n'est pas nécessaire d'appliquer une transformation des données,"
             " comme la normalisation ou la standardisation pour les modèles de forêts aléatoires (Random Forest)"
             " ou de Gradient Boosting, car ces modèles sont généralement peu sensibles à l'échelle des données.")
             
    st.write(" En revanche, une strandardisation sera appliqué sur le modèle de régression logistique.")

    st.header("Modélisation")

    st.subheader("Les métriques")

    st.write("Random Forest")
    st.image("cr_imbalanced_rf.png")

    st.write("GradientBoosting")
    st.image("cr_imbalanced_gb.png")

    st.write("Regression logistique")
    st.image("cr_imbalanced_rg.png")

    st.header("Interprétation")
    st.subheader("features importances")
    st.image("features importances.png")

    st.subheader("features importances - SHAP")
    st.image("features importances shap.png")

    st.subheader("La densité des valeurs Shap")
    st.image("la densité des valeurs shap.png")





