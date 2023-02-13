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
parentid = pd.read_csv('category_tree.csv')
properties1= pd.read_csv('item_properties_part1.csv')
properties2= pd.read_csv('item_properties_part2.csv')
# On rassemble les deux dataframe - properties1 et properties2
properties = pd.concat([properties1, properties2], ignore_index=True)
# Dataframe créée
events_enhanced = pd.read_csv("events_enhanced.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

pages = ["Présentation", "Visualisation", "Pre-processing", "Modélisation"]

st.sidebar.title("Navigation")
page = st.sidebar.radio("Aller vers:", pages)
datasets = ["events", "parentid", "properties"]
clf = None

# présentation
if page == pages[0]:

    st.title("Suivi des utilisateurs de e-commerce sur une période de 4,5 mois")
    st.header("Un projet de pre-processing et de Machine Learning")
    st.write("Vous pouvez retrouver notre projet [ici](https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset)")

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
        st.image("properties.jpg")

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

# Page Visualisation
if page == pages[1]:

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
    st.write("Nous observons un fort déséquilibre de classe")

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
    st.write("C'est en juillet qu'a eu lieu le plus de ventes sur la période étudiée")

    st.subheader("CA en volume / semaine")
    sns.countplot(x='weekday', data=events_sept[events_sept['event'] == 'transaction'], palette='flare')
    plt.xticks(np.arange(7), ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche'])
    plt.xticks(rotation=40)
    st.pyplot()
    st.write("C'est le mercredi qu'a lieu le plus de ventes")

    st.subheader("Visualisation de l'activité journalière")
    sns.countplot(x='hour', data=events[events['event'] == 'transaction'])
    st.pyplot()
    st.write("Les ventes sont généralement réalisées à partir de 15h. Ainsi nous constatons que la période la plus"
             " active est celle qui débute à 15h avec un pic à 17h, pour se stabiliser entre 18h et 21h. le nombre "
             "de vente décroit ensuite lentement durant la nuit jusqu’à être presque nul entre 8h et 13h")


    st.header("Analyse du caractère récurrent des transactions ")
    st.subheader("Nombre de transaction pour un même produit, par tranche")
    st.image("Nb transac for same product.png")


# Traitement de données
    st.header("Analyse de la distribution des évènements par produit")
    most_viewed_items = events[events['event'] == 'view'].itemid.value_counts()
    x = most_viewed_items[:20].index
    y = most_viewed_items[:20].values

    st.subheader("Les produits les plus vus")
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

    st.subheader("Les produits les plus ajoutés au panier")
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

    st.write("Le site internet n’a pas de produit « star » et surtout il lui faut générer beaucoup de trafic et "
             "donc de vues sur les produits pour générer des ventes et donc un chiffre d’affaires.")

    st.header("Analyse de la distribution des évènements par client")
    st.subheader("Nombre de client pour un même produit par tranche")
    st.image("Nb client for a same product by range.png")

# Traitement de données
    most_purchased_clients = events[events['event'] == 'transaction'].visitorid.value_counts()
    x = most_purchased_clients[:20].index
    y = most_purchased_clients[:20].values

    st.subheader("Les clients qui ont le plus acheté")
    sns.barplot(x=x,
                y=y,
                order=x,
                palette="muted")
    plt.xticks(rotation=40)
    st.pyplot()

#preprocessing
if page == pages[2]:
    st.header("Pre-processing : ")

    st.write("Nous avons réalisé plusieurs étapes de pre-processing pour obtenir le DataFrame suivant :")
    st.dataframe(events_enhanced.head(10))

    st.header("Analyse du dataframe 'properties':")

    st.subheader("Comprehension des colonnes 'property' et value'")

    st.write(
        "Les données de la variable 'property' ont été hachées. Par conséquent, seul les modalités 'categoryid' et 'available' sont exploitablent.")
    st.write(
        "Concernant la variable 'value', toutes les valeurs numériques ont été marquées d'un caractère 'n' au début et ont une précision de 3 chiffres après la virgule décimale. Par exemple, '5' deviendra 'n5.000'.")

    st.write("Les valeurs 790 et 888 contiennent le plus d'occurence dans la colonne property")
    st.dataframe(properties['property'].value_counts())

    st.write("Visualisation de la valeur 790")
    properties_790 = properties[properties['property'] == '790']
    st.dataframe(properties_790.head())

    st.write("Visualisation de la valeur 888")
    properties_888 = properties[properties['property'] == '888']
    st.dataframe(properties_888.head())

    st.write(
        "On constate que la modalite 790 est uniquement constitué de valeurs numérique marquées d'un caractère 'n' au début. On peut déduire que cette modalité représente le prix des produits.")
    st.write("La modalité 888 possède des informations hachées. On ne prendra pas en compte ces valeurs. "
                "On peut supposer qu'elle constitue la première promotion effectué sur les prix")

    st.header("Traitement du dataset:")
    st.subheader("Filtrage des modalités")
    st.write("On sélectionne uniquement les éléments pertinents que l'on veut garder")

    st.write("1 - Sur la colonne 'property', les modalites 'categoryid', '790' et 'available'.")

    st.write(
        "2 - Conserver uniquement les lignes du dataset 'properties' pour lesquelles la valeur de la colonne 'itemid' est "
        "similaire à la colonne 'itemid' du dataset 'events'.")

    st.write(
        "3 - Sur la colonne 'value', On supprime les caractères 'n' de toutes les valeurs pour obtenir le prix initial de chaque produit."
        " Par la suite, on transforme les données en nombres flottants.")

    st.write("4 - On renomme les 3 modalités pour obtenir un dataset compréhensible")
    properties = properties[properties.property.isin(["categoryid", "790", "available"]) &
                            properties.itemid.isin(events.itemid.unique())]

    properties.value = properties.value.str.replace("n", "").astype("float")

    properties.property = properties.property.map({"790": "price", "categoryid": "categoryid",
                                                    "available": "available"})
    st.dataframe(properties.head(10))

    st.subheader("Exploration de la variable temporel")
    st.write(
        "Les valeurs des colonnes 'category', 'available' et 'price' ne sont pas constantes. Ces valeurs sont actualisées "
        "au debut de chaque semaine.")
    st.write(
        "L'objectif est d'avoir les bonnes informations de chaque produit au moment correspondant a l'evenement, dans notre dataset. ")

    events = events[events.itemid.isin(properties.itemid.unique())]

    max_timestamp = np.max((events.timestamp.max(), properties.timestamp.max()))
    properties_sorted = properties.sort_values("timestamp")
    properties_sorted["lag_value"] = properties_sorted.groupby(["itemid", "property"]).value.shift(1)
    properties_sorted["lead_timestamp"] = properties_sorted.groupby(["itemid", "property"]).timestamp.shift(-1)

    properties_sorted["is_change"] = np.logical_or(properties_sorted.lag_value.isna(),
                                                    properties_sorted.lag_value != properties_sorted.value)
    properties_sorted = properties_sorted[properties_sorted.is_change]
    properties_sorted["lead_timestamp"] = properties_sorted.groupby(["itemid", "property"]).timestamp.shift(-1)

    properties_sorted["lead_timestamp"].fillna(max_timestamp, inplace=True)
    properties_sorted["lead_timestamp"] = properties_sorted["lead_timestamp"] \
        .astype("int64")
    properties_sorted.rename({"timestamp": "valid_start",
                                "lead_timestamp": "valid_end"}, axis=1, inplace=True)
    properties = properties_sorted.loc[:, ("valid_start", "valid_end",
                                            "itemid", "property", "value")]
    properties["time_valid"] = properties.valid_end - properties.valid_start
    del properties_sorted, max_timestamp
    properties.info()
    st.dataframe(properties.head(10))

    st.subheader("Fusion des datasets 'properties' et 'events'")
    events_enhanced = events.merge(properties, on="itemid")

    st.write(
        "On utilise une notation de filtrage pour sélectionner uniquement les lignes du dataframe 'events_enhanced' où la valeur de la colonne 'timestamp' est supérieure ou égale à la valeur de la colonne 'valid_start' et inférieure à la valeur de la colonne 'valid_end'.")
    st.write(
        "Au final, on ne gardera que les lignes qui ont une valeur de temps qui est dans l'intervalle défini par les colonnes 'valid_start' et 'valid_end' dans le dataframe 'events_enhanced'.")

    events_enhanced = events_enhanced[
        np.logical_and(events_enhanced.timestamp >= events_enhanced.valid_start,
                       events_enhanced.timestamp < events_enhanced.valid_end)]

    events_enhanced = events_enhanced.loc[:, ["timestamp", "visitorid", "itemid",
                                              "event", "property", "value"]]

    events_enhanced = events_enhanced.pivot_table(
        index=["timestamp", "visitorid", "itemid", "event"],
        columns="property", values="value",
        observed=True)

    events_enhanced.columns = list(events_enhanced.columns)
    events_enhanced = events_enhanced.reset_index()
    events_enhanced.rename(index={"property": "index"},
                           inplace=True)

    st.subheader("Remplir les valeurs manquantes")

    del events
    st.dataframe(events_enhanced.head(10))
    st.write("Totalité des Nan créées par le filtre")
    st.dataframe(events_enhanced.isnull().sum())

    st.write(
        "Nous constatons un nombre important de valeurs manquantes dans les colonnes 'categoryid', 'price' et 'available'. ")
    st.write(
        "On décide d'imputer les NaNs avec des valeurs valides sur la période la plus étendue.")

    st.write("1 - Obtenir les propriétés valides le plus longtemps ")
    st.write("2 - Remplir les valeurs manqantes lorsque nécessaire")
    st.write("3 - Renommer les variable en question et les positioner dans le nouveau df 'event_enhanced'")

    top_properties = properties.groupby(["itemid", "property", "value"],
                                        as_index=False).time_valid.sum().sort_values("time_valid") \
        .groupby(["itemid", "property"]).tail(1)

    top_properties = top_properties.pivot_table(index=["itemid"],
                                                columns="property", values="value", observed=True).reset_index()

    events_enhanced = events_enhanced.merge(top_properties, on="itemid")

    events_enhanced.loc[events_enhanced.categoryid_x.isna(),
    ["categoryid_x"]] = events_enhanced["categoryid_y"]

    events_enhanced.loc[events_enhanced.price_x.isna(),
    ["price_x"]] = events_enhanced["price_y"]

    events_enhanced.loc[events_enhanced.available_x.isna(),
    ["available_x"]] = events_enhanced["available_y"]

    events_enhanced.rename({"categoryid_x": "categoryid",
                            "price_x": "price",
                            "available_x": "available"},
                           axis=1, inplace=True)

    events_enhanced = events_enhanced.loc[:, ["timestamp", "visitorid", "itemid",
                                              "event", "categoryid", "available", "price"]]

    del top_properties, properties

    st.write("Vérifier s'il y a toujours des valeurs manquantes")
    st.dataframe(events_enhanced.isnull().sum())

    st.subheader(
        "Transformation de la variable 'timestamp' et attribution des 'categoryid'/'parentid' a chaque 'itemid'. ")
    import datetime

    events_enhanced['timestamp'] = pd.to_datetime(events_enhanced['timestamp'], unit='ms')
    events_enhanced['date'] = events_enhanced['timestamp'].dt.date
    events_enhanced['hour'] = events_enhanced['timestamp'].dt.hour
    events_enhanced['month'] = events_enhanced['timestamp'].dt.month
    events_enhanced['week'] = events_enhanced['timestamp'].dt.week
    events_enhanced['weekday'] = events_enhanced['timestamp'].dt.weekday

    events_enhanced = events_enhanced.merge(parentid, how="left", on=["categoryid"])
    events_enhanced['parentid'].fillna(events_enhanced['parentid'].mode()[0], inplace=True)
    events_enhanced = events_enhanced.reindex(
        columns=['timestamp', 'date', 'month', 'week', 'weekday', 'hour', 'visitorid',
                 'itemid', 'event', 'available', 'price', 'categoryid', 'parentid'])

    events_enhanced.sort_values(['itemid', 'timestamp'], inplace=True)

    st.dataframe(events_enhanced.head(20))

# Page Modélisation
if page == pages[3]:

    models = ["LogisticRegression", "GradientBoostingClassifier", "RandomForestClassifier_resampled"]

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

    st.write("Les coefficients de poids d'un modèle de régression logistique représentent l'importance des variables explicatives "
             "(ou caractéristiques) dans la prédiction de la variable cible (ou variable dépendante). Ils sont utilisés pour déterminer"
             " la relation statistique entre les variables explicatives et la variable cible. ")

    st.write("Chaque coefficient de poids est associé à une variable explicative spécifique et est calculé de manière"
             " à maximiser la probabilité de prédire correctement la variable cible. Les coefficients de poids positifs "
             "indiquent que la variable explicative est positivement corrélée à la variable cible, tandis que "
             "les coefficients de poids négatifs indiquent une corrélation négative")

    st.write("Ces coefficients décrivent la relation statistique entre les variables explicatives et la variable cible. ")

    st.subheader("Coefficient de poids")
    st.image("coefficient de poids.png")

    st.write("Ce graphique nous apprend plusieurs choses que finalement très peu de variables exercent une réelle"
             " influence sur le modèle. Ainsi, nous relevons que le modèle pourrait être simplifié à quatre variables : ")

    st.write ("• Price")
    st.write("• Weekday")
    st.write("• Week")
    st.write("• Available")

    st.write("Concernant la variable « price », elle est très négativement corrélée à la variable cible. "
             "Ainsi, nous comprenons du graphique que plus le prix augmente, moins la prédiction du modèle devient "
             "est sûre. Cela signifie que plus la variable prix est importante dans le jeu de données, "
             "plus la probabilité de réalisation d’une des modalités de la variable cible décroit.")

    st.write("A l’inverse, concernant la variable « available », la corrélation est positive.")

    st.write("Les autres variables weekeday, week, d’importances relatives, confirment ce que nous avions vu avec"
             " les visualisations, à savoir que le trafic du site internet est fortement corrélé à la temporalité "
             "c’est à dire l’heure et le jour de la semaine.")

    st.subheader("features importances - SHAP")

    st.write("Un plot de résumé Shap (SHapley Additive exPlanations) est un graphique utilisé pour visualiser"
             " l'importance relative des différentes caractéristiques d'une entrée pour la prédiction d'un modèle "
             "de machine learning.")

    st.image("features importances shap.png")

    st.write("Ce graphique confirme les observations précédemment réalisées à savoir que les variables prix, "
             "et les variables « temporelles » ont un impact significatif sur le modèle.")

    st.write("Cependant ce graphique indique que les variables de codification telles parentid, categoryid ont "
             "également un impact, ce qui n’était pas mis en avant avec les coefficients de poids.")

    st.write("Ces différences s’expliquent puisqu’un coefficient de poids ne tient pas compte des interactions"
             " entre les caractéristiques. A l’inverse le modèle de SHAP calcule les importances relatives des ces"
             " dernières. Dans le cas de notre modèle, les caractéristiques sont bien interdépendantes, "
             "le modèle de SHAP est donc davantage pertinent.")

    st.write("Par ailleurs, les coefficients de poids d'un modèle sont généralement calculés en utilisant des méthodes "
             "de régression, telles que la régression linéaire4, qui supposent une relation linéaire entre les "
             "caractéristiques et la variable cible. Les plots de résumé SHAP peuvent être utilisés avec des modèles"
             " non linéaires, tels que les arbres de décision ou les réseaux de neurones, pour évaluer l'importance "
             "relative des caractéristiques. De ce fait, étant donné que nous n’avons pas analysé si notre variable "
             "cible suivait effectivement une relation linéaire avec ses caractéristiques, il est difficile de dire si "
             "les coefficients de poids sont réellement pertinents.")

    st.subheader("features importances")

    st.write("Afin de confirmer les résultats obtenus à partir des méthodes SHAP, nous avons réalisé un"
             " feat_importances, méthode d’interprétabilité adaptée.")

    st.write("Les 'feat_importances' sont généralement utilisés pour mesurer l'importance relative des caractéristiques" \
                                   " dans un modèle d'arbre de décision. Ces valeurs indiquent combien chaque " \
                                   "caractéristique a contribué à la prédiction de la variable cible au cours de la " \
                                   "construction de l'arbre.")

    st.image("features importances.png")

    st.write("Ces résultats sont conformes à ceux obtenus avec le SHAP.")

