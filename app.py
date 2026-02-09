import pandas as pd
import numpy as np
import requests
import math

# ML
from gensim.models import Word2Vec
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack

import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

import streamlit as st

df = pd.read_csv(r"C:\Users\Melvin\Desktop\DATA\PORTFOLIO\Recommandation de films\CSV\df_movies_preprocess.csv")
genre_df = pd.read_csv(r"C:\Users\Melvin\Desktop\DATA\PORTFOLIO\Recommandation de films\CSV\genres_binarized.csv")

liste_titres = df['Titre'].values

features = df[['Titre', 'Age du film', 'clean_synopsis_str', 'Note', 'Popularité'] + list(genre_df)]

# preprocessor = ColumnTransformer(
#     transformers=[
#         ('Titre', TfidfVectorizer(), ['Titre']),
#         ('clean_synopsis_str', TfidfVectorizer(), ['clean_synopsis_str']),
#         ('Note', MinMaxScaler(), ['Note'])
#     ],
#     remainder='passthrough' 
# )

titre_tfidf = TfidfVectorizer().fit_transform(df['Titre']) 

synopsis_tfidf = TfidfVectorizer().fit_transform(df['clean_synopsis_str']) * 1.3

note_scaled = MinMaxScaler().fit_transform(df[['Note']]) * 0.75

age_scaled = MinMaxScaler().fit_transform(df[['Age du film']]) * 0.5

pop_scaled = MinMaxScaler().fit_transform(df[['Popularité']])

genre_columns = ['Action', 'Animation', 'Aventure', 'Comédie', 'Crime', 'Documentaire', 'Drame', 'Familial', 'Fantastique', 'Guerre', 'Histoire', 'Horreur', 'Musique', 'Mystère', 'Romance', 'Science-Fiction', 'Thriller', 'Téléfilm', 'Western']
genre_matrix = df[genre_columns].values

features_vectorized = hstack([titre_tfidf, synopsis_tfidf, note_scaled, age_scaled, pop_scaled, genre_matrix])

cosine_sim = cosine_similarity(features_vectorized)


BASE_POSTER_URL = "https://image.tmdb.org/t/p/w500"

def get_recommendations(title, cosine_sim=cosine_sim):
    idx = df.index[df['Titre'] == title].tolist()[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:10]  # 10 films en recommandations
    movie_indices = [i[0] for i in sim_scores]

    # Retourne les titres des films recommandés et leurs chemins de poster
    recommendations = df[['Titre', 'Affiche']].iloc[movie_indices]
    return recommendations


def display_recommendations_with_posters(titles):
    """Affiche les recommandations pour les titres donnés avec leurs affiches."""
    for t in titles:
        try:
            recommendations = get_recommendations(t)
            st.write(f"Recommendations pour {t}:")

            for i in range(0, len(recommendations), 3):
                rows = recommendations.iloc[i:i+3]

                cols = st.columns([1, 0.5, 1, 0.5, 1])
                # Afficher une affiche dans chaque colonne impaire (1, 3, 5, 7, 9)
                # Les colonnes paires (2, 4, 6, 8) agiront comme des espaces
                col_indices = [0, 2, 4]  # Indices des colonnes où afficher les posters
                for idx, (_, row) in zip(col_indices, rows.iterrows()):
                    with cols[idx]:
                        # Construire l'URL complète de l'affiche
                        poster_url = BASE_POSTER_URL + row['Affiche']
                        # Afficher l'affiche avec une largeur spécifiée
                        st.image(poster_url, width=200, caption=row['Titre'])

        except IndexError:
            st.write(f"Film non trouvé: {t}")

#########

def set_background(png_file):
    page_bg_img = f'''
    <style>
    .stApp::before {{
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-image: url("{png_file}");
        background-size: cover;
        opacity: 0.12;  /* Ajustez cette valeur pour changer la transparence */
        z-index: 0;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_background("https://c.wallhere.com/photos/e5/9b/movie_poster_people-1698949.jpg!d")


###########

st.title("CineMatch.io")

# Sélection de films
liste_titres = df['Titre'].tolist() 
title = st.multiselect("Entrer ou Selectionner un titre de film pour obtenir des recommandations:", liste_titres)

def display_recommendations(titles):
    """Affiche les recommandations pour les titres donnés."""
    for t in titles:
        try:
            recommendations = get_recommendations(t)
            st.write(f"Recommendations pour {t}:")

            # Itérer à travers chaque titre recommandé
            for rec in recommendations:
                st.write(rec)

        except IndexError:
            st.write(f"Film non trouvé: {t}")

# Bouton de recommandation
if st.button("Recommander"):
    if title:
        display_recommendations_with_posters(title)
    else:
        st.write("Veuillez entrer un titre de film")
        
