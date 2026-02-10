import pandas as pd
import numpy as np
import pickle
import streamlit as st
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="CineMatch.io",
    page_icon="🎬",
    layout="wide"
)

#########################################
# CONFIGURATION DES CHEMINS
#########################################

# Détection automatique du répertoire de base
try:
    from src.config import DATA_DIR, BASE_POSTER_URL
except ImportError:
    # Fallback si src.config n'existe pas
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / 'data'
    BASE_POSTER_URL = "https://image.tmdb.org/t/p/w500"

MODELS_DIR = DATA_DIR / 'models'

#########################################
# CHARGEMENT DU MODÈLE
#########################################

@st.cache_data
def load_model():
    """
    Charge le modèle de recommandation sauvegardé.
    Utilise le cache de Streamlit pour ne charger qu'une seule fois.
    """
    try:
        # Charger la matrice de similarité
        with open(MODELS_DIR / 'cosine_similarity_matrix.pkl', 'rb') as f:
            cosine_sim = pickle.load(f)
        
        # Charger le DataFrame minimal
        df = pd.read_pickle(MODELS_DIR / 'df_movies_minimal.pkl')
        
        # Charger la liste des titres
        with open(MODELS_DIR / 'liste_titres.pkl', 'rb') as f:
            liste_titres = pickle.load(f)
        
        # Charger les métadonnées (optionnel)
        with open(MODELS_DIR / 'model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        
        return cosine_sim, df, liste_titres, metadata
    
    except FileNotFoundError as e:
        st.error(f"""
        ❌ Fichiers du modèle introuvables!
        
        Assurez-vous d'avoir exécuté le notebook 4 pour créer les fichiers suivants:
        - {MODELS_DIR / 'cosine_similarity_matrix.pkl'}
        - {MODELS_DIR / 'df_movies_minimal.pkl'}
        - {MODELS_DIR / 'liste_titres.pkl'}
        - {MODELS_DIR / 'model_metadata.pkl'}
        
        Erreur: {e}
        """)
        st.stop()
    
    except Exception as e:
        st.error(f"❌ Erreur lors du chargement du modèle: {e}")
        st.stop()

# Chargement du modèle au démarrage
with st.spinner("🔄 Chargement du modèle de recommandation..."):
    cosine_sim, df, liste_titres, metadata = load_model()

st.success(f"✅ Modèle chargé! ({metadata['n_films']} films disponibles)")

#########################################
# FONCTION DE RECOMMANDATION
#########################################

def get_recommendations(title, cosine_sim=cosine_sim, df=df, top_n=9):
    """
    Retourne les films les plus similaires à un film donné.
    
    Args:
        title (str): Titre du film
        cosine_sim (np.array): Matrice de similarité
        df (pd.DataFrame): DataFrame contenant les films
        top_n (int): Nombre de recommandations à retourner
    
    Returns:
        pd.DataFrame: DataFrame avec les colonnes ['Titre', 'Affiche']
    
    Raises:
        IndexError: Si le film n'est pas trouvé
    """
    # Trouve l'index du film
    try:
        idx = df.index[df['Titre'] == title].tolist()[0]
    except IndexError:
        raise IndexError(f"Film non trouvé: {title}")
    
    # Récupère les scores de similarité
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # Trie par score décroissant
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Prend les top_n films (en excluant le film lui-même)
    sim_scores = sim_scores[1:top_n+1]
    
    # Récupère les indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Retourne les recommandations
    recommendations = df[['Titre', 'Affiche']].iloc[movie_indices]
    
    return recommendations

#########################################
# FONCTIONS D'AFFICHAGE
#########################################

def set_background(png_file):
    """Définit l'image de fond de l'application"""
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
        opacity: 0.12;
        z-index: 0;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

def display_recommendations_with_posters(titles):
    """
    Affiche les recommandations pour les titres donnés avec leurs affiches.
    
    Args:
        titles (list): Liste des titres de films
    """
    for t in titles:
        try:
            recommendations = get_recommendations(t)
            
            st.markdown(f"### 🎬 Recommandations pour **{t}**")
            st.markdown("---")
            
            # Affichage par rangées de 3 films
            for i in range(0, len(recommendations), 3):
                rows = recommendations.iloc[i:i+3]
                
                # Créer 5 colonnes: film, espace, film, espace, film
                cols = st.columns([1, 0.5, 1, 0.5, 1])
                col_indices = [0, 2, 4]  # Indices des colonnes pour les films
                
                for idx, (_, row) in zip(col_indices, rows.iterrows()):
                    with cols[idx]:
                        # Construire l'URL complète de l'affiche
                        poster_url = BASE_POSTER_URL + row['Affiche']
                        
                        # Afficher l'affiche
                        st.image(poster_url, use_container_width=True)
                        
                        # Afficher le titre en dessous
                        st.markdown(
                            f"<p style='text-align: center; font-weight: bold;'>{row['Titre']}</p>",
                            unsafe_allow_html=True
                        )
            
            st.markdown("<br>", unsafe_allow_html=True)
            
        except IndexError:
            st.warning(f"⚠️ Film non trouvé: **{t}**")
        except Exception as e:
            st.error(f"❌ Erreur pour {t}: {e}")

#########################################
# INTERFACE UTILISATEUR
#########################################

# Définir le fond
set_background("https://c.wallhere.com/photos/e5/9b/movie_poster_people-1698949.jpg!d")

# Titre de l'application
st.title("🎬 CineMatch.io")
st.markdown("### *Trouvez votre prochain film préféré*")

# Afficher les informations du modèle dans la sidebar
with st.sidebar:
    st.markdown("## 📊 Informations du modèle")
    st.markdown(f"**Nombre de films:** {metadata['n_films']:,}")
    st.markdown(f"**Créé le:** {metadata['date_creation']}")
    st.markdown(f"**Features:** {metadata['n_features']:,}")
    
    st.markdown("---")
    st.markdown("## ⚙️ Pondérations")
    for feature, poids in metadata['poids'].items():
        st.markdown(f"- **{feature.capitalize()}:** {poids}")
    
    st.markdown("---")
    st.markdown("## ℹ️ À propos")
    st.markdown("""
    Ce système de recommandation combine:
    - 📝 Titres et synopsis (TF-IDF)
    - 🎭 Genres des films
    - ⭐ Notes et popularité
    - 📅 Âge des films
    """)

# Sélection de films
st.markdown("---")
st.markdown("### 🔍 Sélectionnez un ou plusieurs films")

selected_titles = st.multiselect(
    "Entrez ou sélectionnez des titres de films pour obtenir des recommandations:",
    options=liste_titres,
    placeholder="Commencez à taper un titre de film...",
    help="Vous pouvez sélectionner plusieurs films pour obtenir des recommandations basées sur chacun d'eux"
)

# Afficher le nombre de films sélectionnés
if selected_titles:
    st.info(f"📌 {len(selected_titles)} film(s) sélectionné(s)")

# Bouton de recommandation
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    recommend_button = st.button(
        "🎯 Obtenir des recommandations",
        type="primary",
        use_container_width=True
    )

# Affichage des recommandations
if recommend_button:
    if selected_titles:
        with st.spinner("🔄 Recherche des meilleurs films pour vous..."):
            display_recommendations_with_posters(selected_titles)
    else:
        st.warning("⚠️ Veuillez sélectionner au moins un film!")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Développé avec ❤️ par CineMatch.io</p>
        <p>Système de recommandation basé sur l'analyse de contenu</p>
    </div>
    """,
    unsafe_allow_html=True
)
