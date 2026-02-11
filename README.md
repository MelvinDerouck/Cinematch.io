# Système de Recommandation de Films par Analyse de Contenu

<p align="center">
  <img src="img/gif_app.gif" alt="Démonstration de l'application CineMatch.io" width="800">
</p>

<p align="center">
  <em>Interface de recommandation : sélection d'un film et affichage des résultats personnalisés</em>
</p>

## Vue d'ensemble du projet

Ce projet implémente un système de recommandation de films basé sur l'analyse de contenu (Content-Based Filtering). Le système combine plusieurs techniques de traitement du langage naturel et d'apprentissage automatique pour identifier les similarités entre films et proposer des recommandations pertinentes.

### Objectif

Développer un système capable de recommander des films similaires à partir d'un ou plusieurs films de référence, en exploitant les informations textuelles (synopsis, titres) et structurées (genres, métadonnées) disponibles.

### Technologies principales

- **Python 3.12**
- **Sentence Transformers** : Pour les embeddings sémantiques
- **Scikit-learn** : Pour le TF-IDF et les métriques de similarité
- **Pandas & NumPy** : Pour la manipulation de données
- **Streamlit** : Pour l'interface utilisateur
- **NLTK / spaCy** : Pour le prétraitement textuel

---

## Architecture du système

### Schéma général

```
Données brutes (TMDB API)
    |
    v
Prétraitement (nettoyage, tokenisation, lemmatisation)
    |
    v
Feature Engineering
    |-- Synopsis : Embeddings sémantiques (384 dimensions)
    |-- Titres : TF-IDF (500 features)
    |-- Genres : Encodage binaire (19 catégories)
    |-- Métadonnées : Normalisation Min-Max (note, popularité, âge)
    |
    v
Combinaison pondérée des features
    |
    v
Calcul de la matrice de similarité cosinus
    |
    v
Sauvegarde du modèle
    |
    v
Application Streamlit
```

---

## Étapes du projet

### 1. Collecte et préparation des données

#### 1.1 Source des données

Les données proviennent de The Movie Database (TMDB) via leur API. Chaque film contient :

- **Titre** : Titre original du film
- **Synopsis** : Description textuelle du contenu
- **Genres** : Liste de catégories (Action, Drame, Comédie, etc.)
- **Note** : Note moyenne des utilisateurs (0-10)
- **Popularité** : Score de popularité TMDB
- **Année de sortie** : Pour calculer l'âge du film
- **Affiche** : URL de l'image d'affiche

#### 1.2 Prétraitement des données

##### Synopsis

Le prétraitement des synopsis suit ces étapes :

1. **Nettoyage de base**
   - Conversion en minuscules
   - Suppression des espaces multiples
   - Suppression des caractères spéciaux

2. **Tokenisation**
   - Utilisation de `RegexpTokenizer` pour extraire les mots
   - Pattern : `\w+` (caractères alphanumériques)

3. **Suppression des stop words**
   - Liste de mots vides en français (NLTK)
   - Filtrage des mots fonctionnels sans valeur sémantique

4. **Filtrage**
   - Suppression des mots rares (apparaissant < 2 fois)
   - Conservation uniquement des mots de longueur >= 2
   - Filtrage des tokens non-alphabétiques

5. **Normalisation** (optionnelle selon l'approche)
   - Lemmatisation : Réduction à la forme canonique (spaCy)
   - Stemming : Réduction à la racine (NLTK Snowball Stemmer)

**Résultat** : Colonne `clean_synopsis_str` contenant les synopsis nettoyés

##### Genres

Les genres sont encodés en format binaire (one-hot encoding) :

```
Film A : [Action, Drame] -> [1, 0, 0, 1, 0, ...]
Film B : [Comédie]       -> [0, 0, 1, 0, 0, ...]
```

**Fichier de sortie** : `genres_binarized.csv`

##### Métadonnées numériques

Création de features dérivées :

- **Age du film** : `année_actuelle - année_sortie`
- **Note** : Conservation de la note TMDB
- **Popularité** : Conservation du score de popularité

**Fichier de sortie** : `df_movies_preprocess.csv`

---

### 2. Feature Engineering

#### 2.1 Embeddings sémantiques pour les synopsis

##### Choix du modèle

Modèle utilisé : `paraphrase-multilingual-MiniLM-L12-v2`

**Caractéristiques** :

- Modèle basé sur BERT
- Entraîné sur 50+ langues incluant le français
- Dimension des embeddings : 384
- Optimisé pour la tâche de paraphrase (détection de similarité sémantique)

##### Processus de création

```python
from sentence_transformers import SentenceTransformer

# Chargement du modèle
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# Création des embeddings (batch processing)
embeddings = model.encode(
    synopsis_list,
    batch_size=32,
    show_progress_bar=True
)

# Résultat : matrice [n_films, 384]
```

##### Normalisation et pondération

1. **Normalisation L2** : Chaque vecteur est normalisé pour avoir une norme euclidienne de 1
2. **Pondération** : Multiplication par un coefficient (1.5) pour augmenter l'importance des synopsis

```python
from sklearn.preprocessing import normalize

synopsis_normalized = normalize(embeddings, norm='l2')
synopsis_weighted = synopsis_normalized * 1.5
```

#### 2.2 TF-IDF pour les titres

##### Justification

Les titres de films sont courts et factuels. L'approche TF-IDF est suffisante car :

- Les titres contiennent peu de contexte sémantique
- La recherche par mots-clés exacts est pertinente
- Efficacité computationnelle supérieure
- Pas de plus-value significative des embeddings sur des textes courts

##### Paramètres

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    max_features=500,      # Limitation à 500 mots les plus importants
    ngram_range=(1, 1),    # Unigrammes uniquement
    lowercase=True,        # Normalisation casse
    strip_accents='unicode' # Suppression accents
)

titre_tfidf = vectorizer.fit_transform(titres)
```

**Résultat** : Matrice sparse [n_films, 500]

#### 2.3 Normalisation des features numériques

Les features numériques sont normalisées avec Min-Max Scaling pour les ramener dans l'intervalle [0, 1] :

```python
from sklearn.preprocessing import MinMaxScaler

# Note (importance : 0.75)
note_scaler = MinMaxScaler()
note_scaled = note_scaler.fit_transform(df[['Note']]) * 0.75

# Age du film (importance : 0.5)
age_scaler = MinMaxScaler()
age_scaled = age_scaler.fit_transform(df[['Age du film']]) * 0.5

# Popularité (importance : 1.0)
pop_scaler = MinMaxScaler()
pop_scaled = pop_scaler.fit_transform(df[['Popularité']]) * 1.0
```

**Formule Min-Max** :

```
x_scaled = (x - x_min) / (x_max - x_min)
```

#### 2.4 Combinaison des features

Les différentes features sont concaténées horizontalement pour former une matrice unique :

```python
from scipy.sparse import hstack, csr_matrix

# Conversion des embeddings en sparse matrix
synopsis_sparse = csr_matrix(synopsis_weighted)

# Concaténation horizontale
features_combined = hstack([
    synopsis_sparse,  # [n, 384] poids 1.5
    titre_tfidf,      # [n, 500] poids 1.0
    genre_matrix,     # [n, 19]  poids 1.0
    note_scaled,      # [n, 1]   poids 0.75
    age_scaled,       # [n, 1]   poids 0.5
    pop_scaled        # [n, 1]   poids 1.0
])

# Résultat : matrice [n_films, ~906 features]
```

---

### 3. Calcul de la similarité

#### 3.1 Métrique de similarité cosinus

La similarité cosinus mesure l'angle entre deux vecteurs dans l'espace multi-dimensionnel :

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

**Propriétés** :

- Valeur dans l'intervalle [-1, 1]
- 1 : Vecteurs identiques (angle de 0°)
- 0 : Vecteurs orthogonaux (angle de 90°)
- -1 : Vecteurs opposés (angle de 180°)

**Avantages** :

- Invariant à la magnitude (normalisation implicite)
- Rapide à calculer
- Bien adapté aux vecteurs de features hétérogènes

#### 3.2 Calcul de la matrice

```python
from sklearn.metrics.pairwise import cosine_similarity

# Calcul de la matrice de similarité complète
similarity_matrix = cosine_similarity(features_combined)

# Résultat : matrice [n_films, n_films]
# similarity_matrix[i, j] = similarité entre film i et film j
```

#### 3.3 Statistiques de similarité

Analyse statistique de la distribution des similarités :

```python
import numpy as np

# Exclusion de la diagonale (similarité avec soi-même)
mask = np.ones_like(similarity_matrix, dtype=bool)
np.fill_diagonal(mask, False)
similarities = similarity_matrix[mask]

# Statistiques descriptives
stats = {
    'moyenne': np.mean(similarities),
    'médiane': np.median(similarities),
    'écart-type': np.std(similarities),
    'min': np.min(similarities),
    'max': np.max(similarities)
}
```

**Interprétation** :

- **Moyenne ~0.25-0.35** : Bonne diversité du catalogue
- **Écart-type ~0.10-0.15** : Variation raisonnable
- **Max <0.95** : Pas de doublons exacts (hors séries/suites)

---

### 4. Fonction de recommandation

#### 4.1 Algorithme

```python
def get_recommendations(title, similarity_matrix, df, top_n=9):
    """
    Recommande les films les plus similaires à un film donné.
    
    Args:
        title (str): Titre du film de référence
        similarity_matrix (np.ndarray): Matrice de similarité précalculée
        df (pd.DataFrame): DataFrame contenant les métadonnées des films
        top_n (int): Nombre de recommandations à retourner
    
    Returns:
        pd.DataFrame: Films recommandés avec titres et affiches
    """
    # Étape 1 : Localisation du film
    idx = df.index[df['Titre'] == title].tolist()[0]
    
    # Étape 2 : Extraction des scores de similarité
    sim_scores = similarity_matrix[idx]
    
    # Étape 3 : Tri par ordre décroissant
    sorted_indices = np.argsort(sim_scores)[::-1]
    
    # Étape 4 : Sélection des top_n (en excluant le film lui-même)
    top_indices = sorted_indices[1:top_n+1]
    
    # Étape 5 : Récupération des métadonnées
    recommendations = df.iloc[top_indices][['Titre', 'Affiche']]
    
    return recommendations
```

## Dépendances

### Fichier requirements.txt

```
pandas==2.1.0
numpy==1.24.3
scikit-learn==1.3.0
scipy==1.11.2
sentence-transformers==2.2.2
streamlit==1.27.0
nltk==3.8.1
spacy==3.6.1
tqdm==4.66.1
python-dotenv==1.0.0
```

---

## Auteur et licence

Projet développé dans le cadre d'un portfolio de Data Science.

License: MIT

---

**Date de dernière mise à jour** : Février 2026

**Version** : 2.0 (avec embeddings sémantiques)
