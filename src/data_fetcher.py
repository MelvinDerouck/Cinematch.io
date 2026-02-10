import requests
from src.config import TMDB_API_TOKEN

def fetch_movies():
    url = "https://api.themoviedb.org/3/discover/movie?include_adult=true&include_video=false&language=fr-FR&page=1&sort_by=popularity.desc"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_TOKEN}"
    }
    response = requests.get(url, headers=headers)
    return response.json()

def fetch_genres():
    url = "https://api.themoviedb.org/3/genre/movie/list?language=fr"
    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_API_TOKEN}"
    }  
    response = requests.get(url, headers=headers)
    return response.json()