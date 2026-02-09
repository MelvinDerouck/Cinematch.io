import os
from pathlib import Path
from dotenv import load_dotenv

# Charger le fichier .env
load_dotenv()

# Configuration
TMDB_API_TOKEN = os.getenv('TMDB_API_TOKEN')
TMDB_API_KEY = os.getenv('TMDB_API_KEY')

# Vérifier que les variables sont chargées
if not TMDB_API_TOKEN:
    raise ValueError(
        "TMDB_API_TOKEN non trouvé. "
    )

# Chemins
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'data'