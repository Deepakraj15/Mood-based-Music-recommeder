import tekore as tk
import os
from dotenv import load_dotenv
import spotipy

def GetAuthorized():    
    load_dotenv()
    from spotipy.oauth2 import SpotifyClientCredentials

    client_id = os.getenv('CLIENT_ID')
    client_secret =os.getenv('CLIENT_SECRET_KEY')

    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
    return sp

def GetSongs(song):
    results = GetAuthorized().search(q=song, type='track')
    print(results['tracks']['items'][0]['name'])

