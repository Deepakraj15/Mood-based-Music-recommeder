import numpy as np
import pandas as pd
from Spotify import GetSongs
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

def GetActualSong(pred_class):
    
    if( pred_class=='Disgust' ):

        Play = music_df[music_df['mood'] =='Sad' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:20].reset_index(drop=True)
        RecommendSongs(Play.iloc[0]['name'],Play)

    if( pred_class=='Happy' or pred_class=='Sad' ):

        Play = music_df[music_df['mood'] =='Happy' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        RecommendSongs(Play.iloc[0]['name'],Play)

    if( pred_class=='Fear' or pred_class=='Angry' ):

        Play = music_df[music_df['mood'] =='Calm' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        RecommendSongs(Play.iloc[0]['name'],Play)

        
    if( pred_class=='Surprise' or pred_class=='Neutral' ):

        Play = music_df[music_df['mood'] =='Energetic' ]
        Play = Play.sort_values(by="popularity", ascending=False)
        Play = Play[:5].reset_index(drop=True)
        RecommendSongs(Play.iloc[0]['name'],Play)

def RecommendSimiliarSongs(song_name, data):
    
    # Getting vector for the input song.
    text_array1 = song_vectorizer.transform(data[data['name']==song_name]['mood']).toarray()
    print(text_array1)

    num_array1 = data[data['name']==song_name].select_dtypes(include=np.number).to_numpy()
    
    # We will store similarity for each row of the dataset.
    sim = []
    for idx, row in data.iterrows():
        name = row['name']
        # Getting vector for current song.
        text_array2 = song_vectorizer.transform(data[data['name']==name]['mood']).toarray()
        num_array2 = data[data['name']==name].select_dtypes(include=np.number).to_numpy()
    
        # Calculating similarities for text as well as numeric features
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)
        
    return sim


def RecommendSongs(song_name, Play):
  # Base case
    if Play[Play['name'] == song_name].shape[0] == 0:
        print('This song is either not so popular or you\
        have entered invalid_name.\n Some songs you may like:\n')
        
        for song in Play.sample(n=5)['name'].values:
            print(song)
        return
    
    Play['similarity_factor'] = RecommendSimiliarSongs(song_name, Play)
    
    Play.sort_values(by=['similarity_factor', 'popularity'],
                    ascending = [False, False],
                    inplace=True)
    
    # First song will be the input song itself as the similarity will be highest.
    print(Play[['name', 'artist']][2:])



music_df =pd.read_csv("./data_moods.csv")
song_vectorizer = CountVectorizer()
song_vectorizer.fit(music_df['mood'])
