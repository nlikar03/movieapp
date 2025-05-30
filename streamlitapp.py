import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

@st.cache_data
def load_data():
    ratings = pd.read_csv('podatki/ml-latest-small/ratings.csv')
    movies = pd.read_csv('podatki/ml-latest-small/movies.csv')
    
    movies['year'] = movies['title'].str.extract(r'\((\d{4})\)')
    return ratings, movies

ratings, movies = load_data()

def get_all_genres():
    genres = set()
    for genre_list in movies['genres'].str.split('|'):
        genres.update(genre_list)
    return sorted(genres)

def get_all_years():
    return sorted(movies['year'].dropna().unique(), reverse=True)

def get_movie_stats(movie_title):
    movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
    movie_ratings = ratings[ratings['movieId'] == movie_id]
    
    return {
        'avg_rating': movie_ratings['rating'].mean(),
        'count': len(movie_ratings),
        'std': movie_ratings['rating'].std()
    }

def get_ratings_over_time(movie_title):
    movie_id = movies[movies['title'] == movie_title]['movieId'].values[0]
    movie_ratings = ratings[ratings['movieId'] == movie_id].copy()
    movie_ratings['date'] = pd.to_datetime(movie_ratings['timestamp'], unit='s')
    movie_ratings['year'] = movie_ratings['date'].dt.year
    return movie_ratings.groupby('year').agg({'rating': ['mean', 'count']}).reset_index()

def get_user_ratings(user_id):
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {}
    return st.session_state.user_ratings.get(user_id, {})

def find_similar_users(user_id, n=5):
    return np.random.choice(ratings['userId'].unique(), n, replace=False)

def calculate_recommendations(user_ratings, similar_users):
    similar_users_ratings = ratings[ratings['userId'].isin(similar_users)]
    top_movies = similar_users_ratings.groupby('movieId')['rating'].mean().sort_values(ascending=False).head(10)
    return movies[movies['movieId'].isin(top_movies.index)].merge(
        pd.DataFrame({'movieId': top_movies.index, 'predicted_rating': top_movies.values}),
        on='movieId'
    ).sort_values('predicted_rating', ascending=False)

def show_analysis():
    st.header("Analiza filmov")
    
    min_ratings = st.slider("Minimalno število ocen", 1, 1000, 10)
    selected_genre = st.selectbox("Izberi žanr", ["Vsi"] + get_all_genres())
    selected_year = st.selectbox("Izberi leto", ["Vsi"] + get_all_years())
    
    movie_stats = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    
    merged = movie_stats.merge(movies, on='movieId')
    filtered = merged[merged['num_ratings'] >= min_ratings]
    
    if selected_genre != "Vsi":
        filtered = filtered[filtered['genres'].str.contains(selected_genre)]
    
    if selected_year != "Vsi":
        filtered = filtered[filtered['year'] == selected_year]
    
    st.write(f"Top 10 filmov po izbranih kriterijih:")
    st.dataframe(filtered.sort_values('avg_rating', ascending=False).head(10)[['title', 'avg_rating', 'num_ratings', 'genres']])

def show_comparison():
    st.header("Primerjava dveh filmov")
    
    movie_list = movies['title'].unique()
    movie1 = st.selectbox("Izberi prvi film", movie_list)
    movie2 = st.selectbox("Izberi drugi film", movie_list)
    
    if movie1 == movie2:
        st.warning("Izberita različna filma za primerjavo")
        return
    
    stats1 = get_movie_stats(movie1)
    stats2 = get_movie_stats(movie2)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader(movie1)
        st.write(f"Povprečna ocena: {stats1['avg_rating']:.2f}")
        st.write(f"Število ocen: {stats1['count']}")
        st.write(f"Standardni odklon: {stats1['std']:.2f}")
    
    with col2:
        st.subheader(movie2)
        st.write(f"Povprečna ocena: {stats2['avg_rating']:.2f}")
        st.write(f"Število ocen: {stats2['count']}")
        st.write(f"Standardni odklon: {stats2['std']:.2f}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    movie1_id = movies[movies['title'] == movie1]['movieId'].values[0]
    movie2_id = movies[movies['title'] == movie2]['movieId'].values[0]
    
    ratings[ratings['movieId'] == movie1_id]['rating'].hist(ax=ax1, bins=5)
    ax1.set_title(movie1)
    ax1.set_ylabel("Število ocen")
    
    ratings[ratings['movieId'] == movie2_id]['rating'].hist(ax=ax2, bins=5)
    ax2.set_title(movie2)
    
    st.pyplot(fig)
    
    time_data1 = get_ratings_over_time(movie1)
    time_data2 = get_ratings_over_time(movie2)
    
    fig, ax = plt.subplots()
    ax.plot(time_data1['year'], time_data1[('rating', 'mean')], label=f"{movie1} - povp. ocena")
    ax.plot(time_data2['year'], time_data2[('rating', 'mean')], label=f"{movie2} - povp. ocena")
    ax.set_xlabel("Leto")
    ax.set_ylabel("Povprečna ocena")
    ax.legend()
    st.subheader("Povprečna ocena skozi leta")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots()
    ax2.plot(time_data1['year'], time_data1[('rating', 'count')], label=f"{movie1} - št. ocen")
    ax2.plot(time_data2['year'], time_data2[('rating', 'count')], label=f"{movie2} - št. ocen")
    ax2.set_xlabel("Leto")
    ax2.set_ylabel("Število ocen")
    ax2.legend()
    st.subheader("Število ocen skozi leta")
    st.pyplot(fig2)


def show_recommendations():
    st.header("Osebna priporočila")
    
    if 'user_id' not in st.session_state:
        st.warning("Za ogled priporočil se prijavite zgoraj desno.")
        return
    
    show_rating_interface()
    
    user_ratings = get_user_ratings(st.session_state['user_id'])
    if len(user_ratings) < 10:
        st.warning(f"Potrebujete še {10 - len(user_ratings)} ocen za priporočila")
        return
    
    similar_users = find_similar_users(st.session_state['user_id'])
    recommendations = calculate_recommendations(user_ratings, similar_users)
    st.write("Vaša priporočila:")
    st.dataframe(recommendations[['title', 'genres', 'predicted_rating']])

def show_login():
    with st.form("Prijava"):
        username = st.text_input("Uporabniško ime")
        password = st.text_input("Geslo", type="password")
        submitted = st.form_submit_button("Prijava")
        
        if submitted:
            st.session_state['user_id'] = username
            st.session_state.user_ratings = {}
            st.success("Uspešno prijavljeni!")
            st.rerun()


def show_rating_interface():
    st.subheader("Oceni filme")
    movie_to_rate = st.selectbox("Izberi film", movies['title'])
    rating = st.slider("Ocena", 1, 5, 3)
    
    if st.button("Shrani oceno"):
        movie_id = movies[movies['title'] == movie_to_rate]['movieId'].values[0]
        if 'user_ratings' not in st.session_state:
            st.session_state.user_ratings = {}
        if st.session_state['user_id'] not in st.session_state.user_ratings:
            st.session_state.user_ratings[st.session_state['user_id']] = {}
        
        st.session_state.user_ratings[st.session_state['user_id']][movie_id] = rating
        st.success(f"Shranjena ocena {rating} za film {movie_to_rate}")

# Glavna aplikacija
def main():
    st.title('MovieLens Analiza in Priporočila')

    if 'user_id' not in st.session_state:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("🔐 Prijava")
        with st.container():
            col1, col2, col3 = st.columns([3, 4, 3])
            with col2:
                with st.form("login_form", clear_on_submit=True):
                    username = st.text_input("Uporabniško ime")
                    password = st.text_input("Geslo", type="password")
                    submitted = st.form_submit_button("Prijava")
                    if submitted:
                        st.session_state['user_id'] = username
                        st.session_state.user_ratings = {}
                        st.success("Uspešno prijavljeni!")
                        st.rerun()
        return

    col1, col2 = st.columns([9, 1])
    with col2:
        st.markdown(f"👤 {st.session_state['user_id']}")
        if st.button("Odjava"):
            del st.session_state['user_id']
            st.rerun()

    page = st.sidebar.radio(
        "Navigacija",
        ["Analiza filmov", "Primerjava filmov", "Priporočila"]
    )

    if page == "Analiza filmov":
        show_analysis()
    elif page == "Primerjava filmov":
        show_comparison()
    else:
        show_recommendations()


if __name__ == "__main__":
    main()