import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking charts
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 7)
plt.rcParams['font.size'] = 10

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/dataset.csv')

# Create charts directory if it doesn't exist
import os
os.makedirs('charts', exist_ok=True)

print("Generating visualizations...\n")

# Chart 1: Popularity Distribution
print("1/15 - Popularity Distribution")
plt.figure(figsize=(12, 7))
plt.hist(df['popularity'], bins=50, edgecolor='black', alpha=0.7, color='#1DB954')
plt.xlabel('Popularity Score', fontsize=12)
plt.ylabel('Number of Tracks', fontsize=12)
plt.title('Distribution of Track Popularity (114,000 Spotify Tracks)', fontsize=14, fontweight='bold')
plt.axvline(df['popularity'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {df["popularity"].mean():.1f}')
plt.axvline(df['popularity'].median(), color='orange', linestyle='--', linewidth=2, label=f'Median: {df["popularity"].median():.1f}')
plt.legend()
plt.tight_layout()
plt.savefig('charts/01_popularity_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 2: Top 15 Genres by Average Popularity
print("2/15 - Top Genres by Popularity")
plt.figure(figsize=(12, 8))
genre_pop = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=True).tail(15)
colors = plt.cm.viridis(np.linspace(0, 1, len(genre_pop)))
genre_pop.plot(kind='barh', color=colors)
plt.xlabel('Average Popularity Score', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.title('Top 15 Genres by Average Popularity', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/02_top_genres_popularity.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 3: Audio Features Distribution
print("3/15 - Audio Features Distribution")
audio_features = ['danceability', 'energy', 'speechiness', 'acousticness',
                  'instrumentalness', 'liveness', 'valence']
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, feature in enumerate(audio_features):
    axes[idx].hist(df[feature], bins=50, edgecolor='black', alpha=0.7, color='#1DB954')
    axes[idx].set_title(f'{feature.capitalize()}', fontweight='bold')
    axes[idx].set_xlabel('Value')
    axes[idx].set_ylabel('Frequency')
    axes[idx].axvline(df[feature].mean(), color='red', linestyle='--', linewidth=2)

# Hide extra subplots
for idx in range(len(audio_features), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Distribution of Audio Features', fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('charts/03_audio_features_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 4: Correlation Heatmap
print("4/15 - Feature Correlations")
plt.figure(figsize=(12, 10))
correlation_features = ['popularity', 'danceability', 'energy', 'loudness',
                        'speechiness', 'acousticness', 'instrumentalness',
                        'liveness', 'valence', 'tempo', 'duration_ms']
corr_matrix = df[correlation_features].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('charts/04_correlation_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 5: Top 20 Artists by Track Count
print("5/15 - Top Artists")
plt.figure(figsize=(12, 8))
all_artists = df['artists'].str.split(';').explode()
artist_counts = all_artists.value_counts().head(20)
colors = plt.cm.plasma(np.linspace(0, 1, len(artist_counts)))
artist_counts.plot(kind='barh', color=colors)
plt.xlabel('Number of Tracks', fontsize=12)
plt.ylabel('Artist', fontsize=12)
plt.title('Top 20 Artists by Track Count', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/05_top_artists.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 6: Danceability vs Energy by Genre (Top 10 Popular Genres)
print("6/15 - Danceability vs Energy")
plt.figure(figsize=(14, 8))
top_genres = df.groupby('track_genre')['popularity'].mean().sort_values(ascending=False).head(10).index
df_top = df[df['track_genre'].isin(top_genres)]

for genre in top_genres:
    genre_data = df_top[df_top['track_genre'] == genre]
    plt.scatter(genre_data['danceability'], genre_data['energy'],
                alpha=0.6, label=genre, s=20)

plt.xlabel('Danceability', fontsize=12)
plt.ylabel('Energy', fontsize=12)
plt.title('Danceability vs Energy (Top 10 Most Popular Genres)', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/06_danceability_vs_energy.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 7: Track Duration Distribution
print("7/15 - Track Duration Distribution")
plt.figure(figsize=(12, 7))
duration_minutes = df['duration_ms'] / 1000 / 60
# Filter out outliers for better visualization
duration_filtered = duration_minutes[duration_minutes < 10]
plt.hist(duration_filtered, bins=60, edgecolor='black', alpha=0.7, color='#1DB954')
plt.xlabel('Duration (minutes)', fontsize=12)
plt.ylabel('Number of Tracks', fontsize=12)
plt.title('Distribution of Track Duration (filtered < 10 min)', fontsize=14, fontweight='bold')
plt.axvline(duration_minutes.mean(), color='red', linestyle='--', linewidth=2,
            label=f'Mean: {duration_minutes.mean():.2f} min')
plt.axvline(duration_minutes.median(), color='orange', linestyle='--', linewidth=2,
            label=f'Median: {duration_minutes.median():.2f} min')
plt.legend()
plt.tight_layout()
plt.savefig('charts/07_duration_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 8: Explicit Content Analysis by Genre
print("8/15 - Explicit Content by Genre")
plt.figure(figsize=(12, 8))
genre_explicit = df.groupby('track_genre')['explicit'].mean().sort_values(ascending=False).head(15) * 100
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(genre_explicit)))
genre_explicit.plot(kind='barh', color=colors)
plt.xlabel('Percentage of Explicit Tracks (%)', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.title('Top 15 Genres by Explicit Content Percentage', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/08_explicit_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 9: Loudness vs Popularity
print("9/15 - Loudness vs Popularity")
plt.figure(figsize=(12, 7))
plt.hexbin(df['loudness'], df['popularity'], gridsize=50, cmap='YlOrRd', mincnt=1)
plt.colorbar(label='Track Count')
plt.xlabel('Loudness (dB)', fontsize=12)
plt.ylabel('Popularity Score', fontsize=12)
plt.title('Loudness vs Popularity', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/09_loudness_vs_popularity.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 10: Valence (Positivity) by Genre
print("10/15 - Valence by Genre")
plt.figure(figsize=(12, 8))
genre_valence = df.groupby('track_genre')['valence'].mean().sort_values(ascending=True)
top_happy = genre_valence.tail(10)
bottom_sad = genre_valence.head(10)
combined = pd.concat([bottom_sad, top_happy])

fig, ax = plt.subplots(figsize=(12, 8))
colors = ['#FF6B6B' if x < 0.5 else '#4ECDC4' for x in combined.values]
combined.plot(kind='barh', color=colors, ax=ax)
plt.xlabel('Average Valence (Musical Positivity)', fontsize=12)
plt.ylabel('Genre', fontsize=12)
plt.title('Most Sad vs Most Happy Genres (by Valence)', fontsize=14, fontweight='bold')
plt.axvline(0.5, color='black', linestyle='--', linewidth=1, alpha=0.5)
plt.tight_layout()
plt.savefig('charts/10_valence_by_genre.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 11: Key Distribution
print("11/15 - Key Distribution")
plt.figure(figsize=(12, 7))
key_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
key_counts = df['key'].value_counts().sort_index()
colors = plt.cm.Set3(np.linspace(0, 1, 12))
plt.bar(range(12), [key_counts.get(i, 0) for i in range(12)], color=colors, edgecolor='black')
plt.xlabel('Musical Key', fontsize=12)
plt.ylabel('Number of Tracks', fontsize=12)
plt.title('Distribution of Musical Keys', fontsize=14, fontweight='bold')
plt.xticks(range(12), key_names)
plt.tight_layout()
plt.savefig('charts/11_key_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 12: Popularity by Audio Feature Quartiles
print("12/15 - Popularity by Feature Quartiles")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
features_to_analyze = ['danceability', 'energy', 'valence', 'acousticness']

for idx, feature in enumerate(features_to_analyze):
    ax = axes[idx // 2, idx % 2]
    df[f'{feature}_quartile'] = pd.qcut(df[feature], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
    quartile_pop = df.groupby(f'{feature}_quartile')['popularity'].mean()

    colors_grad = plt.cm.viridis(np.linspace(0, 1, 4))
    quartile_pop.plot(kind='bar', ax=ax, color=colors_grad, edgecolor='black')
    ax.set_title(f'Popularity by {feature.capitalize()} Quartile', fontweight='bold')
    ax.set_xlabel('Quartile', fontsize=10)
    ax.set_ylabel('Average Popularity', fontsize=10)
    ax.tick_params(axis='x', rotation=45)

plt.suptitle('How Audio Features Affect Popularity', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/12_popularity_by_feature_quartiles.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 13: Genre Landscape - Danceability vs Energy
print("13/15 - Genre Landscape")
plt.figure(figsize=(14, 10))
genre_stats = df.groupby('track_genre').agg({
    'danceability': 'mean',
    'energy': 'mean',
    'popularity': 'mean',
    'track_id': 'count'
}).rename(columns={'track_id': 'count'})

scatter = plt.scatter(genre_stats['danceability'], genre_stats['energy'],
                     s=genre_stats['count']/5, alpha=0.6,
                     c=genre_stats['popularity'], cmap='RdYlGn',
                     edgecolors='black', linewidth=0.5)

# Annotate top 10 popular genres
top_10_genres = genre_stats.nlargest(10, 'popularity')
for genre, row in top_10_genres.iterrows():
    plt.annotate(genre, (row['danceability'], row['energy']),
                fontsize=8, alpha=0.8)

plt.colorbar(scatter, label='Average Popularity')
plt.xlabel('Average Danceability', fontsize=12)
plt.ylabel('Average Energy', fontsize=12)
plt.title('Genre Landscape: Danceability vs Energy (bubble size = track count)',
          fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('charts/13_genre_landscape.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 14: Tempo Distribution by Mode (Major vs Minor)
print("14/15 - Tempo by Mode")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

mode_labels = {0: 'Minor', 1: 'Major'}
for mode in [0, 1]:
    mode_data = df[df['mode'] == mode]['tempo']
    axes[mode].hist(mode_data, bins=50, edgecolor='black', alpha=0.7,
                   color='#1DB954' if mode == 1 else '#FF6B6B')
    axes[mode].set_title(f'{mode_labels[mode]} Key ({len(mode_data):,} tracks)', fontweight='bold')
    axes[mode].set_xlabel('Tempo (BPM)', fontsize=11)
    axes[mode].set_ylabel('Number of Tracks', fontsize=11)
    axes[mode].axvline(mode_data.mean(), color='black', linestyle='--', linewidth=2,
                      label=f'Mean: {mode_data.mean():.1f} BPM')
    axes[mode].legend()

plt.suptitle('Tempo Distribution: Major vs Minor Keys', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('charts/14_tempo_by_mode.png', dpi=300, bbox_inches='tight')
plt.close()

# Chart 15: Feature Importance for Popularity
print("15/15 - Feature Correlation with Popularity")
plt.figure(figsize=(12, 8))
audio_features_corr = ['danceability', 'energy', 'loudness', 'speechiness',
                       'acousticness', 'instrumentalness', 'liveness', 'valence',
                       'tempo', 'duration_ms']
correlations = df[audio_features_corr].corrwith(df['popularity']).sort_values()

colors = ['#FF6B6B' if x < 0 else '#4ECDC4' for x in correlations.values]
correlations.plot(kind='barh', color=colors, edgecolor='black')
plt.xlabel('Correlation Coefficient', fontsize=12)
plt.ylabel('Audio Feature', fontsize=12)
plt.title('Audio Features Correlation with Popularity', fontsize=14, fontweight='bold')
plt.axvline(0, color='black', linestyle='-', linewidth=0.8)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('charts/15_feature_correlation_popularity.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*80)
print("All 15 visualizations created successfully in 'charts/' directory!")
print("="*80)
