"""
============================================================
Hard Task: Beta-VAE and CVAE for Multi-Modal Music Clustering
============================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
    normalized_mutual_info_score,
    confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Random seed for reproducibility
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# Configuration
# ============================================================

class Config:
    """Configuration parameters for hard task."""
    
    # Paths
    AUDIO_FEATURES_PATH = "data/processed/gtzan_features.csv"
    LYRICS_PATH = "data/lyrics/song_lyrics.csv"
    RESULTS_DIR = "results/hard_task"
    MODELS_DIR = "results/models"
    
    # Model parameters
    LATENT_DIM = 32
    HIDDEN_DIMS = [512, 256, 128]
    BETA = 4.0  # Beta for Beta-VAE (higher = more disentanglement)
    
    # Training parameters
    EPOCHS = 200
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    WEIGHT_DECAY = 1e-5
    
    # Feature parameters
    AUDIO_WEIGHT = 0.5
    LYRICS_WEIGHT = 0.3
    GENRE_WEIGHT = 0.2
    
    # Clustering
    N_CLUSTERS = 10
    
    # Lyrics sampling
    LYRICS_SAMPLE_PER_GENRE = 100
    LYRICS_MAX_LENGTH = 500


# ============================================================
# Utility Functions
# ============================================================

def ensure_directory(path):
    """Create directory if it does not exist."""
    if not os.path.exists(path):
        os.makedirs(path)


def compute_cluster_purity(true_labels, pred_labels):
    """
    Compute cluster purity score.
    
    Purity = (1/N) * sum of max class count in each cluster
    
    Args:
        true_labels: Ground truth labels
        pred_labels: Predicted cluster labels
        
    Returns:
        float: Purity score between 0 and 1
    """
    contingency = confusion_matrix(true_labels, pred_labels)
    purity = np.sum(np.max(contingency, axis=0)) / np.sum(contingency)
    return purity


# ============================================================
# Data Loading and Processing
# ============================================================

def load_audio_features(features_path):
    """Load GTZAN audio features."""
    print("\n" + "=" * 60)
    print("Loading Audio Features")
    print("=" * 60)
    
    if not os.path.exists(features_path):
        print(f"Error: File not found - {features_path}")
        return None, None, None
    
    df = pd.read_csv(features_path)
    
    label_columns = ['filename', 'genre', 'filepath']
    feature_columns = [col for col in df.columns if col not in label_columns]
    
    X = df[feature_columns].values
    y = df['genre'].values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"Audio features: {X.shape}")
    print(f"Genres: {np.unique(y)}")
    
    return X, y, feature_columns


def load_lyrics_by_genre(lyrics_path, genres, samples_per_genre=100, max_length=500):
    """
    Load real lyrics from Genius dataset matched by genre.
    
    Args:
        lyrics_path: Path to lyrics CSV
        genres: Array of genre labels from GTZAN
        samples_per_genre: Number of lyrics to sample per genre
        max_length: Maximum characters per lyric
        
    Returns:
        dict: Genre to lyrics mapping
    """
    print("\n" + "=" * 60)
    print("Loading Lyrics by Genre")
    print("=" * 60)
    
    # Genre mapping: GTZAN genre -> Genius tag
    genre_to_tag = {
        'hiphop': 'rap',
        'rock': 'rock',
        'pop': 'pop',
        'country': 'country',
        'metal': 'rock',
        'blues': None,
        'classical': None,
        'disco': None,
        'jazz': None,
        'reggae': None
    }
    
    # Load lyrics in chunks to handle large file
    genre_lyrics = {genre: [] for genre in np.unique(genres)}
    
    print("Reading lyrics file...")
    chunk_size = 50000
    
    try:
        for chunk in pd.read_csv(
            lyrics_path,
            chunksize=chunk_size,
            usecols=['tag', 'lyrics', 'language'],
            on_bad_lines='skip'
        ):
            # Filter English lyrics
            chunk = chunk[chunk['language'] == 'en']
            chunk = chunk.dropna(subset=['lyrics', 'tag'])
            
            for gtzan_genre, genius_tag in genre_to_tag.items():
                if genius_tag and gtzan_genre in genre_lyrics:
                    matching = chunk[chunk['tag'] == genius_tag]['lyrics'].tolist()
                    current_count = len(genre_lyrics[gtzan_genre])
                    needed = samples_per_genre - current_count
                    
                    if needed > 0 and matching:
                        to_add = matching[:needed]
                        genre_lyrics[gtzan_genre].extend(to_add)
            
            # Check if we have enough
            all_full = all(
                len(lyrics) >= samples_per_genre 
                for genre, lyrics in genre_lyrics.items() 
                if genre_to_tag.get(genre)
            )
            if all_full:
                break
                
    except Exception as e:
        print(f"Error reading lyrics: {e}")
    
    # Print statistics
    print("\nLyrics loaded per genre:")
    for genre, lyrics in genre_lyrics.items():
        print(f"  {genre}: {len(lyrics)} lyrics")
    
    return genre_lyrics


def create_lyrics_embeddings_real(genres, genre_lyrics, max_length=500):
    """
    Create lyrics embeddings using real lyrics where available.
    
    Args:
        genres: Array of genre labels for each sample
        genre_lyrics: Dictionary of genre -> lyrics list
        max_length: Maximum characters per lyric
        
    Returns:
        numpy.ndarray: Lyrics embeddings
    """
    print("\n" + "=" * 60)
    print("Creating Lyrics Embeddings")
    print("=" * 60)
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"Sentence Transformer loaded (dim: {embedding_dim})")
    except ImportError:
        os.system("pip install sentence-transformers")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_dim = model.get_sentence_embedding_dimension()
    
    # Fallback descriptions for genres without lyrics
    genre_descriptions = {
        'blues': "Sad emotional music about heartbreak and hard times",
        'classical': "Orchestral symphony with violin piano and elegant harmony",
        'disco': "Dance party music with funky beats and groove",
        'jazz': "Smooth improvisation with saxophone and swing rhythm",
        'reggae': "Island vibes with positive message and laid back rhythm"
    }
    
    embeddings = []
    genre_indices = {genre: 0 for genre in np.unique(genres)}
    
    for genre in tqdm(genres, desc="Creating embeddings"):
        genre_lower = genre.lower()
        
        if genre_lyrics.get(genre_lower) and len(genre_lyrics[genre_lower]) > 0:
            # Use real lyrics
            idx = genre_indices[genre_lower] % len(genre_lyrics[genre_lower])
            lyrics_text = str(genre_lyrics[genre_lower][idx])[:max_length]
            genre_indices[genre_lower] += 1
        elif genre_lower in genre_descriptions:
            # Use fallback description
            lyrics_text = genre_descriptions[genre_lower]
        else:
            lyrics_text = f"This is {genre} music"
        
        embedding = model.encode(lyrics_text, show_progress_bar=False)
        noise = np.random.randn(embedding_dim) * 0.02
        embeddings.append(embedding + noise)
    
    embeddings = np.array(embeddings)
    print(f"Embeddings shape: {embeddings.shape}")
    
    return embeddings


def create_genre_embeddings(genres):
    """
    Create one-hot encoded genre embeddings.
    
    Args:
        genres: Array of genre labels
        
    Returns:
        tuple: (embeddings array, label encoder)
    """
    print("\n" + "=" * 60)
    print("Creating Genre Embeddings")
    print("=" * 60)
    
    le = LabelEncoder()
    genre_encoded = le.fit_transform(genres)
    
    # One-hot encoding
    n_classes = len(le.classes_)
    genre_onehot = np.zeros((len(genres), n_classes))
    genre_onehot[np.arange(len(genres)), genre_encoded] = 1
    
    print(f"Genre embeddings shape: {genre_onehot.shape}")
    print(f"Classes: {le.classes_}")
    
    return genre_onehot, le


def create_multimodal_features(audio, lyrics, genre, audio_w=0.5, lyrics_w=0.3, genre_w=0.2):
    """
    Combine audio, lyrics, and genre features.
    
    Args:
        audio: Audio feature matrix
        lyrics: Lyrics embedding matrix
        genre: Genre embedding matrix
        audio_w: Weight for audio features
        lyrics_w: Weight for lyrics features
        genre_w: Weight for genre features
        
    Returns:
        numpy.ndarray: Combined multi-modal features
    """
    print("\n" + "=" * 60)
    print("Creating Multi-Modal Features")
    print("=" * 60)
    
    # Normalize each modality
    audio_scaler = StandardScaler()
    lyrics_scaler = StandardScaler()
    genre_scaler = StandardScaler()
    
    audio_norm = audio_scaler.fit_transform(audio) * audio_w
    lyrics_norm = lyrics_scaler.fit_transform(lyrics) * lyrics_w
    genre_norm = genre_scaler.fit_transform(genre) * genre_w
    
    # Concatenate
    multimodal = np.concatenate([audio_norm, lyrics_norm, genre_norm], axis=1)
    
    print(f"Audio: {audio.shape[1]}D (weight: {audio_w})")
    print(f"Lyrics: {lyrics.shape[1]}D (weight: {lyrics_w})")
    print(f"Genre: {genre.shape[1]}D (weight: {genre_w})")
    print(f"Multi-modal: {multimodal.shape[1]}D total")
    
    return multimodal


# ============================================================
# Beta-VAE Model
# ============================================================

class BetaVAE(nn.Module):
    """
    Beta-VAE for disentangled latent representations.
    
    Beta > 1 encourages disentanglement by putting more weight
    on the KL divergence term.
    
    Args:
        input_dim: Dimension of input features
        hidden_dims: List of hidden layer dimensions
        latent_dim: Dimension of latent space
        beta: Weight for KL divergence (default: 4.0)
    """
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], latent_dim=32, beta=4.0):
        super(BetaVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in hidden_dims_rev:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        print(f"Beta-VAE: {input_dim}D -> {latent_dim}D (beta={beta})")
    
    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def get_latent(self, x):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu
    
    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss


# ============================================================
# Conditional VAE Model
# ============================================================

class ConditionalVAE(nn.Module):
    """
    Conditional VAE that conditions on genre labels.
    
    The condition (genre) is concatenated to both encoder input
    and decoder input.
    
    Args:
        input_dim: Dimension of input features
        n_classes: Number of condition classes (genres)
        hidden_dims: List of hidden layer dimensions
        latent_dim: Dimension of latent space
    """
    
    def __init__(self, input_dim, n_classes, hidden_dims=[512, 256, 128], latent_dim=32):
        super(ConditionalVAE, self).__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        
        # Encoder (input + condition)
        encoder_input_dim = input_dim + n_classes
        encoder_layers = []
        prev_dim = encoder_input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder (latent + condition)
        decoder_input_dim = latent_dim + n_classes
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = decoder_input_dim
        for h_dim in hidden_dims_rev:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        print(f"CVAE: {input_dim}D + {n_classes} classes -> {latent_dim}D")
    
    def encode(self, x, c):
        x_c = torch.cat([x, c], dim=1)
        h = self.encoder(x_c)
        return self.fc_mu(h), self.fc_logvar(h)
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, c):
        z_c = torch.cat([z, c], dim=1)
        return self.decoder(z_c)
    
    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, c)
        return recon, mu, logvar, z
    
    def get_latent(self, x, c):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x, c)
        return mu
    
    def loss_function(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss, recon_loss, kl_loss


# ============================================================
# Standard Autoencoder (Baseline)
# ============================================================

class Autoencoder(nn.Module):
    """Standard Autoencoder for baseline comparison."""
    
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], latent_dim=32):
        super(Autoencoder, self).__init__()
        
        # Encoder
        encoder_layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder
        decoder_layers = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in hidden_dims_rev:
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        print(f"Autoencoder: {input_dim}D -> {latent_dim}D")
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z
    
    def get_latent(self, x):
        self.eval()
        with torch.no_grad():
            return self.encode(x)


# ============================================================
# Training Functions
# ============================================================

def train_beta_vae(X, config, device):
    """Train Beta-VAE model."""
    print("\n" + "=" * 60)
    print("Training Beta-VAE")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    model = BetaVAE(
        input_dim=X.shape[1],
        hidden_dims=config.HIDDEN_DIMS,
        latent_dim=config.LATENT_DIM,
        beta=config.BETA
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
    
    model.train()
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar, z = model(x)
            loss, recon_l, kl_l = model.loss_function(recon, x, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon += recon_l.item()
            epoch_kl += kl_l.item()
        
        scheduler.step()
        
        history['total_loss'].append(epoch_loss / len(X))
        history['recon_loss'].append(epoch_recon / len(X))
        history['kl_loss'].append(epoch_kl / len(X))
        
        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch + 1:3d}/{config.EPOCHS} | Loss: {epoch_loss / len(X):.4f}")
    
    model.eval()
    latent = model.get_latent(X_tensor.to(device)).cpu().numpy()
    
    print(f"Latent features: {latent.shape}")
    
    return model, latent, history, scaler


def train_cvae(X, conditions, config, device):
    """Train Conditional VAE model."""
    print("\n" + "=" * 60)
    print("Training Conditional VAE")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    C_tensor = torch.FloatTensor(conditions)
    dataset = TensorDataset(X_tensor, C_tensor)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    n_classes = conditions.shape[1]
    model = ConditionalVAE(
        input_dim=X.shape[1],
        n_classes=n_classes,
        hidden_dims=config.HIDDEN_DIMS,
        latent_dim=config.LATENT_DIM
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.EPOCHS)
    
    history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
    
    model.train()
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        
        for batch in dataloader:
            x, c = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            recon, mu, logvar, z = model(x, c)
            loss, recon_l, kl_l = model.loss_function(recon, x, mu, logvar)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        scheduler.step()
        history['total_loss'].append(epoch_loss / len(X))
        
        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch + 1:3d}/{config.EPOCHS} | Loss: {epoch_loss / len(X):.4f}")
    
    model.eval()
    latent = model.get_latent(X_tensor.to(device), C_tensor.to(device)).cpu().numpy()
    
    print(f"Latent features: {latent.shape}")
    
    return model, latent, history, scaler


def train_autoencoder(X, config, device):
    """Train standard Autoencoder for baseline."""
    print("\n" + "=" * 60)
    print("Training Autoencoder (Baseline)")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    
    model = Autoencoder(
        input_dim=X.shape[1],
        hidden_dims=config.HIDDEN_DIMS,
        latent_dim=config.LATENT_DIM
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(config.EPOCHS):
        epoch_loss = 0
        
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, z = model(x)
            loss = criterion(recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        if (epoch + 1) % 40 == 0:
            print(f"Epoch {epoch + 1:3d}/{config.EPOCHS} | Loss: {epoch_loss / len(dataloader):.4f}")
    
    model.eval()
    latent = model.get_latent(X_tensor.to(device)).cpu().numpy()
    
    print(f"Latent features: {latent.shape}")
    
    return model, latent, scaler


# ============================================================
# Baseline Methods
# ============================================================

def pca_baseline(X, n_components=32):
    """PCA dimensionality reduction baseline."""
    print("\n" + "=" * 60)
    print("PCA Baseline")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    explained_var = sum(pca.explained_variance_ratio_)
    print(f"PCA: {X.shape[1]}D -> {n_components}D")
    print(f"Explained variance: {explained_var:.4f}")
    
    return X_pca, pca


def spectral_features_baseline(X, n_clusters=10):
    """Direct spectral clustering on features."""
    print("\n" + "=" * 60)
    print("Spectral Clustering Baseline")
    print("=" * 60)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    spectral = SpectralClustering(
        n_clusters=n_clusters,
        affinity='nearest_neighbors',
        n_neighbors=10,
        random_state=RANDOM_SEED
    )
    labels = spectral.fit_predict(X_scaled)
    
    print(f"Spectral clustering: {n_clusters} clusters")
    
    return labels, X_scaled


# ============================================================
# Clustering
# ============================================================

def perform_kmeans(features, n_clusters=10):
    """Perform K-Means clustering."""
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=20)
    labels = kmeans.fit_predict(features_scaled)
    
    return labels, features_scaled


# ============================================================
# Evaluation
# ============================================================

def evaluate_clustering(features, pred_labels, true_labels, method_name):
    """
    Evaluate clustering with all required metrics.
    
    Metrics:
    - Silhouette Score
    - Normalized Mutual Information (NMI)
    - Adjusted Rand Index (ARI)
    - Cluster Purity
    - Davies-Bouldin Index
    - Calinski-Harabasz Index
    """
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)
    
    # Handle noise points
    mask = pred_labels != -1
    if mask.sum() < 2 or len(np.unique(pred_labels[mask])) < 2:
        return None
    
    metrics = {'Method': method_name}
    
    try:
        metrics['Silhouette'] = silhouette_score(features[mask], pred_labels[mask])
    except:
        metrics['Silhouette'] = np.nan
    
    try:
        metrics['NMI'] = normalized_mutual_info_score(true_encoded[mask], pred_labels[mask])
    except:
        metrics['NMI'] = np.nan
    
    try:
        metrics['ARI'] = adjusted_rand_score(true_encoded[mask], pred_labels[mask])
    except:
        metrics['ARI'] = np.nan
    
    try:
        metrics['Purity'] = compute_cluster_purity(true_encoded[mask], pred_labels[mask])
    except:
        metrics['Purity'] = np.nan
    
    try:
        metrics['Davies-Bouldin'] = davies_bouldin_score(features[mask], pred_labels[mask])
    except:
        metrics['Davies-Bouldin'] = np.nan
    
    try:
        metrics['Calinski-Harabasz'] = calinski_harabasz_score(features[mask], pred_labels[mask])
    except:
        metrics['Calinski-Harabasz'] = np.nan
    
    return metrics


# ============================================================
# Visualization
# ============================================================

def plot_latent_space(features, labels, title, save_path):
    """Plot 2D latent space visualization."""
    ensure_directory(os.path.dirname(save_path))
    
    try:
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30, max_iter=1000)
    except TypeError:
        tsne = TSNE(n_components=2, random_state=RANDOM_SEED, perplexity=30)
    
    features_2d = tsne.fit_transform(features)
    
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0], features_2d[mask, 1],
            c=[colors[i]], label=label, alpha=0.7, s=30
        )
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def plot_cluster_distribution(pred_labels, true_labels, method_name, save_path):
    """Plot genre distribution across clusters."""
    ensure_directory(os.path.dirname(save_path))
    
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)
    
    mask = pred_labels != -1
    cm = confusion_matrix(true_encoded[mask], pred_labels[mask])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=[f'C{i}' for i in range(cm.shape[1])],
        yticklabels=le.classes_
    )
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Genre')
    plt.title(f'Cluster Distribution - {method_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def plot_reconstruction_examples(model, X, scaler, n_examples=5, save_path=None, device='cpu'):
    """Plot reconstruction examples from VAE."""
    ensure_directory(os.path.dirname(save_path))
    
    model.eval()
    X_scaled = scaler.transform(X[:n_examples])
    X_tensor = torch.FloatTensor(X_scaled).to(device)
    
    with torch.no_grad():
        if hasattr(model, 'loss_function'):  # VAE
            recon, _, _, _ = model(X_tensor)
        else:  # Autoencoder
            recon, _ = model(X_tensor)
    
    recon = recon.cpu().numpy()
    
    fig, axes = plt.subplots(n_examples, 2, figsize=(12, 3 * n_examples))
    
    for i in range(n_examples):
        axes[i, 0].bar(range(len(X_scaled[i])), X_scaled[i], alpha=0.7)
        axes[i, 0].set_title(f'Original Sample {i + 1}')
        axes[i, 0].set_ylabel('Value')
        
        axes[i, 1].bar(range(len(recon[i])), recon[i], alpha=0.7, color='orange')
        axes[i, 1].set_title(f'Reconstructed Sample {i + 1}')
        axes[i, 1].set_ylabel('Value')
    
    plt.suptitle('VAE Reconstruction Examples', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def plot_training_comparison(histories, save_path):
    """Plot training history comparison for all models."""
    ensure_directory(os.path.dirname(save_path))
    
    plt.figure(figsize=(10, 6))
    
    for name, history in histories.items():
        plt.plot(history['total_loss'], label=name, linewidth=2)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Comparison', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def plot_metrics_comparison(results_df, save_path):
    """Plot comprehensive metrics comparison."""
    ensure_directory(os.path.dirname(save_path))
    
    metrics = ['Silhouette', 'NMI', 'ARI', 'Purity', 'Davies-Bouldin']
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    methods = results_df['Method'].tolist()
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    
    for idx, metric in enumerate(metrics):
        if metric not in results_df.columns:
            continue
            
        values = results_df[metric].values
        bars = axes[idx].bar(methods, values, color=colors)
        
        title = metric
        if 'Davies' in metric:
            title += ' (Lower=Better)'
        else:
            title += ' (Higher=Better)'
        
        axes[idx].set_title(title, fontweight='bold')
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3, axis='y')
        
        for bar, val in zip(bars, values):
            if not np.isnan(val):
                axes[idx].text(
                    bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8
                )
    
    # Hide unused subplot
    axes[-1].set_visible(False)
    
    plt.suptitle('Clustering Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


def plot_disentanglement_analysis(latent, labels, save_path):
    """Analyze and visualize latent space disentanglement."""
    ensure_directory(os.path.dirname(save_path))
    
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    n_dims = min(8, latent.shape[1])
    
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for i in range(n_dims):
        ax = axes[i]
        for j, label in enumerate(le.classes_):
            mask = labels == label
            ax.hist(latent[mask, i], bins=20, alpha=0.5, label=label, density=True)
        
        ax.set_title(f'Latent Dim {i + 1}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
    
    axes[0].legend(fontsize=6, loc='upper right')
    
    plt.suptitle('Latent Space Disentanglement Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {save_path}")


# ============================================================
# Main Function
# ============================================================

def main():
    """Main function to run Hard Task."""
    
    print("\n" + "=" * 70)
    print("Hard Task: Beta-VAE and CVAE for Multi-Modal Music Clustering")
    print("=" * 70)
    
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create directories
    ensure_directory(config.RESULTS_DIR)
    ensure_directory(config.MODELS_DIR)
    
    # ==================== Data Loading ====================
    
    # Load audio features
    audio_features, genres, _ = load_audio_features(config.AUDIO_FEATURES_PATH)
    if audio_features is None:
        return None
    
    # Load real lyrics by genre
    genre_lyrics = load_lyrics_by_genre(
        config.LYRICS_PATH, genres,
        samples_per_genre=config.LYRICS_SAMPLE_PER_GENRE,
        max_length=config.LYRICS_MAX_LENGTH
    )
    
    # Create lyrics embeddings
    lyrics_embeddings = create_lyrics_embeddings_real(genres, genre_lyrics)
    
    # Create genre embeddings
    genre_embeddings, label_encoder = create_genre_embeddings(genres)
    
    # Create multi-modal features
    multimodal_features = create_multimodal_features(
        audio_features, lyrics_embeddings, genre_embeddings,
        config.AUDIO_WEIGHT, config.LYRICS_WEIGHT, config.GENRE_WEIGHT
    )
    
    # ==================== Train Models ====================
    
    histories = {}
    all_results = []
    
    # 1. Beta-VAE
    print("\n" + "=" * 70)
    print("1. Training Beta-VAE")
    print("=" * 70)
    
    beta_vae, beta_latent, beta_history, beta_scaler = train_beta_vae(
        multimodal_features, config, device
    )
    histories['Beta-VAE'] = beta_history
    
    torch.save(beta_vae.state_dict(), os.path.join(config.MODELS_DIR, "beta_vae.pt"))
    
    # Cluster and evaluate Beta-VAE
    beta_labels, beta_scaled = perform_kmeans(beta_latent, config.N_CLUSTERS)
    beta_metrics = evaluate_clustering(beta_scaled, beta_labels, genres, "Beta-VAE")
    if beta_metrics:
        all_results.append(beta_metrics)
        print(f"Beta-VAE - Silhouette: {beta_metrics['Silhouette']:.4f}, NMI: {beta_metrics['NMI']:.4f}")
    
    # 2. Conditional VAE
    print("\n" + "=" * 70)
    print("2. Training Conditional VAE")
    print("=" * 70)
    
    cvae, cvae_latent, cvae_history, cvae_scaler = train_cvae(
        multimodal_features, genre_embeddings, config, device
    )
    histories['CVAE'] = cvae_history
    
    torch.save(cvae.state_dict(), os.path.join(config.MODELS_DIR, "cvae.pt"))
    
    # Cluster and evaluate CVAE
    cvae_labels, cvae_scaled = perform_kmeans(cvae_latent, config.N_CLUSTERS)
    cvae_metrics = evaluate_clustering(cvae_scaled, cvae_labels, genres, "CVAE")
    if cvae_metrics:
        all_results.append(cvae_metrics)
        print(f"CVAE - Silhouette: {cvae_metrics['Silhouette']:.4f}, NMI: {cvae_metrics['NMI']:.4f}")
    
    # ==================== Baselines ====================
    
    print("\n" + "=" * 70)
    print("3. Running Baseline Methods")
    print("=" * 70)
    
    # 3a. PCA + K-Means
    print("\nPCA + K-Means...")
    pca_features, pca_model = pca_baseline(multimodal_features, config.LATENT_DIM)
    pca_labels, pca_scaled = perform_kmeans(pca_features, config.N_CLUSTERS)
    pca_metrics = evaluate_clustering(pca_scaled, pca_labels, genres, "PCA + K-Means")
    if pca_metrics:
        all_results.append(pca_metrics)
        print(f"PCA+K-Means - Silhouette: {pca_metrics['Silhouette']:.4f}, NMI: {pca_metrics['NMI']:.4f}")
    
    # 3b. Autoencoder + K-Means
    print("\nAutoencoder + K-Means...")
    ae_model, ae_latent, ae_scaler = train_autoencoder(multimodal_features, config, device)
    ae_labels, ae_scaled = perform_kmeans(ae_latent, config.N_CLUSTERS)
    ae_metrics = evaluate_clustering(ae_scaled, ae_labels, genres, "AE + K-Means")
    if ae_metrics:
        all_results.append(ae_metrics)
        print(f"AE+K-Means - Silhouette: {ae_metrics['Silhouette']:.4f}, NMI: {ae_metrics['NMI']:.4f}")
    
    # 3c. Spectral Clustering
    print("\nSpectral Clustering...")
    spectral_labels, spectral_scaled = spectral_features_baseline(multimodal_features, config.N_CLUSTERS)
    spectral_metrics = evaluate_clustering(spectral_scaled, spectral_labels, genres, "Spectral")
    if spectral_metrics:
        all_results.append(spectral_metrics)
        print(f"Spectral - Silhouette: {spectral_metrics['Silhouette']:.4f}, NMI: {spectral_metrics['NMI']:.4f}")
    
    # ==================== Results Summary ====================
    
    print("\n" + "=" * 70)
    print("Results Summary")
    print("=" * 70)
    
    results_df = pd.DataFrame(all_results)
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(os.path.join(config.RESULTS_DIR, "clustering_metrics.csv"), index=False)
    print(f"\nSaved: {config.RESULTS_DIR}/clustering_metrics.csv")
    
    # ==================== Visualizations ====================
    
    print("\n" + "=" * 70)
    print("Creating Visualizations")
    print("=" * 70)
    
    # Latent space plots
    plot_latent_space(
        beta_latent, genres, "Beta-VAE Latent Space",
        os.path.join(config.RESULTS_DIR, "beta_vae_latent.png")
    )
    
    plot_latent_space(
        cvae_latent, genres, "CVAE Latent Space",
        os.path.join(config.RESULTS_DIR, "cvae_latent.png")
    )
    
    plot_latent_space(
        pca_features, genres, "PCA Features",
        os.path.join(config.RESULTS_DIR, "pca_latent.png")
    )
    
    # Cluster distribution
    plot_cluster_distribution(
        beta_labels, genres, "Beta-VAE",
        os.path.join(config.RESULTS_DIR, "beta_vae_clusters.png")
    )
    
    plot_cluster_distribution(
        cvae_labels, genres, "CVAE",
        os.path.join(config.RESULTS_DIR, "cvae_clusters.png")
    )
    
    # Training comparison
    plot_training_comparison(
        histories,
        os.path.join(config.RESULTS_DIR, "training_comparison.png")
    )
    
    # Metrics comparison
    plot_metrics_comparison(
        results_df,
        os.path.join(config.RESULTS_DIR, "metrics_comparison.png")
    )
    
    # Reconstruction examples
    plot_reconstruction_examples(
        beta_vae, multimodal_features, beta_scaler, n_examples=5,
        save_path=os.path.join(config.RESULTS_DIR, "reconstruction_examples.png"),
        device=device
    )
    
    # Disentanglement analysis
    plot_disentanglement_analysis(
        beta_latent, genres,
        os.path.join(config.RESULTS_DIR, "disentanglement_analysis.png")
    )
    
    # ==================== Final Summary ====================
    
    print("\n" + "=" * 70)
    print("Hard Task Completed")
    print("=" * 70)
    
    # Find best method
    best_sil = results_df.loc[results_df['Silhouette'].idxmax()]
    best_nmi = results_df.loc[results_df['NMI'].idxmax()]
    
    print(f"\nBest by Silhouette: {best_sil['Method']} ({best_sil['Silhouette']:.4f})")
    print(f"Best by NMI: {best_nmi['Method']} ({best_nmi['NMI']:.4f})")
    
    print("\nGenerated Files:")
    print(f"  Models:")
    print(f"    - {config.MODELS_DIR}/beta_vae.pt")
    print(f"    - {config.MODELS_DIR}/cvae.pt")
    print(f"  Results:")
    print(f"    - {config.RESULTS_DIR}/clustering_metrics.csv")
    print(f"  Visualizations:")
    print(f"    - {config.RESULTS_DIR}/beta_vae_latent.png")
    print(f"    - {config.RESULTS_DIR}/cvae_latent.png")
    print(f"    - {config.RESULTS_DIR}/pca_latent.png")
    print(f"    - {config.RESULTS_DIR}/beta_vae_clusters.png")
    print(f"    - {config.RESULTS_DIR}/cvae_clusters.png")
    print(f"    - {config.RESULTS_DIR}/training_comparison.png")
    print(f"    - {config.RESULTS_DIR}/metrics_comparison.png")
    print(f"    - {config.RESULTS_DIR}/reconstruction_examples.png")
    print(f"    - {config.RESULTS_DIR}/disentanglement_analysis.png")
    
    return results_df


if __name__ == "__main__":
    results = main()