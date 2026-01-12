"""
============================================================
MEDIUM TASK: Enhanced VAE with Hybrid Features
============================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
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

torch.manual_seed(42)
np.random.seed(42)


def ensure_dir(path):
    """Create directory if it doesn't exist"""
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")


# ============================================================
# LYRICS EMBEDDING
# ============================================================

def create_lyrics_embeddings(genres):
    """Create meaningful lyrics embeddings based on genre"""
    print("\n" + "="*60)
    print("CREATING LYRICS EMBEDDINGS")
    print("="*60)
    
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_dim = model.get_sentence_embedding_dimension()
        print(f"Sentence Transformer loaded (dim={embedding_dim})")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Installing sentence-transformers...")
        os.system("pip install sentence-transformers")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embedding_dim = model.get_sentence_embedding_dimension()
    
    # Genre descriptions
    genre_descriptions = {
        'blues': [
            "Feeling sad and blue, heartbreak and sorrow",
            "Woke up this morning feeling so alone",
            "Life is hard, love is gone, singing the blues"
        ],
        'classical': [
            "Elegant symphony orchestra, beautiful harmony",
            "Peaceful instrumental, refined artistic expression",
            "Orchestral masterpiece, violin and piano"
        ],
        'country': [
            "Pickup truck down dusty road, small town life",
            "Family farm, honest work, simple country living",
            "Cowboys and guitars, heartland story"
        ],
        'disco': [
            "Dance floor lights, funky beat, everybody move",
            "Disco ball spinning, groovy rhythm, party",
            "Get up and dance, feel the music"
        ],
        'hiphop': [
            "Street life hustle, beats and rhymes",
            "Urban flow, microphone skills, hip hop culture",
            "Rhythm and poetry, city streets"
        ],
        'jazz': [
            "Smooth saxophone solo, late night jazz club",
            "Improvisation magic, swing rhythm, cool vibes",
            "Piano keys dancing, bass walking"
        ],
        'metal': [
            "Heavy guitar riffs, powerful drums, intense",
            "Headbanging anthem, aggressive sound",
            "Dark powerful music, screaming vocals"
        ],
        'pop': [
            "Catchy hook, radio hit, everybody singing",
            "Love song melody, upbeat chorus",
            "Dance pop rhythm, feel good music"
        ],
        'reggae': [
            "One love, island vibes, peace and harmony",
            "Jamaican rhythm, positive message",
            "Roots and culture, rastafari spirit"
        ],
        'rock': [
            "Electric guitar solo, rock and roll spirit",
            "Rebellion anthem, stadium crowd",
            "Power chords blazing, drummer pounding"
        ]
    }
    
    # Pre-compute genre embeddings
    genre_embeddings = {}
    for genre, descriptions in genre_descriptions.items():
        genre_embs = [model.encode(desc, show_progress_bar=False) for desc in descriptions]
        genre_embeddings[genre] = genre_embs
    
    embeddings = []
    for i, genre in enumerate(tqdm(genres, desc="Creating embeddings")):
        genre_lower = genre.lower()
        
        if genre_lower in genre_embeddings:
            idx = i % len(genre_embeddings[genre_lower])
            base_embedding = genre_embeddings[genre_lower][idx]
            noise = np.random.randn(embedding_dim) * 0.05
            embedding = base_embedding + noise
        else:
            text = f"This is {genre} music"
            embedding = model.encode(text, show_progress_bar=False)
        
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    print(f"Lyrics embeddings shape: {embeddings.shape}")
    return embeddings


# ============================================================
# VAE MODEL
# ============================================================

class EnhancedVAE(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128], latent_dim=64):
        super(EnhancedVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.25)
            ])
            prev_dim = h_dim
        
        self.encoder = nn.Sequential(*layers)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        layers = []
        hidden_dims_rev = hidden_dims[::-1]
        prev_dim = latent_dim
        for h_dim in hidden_dims_rev:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, input_dim))
        
        self.decoder = nn.Sequential(*layers)
        print(f"EnhancedVAE: {input_dim}D -> {hidden_dims} -> {latent_dim}D")
    
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
        return self.decode(z), mu, logvar, z
    
    def get_latent(self, x):
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


# ============================================================
# DATA LOADING
# ============================================================

def load_audio_features(features_path="data/processed/gtzan_features.csv"):
    print("\n" + "="*60)
    print("LOADING AUDIO FEATURES")
    print("="*60)
    
    if not os.path.exists(features_path):
        print(f"Error: {features_path} not found!")
        return None, None, None
    
    df = pd.read_csv(features_path)
    
    label_cols = ['filename', 'genre', 'filepath']
    feature_cols = [col for col in df.columns if col not in label_cols]
    
    X = df[feature_cols].values
    y = df['genre'].values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"Audio features: {X.shape}")
    print(f"Genres: {np.unique(y)}")
    return X, y, feature_cols


def create_hybrid_features(audio_features, lyrics_embeddings, audio_weight=0.6):
    print("\n" + "="*60)
    print("CREATING HYBRID FEATURES")
    print("="*60)
    
    audio_scaler = StandardScaler()
    lyrics_scaler = StandardScaler()
    
    audio_norm = audio_scaler.fit_transform(audio_features) * audio_weight
    lyrics_norm = lyrics_scaler.fit_transform(lyrics_embeddings) * (1 - audio_weight)
    
    hybrid = np.concatenate([audio_norm, lyrics_norm], axis=1)
    
    print(f"Audio: {audio_features.shape[1]}D (weight={audio_weight})")
    print(f"Lyrics: {lyrics_embeddings.shape[1]}D (weight={1-audio_weight})")
    print(f"Hybrid: {hybrid.shape[1]}D")
    return hybrid


# ============================================================
# TRAINING
# ============================================================

def train_vae(X, latent_dim=64, epochs=150, batch_size=32, lr=5e-4):
    print("\n" + "="*60)
    print("TRAINING ENHANCED VAE")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.FloatTensor(X_scaled)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = EnhancedVAE(X.shape[1], latent_dim=latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    history = {'total_loss': [], 'recon_loss': [], 'kl_loss': []}
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_recon = 0
        epoch_kl = 0
        
        beta = min(1.0, epoch / 50)
        
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            recon, mu, logvar, z = model(x)
            loss, recon_l, kl_l = vae_loss(recon, x, mu, logvar, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
            epoch_recon += recon_l.item()
            epoch_kl += kl_l.item()
        
        scheduler.step()
        
        avg_loss = epoch_loss / len(X)
        history['total_loss'].append(avg_loss)
        history['recon_loss'].append(epoch_recon / len(X))
        history['kl_loss'].append(epoch_kl / len(X))
        
        if (epoch + 1) % 25 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Beta: {beta:.2f}")
    
    print("Training complete!")
    
    model.eval()
    with torch.no_grad():
        latent = model.get_latent(X_tensor.to(device)).cpu().numpy()
    
    print(f"Latent features: {latent.shape}")
    return model, latent, history, scaler


# ============================================================
# CLUSTERING
# ============================================================

def perform_clustering(features, n_clusters=10):
    print("\n" + "="*60)
    print("CLUSTERING")
    print("="*60)
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    results = {}
    
    print("1. K-Means...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    results['K-Means'] = kmeans.fit_predict(features_scaled)
    
    print("2. Agglomerative (Ward)...")
    agg = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward')
    results['Agglomerative-Ward'] = agg.fit_predict(features_scaled)
    
    print("3. Agglomerative (Complete)...")
    agg2 = AgglomerativeClustering(n_clusters=n_clusters, linkage='complete')
    results['Agglomerative-Complete'] = agg2.fit_predict(features_scaled)
    
    print("4. DBSCAN...")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=10)
    nn.fit(features_scaled)
    distances, _ = nn.kneighbors(features_scaled)
    eps = np.percentile(distances[:, -1], 75)
    dbscan = DBSCAN(eps=eps, min_samples=8)
    results['DBSCAN'] = dbscan.fit_predict(features_scaled)
    
    print("Clustering complete!")
    return results, features_scaled


# ============================================================
# EVALUATION
# ============================================================

def evaluate_clustering(features, cluster_results, true_labels):
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)
    
    all_results = []
    
    for method, pred_labels in cluster_results.items():
        mask = pred_labels != -1
        
        if mask.sum() < 2 or len(np.unique(pred_labels[mask])) < 2:
            continue
        
        try:
            sil = silhouette_score(features[mask], pred_labels[mask])
        except:
            sil = np.nan
        
        try:
            db = davies_bouldin_score(features[mask], pred_labels[mask])
        except:
            db = np.nan
        
        try:
            ch = calinski_harabasz_score(features[mask], pred_labels[mask])
        except:
            ch = np.nan
        
        try:
            ari = adjusted_rand_score(true_encoded[mask], pred_labels[mask])
        except:
            ari = np.nan
        
        try:
            nmi = normalized_mutual_info_score(true_encoded[mask], pred_labels[mask])
        except:
            nmi = np.nan
        
        metrics = {
            'Method': method,
            'Silhouette': sil,
            'Davies-Bouldin': db,
            'Calinski-Harabasz': ch,
            'ARI': ari,
            'NMI': nmi
        }
        all_results.append(metrics)
        
        print(f"\n{method}:")
        print(f"  Silhouette:      {sil:.4f}")
        print(f"  Davies-Bouldin:  {db:.4f}")
        print(f"  ARI:             {ari:.4f}")
        print(f"  NMI:             {nmi:.4f}")
    
    return pd.DataFrame(all_results)


# ============================================================
# VISUALIZATION (FIXED)
# ============================================================

def plot_training_history(history, save_path):
    ensure_dir(os.path.dirname(save_path))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['total_loss'], 'b-', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history['recon_loss'], 'g-', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(history['kl_loss'], 'r-', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('KL Divergence Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_clusters(features, cluster_results, true_labels, save_path):
    """Visualize clustering results using t-SNE"""
    ensure_dir(os.path.dirname(save_path))
    
    print("\nComputing t-SNE...")
    
    try:
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    except TypeError:
        try:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        except TypeError:
            tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    
    features_2d = tsne.fit_transform(features)
    
    n_plots = len(cluster_results) + 1
    n_cols = 3
    n_rows = (n_plots + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    # True labels
    unique_labels = np.unique(true_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = true_labels == label
        axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors[i]], label=label, alpha=0.7, s=25)
    axes[0].set_title('True Genres', fontweight='bold')
    axes[0].legend(fontsize=6)
    
    # Cluster results
    for idx, (method, labels) in enumerate(cluster_results.items(), 1):
        unique = np.unique(labels)
        for i, label in enumerate(unique):
            mask = labels == label
            color = 'gray' if label == -1 else plt.cm.tab10(i / max(len(unique), 10))
            axes[idx].scatter(features_2d[mask, 0], features_2d[mask, 1],
                            c=[color], alpha=0.7, s=25)
        axes[idx].set_title(method, fontweight='bold')
    
    for idx in range(n_plots, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Clustering Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_metrics(results_df, save_path):
    """Plot metrics comparison across clustering methods"""
    ensure_dir(os.path.dirname(save_path))
    
    metrics = ['Silhouette', 'Davies-Bouldin', 'ARI', 'NMI']
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    methods = results_df['Method'].tolist()
    colors = plt.cm.Set2(np.linspace(0, 1, len(methods)))
    
    for idx, metric in enumerate(metrics):
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
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                              f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Metrics Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def visualize_genre_dist(pred_labels, true_labels, method_name, save_path):
    """Visualize genre distribution across predicted clusters"""
    ensure_dir(os.path.dirname(save_path))
    
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)
    
    mask = pred_labels != -1
    cm = confusion_matrix(true_encoded[mask], pred_labels[mask])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=[f'C{i}' for i in range(cm.shape[1])],
                yticklabels=le.classes_)
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Genre')
    plt.title(f'Genre Distribution - {method_name}', fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    """Execute medium task pipeline"""
    print("\n" + "="*70)
    print("MEDIUM TASK: Enhanced VAE with Hybrid Features")
    print("="*70)
    
    LATENT_DIM = 64
    N_CLUSTERS = 10
    EPOCHS = 150
    BATCH_SIZE = 32
    
    ensure_dir("results/medium_task")
    ensure_dir("results/models")
    
    audio_features, genres, _ = load_audio_features()
    if audio_features is None:
        return None
    
    lyrics_embeddings = create_lyrics_embeddings(genres)
    hybrid_features = create_hybrid_features(audio_features, lyrics_embeddings)
    
    # Train VAE
    model, latent, history, scaler = train_vae(
        hybrid_features, latent_dim=LATENT_DIM, epochs=EPOCHS, batch_size=BATCH_SIZE
    )
    
    torch.save(model.state_dict(), "results/models/enhanced_vae.pt")
    print("Model saved")
    
    # Plot training
    plot_training_history(history, "results/medium_task/training_history.png")
    
    # Clustering
    cluster_results, features_scaled = perform_clustering(latent, N_CLUSTERS)
    
    # Evaluation
    results_df = evaluate_clustering(features_scaled, cluster_results, genres)
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\n" + results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv("results/medium_task/clustering_metrics.csv", index=False)
    print("\nSaved: results/medium_task/clustering_metrics.csv")
    
    # Visualizations
    print("\n" + "="*70)
    print("CREATING VISUALIZATIONS")
    print("="*70)
    
    visualize_clusters(features_scaled, cluster_results, genres, 
                      "results/medium_task/clustering_comparison.png")
    visualize_metrics(results_df, "results/medium_task/metrics_comparison.png")
    best_method = results_df.loc[results_df['Silhouette'].idxmax(), 'Method']
    visualize_genre_dist(cluster_results[best_method], genres, best_method,
                        "results/medium_task/genre_distribution.png")
    
    print("\n" + "="*70)
    print("MEDIUM TASK COMPLETED")
    print("="*70)
    print("\nGenerated Files:")
    print("  - results/models/enhanced_vae.pt")
    print("  - results/medium_task/clustering_metrics.csv")
    print("  - results/medium_task/training_history.png")
    print("  - results/medium_task/clustering_comparison.png")
    print("  - results/medium_task/metrics_comparison.png")
    print("  - results/medium_task/genre_distribution.png")
    
    return results_df


if __name__ == "__main__":
    results = main()