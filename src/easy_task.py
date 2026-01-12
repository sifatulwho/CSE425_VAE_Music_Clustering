"""
============================================================
EASY TASK: Basic VAE for Music Clustering
============================================================
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# ============================================================
# STEP 1: BASIC VAE MODEL
# ============================================================

class BasicVAE(nn.Module):
    """
    Basic Variational Autoencoder for music feature extraction
    
    Architecture:
        Encoder: Input -> 256 -> 128 -> (mu, logvar)
        Decoder: Latent -> 128 -> 256 -> Output
    """
    
    def __init__(self, input_dim, hidden_dim1=256, hidden_dim2=128, latent_dim=32):
        super(BasicVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dim2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim2, latent_dim)
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim2),
            nn.BatchNorm1d(hidden_dim2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim2, hidden_dim1),
            nn.BatchNorm1d(hidden_dim1),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim1, input_dim)
        )
        
        print(f"BasicVAE initialized:")
        print(f"  Input dimension: {input_dim}")
        print(f"  Latent dimension: {latent_dim}")
        print(f"  Architecture: {input_dim} -> 256 -> 128 -> {latent_dim} -> 128 -> 256 -> {input_dim}")
    
    def encode(self, x):
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation through sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar, z
    
    def get_latent(self, x):
        """Get latent representation (mu) for clustering"""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


def vae_loss_function(recon_x, x, mu, logvar):
    """
    VAE Loss = Reconstruction Loss + KL Divergence
    
    Reconstruction Loss: MSE between input and reconstruction
    KL Divergence: Regularizes latent space to be close to N(0,1)
    """
    # Reconstruction loss
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + kl_loss, recon_loss, kl_loss


# ============================================================
# STEP 2: DATA LOADING
# ============================================================

def load_gtzan_features(features_path="data/processed/gtzan_features.csv"):
    """
    Load GTZAN features from CSV
    
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Genre labels
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("LOADING GTZAN DATASET")
    print("="*60)
    
    if not os.path.exists(features_path):
        print(f"Error: Features file not found at {features_path}")
        print("Please run 'python src/dataset.py' first to extract features.")
        return None, None, None
    
    # Load CSV
    df = pd.read_csv(features_path)
    print(f"Loaded {len(df)} samples")
    
    # Separate features and labels
    label_cols = ['filename', 'genre', 'filepath']
    feature_cols = [col for col in df.columns if col not in label_cols]
    
    X = df[feature_cols].values
    y = df['genre'].values
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Number of features: {len(feature_cols)}")
    print(f"Genres: {np.unique(y)}")
    print(f"Samples per genre: {len(y) // len(np.unique(y))}")
    
    return X, y, feature_cols


# ============================================================
# STEP 3: VAE TRAINING
# ============================================================

def train_vae(X, latent_dim=32, epochs=100, batch_size=32, learning_rate=1e-3):
    """
    Train the Basic VAE
    
    Args:
        X: Feature matrix
        latent_dim: Dimension of latent space
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
    
    Returns:
        model: Trained VAE model
        latent_features: Latent representations
        history: Training history
    """
    print("\n" + "="*60)
    print("TRAINING BASIC VAE")
    print("="*60)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert to PyTorch tensors
    X_tensor = torch.FloatTensor(X_scaled)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    input_dim = X.shape[1]
    model = BasicVAE(input_dim=input_dim, latent_dim=latent_dim)
    
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': []
    }
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    print("-" * 40)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        total_recon = 0
        total_kl = 0
        
        for batch in dataloader:
            x = batch[0].to(device)
            
            # Forward pass
            recon, mu, logvar, z = model(x)
            
            # Compute loss
            loss, recon_loss, kl_loss = vae_loss_function(recon, x, mu, logvar)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_kl += kl_loss.item()
        
        # Average losses
        avg_loss = total_loss / len(X)
        avg_recon = total_recon / len(X)
        avg_kl = total_kl / len(X)
        
        history['total_loss'].append(avg_loss)
        history['recon_loss'].append(avg_recon)
        history['kl_loss'].append(avg_kl)
        
        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | Loss: {avg_loss:.4f} | Recon: {avg_recon:.4f} | KL: {avg_kl:.4f}")
    
    print("-" * 40)
    print("Training complete!")
    
    # Extract latent features
    model.eval()
    X_tensor = X_tensor.to(device)
    latent_features = model.get_latent(X_tensor).cpu().numpy()
    
    print(f"Latent features shape: {latent_features.shape}")
    
    return model, latent_features, history, scaler


# ============================================================
# STEP 4: CLUSTERING
# ============================================================

def perform_clustering(features, n_clusters=10):
    """
    Perform K-Means clustering on features
    
    Args:
        features: Feature matrix
        n_clusters: Number of clusters
    
    Returns:
        labels: Cluster labels
        model: Fitted K-Means model
    """
    print(f"\nPerforming K-Means clustering with k={n_clusters}...")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    print(f"Clustering complete!")
    print(f"Cluster distribution: {np.bincount(labels)}")
    
    return labels, kmeans


def pca_baseline(X, n_components=32, n_clusters=10):
    """
    Baseline: PCA + K-Means
    
    Args:
        X: Original feature matrix
        n_components: Number of PCA components
        n_clusters: Number of clusters
    
    Returns:
        labels: Cluster labels
        pca_features: PCA-reduced features
    """
    print("\n" + "-"*40)
    print("BASELINE: PCA + K-Means")
    print("-"*40)
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    n_components = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=n_components)
    pca_features = pca.fit_transform(X_scaled)
    
    print(f"PCA: {X.shape[1]} -> {n_components} dimensions")
    print(f"Explained variance ratio: {sum(pca.explained_variance_ratio_):.4f}")
    
    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pca_features)
    
    print(f"K-Means clustering complete!")
    
    return labels, pca_features


# ============================================================
# STEP 5: EVALUATION
# ============================================================

def evaluate_clustering(features, pred_labels, true_labels, method_name=""):
    """
    Evaluate clustering using required metrics
    
    Args:
        features: Feature matrix used for clustering
        pred_labels: Predicted cluster labels
        true_labels: True genre labels
        method_name: Name of the method
    
    Returns:
        Dictionary of metrics
    """
    # Encode true labels to integers
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(true_labels)
    
    # Calculate metrics
    silhouette = silhouette_score(features, pred_labels)
    calinski = calinski_harabasz_score(features, pred_labels)
    
    # Additional metrics for comparison with ground truth
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    ari = adjusted_rand_score(true_labels_encoded, pred_labels)
    nmi = normalized_mutual_info_score(true_labels_encoded, pred_labels)
    
    results = {
        'method': method_name,
        'silhouette_score': silhouette,
        'calinski_harabasz_index': calinski,
        'adjusted_rand_index': ari,
        'normalized_mutual_info': nmi
    }
    
    return results


def print_evaluation(results):
    """Print evaluation results in a formatted way"""
    print("\n" + "="*50)
    print(f"EVALUATION: {results['method']}")
    print("="*50)
    print(f"  Silhouette Score:         {results['silhouette_score']:.4f}  (Higher is better, range: [-1, 1])")
    print(f"  Calinski-Harabasz Index:  {results['calinski_harabasz_index']:.4f}  (Higher is better)")
    print(f"  Adjusted Rand Index:      {results['adjusted_rand_index']:.4f}  (Higher is better, range: [-1, 1])")
    print(f"  Normalized Mutual Info:   {results['normalized_mutual_info']:.4f}  (Higher is better, range: [0, 1])")
    print("="*50)


# ============================================================
# STEP 6: VISUALIZATION
# ============================================================

def visualize_latent_space(features, labels, title, save_path, method='tsne'):
    """
    Visualize latent space using t-SNE or PCA
    
    Args:
        features: Feature matrix
        labels: Labels for coloring
        title: Plot title
        save_path: Path to save figure
        method: 'tsne' or 'pca'
    """
    print(f"\nCreating {method.upper()} visualization...")
    
    # Reduce to 2D
    if method == 'tsne':
        perplexity = min(30, len(features) - 1)
        reducer = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    features_2d = reducer.fit_transform(features)
    
    # Create plot
    plt.figure(figsize=(12, 10))
    
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = labels == label
        plt.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=label,
            alpha=0.7,
            s=60,
            edgecolors='white',
            linewidth=0.5
        )
    
    plt.xlabel(f'{method.upper()} Dimension 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Dimension 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title='Labels')
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def visualize_training_history(history, save_path):
    """Plot training history"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Total loss
    axes[0].plot(history['total_loss'], color='blue', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Total Loss')
    axes[0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[1].plot(history['recon_loss'], color='green', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Reconstruction Loss')
    axes[1].grid(True, alpha=0.3)
    
    # KL loss
    axes[2].plot(history['kl_loss'], color='red', linewidth=2)
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Loss')
    axes[2].set_title('KL Divergence Loss')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def visualize_comparison(vae_results, pca_results, save_path):
    """Compare VAE vs PCA+K-Means results"""
    
    metrics = ['silhouette_score', 'calinski_harabasz_index', 'adjusted_rand_index', 'normalized_mutual_info']
    metric_names = ['Silhouette\nScore', 'Calinski-Harabasz\nIndex', 'Adjusted Rand\nIndex', 'Normalized\nMutual Info']
    
    vae_values = [vae_results[m] for m in metrics]
    pca_values = [pca_results[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, vae_values, width, label='VAE + K-Means', color='steelblue')
    bars2 = ax.bar(x + width/2, pca_values, width, label='PCA + K-Means', color='coral')
    
    ax.set_ylabel('Score')
    ax.set_title('Comparison: VAE vs PCA Baseline', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def visualize_cluster_distribution(pred_labels, true_labels, save_path):
    """Visualize how genres are distributed across clusters"""
    
    # Create confusion matrix
    from sklearn.metrics import confusion_matrix
    
    le = LabelEncoder()
    true_encoded = le.fit_transform(true_labels)
    
    cm = confusion_matrix(true_encoded, pred_labels)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[f'Cluster {i}' for i in range(cm.shape[1])],
                yticklabels=le.classes_)
    plt.xlabel('Predicted Cluster')
    plt.ylabel('True Genre')
    plt.title('Genre Distribution Across Clusters', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


# ============================================================
# STEP 7: MAIN FUNCTION
# ============================================================

def main():
    """
    Main function to run the Easy Task
    """
    print("\n" + "="*70)
    print("   EASY TASK: Basic VAE for Music Clustering")
    print("   Course: Neural Networks")
    print("="*70)
    
    # Configuration
    LATENT_DIM = 32
    N_CLUSTERS = 10  # GTZAN has 10 genres
    EPOCHS = 100
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-3
    
    # ==================== STEP 1: Load Data ====================
    X, y, feature_names = load_gtzan_features()
    
    if X is None:
        return
    
    # ==================== STEP 2: Train VAE ====================
    model, latent_features, history, scaler = train_vae(
        X, 
        latent_dim=LATENT_DIM, 
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE
    )
    
    # Save model
    os.makedirs("results/models", exist_ok=True)
    torch.save(model.state_dict(), "results/models/basic_vae.pt")
    print("Model saved to: results/models/basic_vae.pt")
    
    # ==================== STEP 3: Clustering ====================
    print("\n" + "="*60)
    print("CLUSTERING")
    print("="*60)
    
    # VAE + K-Means
    print("\n--- VAE + K-Means ---")
    vae_labels, _ = perform_clustering(latent_features, n_clusters=N_CLUSTERS)
    
    # Baseline: PCA + K-Means
    pca_labels, pca_features = pca_baseline(X, n_components=LATENT_DIM, n_clusters=N_CLUSTERS)
    
    # ==================== STEP 4: Evaluation ====================
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)
    
    # Evaluate VAE + K-Means
    vae_results = evaluate_clustering(latent_features, vae_labels, y, "VAE + K-Means")
    print_evaluation(vae_results)
    
    # Evaluate PCA + K-Means (baseline)
    pca_results = evaluate_clustering(pca_features, pca_labels, y, "PCA + K-Means (Baseline)")
    print_evaluation(pca_results)
    
    # ==================== STEP 5: Comparison ====================
    print("\n" + "="*60)
    print("COMPARISON: VAE vs Baseline")
    print("="*60)
    
    comparison_df = pd.DataFrame([vae_results, pca_results])
    print("\n" + comparison_df.to_string(index=False))
    
    # Determine winner for each metric
    print("\nSummary:")
    metrics_to_compare = ['silhouette_score', 'calinski_harabasz_index', 'adjusted_rand_index', 'normalized_mutual_info']
    
    for metric in metrics_to_compare:
        vae_val = vae_results[metric]
        pca_val = pca_results[metric]
        winner = "VAE" if vae_val > pca_val else "PCA"
        diff = abs(vae_val - pca_val)
        print(f"  {metric}: {winner} wins (difference: {diff:.4f})")
    
    # Save results to CSV
    os.makedirs("results", exist_ok=True)
    comparison_df.to_csv("results/clustering_metrics.csv", index=False)
    print("\nMetrics saved to: results/clustering_metrics.csv")
    
    # ==================== STEP 6: Visualization ====================
    print("\n" + "="*60)
    print("VISUALIZATION")
    print("="*60)
    
    # Create visualization directory
    os.makedirs("results/easy_task", exist_ok=True)
    
    # 1. Training history
    visualize_training_history(
        history, 
        "results/easy_task/training_history.png"
    )
    
    # 2. Latent space with true genres (t-SNE)
    visualize_latent_space(
        latent_features, y,
        "VAE Latent Space (Colored by True Genre)",
        "results/easy_task/latent_space_true_genres_tsne.png",
        method='tsne'
    )
    
    # 3. Latent space with predicted clusters (t-SNE)
    cluster_labels = [f"Cluster {i}" for i in vae_labels]
    visualize_latent_space(
        latent_features, cluster_labels,
        "VAE Latent Space (Colored by Predicted Cluster)",
        "results/easy_task/latent_space_clusters_tsne.png",
        method='tsne'
    )
    
    # 4. PCA baseline visualization
    visualize_latent_space(
        pca_features, y,
        "PCA Features (Colored by True Genre)",
        "results/easy_task/pca_space_true_genres.png",
        method='pca'
    )
    
    # 5. Comparison bar chart
    visualize_comparison(
        vae_results, pca_results,
        "results/easy_task/vae_vs_pca_comparison.png"
    )
    
    # 6. Cluster distribution heatmap
    visualize_cluster_distribution(
        vae_labels, y,
        "results/easy_task/cluster_distribution.png"
    )
    

    print("\n" + "="*70)
    print("EASY TASK COMPLETED")
    print("="*70)
    print("\nGenerated Files:")
    print("  Models:")
    print("    - results/models/basic_vae.pt")
    print("  Metrics:")
    print("    - results/clustering_metrics.csv")
    print("  Visualizations:")
    print("    - results/easy_task/training_history.png")
    print("    - results/easy_task/latent_space_true_genres_tsne.png")
    print("    - results/easy_task/latent_space_clusters_tsne.png")
    print("    - results/easy_task/pca_space_true_genres.png")
    print("    - results/easy_task/vae_vs_pca_comparison.png")
    print("    - results/easy_task/cluster_distribution.png")
    print("\n" + "="*70)
    
    return {
        'model': model,
        'latent_features': latent_features,
        'vae_labels': vae_labels,
        'pca_labels': pca_labels,
        'vae_results': vae_results,
        'pca_results': pca_results,
        'history': history
    }


if __name__ == "__main__":
    results = main()