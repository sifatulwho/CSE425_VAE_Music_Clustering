"""
Dataset Module
Handles data loading and feature extraction for Audio (GTZAN) and Lyrics (Genius).
"""

import os
import numpy as np
import pandas as pd
import librosa
import warnings
from tqdm import tqdm

# Suppress audio processing warnings
warnings.filterwarnings('ignore')


class GTZANProcessor:
    """
    Handles feature extraction from the GTZAN audio dataset.
    Extracts statistical summaries (mean/std) of spectral and rhythmic features.
    """
    
    def __init__(self, data_path, sr=22050, duration=30):
        """
        Args:
            data_path (str): Path to the 'genres_original' folder.
            sr (int): Sample rate for audio loading.
            duration (int): Duration in seconds to load.
        """
        self.data_path = data_path
        self.sr = sr
        self.duration = duration
        self.genres = self._find_genres()
        print(f"Initialized GTZAN Processor. Found {len(self.genres)} genres.")
    
    def _find_genres(self):
        """Identify genre subdirectories."""
        genres = []
        if not os.path.exists(self.data_path):
            return []
            
        for item in os.listdir(self.data_path):
            item_path = os.path.join(self.data_path, item)
            if os.path.isdir(item_path):
                genres.append(item)
        return sorted(genres)
    
    def extract_features_from_file(self, audio_path):
        """
        Process a single audio file to extract MFCC, Chroma, and Spectral features.
        
        Args:
            audio_path (str): Path to the .wav file.
            
        Returns:
            dict: Extracted features or None if file is corrupted.
        """
        try:
            # Load audio signal
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            
            features = {}
            
            # 1. MFCC (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for i in range(13):
                features[f'mfcc_{i}_mean'] = np.mean(mfcc[i])
                features[f'mfcc_{i}_std'] = np.std(mfcc[i])
            
            # 2. Chroma Features
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for i in range(12):
                features[f'chroma_{i}_mean'] = np.mean(chroma[i])
                features[f'chroma_{i}_std'] = np.std(chroma[i])
            
            # 3. Spectral Features
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            features['spectral_centroid_mean'] = np.mean(spectral_centroid)
            features['spectral_centroid_std'] = np.std(spectral_centroid)
            
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
            features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
            
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
            features['spectral_rolloff_std'] = np.std(spectral_rolloff)
            
            spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            for i in range(7):
                features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
            
            # 4. Zero Crossing Rate
            zcr = librosa.feature.zero_crossing_rate(y)
            features['zcr_mean'] = np.mean(zcr)
            features['zcr_std'] = np.std(zcr)
            
            # 5. RMS Energy
            rms = librosa.feature.rms(y=y)
            features['rms_mean'] = np.mean(rms)
            features['rms_std'] = np.std(rms)
            
            # 6. Tempo
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features['tempo'] = float(tempo)
            
            return features
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            return None
    
    def process_dataset(self, save_path="data/processed"):
        """
        Iterate through all genres and extraction features. Saves result to CSV.
        """
        os.makedirs(save_path, exist_ok=True)
        all_data = []
        
        print(f"Starting feature extraction from: {self.data_path}")
        
        for genre in self.genres:
            genre_path = os.path.join(self.data_path, genre)
            audio_files = [f for f in os.listdir(genre_path) if f.endswith('.wav')]
            
            # Progress bar for each genre
            for audio_file in tqdm(audio_files, desc=f"Processing {genre}"):
                audio_path = os.path.join(genre_path, audio_file)
                features = self.extract_features_from_file(audio_path)
                
                if features is not None:
                    features['filename'] = audio_file
                    features['genre'] = genre
                    features['filepath'] = audio_path
                    all_data.append(features)
        
        # Save to CSV
        df = pd.DataFrame(all_data)
        output_file = os.path.join(save_path, "gtzan_features.csv")
        df.to_csv(output_file, index=False)
        
        print(f"\nProcessing Complete.")
        print(f"Total samples: {len(df)}")
        print(f"Features per sample: {len(df.columns) - 3}")
        print(f"Saved to: {output_file}")
        
        return df


class LyricsDataLoader:
    """
    Simple loader for the Lyrics CSV to ensure it exists and is readable.
    """
    @staticmethod
    def check_lyrics_file(lyrics_path):
        """Verifies the lyrics file exists."""
        if os.path.exists(lyrics_path):
            print(f"Lyrics file found at: {lyrics_path}")
            return True
        else:
            print(f"Warning: Lyrics file not found at {lyrics_path}")
            return False


def find_gtzan_path():
    """Helper to locate the GTZAN dataset in common directories."""
    possible_paths = [
        "data/audio/genres_original",
        "data/audio/Data/genres_original",
        "data/genres_original"
    ]
    
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if it looks like GTZAN (contains 'blues', 'rock', etc.)
            contents = os.listdir(path)
            if 'blues' in contents and 'rock' in contents:
                return path
    return None


def main():
    """CLI Entry point for dataset processing."""
    print("="*60)
    print("GTZAN Feature Extraction Pipeline")
    print("="*60)
    
    # 1. Locate Audio Data
    gtzan_path = find_gtzan_path()
    if gtzan_path is None:
        print("Error: GTZAN dataset not found.")
        print("Please ensure the dataset is extracted to 'data/audio/genres_original'.")
        return
    
    # 2. Check if processing is needed
    output_csv = "data/processed/gtzan_features.csv"
    if os.path.exists(output_csv):
        print(f"Processed features already exist: {output_csv}")
        choice = input("Do you want to re-process the dataset? (y/N): ").strip().lower()
        if choice != 'y':
            print("Skipping extraction.")
            return

    # 3. Run Extraction
    processor = GTZANProcessor(gtzan_path)
    processor.process_dataset()


if __name__ == "__main__":
    main()