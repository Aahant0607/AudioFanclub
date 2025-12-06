# ==============================================================================
# FINAL AUDIO ANALYSIS SUITE: ALL FILES (ORTHOGONAL, SPATIAL, EXTRAPOLATION)
# ==============================================================================
# AIM:
# 1. Compositing audio data into Orthogonal Components (PCA).
# 2. Complete Analysis: Time Domain, Frequency Domain, & Feature Space for ALL files.
# 3. Spatial Extrapolation: Finding the distribution of voices in every file.
# ==============================================================================

import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import scipy.fftpack as fftpack
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import pandas as pd
import plotly.express as px
from google.colab import files
import warnings

# Suppress warnings for clearer output
warnings.filterwarnings('ignore')

# ==========================================
# STEP 1: UPLOAD & DATA INGESTION
# ==========================================
print("--- STEP 1: UPLOAD & PRE-PROCESSING ---")
print("Please upload ALL 7 files (Voice 1-5, Double, Triple)...")
uploaded = files.upload()

print("\n> Processing Filenames...")
voice_files = []
double_file = None
triple_file = None

# Smart detection of files based on your timestamps
for filename in uploaded.keys():
    if "10.21" in filename:
        double_file = filename
    elif "10.22" in filename:
        triple_file = filename
    else:
        voice_files.append(filename)

voice_files.sort() # Ensure logical order for voices

# Map raw filenames to expected keys
file_map = {}
if double_file: file_map['double'] = double_file
if triple_file: file_map['triple'] = triple_file
for i, f in enumerate(voice_files):
    file_map[f'voice{i+1}'] = f

# Load ALL Audio files
audio = {}
sr_target = 16000
# We define the order we want to process
all_keys = [f'voice{i+1}.wav' for i in range(len(voice_files))] + ['double.wav', 'triple.wav']

print("> Loading Signal Data...")
for key in all_keys:
    base = key.replace('.wav', '')
    if base in file_map:
        try:
            # Load file (handling .aac automatically via librosa)
            y, sr = librosa.load(file_map[base], sr=sr_target, mono=True)
            audio[key] = {'y': y, 'sr': sr}
        except Exception as e:
            print(f"Error loading {key}: {e}")
    else:
        print(f"Warning: Source for {key} not found (check upload selection).")

print(f"Successfully loaded {len(audio)} audio files.")

# ==========================================
# STEP 2: COMPLETE TIME & FREQ ANALYSIS (ALL FILES)
# ==========================================
print("\n--- STEP 2: SPECTRAL & TEMPORAL ANALYSIS (ALL FILES) ---")

# Create a grid of plots: Rows = Files, Cols = 2 (Time vs Freq)
num_files = len(audio)
fig, axes = plt.subplots(num_files, 2, figsize=(15, 3 * num_files))
fig.suptitle("Time & Frequency Domain Analysis", fontsize=16)

for idx, fname in enumerate(audio.keys()):
    y = audio[fname]['y']
    sr = audio[fname]['sr']
    
    # 2A. Time Domain (Waveform) - Left Column
    librosa.display.waveshow(y, sr=sr, ax=axes[idx, 0], alpha=0.6)
    axes[idx, 0].set_title(f'Time Domain: {fname}')
    axes[idx, 0].set_ylabel('Amplitude')
    
    # 2B. Frequency Domain (Spectrogram) - Right Column
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[idx, 1])
    axes[idx, 1].set_title(f'Freq Domain: {fname}')

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
plt.show()

# ==========================================
# STEP 3: FEATURE EXTRACTION (MFCC)
# ==========================================
print("\n--- STEP 3: EXTRACTING FEATURES (MFCC) ---")

def get_mfcc(y, sr):
    # Calculate MFCCs
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    log_S = librosa.power_to_db(S, ref=np.max)
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    return mfcc

mfccs = {}
for k in audio:
    mfccs[k] = get_mfcc(audio[k]['y'], audio[k]['sr'])

print(f"MFCCs extracted for {len(mfccs)} files.")

# ==========================================
# STEP 4: ORTHOGONAL COMPOSITION (PCA)
# ==========================================
print("\n--- STEP 4: ORTHOGONAL COMPONENT ANALYSIS (PCA 3D) ---")
# "Compositing random data in orthogonal component"

pca_data = []

# Prepare data for PCA for ALL files
for name, features in mfccs.items():
    # Transpose to (Time, Features)
    X = features.T 
    
    # We project the high-dim audio into 3 Orthogonal Components
    # Note: We fit PCA on the individual file to see its internal variance structure
    # or you can fit on a global set. Here we visualize the internal structure.
    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)
    
    # Subsample for plotting performance (every 5th frame)
    step = 5
    for i in range(0, len(X_pca), step):
        pca_data.append({
            'PC1': X_pca[i, 0],
            'PC2': X_pca[i, 1],
            'PC3': X_pca[i, 2],
            'Source': name,
            'Time': i
        })

df_pca = pd.DataFrame(pca_data)

# Interactive 3D Plot
fig = px.scatter_3d(
    df_pca, x='PC1', y='PC2', z='PC3', color='Source',
    opacity=0.5, size_max=3,
    title='Spatial Distribution in Orthogonal Domain (All Files)'
)
fig.update_layout(margin=dict(l=0, r=0, b=0, t=30))
fig.show()

# ==========================================
# STEP 5: ADVANCED SPATIAL DISTRIBUTION & EXTRAPOLATION
# ==========================================
print("\n--- STEP 5: SPATIAL DISTRIBUTION (DIARIZATION & QUANTIFICATION) ---")

from scipy import stats
import seaborn as sns

# 1. Train GMMs on Single Voices (The "Centralized Acoustic Waves")
single_voices = [k for k in mfccs.keys() if 'voice' in k]
single_voices.sort()
gmm_models = {}

print("Training centralized distribution models...")
for v in single_voices:
    X_train = mfccs[v].T
    # Increased components to 16 for better texture capture
    gmm = GaussianMixture(n_components=16, covariance_type='diag', random_state=42)
    gmm.fit(X_train)
    gmm_models[v] = gmm

# 2. Prediction & Smoothing Function
def predict_and_smooth(target_key, window_size=15):
    X_test = mfccs[target_key].T
    
    # Calculate Log-Likelihood for each voice
    scores = np.array([model.score_samples(X_test) for model in gmm_models.values()])
    
    # Raw Prediction (Frame-by-frame)
    raw_preds = np.argmax(scores, axis=0)
    
    # Smooth the result (Majority vote over a window to remove jitter)
    # This makes the "Spatial Distribution" meaningful in the Time Domain
    if len(raw_preds) > window_size:
        smoothed_preds = stats.mode(
            np.lib.stride_tricks.sliding_window_view(raw_preds, window_size), 
            axis=1
        )[0]
        # Pad the edges to match original length
        pad = (len(raw_preds) - len(smoothed_preds)) // 2
        smoothed_preds = np.pad(smoothed_preds, (pad, len(raw_preds)-len(smoothed_preds)-pad), mode='edge')
    else:
        smoothed_preds = raw_preds
        
    return smoothed_preds

# 3. Analyze ALL files
results_matrix = []
timeline_data = {}

sorted_keys = sorted(mfccs.keys())

print("Extrapolating distributions...")

for target in sorted_keys:
    # Get smoothed predictions
    preds = predict_and_smooth(target)
    timeline_data[target] = preds
    
    # Calculate Distribution % (How much of Voice X is in File Y?)
    counts = np.bincount(preds, minlength=len(single_voices))
    total = np.sum(counts)
    percentages = (counts / total) * 100
    results_matrix.append(percentages)

# ==========================================
# VISUALIZATION 1: THE TIMELINES (Who spoke when?)
# ==========================================
fig, axes = plt.subplots(len(sorted_keys), 1, figsize=(15, 1.2 * len(sorted_keys)))
if len(sorted_keys) == 1: axes = [axes]

fig.suptitle("Temporal Distribution of Voices (Smoothed)", fontsize=16)

# Define a consistent colormap
cmap = plt.cm.get_cmap('tab10', len(single_voices))

for idx, target in enumerate(sorted_keys):
    preds = timeline_data[target]
    
    # Plot as a color strip
    ax = axes[idx]
    im = ax.imshow(preds.reshape(1, -1), aspect='auto', cmap=cmap, interpolation='nearest', vmin=0, vmax=len(single_voices)-1)
    
    ax.set_yticks([])
    ax.set_ylabel(target, rotation=0, labelpad=80, fontsize=11, fontweight='bold', ha='right')
    
    # Add borders to segments
    ax.grid(False)
    
    if idx == len(sorted_keys) - 1:
        ax.set_xlabel('Time Frames (Approx 16ms per frame)')
    else:
        ax.set_xticks([])

# Legend
import matplotlib.patches as mpatches
patches = [mpatches.Patch(color=cmap(i), label=single_voices[i]) for i in range(len(single_voices))]
fig.legend(handles=patches, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=len(single_voices))
plt.subplots_adjust(top=0.85)
plt.show()

# ==========================================
# VISUALIZATION 2: THE COMPOSITION MATRIX (Quantitative)
# ==========================================
# This answers: "What is the distribution of the centralized acoustic wave?"
print("\n> Quantitative Distribution Analysis:")

df_matrix = pd.DataFrame(results_matrix, index=sorted_keys, columns=single_voices)

plt.figure(figsize=(10, 6))
sns.heatmap(df_matrix, annot=True, fmt=".1f", cmap="Blues", cbar_kws={'label': 'Percentage Composition (%)'})
plt.title("Spatial Distribution Matrix: Voice Composition per File")
plt.ylabel("Target Audio File")
plt.xlabel("Detected Voice Source")
plt.show()
