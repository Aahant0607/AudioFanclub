# AudioFanclub

AudioFanclub is a research-grade voice analysis system designed to transform raw audio files into clear, meaningful insights. Unlike systems focused on isolated DSP tricks, AudioFanclub features an end-to-end pipeline for automatic audio loading, cleaning, feature extraction, complexity reduction, speaker modeling, and rich visualization — enabling high-precision, forensic-grade audio processing for research, security, analytics, and academic projects.

---

## Key Features

- **Automatic Audio Loading & Preprocessing:** Reads a variety of audio formats and applies robust cleaning/preparation to prepare input for analysis.
- **MFCC Extraction:** Derives Mel-Frequency Cepstral Coefficients (MFCCs) to capture the unique texture of each voice — a fundamental feature for speaker identification and clustering.
- **Dimensionality Reduction (PCA):** MFCCs are projected via Principal Component Analysis (PCA), yielding clean 3D clusters that visually separate speakers.
- **Speaker Modeling (GMM):** Gaussian Mixture Models (GMMs) learn statistical voice patterns for each speaker, supporting accurate frame-by-frame identification, even in noisy or overlapped recordings.
- **Speaker Diarization & Visualization:** Generates timelines, diarization strips, and percentage speaker breakdowns. For multi-speaker audio, produces visualizations and heatmaps for insightful representation of conversation flow and participation.

### Pipeline Overview

1. **Loading & Cleaning:**  
   `audio = load_and_clean(file_path)`  
   Reads the input file and applies noise reduction or filtering as needed.

2. **Feature Extraction:**  
   `mfccs = extract_mfcc(audio)`  
   Calculates MFCCs for short frames throughout the audio, capturing key vocal characteristics.

3. **Complexity Reduction:**  
   `mfcc_pca = run_pca(mfccs, n_components=3)`  
   Reduces feature dimensionality to 3; enables visual clusters in subsequent plots.

4. **Speaker Modeling:**  
   `gmm = GaussianMixture(n_components=num_speakers)`  
   Fits a GMM model to the reduced features, learning the statistical distribution of voice patterns.

5. **Speaker Identification:**  
   `labels = gmm.predict(mfcc_pca)`  
   Assigns each frame to a speaker cluster; enables fine-grained diarization.

6. **Visualization:**  
   - **Timeline:** Plots speaker activity over time.
   - **Diarization Strip:** Color-coded frames for easy inspection.
   - **Percentage Breakdown:** Calculates and visualizes how much each speaker contributed.

---

## Example Usage

```python
# Load and clean raw audio file
audio = load_and_clean('example.wav')

# Extract MFCC features
mfccs = extract_mfcc(audio)

# Reduce to 3D using PCA for clustering
mfcc_pca = run_pca(mfccs, n_components=3)

# Fit GMM and identify speakers
gmm = GaussianMixture(n_components=2)
labels = gmm.fit_predict(mfcc_pca)

# Generate diarization strip
plot_diarization(labels)

# Visualize percentage breakdown
plot_speaker_contributions(labels)
```

---

## Code Explanation

- **`load_and_clean(file_path)`**  
  Loads audio from disk and applies necessary preprocessing such as normalization and noise filtering for robust feature extraction.

- **`extract_mfcc(audio)`**  
  Splits audio into frames and calculates MFCCs, capturing timbral qualities specific to individual speakers.

- **`run_pca(mfccs, n_components=3)`**  
  Projects MFCCs down to a 3D space using PCA for clear visualization and clustering.

- **`GaussianMixture(n_components)`**  
  Implements a probabilistic model to learn and represent speaker voice distributions. Handles overlapping speech and background noise.

- **`gmm.predict(mfcc_pca)`**  
  Assigns each feature vector to a speaker model, supporting robust speaker diarization.

- **Visualization Functions**  
  Helper functions generate:
  - **Timelines:** When each speaker is active.
  - **Diarization strips:** Easy inspection of dominant voices over time.
  - **Heatmaps and breakdowns:** Visual summary of who spoke and for how long.

---

## Applications

- **Research/Academia:** High-resolution speaker studies, sociolinguistics, group dynamics.
- **Security/Forensics:** Forensic audio, surveillance, speaker attribution.
- **Media Analytics:** Panel discussions, podcast analysis, meeting minutes.
- **Health/Diagnostics:** Therapy session tracking, vocal biometrics.

---

## Future Directions

- **HMM-based Diarization:** Integration of Hidden Markov Models for sequential modeling.
- **Spectral Clustering:** Improved multi-speaker separation via spectral techniques.
- **End-to-End Neural Models:** Deep learning-based voice diarization for even higher accuracy.

---

## Requirements

- Python (>=3.8)
- `librosa`, `scikit-learn`, `numpy`, `matplotlib` *(see requirements.txt for full list)*

---

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/Aahant0607/AudioFanclub.git
   cd AudioFanclub
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run a sample analysis:
   ```bash
   python main.py --input your_audio_file.wav
   ```

---

## License

See `LICENSE` for details.

## Acknowledgments

Inspired by academic work in audio DSP, machine learning, and speaker diarization.

---

## Contributors

- Aahant Kumar
- Aryan Malhotra
- Aasif Mhd
- Anand Singh
- Arkajyoti  Deb
- Atul Yadav

---

**For questions, feature requests, or contributions:**  
Open an issue or submit a PR on [GitHub](https://github.com/Aahant0607/AudioFanclub).
