# =========================================================
# Live Auditory Cortex Simulation
# Author: Abolfazl Mastaalizadeh
# Version: 1.0
# 
# Purpose:
#   This project simulates the activity of neurons in the auditory cortex,
#   modeling how the brain processes sound frequencies using a Self-Organizing Map (SOM).
# 
# 
# Key Features:
#   - Real-time audio capture from microphone
#   - Feature extraction (low/mid energy, intensity)
#   - Online training of a 2D SOM to represent tonotopic organization
#   - Live visualization: waveform, spectrogram, feature timelines, SOM map, 3D neuron scatter
# 
# Technical Notes:
#   - SOM weights are saved periodically for analysis and reproducibility
#   - Plots are organized and fixed-size for consistent display
#   - Press 'q' in matplotlib windows to safely stop capture and save results
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from scipy.fft import rfft
from scipy.signal import spectrogram
from collections import deque
import threading
import time
import os
from matplotlib.animation import FuncAnimation
import pyaudio

# ---------------- Config ----------------
CHUNK = 2048
RATE = 44100
CHANNELS = 1
DISPLAY_WINDOW_SEC = 4
SOM_SIZE_X = 20
SOM_SIZE_Y = 20
FEATURE_DIM = 3
SAVE_DIR = "auditory_som_results"
AUTO_SAVE_INTERVAL = 60  # seconds
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------------- SOM init ----------------
som = MiniSom(SOM_SIZE_X, SOM_SIZE_Y, FEATURE_DIM,
              sigma=3.0, learning_rate=0.5, neighborhood_function='gaussian')
som.random_weights_init(np.random.rand(SOM_SIZE_X * SOM_SIZE_Y, FEATURE_DIM))

# ---------------- Shared structures ----------------
audio_buffer = deque(maxlen=int(RATE * DISPLAY_WINDOW_SEC / CHUNK) * CHUNK + CHUNK)
features_history = []
training_lock = threading.Lock()
stop_event = threading.Event()
trained_steps = 0

# ---------------- Feature extraction ----------------
def extract_features(chunk):
    fft = np.abs(rfft(chunk))
    freqs = np.linspace(0, RATE/2, len(fft))
    bands = [50, 200, 800, 3200, 10000]
    energy_bands = []
    for i in range(len(bands)-1):
        mask = (freqs >= bands[i]) & (freqs < bands[i+1])
        energy_bands.append(np.sum(fft[mask]) if np.any(mask) else 0.0)
    intensity = np.mean(np.abs(chunk))
    low_energy = np.log(energy_bands[0] + 1.0)
    mid_energy = np.log(energy_bands[2] + 1.0) if len(energy_bands) > 2 else 0.0
    return np.array([low_energy, mid_energy, intensity], dtype=float)

# ---------------- Audio producer ----------------
def audio_producer():
    p = pyaudio.PyAudio()
    try:
        stream = p.open(format=pyaudio.paFloat32,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)
    except Exception as e:
        print("ERROR opening microphone stream:", e)
        stop_event.set()
        return

    print("🎤 Listening to microphone — press 'q' in the visualization window to stop.")
    try:
        while not stop_event.is_set():
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                samples = np.frombuffer(data, dtype=np.float32)
                audio_buffer.extend(samples)
            except Exception as read_err:
                print("Warning: audio read error (ignored):", read_err)
                time.sleep(0.01)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

# ---------------- Trainer ----------------
def trainer():
    global trained_steps
    while not stop_event.is_set():
        if len(audio_buffer) >= CHUNK:
            chunk = np.array([audio_buffer.popleft() for _ in range(CHUNK)], dtype=np.float32)
            feat = extract_features(chunk)
            with training_lock:
                features_history.append(feat.copy())
                som.train(feat.reshape(1, -1), 1)
                trained_steps += 1
        else:
            time.sleep(0.005)

# ---------------- Auto-save SOM weights ----------------
def auto_save_weights():
    while not stop_event.is_set():
        time.sleep(AUTO_SAVE_INTERVAL)
        with training_lock:
            weights = som.get_weights()
            np.save(os.path.join(SAVE_DIR, f"autosave_som_{int(time.time())}.npy"), weights)
            print(f"Auto-saved SOM weights at step {trained_steps}")

# ---------------- Live visualization ----------------
def live_visualization():
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    ax_wave, ax_spec = axes[0]
    ax_energy, ax_map = axes[1]
    ax_3d = fig.add_subplot(3,2,5, projection='3d')  # occupy lower-left
    ax_dummy = fig.add_subplot(3,2,6)  # empty placeholder
    ax_dummy.axis('off')

    # Waveform
    wave_line, = ax_wave.plot([], [], lw=1)
    ax_wave.set_title(f"Live Waveform ({DISPLAY_WINDOW_SEC}s)")
    ax_wave.set_xlim(0, RATE * DISPLAY_WINDOW_SEC)
    ax_wave.set_ylim(-1.5, 1.5)
    ax_wave.grid(alpha=0.3)

    # Spectrogram
    ax_spec.set_title("Live Spectrogram")
    ax_spec.set_ylim(0, 8000)

    # Features timeline
    line_low, = ax_energy.plot([], [], label="Low (log)")
    line_mid, = ax_energy.plot([], [], label="Mid (log)")
    line_int, = ax_energy.plot([], [], label="Intensity")
    ax_energy.set_title("Feature Timeline")
    ax_energy.legend()
    ax_energy.grid(alpha=0.3)

    # SOM Map
    im = ax_map.imshow(np.zeros((SOM_SIZE_X, SOM_SIZE_Y)), cmap='plasma', origin='lower')
    ax_map.set_title("Tonotopic Map")
    fig.colorbar(im, ax=ax_map, label="Log(Low energy)")

    # 3D scatter labels
    ax_3d.set_xlabel("Low")
    ax_3d.set_ylabel("Mid")
    ax_3d.set_zlabel("Intensity")

    # Key press handler
    def on_key(event):
        if event.key.lower() == 'q':
            print("'q' pressed — stopping.")
            stop_event.set()
    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(frame):
        if stop_event.is_set():
            plt.close(fig)
            return

        # Waveform
        if len(audio_buffer) > 0:
            data = np.array(audio_buffer)
            n = min(len(data), RATE * DISPLAY_WINDOW_SEC)
            wave = data[-n:]
            wave_line.set_data(np.arange(n), wave)
            ax_wave.set_xlim(0, n)
            ymin, ymax = wave.min(), wave.max()
            if ymin == ymax: ymax += 0.1
            ax_wave.set_ylim(ymin*1.2, ymax*1.2)

        # Spectrogram
        if len(audio_buffer) >= CHUNK:
            seg = np.array(list(audio_buffer)[-int(RATE * DISPLAY_WINDOW_SEC):])
            try:
                f, t, Sxx = spectrogram(seg, fs=RATE, nperseg=1024, noverlap=512)
                ax_spec.clear()
                ax_spec.pcolormesh(t, f, 10*np.log10(Sxx + 1e-12), shading='gouraud', cmap='magma')
                ax_spec.set_ylim(0, 8000)
                ax_spec.set_title("Live Spectrogram")
            except: pass

        # Feature timeline
        with training_lock:
            if len(features_history) > 0:
                feats = np.array(features_history)
                frames = np.arange(len(feats))
                line_low.set_data(frames, feats[:,0])
                line_mid.set_data(frames, feats[:,1])
                line_int.set_data(frames, feats[:,2])
                ax_energy.set_xlim(0, max(10, len(frames)))
                vmin, vmax = feats.min(), feats.max()
                ax_energy.set_ylim(vmin - 0.1*abs(vmin), vmax + 0.1*abs(vmax))

        # SOM map + 3D
        weights = som.get_weights()
        im.set_data(weights[:, :, 0])
        try:
            im.set_clim(np.nanmin(weights[:,:,0]), np.nanmax(weights[:,:,0]))
        except: pass
        ax_map.set_title(f"Tonotopic Map — Trained: {trained_steps} steps")

        flat = weights.reshape(-1, FEATURE_DIM)
        ax_3d.clear()
        ax_3d.scatter(flat[:,0], flat[:,1], flat[:,2], c=flat[:,0], cmap='viridis', s=40)
        ax_3d.set_xlabel("Low")
        ax_3d.set_ylabel("Mid")
        ax_3d.set_zlabel("Intensity")

        return wave_line, line_low, line_mid, line_int, im

    ani = FuncAnimation(fig, update, interval=200, blit=False)
    plt.tight_layout()
    plt.show()

# ---------------- Run everything ----------------
if __name__ == "__main__":
    print("Starting live auditory cortex simulation.")
    threading.Thread(target=audio_producer, daemon=True).start()
    threading.Thread(target=trainer, daemon=True).start()
    threading.Thread(target=auto_save_weights, daemon=True).start()

    try:
        live_visualization()
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        time.sleep(0.3)

        # Save final weights and figures
        with training_lock:
            final_weights = som.get_weights()
            np.save(os.path.join(SAVE_DIR, "final_som_weights.npy"), final_weights)

        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,5))
        ax1.imshow(final_weights[:,:,0], cmap='plasma', origin='lower')
        ax1.set_title("Final Tonotopic Map")
        plt.colorbar(ax1.images[0], ax=ax1, label="Low energy (log)")

        with training_lock:
            if len(features_history) > 0:
                feats = np.array(features_history)
                ax2.plot(feats[:,0], label="Low (log)")
                ax2.plot(feats[:,1], label="Mid (log)")
                ax2.plot(feats[:,2], label="Intensity")
                ax2.legend()
                ax2.set_title("Feature Timeline")
            else:
                ax2.text(0.5, 0.5, "No features collected", ha='center', va='center')

        final_fig_path = os.path.join(SAVE_DIR, "final_tonotopic_map.png")
        fig.savefig(final_fig_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Done. Results saved to {SAVE_DIR}")
