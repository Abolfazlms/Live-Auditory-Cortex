# =========================================================
#  Real-Time Auditory Cortex Simulation using Self-Organizing Maps
#  Live microphone input → Online SOM training → Evolving Tonotopic Organization
#
#  Author        : Abolfazl Mastaalizadeh
#  Version       : 1.2.0
#  Date          : November 2025
#  License       : MIT
#  Project       : https://github.com/yourusername/auditory-som-live
#
#  Features:
#   • Pure live microphone input (no pre-recorded files)
#   • Real-time auditory feature extraction (log-energy bands + intensity)
#   • Incremental online training of a 20×20 SOM
#   • Fully interactive visualization with multiple live plots
#   • Graceful shutdown via 'q' key on matplotlib window
#   • Automatic saving of final weights and high-quality summary figure
#
#  Scientific Inspiration:
#   • Tonotopic and periodotopic maps in mammalian primary auditory cortex (A1)
#   • Self-organization principles in sensory cortices (Kohonen, 1982; Miikkulainen et al.)
#   • Unsupervised feature learning in biological neural systems
#
#  Dependencies:
#   pip install -r requirements.txt
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

# ---------------- Audio producer (microphone only) ----------------
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
                # transient error reading audio — print and continue
                print("Warning: audio read error (ignored):", read_err)
                time.sleep(0.01)
                continue
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except Exception:
            pass
        p.terminate()

# ---------------- Trainer ----------------
def trainer():
    global trained_steps
    while not stop_event.is_set():
        if len(audio_buffer) >= CHUNK:
            # pop CHUNK samples
            chunk = np.array([audio_buffer.popleft() for _ in range(CHUNK)], dtype=np.float32)
            feat = extract_features(chunk)
            with training_lock:
                features_history.append(feat.copy())
                # one-step training
                som.train(feat.reshape(1, -1), 1)
                trained_steps += 1
        else:
            time.sleep(0.005)

# ---------------- Live visualization with key handler ----------------
def live_visualization():
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1,1,1.2])

    ax_wave = fig.add_subplot(gs[0, :2])
    ax_spec = fig.add_subplot(gs[0, 2])
    ax_energy = fig.add_subplot(gs[1, :2])
    ax_map = fig.add_subplot(gs[1, 2])
    ax_3d = fig.add_subplot(gs[2, :], projection='3d')

    wave_line, = ax_wave.plot([], [], lw=1)
    ax_wave.set_title(f"Live Waveform (last {DISPLAY_WINDOW_SEC}s)")
    ax_wave.set_xlim(0, RATE * DISPLAY_WINDOW_SEC)
    ax_wave.set_ylim(-1.5, 1.5)
    ax_wave.grid(alpha=0.3)

    ax_spec.set_title("Live Spectrogram")
    ax_spec.set_ylim(0, 8000)

    line_low, = ax_energy.plot([], [], label="Low (log)")
    line_mid, = ax_energy.plot([], [], label="Mid (log)")
    line_int, = ax_energy.plot([], [], label="Intensity")
    ax_energy.set_title("Extracted Features Over Time")
    ax_energy.legend()
    ax_energy.grid(alpha=0.3)

    im = ax_map.imshow(np.zeros((SOM_SIZE_X, SOM_SIZE_Y)), cmap='plasma', origin='lower')
    ax_map.set_title("Live Tonotopic Map (Low-frequency axis)")
    fig.colorbar(im, ax=ax_map, label="Log(Low energy)")

    ax_3d.set_xlabel("Low")
    ax_3d.set_ylabel("Mid")
    ax_3d.set_zlabel("Intensity")

    # Key press handler — when 'q' pressed in the figure window, stop_event.set()
    def on_key(event):
        if event.key == 'q' or event.key == 'Q':
            print("'q' pressed in figure — stopping capture and entering analysis.")
            stop_event.set()

    fig.canvas.mpl_connect('key_press_event', on_key)

    def update(frame):
        if stop_event.is_set():
            # close window gracefully
            plt.close(fig)
            return

        # Waveform
        if len(audio_buffer) > 0:
            data = np.array(audio_buffer)
            n = min(len(data), RATE * DISPLAY_WINDOW_SEC)
            wave = data[-n:]
            wave_line.set_data(np.arange(n), wave)
            ax_wave.set_xlim(0, n)
            # avoid zero-range
            ymin, ymax = wave.min(), wave.max()
            if ymin == ymax:
                ax_wave.set_ylim(ymin - 0.1, ymax + 0.1)
            else:
                ax_wave.set_ylim(ymin * 1.2, ymax * 1.2)

        # Spectrogram
        if len(audio_buffer) >= CHUNK:
            seg = np.array(list(audio_buffer)[-int(RATE * DISPLAY_WINDOW_SEC):])
            try:
                f, t, Sxx = spectrogram(seg, fs=RATE, nperseg=1024, noverlap=512)
                ax_spec.clear()
                ax_spec.pcolormesh(t, f, 10*np.log10(Sxx + 1e-12), shading='gouraud', cmap='magma')
                ax_spec.set_ylim(0, 8000)
                ax_spec.set_title("Live Spectrogram")
                ax_spec.set_xlabel("Time (s)")
                ax_spec.set_ylabel("Frequency (Hz)")
            except Exception as e:
                # in case spectrogram fails for small segment lengths
                pass

        # Feature timeline
        with training_lock:
            if len(features_history) > 0:
                feats = np.array(features_history)
                if feats.ndim == 2 and feats.shape[0] > 0:
                    frames = np.arange(len(feats))
                    line_low.set_data(frames, feats[:,0])
                    line_mid.set_data(frames, feats[:,1])
                    line_int.set_data(frames, feats[:,2])
                    ax_energy.set_xlim(0, max(10, len(frames)))
                    vmin = feats.min()
                    vmax = feats.max()
                    if vmin == vmax:
                        ax_energy.set_ylim(vmin - 0.1, vmax + 0.1)
                    else:
                        ax_energy.set_ylim(vmin - 0.1*abs(vmin+1e-9), vmax + 0.1*abs(vmax+1e-9))

        # SOM map + 3D
        weights = som.get_weights()
        im.set_data(weights[:, :, 0])
        # protect against NaN
        try:
            im.set_clim(np.nanmin(weights[:, :, 0]), np.nanmax(weights[:, :, 0]))
        except ValueError:
            pass
        ax_map.set_title(f"Tonotopic Map — Trained: {trained_steps} steps")

        flat = weights.reshape(-1, FEATURE_DIM)
        ax_3d.clear()
        ax_3d.scatter(flat[:,0], flat[:,1], flat[:,2], c=flat[:,0], cmap='viridis', s=40)
        ax_3d.set_xlabel("Low (log)")
        ax_3d.set_ylabel("Mid (log)")
        ax_3d.set_zlabel("Intensity")

        return wave_line, line_low, line_mid, line_int, im

    ani = FuncAnimation(fig, update, interval=200, blit=False)
    plt.tight_layout()
    plt.show()

# ---------------- Run everything ----------------
if __name__ == "__main__":
    print("Starting live auditory cortex simulation (microphone).")
    prod_thread = threading.Thread(target=audio_producer, daemon=True)
    train_thread = threading.Thread(target=trainer, daemon=True)
    prod_thread.start()
    train_thread.start()

    try:
        live_visualization()
    except KeyboardInterrupt:
        print("\nInterrupted by user (KeyboardInterrupt).")
        stop_event.set()
    finally:
        # ensure threads stop
        stop_event.set()
        time.sleep(0.3)

        # Save final weights and summary safely (only if any features collected)
        final_weights = som.get_weights()
        np.save(os.path.join(SAVE_DIR, "final_som_weights.npy"), final_weights)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        ax1.imshow(final_weights[:, :, 0], cmap='plasma', origin='lower')
        ax1.set_title("Final Tonotopic Map (Low-frequency axis)")
        ax1.set_xlabel("Neuron X")
        ax1.set_ylabel("Neuron Y")
        plt.colorbar(ax1.images[0], ax=ax1, label="Log(Low energy)")

        with training_lock:
            if len(features_history) > 0:
                feats = np.array(features_history)
                if feats.ndim == 2 and feats.shape[0] > 0:
                    ax2.plot(feats[:, 0], label="Low energy (log)")
                    ax2.plot(feats[:, 1], label="Mid energy (log)")
                    ax2.plot(feats[:, 2], label="Intensity")
                    ax2.legend()
                    ax2.set_title("Extracted Features Over Time")
            else:
                ax2.text(0.5, 0.5, "No features collected", ha='center', va='center')

        final_fig_path = os.path.join(SAVE_DIR, "final_tonotopic_map.png")
        fig.savefig(final_fig_path, dpi=300, bbox_inches='tight')
        plt.show()

        print(f"Done. Results saved to {SAVE_DIR}")
