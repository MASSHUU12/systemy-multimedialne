import glob
import os
import re
from datetime import datetime
from typing import List, Tuple, Literal

import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import rfft, rfftfreq

plt.ioff()

EPS = np.finfo(np.float32).eps

InterpolationMethod = Literal["linear", "cubic", "nearest"]
SignalData = np.ndarray


def _sanitize_filename(s: str, max_len: int = 120) -> str:
    s_clean = re.sub(r"[^\w\-_\. ]", "_", s).strip()
    if len(s_clean) > max_len:
        s_clean = s_clean[:max_len]
    return s_clean


def quantize(data: SignalData, bit_depth: int) -> SignalData:
    bit_depth = int(np.clip(bit_depth, 2, 32))
    x = data.astype(np.float64)
    xmin, xmax = x.min(), x.max()

    if xmax == xmin:
        return data.copy().astype(np.float32)

    levels = 2**bit_depth
    x_normalized = (x - xmin) / (xmax - xmin)
    x_quantized_normalized = np.round(x_normalized * (levels - 1)) / (levels - 1)
    x_quantized = x_quantized_normalized * (xmax - xmin) + xmin

    return x_quantized.astype(np.float32)


def decimate(data: SignalData, fs: float, factor: int) -> Tuple[SignalData, float]:
    factor = int(max(1, factor))
    decimated_data = data[::factor].astype(np.float32)
    new_fs = fs / factor
    return decimated_data, new_fs


def interpolate(
    data: SignalData, fs: float, new_fs: float, method: InterpolationMethod = "linear"
) -> Tuple[SignalData, float]:
    if len(data) < 2 or fs <= 0 or new_fs <= 0:
        return data.astype(np.float32), fs

    original_duration = (len(data) - 1) / fs
    t_original = np.linspace(0.0, original_duration, len(data))

    new_num_samples = max(2, int(round(len(data) * (new_fs / fs))))
    t_new = np.linspace(0.0, original_duration, new_num_samples)

    interp_func = interp1d(
        t_original,
        data.astype(np.float64),
        kind=method,
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_data = interp_func(t_new).astype(np.float32)
    return interpolated_data, new_fs


def select_fragment(data: SignalData, fs: float, periods: int = 4) -> SignalData:
    n_samples = len(data)
    if n_samples < 16 or fs <= 0:
        return data

    yf = rfft(data * np.hanning(n_samples))
    xf = rfftfreq(n_samples, 1 / fs)

    if len(xf) < 2:
        return data

    idx = np.argmax(np.abs(yf[1:])) + 1
    f0 = max(xf[idx], 1.0)

    period_duration = 1.0 / f0
    fragment_duration = periods * period_duration
    fragment_length = int(max(16, min(n_samples, round(fragment_duration * fs))))

    return data[:fragment_length]


def plot_time_and_spectrum(
    data: SignalData, fs: float, title: str = "", output_dir: str = "./plots"
) -> str:
    data_float = np.asarray(data).astype(np.float32)
    if len(data_float) < 2 or fs <= 0:
        print(
            f"WARNING: Cannot generate plot for '{title}' (insufficient data or invalid Fs)."
        )
        return ""

    fragment = select_fragment(data_float, fs, periods=4)
    t = np.linspace(0.0, (len(fragment) - 1) / fs, len(fragment))

    n_full = len(data_float)
    spec = np.abs(rfft(data_float))
    db = 20.0 * np.log10(spec + EPS)
    freqs = rfftfreq(n_full, 1 / fs)

    if np.any(np.isinf(db)) or np.any(np.isnan(db)):
        print(f"WARNING: Issue with spectrum for '{title}' (check Nyquist / Fs)!")

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(t, fragment)
    plt.title(f"{title} — time-domain signal (a few periods)")
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(freqs, db)
    plt.title("Single-sided spectrum (dB scale)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude [dB]")
    plt.grid(True)

    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    safe_title = _sanitize_filename(title or "plot")
    filename = f"{safe_title}_Fs{int(round(fs))}_{timestamp}.png"
    out_path = os.path.join(output_dir, filename)

    try:
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot: {out_path}")
    except Exception as e:
        print(f"ERROR: Failed to save plot '{title}' to '{out_path}': {e}")
        out_path = ""
    finally:
        plt.close()

    return out_path


def run_sine_wave_experiments():
    sin_files = sorted(glob.glob("./SOUND_SIN/SIN/sin_*.wav"))
    if not sin_files:
        print("No sin_*.wav files found in ./SOUND_SIN/SIN/.")
        return

    BIT_DEPTHS = [4, 8, 16, 24]
    DECIMATION_FACTORS = [2, 4, 6, 10, 24]
    INTERPOLATION_FREQS = [2000, 4000, 8000, 11999, 16000, 16953, 24000, 41000]

    plots_dir = "./plots/sine_experiments"
    os.makedirs(plots_dir, exist_ok=True)

    for filepath in sin_files:
        data, fs = sf.read(filepath, always_2d=False, dtype="float32")
        filename = os.path.basename(filepath)
        print(f"\n=== Processing: {filename} | Fs={fs} Hz ===")

        for bit in BIT_DEPTHS:
            quantized_data = quantize(data, bit)
            plot_time_and_spectrum(
                quantized_data,
                fs,
                f"{filename}: quantization {bit}-bit",
                output_dir=plots_dir,
            )

        for factor in DECIMATION_FACTORS:
            decimated_data, new_fs = decimate(data, fs, factor)
            plot_time_and_spectrum(
                decimated_data,
                new_fs,
                f"{filename}: decimation n={factor}",
                output_dir=plots_dir,
            )

        for f_new in INTERPOLATION_FREQS:
            for method in ("linear", "cubic"):
                interpolated_data, new_fs = interpolate(data, fs, f_new, method=method)
                plot_time_and_spectrum(
                    interpolated_data,
                    new_fs,
                    f"{filename}: interpolation {method}, {int(round(f_new))} Hz",
                    output_dir=plots_dir,
                )


def run_listening_tests():
    base_path = "./SOUND_SING/SING/"
    candidates = (
        sorted(glob.glob(os.path.join(base_path, "sing_low*.wav")))[:1]
        + sorted(glob.glob(os.path.join(base_path, "sing_medium*.wav")))[:1]
        + sorted(glob.glob(os.path.join(base_path, "sing_high*.wav")))[:1]
    )

    if not candidates:
        print("No sing_*.wav files for listening found in ./SOUND_SING/SING/.")
        return

    BIT_DEPTHS = [4, 8]
    DECIMATION_FACTORS = [4, 6, 10, 24]
    INTERPOLATION_FREQS = [4000, 8000, 11999, 16000, 16953]

    output_dir = "./output_listening_tests"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output files will be saved to: {output_dir}")

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    for filepath in candidates:
        data, fs = sf.read(filepath, always_2d=False, dtype="float32")
        filename_base = os.path.splitext(os.path.basename(filepath))[0]
        print(f"\n[LISTENING] Processing: {filename_base}.wav | Fs={fs} Hz")

        for bit in BIT_DEPTHS:
            out_path = os.path.join(output_dir, f"{filename_base}_Q{bit}.wav")
            quantized_data = quantize(data, bit)
            sf.write(out_path, quantized_data, int(fs))
            print(f"Saved: {out_path}")

            plot_time_and_spectrum(
                quantized_data,
                fs,
                f"{filename_base}: quantization {bit}-bit",
                output_dir=plots_dir,
            )

        for factor in DECIMATION_FACTORS:
            out_path = os.path.join(output_dir, f"{filename_base}_DEC{factor}.wav")
            decimated_data, new_fs = decimate(data, fs, factor)
            sf.write(out_path, decimated_data, int(round(new_fs)))
            print(f"Saved: {out_path}")

            plot_time_and_spectrum(
                decimated_data,
                new_fs,
                f"{filename_base}: decimation n={factor}",
                output_dir=plots_dir,
            )

        for f_new in INTERPOLATION_FREQS:
            for method in ("linear", "cubic"):
                out_path = os.path.join(
                    output_dir, f"{filename_base}_INT_{method}_{int(round(f_new))}.wav"
                )
                interpolated_data, new_fs = interpolate(data, fs, f_new, method=method)
                sf.write(out_path, interpolated_data, int(round(new_fs)))
                print(f"Saved: {out_path}")

                plot_time_and_spectrum(
                    interpolated_data,
                    new_fs,
                    f"{filename_base}: interpolation {method}, {int(round(f_new))} Hz",
                    output_dir=plots_dir,
                )


def main():
    print("Sine wave experiments (plots saved to ./plots)")
    run_sine_wave_experiments()

    print(
        "\nPreparing material for listening tests (audio -> ./output_listening_tests, plots -> ./output_listening_tests/plots)"
    )
    run_listening_tests()


if __name__ == "__main__":
    main()
