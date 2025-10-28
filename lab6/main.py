import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import os


def kwant(x, bit):
    x = np.rint(x).astype(np.float64)
    lim = 2 ** (bit - 1)
    return np.clip(x, -lim, lim - 1)


def A_law_compress(x, A=87.6):
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    ax = np.abs(x)
    y = np.zeros_like(x)
    idx = ax < (1.0 / A)
    y[idx] = (A * ax[idx]) / (1.0 + np.log(A))
    y[~idx] = (1.0 + np.log(A * ax[~idx])) / (1.0 + np.log(A))
    return s * y


def A_law_decompress(y, A=87.6):
    y = np.asarray(y, dtype=np.float64)
    s = np.sign(y)
    ay = np.abs(y)
    x = np.zeros_like(y)
    th = 1.0 / (1.0 + np.log(A))
    idx = ay < th
    x[idx] = ay[idx] * (1.0 + np.log(A)) / A
    x[~idx] = (1.0 / A) * np.exp(ay[~idx] * (1.0 + np.log(A)) - 1.0)
    return s * x


def mu_law_compress(x, mu=255.0):
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    ax = np.abs(x)
    y = np.log1p(mu * ax) / np.log1p(mu)
    return s * y


def mu_law_decompress(y, mu=255.0):
    y = np.asarray(y, dtype=np.float64)
    s = np.sign(y)
    ay = np.abs(y)
    x = np.expm1(ay * np.log1p(mu)) / mu
    return s * x


def no_pred(X):
    return X[-1]


def mean_pred(X):
    return np.mean(X)


def median_pred(X):
    return np.median(X)


def DPCM_compress(x, bit, predictor=None, n=0):
    x = np.asarray(x, dtype=np.float64)
    y = np.zeros(x.shape, dtype=np.float64)

    if predictor is None or n <= 0:
        e = 0.0
        for i in range(x.shape[0]):
            y[i] = kwant(x[i] - e, bit)
            e += y[i]
        return y

    xp = np.zeros(x.shape, dtype=np.float64)
    e = 0.0
    for i in range(x.shape[0]):
        y[i] = kwant(x[i] - e, bit)
        xp[i] = y[i] + e
        idx = np.arange(max(0, i - n + 1), i + 1)
        e = predictor(xp[idx]) if idx.size > 0 else 0.0
    return y


def DPCM_decompress(y, predictor=None, n=0):
    y = np.asarray(y, dtype=np.float64)

    if predictor is None or n <= 0:
        return np.cumsum(y)

    xp = np.zeros(y.shape, dtype=np.float64)
    e = 0.0
    for i in range(y.shape[0]):
        xp[i] = y[i] + e
        idx = np.arange(max(0, i - n + 1), i + 1)
        e = predictor(xp[idx]) if idx.size > 0 else 0.0
    return xp


def process_audio_files(files, output_dir="results"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for fname in files:
        if not os.path.exists(fname):
            print(f"Warning: File {fname} was not found. Skipping.")
            continue

        print(f"\n--- Processing file: {fname} ---")
        x, fs = sf.read(fname, dtype="float32")

        base_name = os.path.splitext(os.path.basename(fname))[0]
        file_output_dir = os.path.join(output_dir, base_name)
        if not os.path.exists(file_output_dir):
            os.makedirs(file_output_dir)

        yA = A_law_compress(x)
        xA = A_law_decompress(yA)
        sf.write(os.path.join(file_output_dir, f"{base_name}_A-law.wav"), xA, fs)

        yU = mu_law_compress(x)
        xU = mu_law_decompress(yU)
        sf.write(os.path.join(file_output_dir, f"{base_name}_mu-law.wav"), xU, fs)

        print("Generated files for A-law and μ-law.")

        for bits in range(8, 1, -1):
            print(f"Generating DPCM files for {bits} bits...")

            yD = DPCM_compress(x, bits)
            xD = DPCM_decompress(yD)
            sf.write(
                os.path.join(file_output_dir, f"{base_name}_DPCM_{bits}bit.wav"), xD, fs
            )

            yDP = DPCM_compress(x, bits, predictor=mean_pred, n=3)
            xDP = DPCM_decompress(yDP, predictor=mean_pred, n=3)
            sf.write(
                os.path.join(file_output_dir, f"{base_name}_DPCM_pred_{bits}bit.wav"),
                xDP,
                fs,
            )
    print("\nFinished processing all files.")


def generate_plots(output_dir="results"):
    print("\n--- Generating comparison plots ---")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    x = np.linspace(-1, 1, 1000)

    plt.figure(figsize=(7, 5))
    plt.plot(x, A_law_compress(x), label="A-law")
    plt.plot(x, mu_law_compress(x), label="μ-law")
    plt.title("A-law and μ-law Compression Characteristics")
    plt.xlabel("Input sample value")
    plt.ylabel("Compressed sample value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_law_overview.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(x, A_law_compress(x), label="A-law")
    plt.plot(x, mu_law_compress(x), label="μ-law")
    plt.xlim(-0.1, 0.1)
    plt.ylim(-0.5, 0.5)
    plt.title("A-law and μ-law Compression Characteristics (near zero)")
    plt.xlabel("Input sample value")
    plt.ylabel("Compressed sample value")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_law_zoom.png"), dpi=200)
    plt.close()

    x_sin = np.linspace(-1, 1, 1000)
    y_sin = 0.9 * np.sin(np.pi * x_sin * 4)

    yA_rec = A_law_decompress(A_law_compress(y_sin))
    yU_rec = mu_law_decompress(mu_law_compress(y_sin))
    yD_rec = DPCM_decompress(DPCM_compress(y_sin, 8))

    plt.figure(figsize=(8, 5))
    plt.plot(x_sin, y_sin, label="Original", linewidth=2, linestyle="--")
    plt.plot(x_sin, yA_rec, label="A-law (reconstruction)")
    plt.plot(x_sin, yU_rec, label="μ-law (reconstruction)")
    plt.plot(x_sin, yD_rec, label="DPCM 8-bit (reconstruction)")
    plt.title("Comparison of signal reconstruction quality")
    plt.xlabel("Time (normalized)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "plot_reconstruction_compare.png"), dpi=200)
    plt.close()

    print("Finished generating plots.")


if __name__ == "__main__":
    files_to_process = [
        "SING/sing_low1.wav",
        "SING/sing_medium1.wav",
        "SING/sing_high1.wav",
        "SING/sing_low2.wav",
        "SING/sing_medium2.wav",
        "SING/sing_high2.wav",
    ]

    process_audio_files(files_to_process, output_dir="results")
    generate_plots(output_dir="results")
