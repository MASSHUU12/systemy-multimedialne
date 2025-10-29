import argparse
import numpy as np
import scipy.fftpack
import cv2
import matplotlib.pyplot as plt
import os
import re

QY = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 36, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

QC = np.array(
    [
        [17, 18, 24, 47, 99, 99, 99, 99],
        [18, 21, 26, 66, 99, 99, 99, 99],
        [24, 26, 56, 99, 99, 99, 99, 99],
        [47, 66, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
        [99, 99, 99, 99, 99, 99, 99, 99],
    ]
)

QN = np.ones((8, 8))


ZIGZAG_TEMPLATE = np.array(
    [
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ]
)
ZIGZAG_INDICES = ZIGZAG_TEMPLATE.flatten()
INVERSE_ZIGZAG_INDICES = np.argsort(ZIGZAG_INDICES)


class JpegData:
    def __init__(self, OGShape, Ratio="4:4:4", QY=np.ones((8, 8)), QC=np.ones((8, 8))):
        self.shape = OGShape
        self.Y = np.array([])
        self.Cb = np.array([])
        self.Cr = np.array([])
        self.ChromaRatio = Ratio
        self.QY = QY
        self.QC = QC


def dct2(a):
    return scipy.fftpack.dct(
        scipy.fftpack.dct(a.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"
    )


def idct2(a):
    return scipy.fftpack.idct(
        scipy.fftpack.idct(a.astype(float), axis=0, norm="ortho"), axis=1, norm="ortho"
    )


def view_as_blocks(arr, block_shape=(8, 8)):
    h, w = arr.shape
    bh, bw = block_shape
    return arr.reshape(h // bh, bh, w // bw, bw).swapaxes(1, 2).reshape(-1, bh, bw)


def inverse_view_as_blocks(blocks, image_shape):
    h, w = image_shape
    bh, bw = blocks.shape[1], blocks.shape[2]
    return blocks.reshape(h // bh, w // bw, bh, bw).swapaxes(1, 2).reshape(h, w)


def CompressLayer(layer, Q_table):
    if layer.shape[0] % 8 != 0 or layer.shape[1] % 8 != 0:
        raise ValueError("Layer dimensions must be multiples of 8.")

    blocks = view_as_blocks(layer.astype(float))

    dct_blocks = dct2(blocks)

    quantized_blocks = np.round(dct_blocks / Q_table).astype(np.int32)

    flattened_blocks = quantized_blocks.reshape(-1, 64)
    zigzagged_data = flattened_blocks[:, ZIGZAG_INDICES]

    return zigzagged_data.flatten()


def DecompressLayer(compressed_data, Q_table, layer_shape):
    h, w = layer_shape

    vectors = compressed_data.reshape(-1, 64)
    unzigzagged_vectors = vectors[:, INVERSE_ZIGZAG_INDICES]

    quantized_blocks = unzigzagged_vectors.reshape(-1, 8, 8)
    dequantized_blocks = quantized_blocks.astype(float) * Q_table

    dct_blocks = idct2(dequantized_blocks)

    layer = inverse_view_as_blocks(dct_blocks, layer_shape)

    return layer


def CompressJPEG(RGB, Ratio="4:4:4", use_quantization_tables=True):
    original_shape = RGB.shape

    if use_quantization_tables:
        Q_Y, Q_C = QY, QC
    else:
        Q_Y, Q_C = QN, QN

    JPEG = JpegData(original_shape, Ratio, Q_Y, Q_C)

    YCrCb = cv2.cvtColor(RGB, cv2.COLOR_RGB2YCrCb).astype(int)
    Y = YCrCb[:, :, 0] - 128
    Cr = YCrCb[:, :, 1] - 128
    Cb = YCrCb[:, :, 2] - 128

    if Ratio == "4:2:2":
        Cr_subsampled = Cr[:, ::2]
        Cb_subsampled = Cb[:, ::2]
    else:
        Cr_subsampled = Cr
        Cb_subsampled = Cb

    JPEG.Y = CompressLayer(Y, JPEG.QY)
    JPEG.Cr = CompressLayer(Cr_subsampled, JPEG.QC)
    JPEG.Cb = CompressLayer(Cb_subsampled, JPEG.QC)

    return JPEG


def DecompressJPEG(JPEG):
    h, w, _ = JPEG.shape
    y_shape = (h, w)

    if JPEG.ChromaRatio == "4:2:2":
        c_shape = (h, w // 2)
    else:
        c_shape = (h, w)

    Y = DecompressLayer(JPEG.Y, JPEG.QY, y_shape)
    Cr_subsampled = DecompressLayer(JPEG.Cr, JPEG.QC, c_shape)
    Cb_subsampled = DecompressLayer(JPEG.Cb, JPEG.QC, c_shape)

    if JPEG.ChromaRatio == "4:2:2":
        Cr = np.repeat(Cr_subsampled, 2, axis=1)
        Cb = np.repeat(Cb_subsampled, 2, axis=1)
    else:
        Cr = Cr_subsampled
        Cb = Cb_subsampled

    Y += 128
    Cr += 128
    Cb += 128

    Y = Y[:h, :w]
    Cr = Cr[:h, :w]
    Cb = Cb[:h, :w]

    YCrCb = np.dstack([Y, Cr, Cb]).clip(0, 255).astype(np.uint8)
    RGB = cv2.cvtColor(YCrCb, cv2.COLOR_YCrCb2RGB)

    return RGB


def process_image(image_rgb, display=True, output_basename="full_comparison"):
    if not isinstance(image_rgb, np.ndarray):
        raise TypeError("image must be a numpy array (HxWx3 RGB).")
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image must have shape HxWx3 (RGB).")
    if image_rgb.dtype != np.uint8:
        image_rgb = image_rgb.astype(np.uint8)

    h, w, _ = image_rgb.shape
    new_h = (h // 8) * 8
    new_w = (w // 16) * 16
    image_rgb_cropped = image_rgb[:new_h, :new_w]

    print(f"Original image size: {h}x{w}")
    print(f"Cropped size for compatibility: {new_h}x{new_w}")

    jpeg_data1 = CompressJPEG(
        image_rgb_cropped, Ratio="4:4:4", use_quantization_tables=False
    )
    reconstructed1 = DecompressJPEG(jpeg_data1)

    jpeg_data2 = CompressJPEG(
        image_rgb_cropped, Ratio="4:4:4", use_quantization_tables=True
    )
    reconstructed2 = DecompressJPEG(jpeg_data2)

    jpeg_data3 = CompressJPEG(
        image_rgb_cropped, Ratio="4:2:2", use_quantization_tables=False
    )
    reconstructed3 = DecompressJPEG(jpeg_data3)

    jpeg_data4 = CompressJPEG(
        image_rgb_cropped, Ratio="4:2:2", use_quantization_tables=True
    )
    reconstructed4 = DecompressJPEG(jpeg_data4)

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))

    axs[0, 0].imshow(image_rgb_cropped)
    axs[0, 0].set_title("Original")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(reconstructed1)
    axs[0, 1].set_title("4:4:4, no quantization")
    axs[0, 1].axis("off")

    axs[0, 2].imshow(reconstructed2)
    axs[0, 2].set_title("4:4:4, with quantization")
    axs[0, 2].axis("off")

    axs[1, 0].axis("off")

    axs[1, 1].imshow(reconstructed3)
    axs[1, 1].set_title("4:2:2, no quantization")
    axs[1, 1].axis("off")

    axs[1, 2].imshow(reconstructed4)
    axs[1, 2].set_title("4:2:2, with quantization")
    axs[1, 2].axis("off")

    plt.tight_layout()

    output_filename = f"{output_basename}_full_comparison.png"
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved full image comparison to '{output_filename}'")

    if display:
        plt.show()


def analyze_and_display_fragment(
    fragment, title, display=True, output_basename="fragment"
):
    h, w, _ = fragment.shape
    new_h = (h // 8) * 8
    new_w = (w // 16) * 16
    if new_h == 0 or new_w == 0:
        print(
            f"Skipping fragment '{title}' because its dimensions are too small after cropping."
        )
        return

    fragment_cropped = fragment[:new_h, :new_w]

    jpeg_data1 = CompressJPEG(
        fragment_cropped, Ratio="4:4:4", use_quantization_tables=False
    )
    reconstructed1 = DecompressJPEG(jpeg_data1)

    jpeg_data2 = CompressJPEG(
        fragment_cropped, Ratio="4:4:4", use_quantization_tables=True
    )
    reconstructed2 = DecompressJPEG(jpeg_data2)

    jpeg_data3 = CompressJPEG(
        fragment_cropped, Ratio="4:2:2", use_quantization_tables=False
    )
    reconstructed3 = DecompressJPEG(jpeg_data3)

    jpeg_data4 = CompressJPEG(
        fragment_cropped, Ratio="4:2:2", use_quantization_tables=True
    )
    reconstructed4 = DecompressJPEG(jpeg_data4)

    fig, axs = plt.subplots(1, 5, figsize=(20, 4))
    fig.suptitle(title, fontsize=16)

    axs[0].imshow(fragment_cropped)
    axs[0].set_title("Original")
    axs[0].axis("off")

    axs[1].imshow(reconstructed1)
    axs[1].set_title("4:4:4, No Quant")
    axs[1].axis("off")

    axs[2].imshow(reconstructed2)
    axs[2].set_title("4:4:4, Quant")
    axs[2].axis("off")

    axs[3].imshow(reconstructed3)
    axs[3].set_title("4:2:2, No Quant")
    axs[3].axis("off")

    axs[4].imshow(reconstructed4)
    axs[4].set_title("4:2:2, Quant")
    axs[4].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    sanitized_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_").lower()
    output_filename = f"{output_basename}_{sanitized_title}.png"
    fig.savefig(output_filename, bbox_inches="tight")
    print(f"Saved fragment analysis to '{output_filename}'")

    if display:
        plt.show()


def _parse_args():
    parser = argparse.ArgumentParser(description="Simple JPEG-like compression demo.")
    parser.add_argument(
        "--image-path",
        "-i",
        type=str,
        default="lena.png",
        help="Path to input image. If loading fails, a placeholder image is used.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not display the comparison figures.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    try:
        image_bgr = cv2.imread(args.image_path)
        if image_bgr is None:
            raise FileNotFoundError
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        image_basename = os.path.splitext(os.path.basename(args.image_path))[0]
    except FileNotFoundError:
        print(
            f"Warning: could not load image '{args.image_path}'. Creating a placeholder test image."
        )
        image_rgb = np.zeros((256, 256, 3), dtype=np.uint8)
        image_rgb[:, :128] = [255, 0, 0]
        image_rgb[:, 128:] = [0, 0, 255]
        image_rgb[128:, :] = [0, 255, 0]
        image_basename = "placeholder_image"

    process_image(
        image_rgb, display=not args.no_display, output_basename=image_basename
    )

    print("\n--- Analyzing Image Fragments ---")

    # Cat 1
    # fragments_to_analyze = [
    #     {
    #         "title": "Fragment 1: Detail (Eye area)",
    #         "center": (900, 80),
    #     },
    #     {
    #         "title": "Fragment 2: Smooth Area (Leg)",
    #         "center": (435, 438),
    #     },
    #     {
    #         "title": "Fragment 3: High Contrast (Ear)",
    #         "center": (808, 53),
    #     },
    # ]

    # Cat 2
    # fragments_to_analyze = [
    #     {
    #         "title": "Fragment 1: Detail (Eye area)",
    #         "center": (470, 220),
    #     },
    #     {
    #         "title": "Fragment 2: Smooth Area (Fur)",
    #         "center": (600, 400),
    #     },
    #     {
    #         "title": "Fragment 3: High Contrast (Nose)",
    #         "center": (530, 255),
    #     },
    # ]

    # Cat 3
    # fragments_to_analyze = [
    #     {
    #         "title": "Fragment 1: Detail (Eye area)",
    #         "center": (550, 666),
    #     },
    #     {
    #         "title": "Fragment 2: Smooth Area (Ear)",
    #         "center": (880, 550),
    #     },
    #     {
    #         "title": "Fragment 3: High Contrast (Nose)",
    #         "center": (600, 830),
    #     },
    # ]

    # Cat 4
    fragments_to_analyze = [
        {
            "title": "Fragment 1: Detail (Eye area)",
            "center": (610, 200),
        },
        {
            "title": "Fragment 2: Smooth Area (Fur)",
            "center": (540, 400),
        },
        {
            "title": "Fragment 3: High Contrast (Nose)",
            "center": (550, 250),
        },
    ]

    img_h, img_w, _ = image_rgb.shape
    for fragment_info in fragments_to_analyze:
        center_x, center_y = fragment_info["center"]
        size_w, size_h = fragment_info.get("size", (128, 128))

        half_w = size_w // 2
        half_h = size_h // 2

        start_x = max(0, center_x - half_w)
        end_x = min(img_w, center_x + half_w)
        start_y = max(0, center_y - half_h)
        end_y = min(img_h, center_y + half_h)

        if start_x >= end_x or start_y >= end_y:
            print(
                f"Skipping fragment '{fragment_info['title']}' as it is outside image bounds."
            )
            continue

        fragment = image_rgb[start_y:end_y, start_x:end_x]
        analyze_and_display_fragment(
            fragment,
            fragment_info["title"],
            display=not args.no_display,
            output_basename=image_basename,
        )
