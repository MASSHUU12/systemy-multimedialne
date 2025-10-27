import os
from typing import List, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

PALETTE_8_COLOR = np.array(
    [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ],
    dtype=np.float32,
)

PALETTE_16_COLOR = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 1.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.5],
        [0.0, 1.0, 0.0],
        [0.5, 0.0, 0.0],
        [0.0, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        [0.5, 0.0, 0.5],
        [1.0, 0.0, 0.0],
        [0.75, 0.75, 0.75],
        [0.0, 0.5, 0.5],
        [1.0, 1.0, 1.0],
        [1.0, 1.0, 0.0],
    ],
    dtype=np.float32,
)


def _load_image_as_float(path: str) -> npt.NDArray[np.float32]:
    with Image.open(path) as im:
        if im.mode == "L":
            array = np.asarray(im, dtype=np.float32) / 255.0
            return array[..., None]
        im_rgb = im.convert("RGB")
        return np.asarray(im_rgb, dtype=np.float32) / 255.0


def _prepare_image_and_palette(
    image: npt.NDArray[np.float32], palette: npt.NDArray[np.float32]
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
    img_copy = image.copy()
    if img_copy.ndim == 2:
        img_copy = img_copy[..., None]

    pal_copy = palette.copy()
    if pal_copy.ndim == 1:
        pal_copy = pal_copy.reshape(-1, 1)

    if img_copy.shape[2] != pal_copy.shape[1]:
        raise ValueError(
            "Number of image and palette channels must match. "
            f"Image: {img_copy.shape[2]}, Palette: {pal_copy.shape[1]}"
        )

    return img_copy, pal_copy


def rgb_to_gray(image_rgb: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        return image_rgb
    r, g, b = image_rgb[..., 0], image_rgb[..., 1], image_rgb[..., 2]
    gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return gray[..., None]


def create_grayscale_palette(bits: int) -> npt.NDArray[np.float32]:
    num_colors = 2**bits
    return np.linspace(0, 1, num_colors, dtype=np.float32).reshape(num_colors, 1)


def find_closest_color(
    pixel_color: npt.NDArray[np.float32], palette: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    if palette.ndim == 1:
        palette = palette.reshape(-1, 1)

    distances = np.linalg.norm(palette - pixel_color, axis=1)
    closest_color_index = np.argmin(distances)
    return palette[closest_color_index]


def quantize_image(
    image: npt.NDArray[np.float32], palette: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    img, pal = _prepare_image_and_palette(image, palette)
    H, W, C = img.shape

    flat_img = img.reshape(-1, C)

    img_norm_sq = np.sum(flat_img**2, axis=1, keepdims=True)
    pal_norm_sq = np.sum(pal**2, axis=1, keepdims=True).T
    dot_product = flat_img @ pal.T

    distances_sq = img_norm_sq - 2 * dot_product + pal_norm_sq

    indices = np.argmin(distances_sq, axis=1)
    quantized_flat = pal[indices]

    return quantized_flat.reshape(H, W, C)


def random_dithering(image: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    if image.shape[2] != 1:
        gray_image = rgb_to_gray(image)
    else:
        gray_image = image

    H, W, _ = gray_image.shape
    random_matrix = np.random.rand(H, W).astype(np.float32)

    dithered_image = (gray_image[..., 0] >= random_matrix).astype(np.float32)

    return dithered_image[..., None]


def _get_bayer_matrix(n: int) -> npt.NDArray[np.float32]:
    base_matrix = np.array([[0, 2], [3, 1]], dtype=np.float32)
    m = base_matrix
    for _ in range(1, n):
        m_tl = 4 * m + 0
        m_tr = 4 * m + 2
        m_bl = 4 * m + 3
        m_br = 4 * m + 1
        m = np.block([[m_tl, m_tr], [m_bl, m_br]])
    return m


def ordered_dithering(
    image: npt.NDArray[np.float32],
    palette: npt.NDArray[np.float32],
    order_n: int = 2,
    strength: float = 1.0,
) -> npt.NDArray[np.float32]:
    img, pal = _prepare_image_and_palette(image, palette)
    H, W, C = img.shape

    bayer_matrix = _get_bayer_matrix(order_n)
    matrix_size = bayer_matrix.shape[0]

    threshold_map = (bayer_matrix + 1.0) / (matrix_size**2) - 0.5

    tiled_map = np.tile(threshold_map, (H // matrix_size + 1, W // matrix_size + 1))
    cropped_map = tiled_map[:H, :W]

    modified_image = img + strength * cropped_map[..., None]
    modified_image = np.clip(modified_image, 0.0, 1.0)

    return quantize_image(modified_image, pal)


def floyd_steinberg_dithering(
    image: npt.NDArray[np.float32], palette: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    img, pal = _prepare_image_and_palette(image, palette)
    H, W, C = img.shape

    for y in range(H):
        for x in range(W):
            old_pixel = img[y, x].copy()
            new_pixel = find_closest_color(old_pixel, pal)
            img[y, x] = new_pixel

            quant_error = old_pixel - new_pixel

            if x + 1 < W:
                img[y, x + 1] += quant_error * (7 / 16)
            if y + 1 < H:
                if x > 0:
                    img[y + 1, x - 1] += quant_error * (3 / 16)
                img[y + 1, x] += quant_error * (5 / 16)
                if x + 1 < W:
                    img[y + 1, x + 1] += quant_error * (1 / 16)

    return np.clip(img, 0.0, 1.0)


def _convert_to_pil_images(
    arrays: List[npt.NDArray[np.float32]], target_height: int = 320
) -> List[Image.Image]:
    pil_images = []
    for arr in arrays:
        if arr.ndim == 2 or arr.shape[2] == 1:
            arr = np.repeat(arr, 3, axis=2)

        img_uint8 = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
        img = Image.fromarray(img_uint8, "RGB")

        w, h = img.size
        new_width = int(w * target_height / h)
        img = img.resize((new_width, target_height), Image.Resampling.BICUBIC)
        pil_images.append(img)
    return pil_images


def create_comparison_strip(
    images: List[npt.NDArray[np.float32]],
    labels: List[str],
    output_path: str,
) -> None:
    pil_images = _convert_to_pil_images(images)

    padding = 8
    label_height = 28

    total_width = sum(im.width for im in pil_images) + padding * (len(images) - 1)
    max_height = max(im.height for im in pil_images) + label_height

    canvas = Image.new("RGB", (total_width, max_height), (20, 20, 20))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=12)
    except AttributeError:
        font = ImageFont.load_default()

    current_x = 0
    for img, label in zip(pil_images, labels):
        canvas.paste(img, (current_x, 0))

        label_bg_box = [current_x, img.height, current_x + img.width, max_height]
        draw.rectangle(label_bg_box, fill=(0, 0, 0))

        text_pos = (current_x + 6, img.height + 6)
        draw.text(text_pos, label, fill=(220, 220, 220), font=font)

        current_x += img.width + padding

    canvas.save(output_path, quality=92)
    print(f"Saved comparison strip: {output_path}")


def run_sanity_check():
    print("Running sanity check...")
    test_image = np.random.rand(32, 32, 1).astype(np.float32)
    binary_palette = create_grayscale_palette(1)

    dithered = floyd_steinberg_dithering(test_image, binary_palette)
    unique_colors = np.unique(dithered).size

    if unique_colors != 2:
        raise RuntimeError(
            f"Floyd-Steinberg test failed! Expected 2 unique colors, got {unique_colors}."
        )
    print("Sanity check completed successfully.")


def main():
    run_sanity_check()

    INPUT_GRAY_DIR = "IMG_GS"
    INPUT_COLOR_DIR = "IMG_SMALL"
    OUTPUT_DIR = "output"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    gray_images = sorted([f.path for f in os.scandir(INPUT_GRAY_DIR) if f.is_file()])
    color_images = sorted([f.path for f in os.scandir(INPUT_COLOR_DIR) if f.is_file()])[
        :4
    ]

    for path in gray_images:
        base_name = os.path.splitext(os.path.basename(path))[0]
        original_img = _load_image_as_float(path)
        gray_img = rgb_to_gray(original_img)

        for bits in [1, 2, 4]:
            palette = create_grayscale_palette(bits)

            if bits == 1:
                images_to_compare = [
                    gray_img,
                    quantize_image(gray_img, palette),
                    random_dithering(gray_img),
                    ordered_dithering(gray_img, palette, order_n=2),
                    floyd_steinberg_dithering(gray_img, palette),
                ]
                labels = [
                    "Original",
                    "Quantization",
                    "Random",
                    "Ordered",
                    "Floyd-Steinberg",
                ]
            else:
                images_to_compare = [
                    gray_img,
                    quantize_image(gray_img, palette),
                    ordered_dithering(gray_img, palette, order_n=2),
                    floyd_steinberg_dithering(gray_img, palette),
                    ordered_dithering(gray_img, palette, order_n=3),
                ]
                labels = [
                    "Original",
                    f"Quantization {bits}b",
                    f"Ordered N2 {bits}b",
                    f"Floyd-Steinberg {bits}b",
                    f"Ordered N3 {bits}b",
                ]

            output_filename = f"COMP_{base_name}_{bits}bit.jpg"
            create_comparison_strip(
                images_to_compare, labels, os.path.join(OUTPUT_DIR, output_filename)
            )

    for path in color_images:
        base_name = os.path.splitext(os.path.basename(path))[0]
        color_img = _load_image_as_float(path)

        for pal, pal_name in [(PALETTE_8_COLOR, "pal8"), (PALETTE_16_COLOR, "pal16")]:
            images_to_compare = [
                color_img,
                quantize_image(color_img, pal),
                ordered_dithering(color_img, pal, order_n=2),
                floyd_steinberg_dithering(color_img, pal),
            ]
            labels = [
                "Original",
                f"Quantization {pal_name}",
                f"Ordered {pal_name}",
                f"Floyd-Steinberg {pal_name}",
            ]
            output_filename = f"COMP_{base_name}_{pal_name}.jpg"
            create_comparison_strip(
                images_to_compare, labels, os.path.join(OUTPUT_DIR, output_filename)
            )


if __name__ == "__main__":
    main()
