import numpy as np
import matplotlib.pyplot as plt
import cv2


def print_img_info(img):
    print(img.dtype)
    print(img.shape)
    print(np.min(img), np.max(img))
    print()


def imgToUint8(img):
    if np.issubdtype(img.dtype, np.unsignedinteger):
        return img
    if np.issubdtype(img.dtype, np.floating):
        out = np.clip(np.round(img * 255.0), 0, 255).astype(np.uint8)
        return out
    if np.issubdtype(img.dtype, np.signedinteger) or np.issubdtype(img.dtype, np.integer):
        out = np.clip(img, 0, 255).astype(np.uint8)
        return out
    return img.astype(np.uint8)


def imgToFloat(img):
    if np.issubdtype(img.dtype, np.floating):
        return img
    if np.issubdtype(img.dtype, np.integer):
        maxv = np.iinfo(img.dtype).max
        return img.astype(np.float32) / float(maxv)
    return img.astype(np.float32)


def show_image_3x3(img, title=None, save_path=None, show=True):
    img_f = imgToFloat(img)
    if img_f.ndim != 3 or img_f.shape[2] < 0:
        raise ValueError("Expected a color image with 3 channels (RGB).")

    R = img_f[:,:,0]
    G = img_f[:,:,1]
    B = img_f[:,:,2]

    Y1 = 0.299 * R + 0.587 * G + 0.114 * B
    Y2 = 0.2126 * R + 0.7152 * G + 0.0722 * B

    red_only = np.zeros_like(img_f)
    red_only[:,:,0] = R
    green_only = np.zeros_like(img_f)
    green_only[:,:,1] = G
    blue_only = np.zeros_like(img_f)
    blue_only[:,:,2] = B

    fig = plt.figure(figsize=(10,10))
    if title:
        fig.suptitle(title, fontsize=14)

    # Row 1: Original, Y1, Y2
    ax = plt.subplot(3,3,1)
    ax.imshow(img_f)
    ax.set_title("O - original")
    ax.axis('off')

    ax = plt.subplot(3,3,2)
    ax.imshow(Y1, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax.set_title("Y1 (0.299,0.587,0.114)")
    ax.axis('off')

    ax = plt.subplot(3,3,3)
    ax.imshow(Y2, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax.set_title("Y2 (0.2126,0.7152,0.0722)")
    ax.axis('off')

    # Row 2: R, G, B single-layer grayscale
    ax = plt.subplot(3,3,4)
    ax.imshow(R, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax.set_title("R channel (gray)")
    ax.axis('off')

    ax = plt.subplot(3,3,5)
    ax.imshow(G, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax.set_title("G channel (gray)")
    ax.axis('off')

    ax = plt.subplot(3,3,6)
    ax.imshow(B, cmap=plt.cm.gray, vmin=0, vmax=1)
    ax.set_title("B channel (gray)")
    ax.axis('off')

    # Row 3: color images with only one channel preserved
    ax = plt.subplot(3,3,7)
    ax.imshow(red_only)
    ax.set_title("R only (color)")
    ax.axis('off')

    ax = plt.subplot(3,3,8)
    ax.imshow(green_only)
    ax.set_title("G only (color)")
    ax.axis('off')

    ax = plt.subplot(3,3,9)
    ax.imshow(blue_only)
    ax.set_title("B only (color)")
    ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved figure to: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def process_and_save_fragments(img_path, fragments, out_prefix="frag", show=False):
    img = plt.imread(img_path)

    for i, f in enumerate(fragments):
        w1, k1, w2, k2 = f
        fragment = img[w1:w2, k1:k2].copy()
        h, w = fragment.shape[:2]
        if (h, w) != (200, 200):
            print(f"Warning: fragment {i} has size {(h,w)} (expected 200x200). Proceeding anyway.")
        save_name = f"{out_prefix}_{i:02d}.png"
        show_image_3x3(fragment, title=f"{out_prefix} fragment {i}", save_path=save_name, show=show)


if __name__ == "__main__":
    img_b1 = plt.imread('./imgs/B01.png')  
    img_b2 = plt.imread('./imgs/B02.jpg')

    show_image_3x3(img_b1, title="B01 - full image (Zadanie 2)", save_path="B01_overview.png", show=True)

    # Format: [row_start, col_start, row_end, col_end]
    fragments_B01 = [
        [100, 200, 300, 400],
        [400, 50, 600, 250],
    ]

    fragments_B02 = [
        [150, 120, 350, 320],
    ]

    process_and_save_fragments('./imgs/B01.png', fragments_B01, out_prefix="B01_frag", show=False)
    process_and_save_fragments('./imgs/B02.jpg', fragments_B02, out_prefix="B02_frag", show=False)

    print("Done.")
