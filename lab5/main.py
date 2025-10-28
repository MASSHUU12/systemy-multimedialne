from typing import Any
import numpy as np
import os
import argparse
import time
import struct
from PIL import Image


def _pack_shape_info(arr: np.ndarray) -> np.ndarray:
    header_parts = [struct.pack("!B", arr.ndim)]
    for dim in arr.shape:
        header_parts.append(struct.pack("!I", dim))

    return np.frombuffer(b"".join(header_parts), dtype=np.uint8)


def _unpack_shape_info(encoded_data: np.ndarray) -> tuple[tuple[int, ...], int]:
    enc_bytes = encoded_data.tobytes()

    if len(enc_bytes) < 1:
        raise ValueError("Encoded data is too short to contain shape information.")
    ndim = struct.unpack("!B", enc_bytes[0:1])[0]

    header_size = 1 + ndim * 4
    if len(enc_bytes) < header_size:
        raise ValueError("Encoded data is too short for the specified dimensions.")

    shape = struct.unpack(f'!{"I"*ndim}', enc_bytes[1:header_size])

    return shape, header_size


def _repeat_length(arr: np.ndarray, start: int) -> int:
    n = arr.size
    val = arr[start]
    cnt = 1
    i = start + 1
    while i < n:
        if arr[i] == val:
            cnt += 1
            i += 1
        else:
            break
    return cnt


def _nonrepeat_length(arr: np.ndarray, start: int) -> int:
    n = arr.size
    i = start
    while i < n - 1 and arr[i] != arr[i + 1]:
        i += 1
    return i - start + 1


def byterun_encode(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=np.uint8)
    shape_header = _pack_shape_info(arr)

    flat = arr.flatten()
    n = flat.size
    if n == 0:
        return shape_header

    buf = []
    i = 0
    while i < n:
        if i + 1 < n and flat[i] == flat[i + 1]:
            cnt = _repeat_length(flat, i)
            val = int(flat[i])
            while cnt > 128:
                chunk = 128
                control = 257 - chunk
                buf.append(control)
                buf.append(val)
                cnt -= chunk
            if cnt > 1:
                control = 257 - cnt
                buf.append(control)
                buf.append(val)
            elif cnt == 1:
                buf.append(0)
                buf.append(val)
            i += _repeat_length(flat, i)
        else:
            length = _nonrepeat_length(flat, i)
            start = i
            while length > 0:
                chunk = min(length, 128)
                buf.append(chunk - 1)
                buf.extend(flat[start : start + chunk].tolist())
                start += chunk
                length -= chunk
            i = start

    compressed_data = np.array(buf, dtype=np.uint8)
    return np.concatenate((shape_header, compressed_data))


def byterun_decode(encoded: Any) -> np.ndarray:
    enc_arr = np.asarray(encoded, dtype=np.uint8)

    original_shape, header_size = _unpack_shape_info(enc_arr)
    enc = enc_arr[header_size:]

    m = enc.size
    if m == 0:
        return np.array([], dtype=np.uint8).reshape(original_shape)

    out = []
    i = 0
    while i < m:
        ctrl = int(enc[i])
        i += 1

        if ctrl <= 127:
            cnt = ctrl + 1
            if i + cnt > m:
                raise ValueError("Encoded literal length exceeds available payload")
            out.extend(enc[i : i + cnt].tolist())
            i += cnt
        elif ctrl == 128:
            continue
        else:
            cnt = 257 - ctrl
            if i >= m:
                raise ValueError("Encoded run missing its value")
            val = int(enc[i])
            i += 1
            out.extend([val] * cnt)

    decoded_flat = np.array(out, dtype=np.uint8)
    return decoded_flat.reshape(original_shape)


def rle_encode(data: Any) -> np.ndarray:
    arr = np.asarray(data, dtype=np.uint8)
    shape_header = _pack_shape_info(arr)

    flat = arr.flatten()
    n = flat.size
    if n == 0:
        return shape_header

    buf = []
    i = 0
    while i < n:
        val = int(flat[i])
        cnt = 1
        j = i + 1
        while j < n:
            if flat[j] == val:
                cnt += 1
                j += 1
            else:
                break

        while cnt > 255:
            buf.append(255)
            buf.append(val)
            cnt -= 255

        if cnt > 0:
            buf.append(cnt)
            buf.append(val)
        i = j

    compressed_data = np.array(buf, dtype=np.uint8)
    return np.concatenate((shape_header, compressed_data))


def rle_decode(encoded: Any) -> np.ndarray:
    enc_arr = np.asarray(encoded, dtype=np.uint8)

    original_shape, header_size = _unpack_shape_info(enc_arr)
    enc = enc_arr[header_size:]

    if enc.size == 0:
        return np.array([], dtype=np.uint8).reshape(original_shape)
    if enc.size % 2 != 0:
        raise ValueError("Encoded data length must be even (pairs of count and value).")

    out = []
    for i in range(0, enc.size, 2):
        cnt = int(enc[i])
        val = int(enc[i + 1])
        out.extend([val] * cnt)

    decoded_flat = np.array(out, dtype=np.uint8)
    return decoded_flat.reshape(original_shape)


def _round_trip(encode_fn, decode_fn, arr):
    original_arr = np.asarray(arr, dtype=np.uint8)
    enc = encode_fn(original_arr)
    dec = decode_fn(enc)

    if original_arr.shape != dec.shape:
        raise AssertionError(
            f"Round-trip shape mismatch\n"
            f"Original shape: {original_arr.shape}\n"
            f"Decoded shape:  {dec.shape}"
        )

    if not np.array_equal(original_arr, dec):
        mismatch_idx = np.where(original_arr.flatten() != dec.flatten())[0][0]
        raise AssertionError(
            f"Round-trip content mismatch at flat index {mismatch_idx}."
        )


def _test_on_array(
    label: str, arr: np.ndarray, show_sizes: bool = True, timed: bool = True
):
    arr = np.asarray(arr, dtype=np.uint8)
    n = int(arr.size)
    print(f"\nTesting {label}: elements={n}, shape={arr.shape}, dtype={arr.dtype}")

    try:
        t0 = time.perf_counter()
        enc_b = byterun_encode(arr)
        t1 = time.perf_counter()
        dec_b = byterun_decode(enc_b)
        t2 = time.perf_counter()
        _round_trip(byterun_encode, byterun_decode, arr)
        if show_sizes:
            print(f"  byterun encoded bytes: {enc_b.size} (original: {n})")
        if n > 0:
            ratio = enc_b.size / n
            reduction_pct = (1.0 - ratio) * 100.0
            sign = "reduction" if reduction_pct >= 0 else "expansion"
            print(
                f"  byterun compression: {enc_b.size}/{n} bytes = {ratio:.4f} ({abs(reduction_pct):.1f}% {sign})"
            )
        if timed:
            print(f"  byterun encode time: {(t1-t0):.6f}s, decode time: {(t2-t1):.6f}s")
    except Exception as e:
        print(f"  byterun test FAILED: {e}")
        raise

    try:
        t0 = time.perf_counter()
        enc_r = rle_encode(arr)
        t1 = time.perf_counter()
        dec_r = rle_decode(enc_r)
        t2 = time.perf_counter()
        _round_trip(rle_encode, rle_decode, arr)
        if show_sizes:
            print(f"  rle encoded bytes: {enc_r.size} (original: {n})")
        if n > 0:
            ratio = enc_r.size / n
            reduction_pct = (1.0 - ratio) * 100.0
            sign = "reduction" if reduction_pct >= 0 else "expansion"
            print(
                f"  rle compression: {enc_r.size}/{n} bytes = {ratio:.4f} ({abs(reduction_pct):.1f}% {sign})"
            )
        if timed:
            print(f"  rle encode time: {(t1-t0):.6f}s, decode time: {(t2-t1):.6f}s")
    except Exception as e:
        print(f"  rle test FAILED: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Test byterun and rle implementations with shape preservation."
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Image files to test the algorithms on. If none provided, run internal base tests.",
    )
    parser.add_argument(
        "--no-time", action="store_true", help="Disable timing measurements in output."
    )
    parser.add_argument(
        "--no-sizes", action="store_true", help="Do not print encoded sizes."
    )
    args = parser.parse_args()

    if not args.files:
        print("--- Running internal base tests ---")
        arrays = [
            np.array([1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1], dtype=np.uint8),
            np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=np.uint8),
            np.zeros((10, 52), dtype=np.uint8),
            np.arange(0, 256, 1, dtype=np.uint8).reshape(16, 16),
            np.ones((2, 5), dtype=np.uint8),
        ]
        for idx, arr in enumerate(arrays, start=1):
            _test_on_array(
                f"Base test {idx}",
                arr,
                show_sizes=(not args.no_sizes),
                timed=(not args.no_time),
            )
        print("\n--- All base tests passed successfully! ---")
    else:
        print("--- Testing on provided image files ---")
        for path in args.files:
            if not os.path.isfile(path):
                print(f"\nSkipping '{path}': not a file.")
                continue
            try:
                with Image.open(path) as img:
                    arr = np.array(img.convert("RGB"))

                label = f"file:{os.path.basename(path)}"
                _test_on_array(
                    label, arr, show_sizes=(not args.no_sizes), timed=(not args.no_time)
                )
            except Exception as e:
                print(f"\nError processing file '{path}': {e}")


if __name__ == "__main__":
    main()
