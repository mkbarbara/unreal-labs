import os
import cv2
import numpy as np
from utils.logger import setup_logger
from pathlib import Path

logger = setup_logger(__name__)

# ---------------------- Tunable parameters (color-agnostic) ----------------------
SAMPLE_MAX_FRAMES = 240                         # maximum frames to sample
SAMPLE_STRIDE = 2                               # sample every N-th frame

# Gradient / persistence
GAUSS_BLUR = 1                                  # 0/1 -> small denoise before Sobel
SOBEL_KSIZE = 3
EDGE_TAU = 60                                   # threshold on Sobel magnitude (8-bit) for edge presence
PERSIST_QUANTILE = 0.65                         # how often a pixel must be an edge to be "persistent" (0..1)
SCORE_MIN = 0.15                                # threshold on combined stability score (see below)

# Static (low variance in intensity) selection
GRAY_STD_THR = 6.0                              # pixels with temporal stddev <= this are considered static

# Morphology (post-processing)
OPEN_ITERS = 1
CLOSE_ITERS = 1
DILATE_ITERS = 0
MIN_COMPONENT_AREA = 80                         # remove tiny specks

# --- New guardrails ---
EDGE_TO_FILL_MAX_DIST = 3        # max pixels to grow from edge seeds into filled regions
MAX_COMPONENT_AREA_FRAC = 0.06   # drop components larger than this fraction of the frame

# ---------------------------------------------------------------------------------

def read_sampled_frames(path, max_frames=SAMPLE_MAX_FRAMES, stride=SAMPLE_STRIDE):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")

    frames = []
    idx = 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    # pick up to max_frames, spacing by stride, but also limited by actual length
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break
        if idx % stride == 0:
            ok, f = cap.retrieve()
            if not ok:
                break
            frames.append(f)
            if len(frames) >= max_frames:
                break
        idx += 1
    cap.release()
    if len(frames) < 3:
        raise RuntimeError("Not enough frames sampled; need at least 3 frames for temporal stats.")
    return frames


def _estimate_transform(from_gray, to_gray, mode="homography"):
    # ORB features + RANSAC
    orb = cv2.ORB_create(2000)
    k1, d1 = orb.detectAndCompute(from_gray, None)
    k2, d2 = orb.detectAndCompute(to_gray, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return None

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(d1, d2, k=2)
    good = []
    for m,n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)
    if len(good) < 8:
        return None

    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    if mode == "homography":
        H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 3.0)  # map 'to' -> 'from'
        return ("homography", H) if H is not None else None
    else:
        A, mask = cv2.estimateAffinePartial2D(dst, src, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if A is None:
            return None
        return ("affine", A.astype(np.float32))


def stabilize_frames(frames):
    h, w = frames[0].shape[:2]
    ref_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    stabilized = [frames[0]]
    for i in range(1, len(frames)):
        g = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        tr = _estimate_transform(ref_gray, g, "homography")
        if tr is None:
            # fallback: no transform
            stabilized.append(frames[i])
            continue
        tmode, T = tr
        if tmode == "homography":
            warped = cv2.warpPerspective(frames[i], T, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        else:
            warped = cv2.warpAffine(frames[i], T, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        stabilized.append(warped)
    return stabilized


def sobel_mag_u8(gray):
    if GAUSS_BLUR:
        gray = cv2.GaussianBlur(gray, (3,3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=SOBEL_KSIZE)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=SOBEL_KSIZE)
    mag = cv2.magnitude(gx, gy)
    # scale to 0..255 u8
    m = np.clip((mag / (mag.max() + 1e-6)) * 255.0, 0, 255).astype(np.uint8)
    return m


def build_persistent_edges(frames):
    """
    Returns:
      - E_persist (uint8 mask 0/255): pixels that are persistent edges
      - gray_stack (T,H,W) uint8
      - mag_stack (T,H,W) uint8 Sobel magnitudes
    """
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    mags  = [sobel_mag_u8(g) for g in grays]

    mag_stack = np.stack(mags, axis=0)         # (T,H,W)
    gray_stack = np.stack(grays, axis=0)       # (T,H,W)

    # Edge presence over time
    edges_bin = (mag_stack >= EDGE_TAU).astype(np.uint8)
    Q = edges_bin.mean(axis=0)                 # frequency [0..1]

    # Stability: low variance of magnitude
    var_g = mag_stack.astype(np.float32).var(axis=0)  # variance over time
    # normalize variance to 0..1 for combination
    var_norm = (var_g - var_g.min()) / (var_g.max() - var_g.min() + 1e-6)
    stability = Q * (1.0 - var_norm)          # high when frequent AND stable

    E = (Q >= PERSIST_QUANTILE) & (stability >= SCORE_MIN)
    E = (E.astype(np.uint8) * 255)

    # Clean up thin noise
    if OPEN_ITERS > 0:
        E = cv2.morphologyEx(E, cv2.MORPH_OPEN, np.ones((3,3), np.uint8), iterations=OPEN_ITERS)
    if CLOSE_ITERS > 0:
        E = cv2.morphologyEx(E, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=CLOSE_ITERS)

    return E, gray_stack, mag_stack


def low_variance_static(gray_stack):
    """
    Returns a mask of pixels whose intensity is temporally stable (std <= GRAY_STD_THR).
    """
    std = gray_stack.astype(np.float32).std(axis=0)
    M_static = (std <= GRAY_STD_THR).astype(np.uint8) * 255
    # Smooth small gaps
    M_static = cv2.morphologyEx(M_static, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8), iterations=1)
    return M_static


def keep_components_touching_seeds(
    candidates_mask,
    seed_mask,
    max_dist=EDGE_TO_FILL_MAX_DIST,
):
    cand = (candidates_mask > 0).astype(np.uint8) * 255
    seeds = (seed_mask > 0).astype(np.uint8) * 255

    # distance-limited grow
    inv_seeds = (seeds == 0).astype(np.uint8) * 255
    dt = cv2.distanceTransform(inv_seeds, cv2.DIST_L2, 3)
    near = (dt <= float(max_dist)).astype(np.uint8) * 255

    # take only static pixels that are near persistent edges
    gate = cv2.bitwise_and(cand, near)

    # we decided to use this directly because it already keeps all letters
    return gate


def save_colored_overlay(frames, mask, out_path):
    stack = np.stack(frames, axis=0).astype(np.uint8)   # (T,H,W,3) BGR
    H, W = mask.shape
    rgba = np.zeros((H, W, 4), np.uint8)

    # where we actually have text
    text_idx = mask > 0

    # compute median ONLY over text pixels, per channel
    for c in range(3):  # B,G,R
        ch = stack[:, :, :, c]          # (T,H,W)
        vals = ch[:, text_idx]          # (T, Ntext)
        med = np.median(vals, axis=0).astype(np.uint8)   # (Ntext,)
        tmp = np.zeros((H, W), np.uint8)
        tmp[text_idx] = med
        rgba[:, :, c] = tmp

    # alpha from mask
    rgba[:, :, 3] = (mask > 0).astype(np.uint8) * 255

    ok = cv2.imwrite(out_path, rgba)
    if not ok:
        raise RuntimeError(f"Failed to save {out_path}")


def extract_text_layer(
    work_dir: Path,
    input_video_path: str,
):
    """
    Extracts the text layer from a video file and saves the processed results in a specified
    working directory.

    Args:
        work_dir: Path to the directory where the output files will be saved
        input_video_path: Optional path to input video for cache key generation
    """
    os.makedirs(work_dir, exist_ok=True)
    logger.info(f"Reading & sampling: {input_video_path}")
    frames = read_sampled_frames(input_video_path)

    logger.info("Stabilizing frames (global alignment)...")
    frames = stabilize_frames(frames)

    logger.info("Computing persistent edges...")
    E_persist, gray_stack, mag_stack = build_persistent_edges(frames)

    logger.info("Selecting static (low-variance) pixels...")
    M_static = low_variance_static(gray_stack)

    logger.info("Keeping static components that touch persistent edges...")
    M_text = keep_components_touching_seeds(M_static, E_persist)

    # also save a colored overlay with alpha
    overlay_path = os.path.join(work_dir, "text_rgba.png")
    save_colored_overlay(frames, M_text, overlay_path)
    logger.info(f"Saved colored overlay -> {overlay_path}")
