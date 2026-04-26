import os
import pickle
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dlib
import numpy as np

from constants import EMBEDDING_THREADS

_MODEL_DIR     = Path(__file__).parent.parent / "data"
_LANDMARK_PATH = str(_MODEL_DIR / "shape_predictor_68_face_landmarks.dat")
_RECOG_PATH    = str(_MODEL_DIR / "dlib_face_recognition_resnet_model_v1.dat")
_CACHE_FILE    = _MODEL_DIR / ".face_embedding_cache.pkl"

_thread_local = threading.local()
_disk_cache   = None
_cache_lock   = threading.Lock()
_print_lock   = threading.Lock()


def _get_dlib():
    """Each thread gets its own dlib instances — dlib is not thread-safe."""
    if not hasattr(_thread_local, "detector"):
        _thread_local.detector   = dlib.get_frontal_face_detector()
        _thread_local.predictor  = dlib.shape_predictor(_LANDMARK_PATH)
        _thread_local.recognizer = dlib.face_recognition_model_v1(_RECOG_PATH)
    return _thread_local.detector, _thread_local.predictor, _thread_local.recognizer


def _get_cache():
    global _disk_cache
    if _disk_cache is None:
        with _cache_lock:
            if _disk_cache is None:
                if _CACHE_FILE.exists():
                    try:
                        with open(_CACHE_FILE, "rb") as f:
                            _disk_cache = pickle.load(f)
                    except Exception:
                        print("Cache corrupted, rebuilding...")
                        _disk_cache = {}
                else:
                    _disk_cache = {}
    return _disk_cache


def _save_cache(cache):
    with open(_CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def _compute_embedding(path_str):
    detector, predictor, recognizer = _get_dlib()
    try:
        img  = dlib.load_rgb_image(path_str)
        dets = detector(img, 1)
        if len(dets) == 0:
            return np.zeros(128, dtype=np.float32)
        det   = max(dets, key=lambda d: d.width() * d.height())
        shape = predictor(img, det)
        vec   = recognizer.compute_face_descriptor(img, shape)
        return np.array(vec, dtype=np.float32)
    except Exception:
        return np.zeros(128, dtype=np.float32)


def preprocess_file(path, target_size=None):
    path_str = str(path)
    cache = _get_cache()
    if path_str not in cache:
        cache[path_str] = _compute_embedding(path_str)
        _save_cache(cache)
    return cache[path_str]


def preprocess_all_dirs(person_dirs):
    """Compute dlib embeddings for all images across all person dirs in one batch."""
    cache = _get_cache()

    dir_to_paths = {}
    all_paths = []
    for d in person_dirs:
        d_str = str(d)
        try:
            paths = [os.path.join(d_str, f) for f in sorted(os.listdir(d_str))]
        except (FileNotFoundError, NotADirectoryError):
            paths = []
        dir_to_paths[d_str] = paths
        all_paths.extend(paths)

    to_compute = [p for p in all_paths if p not in cache]

    if to_compute:
        total = len(to_compute)
        done = [0]
        lock = threading.Lock()
        step = max(1, total // 100)

        def compute_and_track(p):
            emb = _compute_embedding(p)
            with lock:
                done[0] += 1
                if done[0] % step == 0 or done[0] == total:
                    pct = done[0] * 100 // total
                    print(f"\rprogress: {pct}%", end="", flush=True)
            return p, emb

        with ThreadPoolExecutor(max_workers=EMBEDDING_THREADS) as ex:
            for path, emb in ex.map(compute_and_track, to_compute):
                cache[path] = emb

        print()
        _save_cache(cache)
    else:
        print("progress: 100% (all cached)")

    return {
        d_str: np.array([cache[p] for p in paths], dtype=np.float32)
        for d_str, paths in dir_to_paths.items()
    }


def preprocess_dir(directory, target_size=None):
    result = preprocess_all_dirs([directory])
    return result.get(str(directory), np.zeros((0, 128), dtype=np.float32))
