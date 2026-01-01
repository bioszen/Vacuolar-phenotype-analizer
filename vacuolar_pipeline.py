# -*- coding: utf-8 -*-
"""
vacuolar_pipeline.py

DYFNet-style pipeline adapted for yeast, confocal FM4-64:
1) EXTRACT cell ROIs (crops) from:
   - a folder with .ome.tif/.tif, or
   - a .lif file
2) TRAIN a cell classifier (A/B/C or fused/partial, etc.) using crops + per-cell labels.
3) APPLY the classifier to new images and AGGREGATE per-image counts and proportions.

Important:
- To replicate the paper's approach (cell-level classification then counting),
  you NEED per-cell labels to train the classifier.
- This script automatically generates crops and a CSV template for labeling
  (e.g., in Excel). Then you train and apply.

Recommended install (in your venv):
  pip install tensorflow tifffile pandas numpy scikit-image
  pip install readlif pillow   # only if you use --lif_file

Quick usage:
  # 1) Extract ROIs from a LIF
  python vacuolar_pipeline.py extract --lif_file "D:\data\file.lif" --out_dir "D:\data\crops_v1" --crop_size 96 --max_rois 50000

  # 2) Manual labeling: open crops_v1\labels_template.csv and fill the "label" column
  #    with A/B/C (or 0/1/2) for each crop.

  # 3) Train
  python vacuolar_pipeline.py train --crops_dir "D:\data\crops_v1\crops" --labels_csv "D:\data\crops_v1\labels_filled.csv" --out_model "D:\data\cellclf_v1.keras" --img_size 96 --epochs 80

  # 4) Apply to new images and export per-image counts/proportions
  python vacuolar_pipeline.py apply --lif_file "D:\data\new.lif" --model "D:\data\cellclf_v1.keras" --out_csv "D:\data\conteo_por_imagen.csv"
"""
import os
import csv
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd
import tifffile

# scikit-image
from skimage.filters import gaussian
from skimage.feature import blob_log

import tensorflow as tf


# ------------------------- IO helpers -------------------------

def ensure_dir(d: str) -> None:
    os.makedirs(d, exist_ok=True)

def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        raise RuntimeError("No hay filas para escribir.")
    ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)

def read_image_2ch_from_tif(path: str) -> np.ndarray:
    """
    Read OME-TIFF/TIFF and return (H,W,2) using Z MIP when applicable.
    Assumes 2 channels or 1 channel (duplicates).
    """
    arr = tifffile.imread(path)
    arr = np.asarray(arr)

    # 4D
    if arr.ndim == 4:
        if arr.shape[1] in (1,2):      # (Z,C,Y,X)
            arr = np.max(arr, axis=0)  # (C,Y,X)
            arr = np.moveaxis(arr, 0, -1)  # (Y,X,C)
        elif arr.shape[0] in (1,2):    # (C,Z,Y,X)
            arr = np.max(arr, axis=1)  # (C,Y,X)
            arr = np.moveaxis(arr, 0, -1)
        else:
            # fallback: max over axis 0
            arr = np.max(arr, axis=0)

    # 3D
    if arr.ndim == 3:
        # (C,Y,X)
        if arr.shape[0] in (1,2):
            arr = np.moveaxis(arr, 0, -1)
        # (Z,Y,X)
        elif arr.shape[0] > 5 and arr.shape[-1] not in (1,2):
            arr = np.max(arr, axis=0)
            arr = np.expand_dims(arr, -1)

    if arr.ndim == 2:
        arr = np.expand_dims(arr, -1)

    if arr.shape[-1] == 1:
        arr = np.concatenate([arr, arr], axis=-1)
    if arr.shape[-1] > 2:
        arr = arr[..., :2]
    if arr.shape[-1] != 2:
        raise ValueError(f"No se pudo obtener 2 canales desde {path}. shape={arr.shape}")

    # normalize to 0..1
    arr = arr.astype(np.float32)
    vmax = float(arr.max()) if arr.size else 1.0
    if vmax > 255.0:
        arr /= 65535.0
    elif vmax > 1.0:
        arr /= 255.0
    return arr


def read_images_from_lif(lif_file: str):
    """
    Iterator of (image_name, image_index, image_2ch_float01[H,W,2]).
    Uses readlif to avoid BioFormats.
    """
    try:
        from readlif.reader import LifFile
    except Exception as e:
        raise SystemExit("Falta readlif. Instala con: pip install readlif pillow\nDetalle: " + str(e))

    lf = LifFile(lif_file)
    imgs = list(lf.get_iter_image())
    if not imgs:
        raise RuntimeError("No se encontraron imágenes dentro del LIF.")

    def get_name(im, fallback):
        for key in ("name", "Name", "ImageName", "image_name"):
            try:
                v = im.image_info.get(key)
                if isinstance(v, str) and v.strip():
                    return v.strip()
            except Exception:
                pass
        v = getattr(im, "name", None)
        if isinstance(v, str) and v.strip():
            return v.strip()
        return fallback

    for i, im in enumerate(imgs):
        name = get_name(im, f"image_{i:03d}")

        # dims: (x,y,z,t,m) in readlif
        dims = getattr(im, "dims", None)
        z_count = 1
        c_count = 1
        if isinstance(dims, (tuple, list)) and len(dims) >= 5:
            z_count = int(dims[2]) if dims[2] else 1
            c_count = int(dims[4]) if dims[4] else 1
        c_use = min(2, c_count)

        chans = []
        for c in range(c_use):
            planes = []
            for z in range(z_count):
                frame = im.get_frame(z=z, t=0, c=c)
                a = np.asarray(frame)
                if a.ndim == 3:  # RGB -> channel 0
                    a = a[..., 0]
                planes.append(a)
            mip = np.max(np.stack(planes, axis=0), axis=0) if len(planes) > 1 else planes[0]
            chans.append(mip)

        if len(chans) == 1:
            chans = [chans[0], chans[0]]

        arr = np.stack(chans[:2], axis=-1).astype(np.float32)

        vmax = float(arr.max()) if arr.size else 1.0
        if vmax > 255.0:
            arr /= 65535.0
        elif vmax > 1.0:
            arr /= 255.0

        yield name, i, arr


# ------------------------- ROI extraction -------------------------

def detect_cells_blobs(img2ch: np.ndarray, sigma_min: float, sigma_max: float, threshold: float) -> np.ndarray:
    """
    Simple LoG (blob_log) detector. Returns Nx3: (y,x,sigma).
    Applied to a detection image derived from the 2 channels.
    """
    # detection image: sum of channels (switch to FM4-64 only if desired)
    det = img2ch[..., 0] + img2ch[..., 1]
    det = gaussian(det, sigma=1.0, preserve_range=True)

    blobs = blob_log(det, min_sigma=sigma_min, max_sigma=sigma_max, num_sigma=10, threshold=threshold)
    # blobs: y, x, sigma
    return blobs.astype(np.float32)

def crop_around(img2ch: np.ndarray, cy: float, cx: float, crop_size: int) -> Optional[np.ndarray]:
    H, W, C = img2ch.shape
    r = crop_size // 2
    y0 = int(round(cy)) - r
    x0 = int(round(cx)) - r
    y1 = y0 + crop_size
    x1 = x0 + crop_size
    if y0 < 0 or x0 < 0 or y1 > H or x1 > W:
        return None
    return img2ch[y0:y1, x0:x1, :]

def save_crop_png(crop2ch: np.ndarray, out_path: str) -> None:
    """
    Save crop as PNG by stacking 2 channels to RGB:
      R = ch1, G = ch0, B = 0 (for visualization/labeling)
    """
    ensure_dir(os.path.dirname(out_path) or ".")
    ch0 = crop2ch[..., 0]
    ch1 = crop2ch[..., 1]
    rgb = np.stack([ch1, ch0, np.zeros_like(ch0)], axis=-1)
    rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)

    # write without PIL: use tf.io.encode_png
    png = tf.io.encode_png(rgb).numpy()
    with open(out_path, "wb") as f:
        f.write(png)

def cmd_extract(args):
    out_dir = args.out_dir
    crops_dir = os.path.join(out_dir, "crops")
    ensure_dir(crops_dir)

    rows = []
    crop_id = 0

    def process_one(image_key: str, img2ch: np.ndarray):
        nonlocal crop_id, rows
        blobs = detect_cells_blobs(img2ch, args.sigma_min, args.sigma_max, args.threshold)
        # sort by sigma (largest first) or by local intensity (simple)
        if blobs.size == 0:
            return
        # limit per image
        if args.max_rois_per_image > 0:
            blobs = blobs[:args.max_rois_per_image]

        for (y, x, s) in blobs:
            cr = crop_around(img2ch, y, x, args.crop_size)
            if cr is None:
                continue
            fn = f"{image_key}__crop_{crop_id:07d}.png"
            path = os.path.join(crops_dir, fn)
            save_crop_png(cr, path)
            rows.append({
                "crop_file": fn,
                "image_key": image_key,
                "y": float(y),
                "x": float(x),
                "sigma": float(s),
                "label": ""  # fill later (A/B/C)
            })
            crop_id += 1
            if args.max_rois > 0 and crop_id >= args.max_rois:
                return

    # source: lif or folder
    if args.lif_file:
        for name, idx, img2ch in read_images_from_lif(args.lif_file):
            image_key = f"{name}"
            process_one(image_key, img2ch)
            if args.max_rois > 0 and crop_id >= args.max_rois:
                break
    else:
        exts = (".tif", ".tiff", ".ome.tif", ".ome.tiff")
        files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(exts)]
        files.sort()
        for f in files:
            img2ch = read_image_2ch_from_tif(os.path.join(args.images_dir, f))
            image_key = f
            process_one(image_key, img2ch)
            if args.max_rois > 0 and crop_id >= args.max_rois:
                break

    if not rows:
        raise SystemExit("No se generaron crops. Prueba ajustar --threshold o --sigma_* o verifica canales.")

    labels_path = os.path.join(out_dir, "labels_template.csv")
    write_csv(labels_path, rows)

    print("\n✅ Extracción lista")
    print(f"  Crops:   {crops_dir}")
    print(f"  Labels:  {labels_path}")
    print("\nSiguiente paso:")
    print("  1) Copia labels_template.csv -> labels_filled.csv")
    print("  2) Rellena columna 'label' con A/B/C para cada crop (puede ser texto o 0/1/2).")
    print("  3) Entrena con: python pipeline_vacuolas_celllevel.py train ...")


# ------------------------- Training cell classifier -------------------------

def load_crop_png(path: str, img_size: int) -> np.ndarray:
    raw = tf.io.read_file(path)
    img = tf.io.decode_png(raw, channels=3)  # RGB
    img = tf.image.convert_image_dtype(img, tf.float32)  # 0..1
    img = tf.image.resize(img, (img_size, img_size), antialias=True)
    return img.numpy().astype(np.float32)

def make_class_map(labels: List[str]) -> Dict[str,int]:
    uniq = sorted(set(labels))
    return {k:i for i,k in enumerate(uniq)}

def cmd_train(args):
    df = pd.read_csv(args.labels_csv)
    if "crop_file" not in df.columns or "label" not in df.columns:
        raise SystemExit("labels_csv debe tener columnas crop_file,label (sale del extract).")

    # --- use ONLY labeled rows (e.g., A/B/C) ---
    raw = df["label"].copy()
    lab = raw.astype(str).str.strip()
    lab_up = lab.str.upper()

    allowed = [x.strip().upper() for x in str(args.allowed_labels).split(",") if x.strip() != ""]
    if not allowed:
        raise SystemExit("allowed_labels está vacío. Usa por ejemplo: --allowed_labels A,B,C")

    keep = lab_up.isin(set(allowed))
    n_total = len(df)
    df = df[keep].copy()
    df["label"] = lab_up[keep].values  # normalized to uppercase

    n_kept = len(df)
    n_drop = n_total - n_kept
    print(f"Etiquetas permitidas: {allowed}")
    print(f"Filas totales: {n_total} | Usadas para entrenar: {n_kept} | Excluidas: {n_drop}")

    if n_kept < 200:
        print(f"⚠️  Hay pocas etiquetas válidas ({n_kept}). Entrenará, pero lo ideal es >1000 crops etiquetados.")

    # class map (only labels that are present and allowed)
    class_map = make_class_map(df["label"].tolist())
    df["y"] = df["label"].map(class_map).astype(int)
    n_classes = len(class_map)
    print("Clases (presentes):", class_map)

    crop_paths = [os.path.join(args.crops_dir, f) for f in df["crop_file"].tolist()]
    y = df["y"].values.astype(np.int32)

    # split
    rng = np.random.RandomState(args.seed)
    idx = np.arange(len(crop_paths))
    rng.shuffle(idx)
    split = int((1.0 - args.val_frac) * len(idx))
    tr, va = idx[:split], idx[split:]

    def ds_from_indices(indices, training: bool):
        paths = [crop_paths[i] for i in indices]
        yy = y[indices]
        ds = tf.data.Dataset.from_tensor_slices((paths, yy))

        def _map(p, lab):
            def _py(pp):
                return load_crop_png(pp.decode("utf-8"), args.img_size)
            img = tf.numpy_function(_py, [p], Tout=tf.float32)
            img.set_shape((args.img_size, args.img_size, 3))
            if training and args.augment:
                img = tf.image.random_flip_left_right(img)
                img = tf.image.random_flip_up_down(img)
                img = tf.image.random_brightness(img, 0.08)
                img = tf.image.random_contrast(img, 0.9, 1.1)
                img = tf.clip_by_value(img, 0, 1)
            return img, tf.one_hot(lab, depth=n_classes)

        ds = ds.map(_map, num_parallel_calls=4)
        if training:
            ds = ds.shuffle(1024, seed=args.seed, reshuffle_each_iteration=True)
        ds = ds.batch(args.batch_size).prefetch(2)
        return ds

    train_ds = ds_from_indices(tr, True)
    val_ds = ds_from_indices(va, False)

    # simple, robust model
    base = tf.keras.applications.EfficientNetV2B0(include_top=False, weights="imagenet", input_shape=(args.img_size, args.img_size, 3), pooling="avg")
    base.trainable = False

    inp = tf.keras.layers.Input(shape=(args.img_size, args.img_size, 3))
    x = tf.keras.applications.efficientnet_v2.preprocess_input(inp * 255.0)
    x = base(x)
    x = tf.keras.layers.Dropout(args.dropout)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(args.dropout)(x)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", name="class")(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    ensure_dir(os.path.dirname(args.out_model) or ".")
    ckpt = tf.keras.callbacks.ModelCheckpoint(args.out_model, monitor="val_accuracy", save_best_only=True, verbose=1)
    early = tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=args.patience, restore_best_weights=True, verbose=1)

    print("\n=== Entrenamiento clasificador por célula ===")
    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=[ckpt, early], verbose=1)

    # save class_map
    map_path = os.path.splitext(args.out_model)[0] + "_class_map.csv"
    pd.DataFrame({"label": list(class_map.keys()), "class_id": list(class_map.values())}).to_csv(map_path, index=False)
    print(f"\n✅ Modelo guardado: {args.out_model}")
    print(f"✅ Mapa de clases: {map_path}")


# ------------------------- Apply + aggregation -------------------------

def load_class_map(model_path: str) -> Dict[int,str]:
    map_path = os.path.splitext(model_path)[0] + "_class_map.csv"
    df = pd.read_csv(map_path)
    inv = {int(r["class_id"]): str(r["label"]) for _, r in df.iterrows()}
    return inv

def classify_crops(model: tf.keras.Model, crop_paths: List[str], img_size: int, batch_size: int) -> np.ndarray:
    def gen():
        for p in crop_paths:
            raw = tf.io.read_file(p)
            img = tf.io.decode_png(raw, channels=3)
            img = tf.image.convert_image_dtype(img, tf.float32)
            img = tf.image.resize(img, (img_size, img_size), antialias=True)
            yield img

    ds = tf.data.Dataset.from_generator(gen, output_signature=tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32))
    ds = ds.batch(batch_size)
    probs = model.predict(ds, verbose=0)
    return probs

def cmd_apply(args):
    model = tf.keras.models.load_model(args.model, compile=False, safe_mode=False)
    inv = load_class_map(args.model)

    out_rows = []

    def process_one(image_key: str, img2ch: np.ndarray):
        blobs = detect_cells_blobs(img2ch, args.sigma_min, args.sigma_max, args.threshold)
        if blobs.size == 0:
            out_rows.append({"image_key": image_key, "n_cells": 0})
            return

        if args.max_rois_per_image > 0:
            blobs = blobs[:args.max_rois_per_image]

        crops_tmp_dir = os.path.join(args.tmp_dir, "crops_tmp")
        ensure_dir(crops_tmp_dir)

        crop_paths = []
        for j, (y, x, s) in enumerate(blobs):
            cr = crop_around(img2ch, y, x, args.crop_size)
            if cr is None:
                continue
            fn = f"{image_key}__tmp_{j:05d}.png"
            path = os.path.join(crops_tmp_dir, fn)
            save_crop_png(cr, path)
            crop_paths.append(path)

        if not crop_paths:
            out_rows.append({"image_key": image_key, "n_cells": 0})
            return

        probs = classify_crops(model, crop_paths, args.img_size, args.batch_size)
        yhat = np.argmax(probs, axis=1)

        # counts
        counts = {}
        for cid in yhat:
            lab = inv.get(int(cid), str(cid))
            counts[lab] = counts.get(lab, 0) + 1

        n = int(len(yhat))
        row = {"image_key": image_key, "n_cells": n}
        # add counts + proportions (p*)
        for lab in sorted(inv.values()):
            c = int(counts.get(lab, 0))
            row[f"{lab}"] = c
            row[f"p{lab}"] = (c / n) if n > 0 else 0.0
        out_rows.append(row)

    # source
    if args.lif_file:
        for name, idx, img2ch in read_images_from_lif(args.lif_file):
            process_one(name, img2ch)
    else:
        exts = (".tif", ".tiff", ".ome.tif", ".ome.tiff")
        files = [f for f in os.listdir(args.images_dir) if f.lower().endswith(exts)]
        files.sort()
        for f in files:
            img2ch = read_image_2ch_from_tif(os.path.join(args.images_dir, f))
            process_one(f, img2ch)

    df = pd.DataFrame(out_rows)
    ensure_dir(os.path.dirname(args.out_csv) or ".")
    df.to_csv(args.out_csv, index=False)
    print(f"✅ Agregación lista. CSV: {args.out_csv}")


# ------------------------- CLI -------------------------

def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    # extract
    pe = sub.add_parser("extract", help="Extrae crops por célula + labels_template.csv")
    pe.add_argument("--images_dir", default="", help="Carpeta con .tif/.ome.tif")
    pe.add_argument("--lif_file", default="", help="Archivo .lif")
    pe.add_argument("--out_dir", required=True, help="Carpeta salida (crea crops/ y labels_template.csv)")
    pe.add_argument("--crop_size", type=int, default=96)
    pe.add_argument("--sigma_min", type=float, default=3.0)
    pe.add_argument("--sigma_max", type=float, default=12.0)
    pe.add_argument("--threshold", type=float, default=0.06, help="Más bajo = detecta más (y más falsos)")
    pe.add_argument("--max_rois", type=int, default=0, help="0=sin límite global")
    pe.add_argument("--max_rois_per_image", type=int, default=0, help="0=sin límite por imagen")

    # train
    pt = sub.add_parser("train", help="Entrena clasificador por célula con crops + labels")
    pt.add_argument("--crops_dir", required=True, help=".../crops")
    pt.add_argument("--labels_csv", required=True, help="labels_filled.csv")
    pt.add_argument("--allowed_labels", default="A,B,C", help="Solo entrenar con estas etiquetas (coma-separadas). Ej: A,B,C")
    pt.add_argument("--out_model", required=True, help="Salida .keras")
    pt.add_argument("--img_size", type=int, default=96)
    pt.add_argument("--batch_size", type=int, default=32)
    pt.add_argument("--epochs", type=int, default=80)
    pt.add_argument("--val_frac", type=float, default=0.2)
    pt.add_argument("--lr", type=float, default=3e-4)
    pt.add_argument("--dropout", type=float, default=0.25)
    pt.add_argument("--augment", action="store_true")
    pt.add_argument("--patience", type=int, default=12)
    pt.add_argument("--seed", type=int, default=42)

    # apply
    pa = sub.add_parser("apply", help="Aplica clasificador por célula y agrega conteos/proporciones por imagen")
    pa.add_argument("--images_dir", default="", help="Carpeta con .tif/.ome.tif")
    pa.add_argument("--lif_file", default="", help="Archivo .lif")
    pa.add_argument("--model", required=True)
    pa.add_argument("--out_csv", required=True)
    pa.add_argument("--img_size", type=int, default=96)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--crop_size", type=int, default=96)
    pa.add_argument("--sigma_min", type=float, default=3.0)
    pa.add_argument("--sigma_max", type=float, default=12.0)
    pa.add_argument("--threshold", type=float, default=0.06)
    pa.add_argument("--max_rois_per_image", type=int, default=0)
    pa.add_argument("--tmp_dir", default=".", help="Carpeta temp para crops de inferencia")

    return p

def main():
    args = build_parser().parse_args()

    if args.cmd == "extract":
        if bool(args.images_dir) == bool(args.lif_file):
            raise SystemExit("Usa exactamente uno: --images_dir o --lif_file")
        cmd_extract(args)
    elif args.cmd == "train":
        cmd_train(args)
    elif args.cmd == "apply":
        if bool(args.images_dir) == bool(args.lif_file):
            raise SystemExit("Usa exactamente uno: --images_dir o --lif_file")
        cmd_apply(args)
    else:
        raise SystemExit("Comando inválido.")

if __name__ == "__main__":
    main()
