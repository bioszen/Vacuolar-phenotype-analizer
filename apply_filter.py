#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
apply_filter.py

Goal
- Use a pre-trained .keras model (no retraining) to quantify A/B/C phenotypes per image in a .LIF.
- Write a single CSV with:
  * Unfiltered counts (all detected cells)
  * Per-image proportions (pA/pB/pC based on A+B+C)
  * Confidence-filtered counts (only "safe" crops)
  * Filtered per-image proportions
- Column order: A, B, C, n_cells, pA, pB, pC, then filtered columns to the right.

Notes
- To avoid changing your main pipeline, this script reuses the `extract` step from vacuolar_pipeline.py
  to detect cells and generate temporary crops; by default it deletes that temp folder at the end.
- To keep it for inspection, use --keep_workdir.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
import re

import numpy as np
import pandas as pd
import tensorflow as tf


def _sanitize_path_str(s: str) -> str:
    """Avoid errors from embedded newlines or odd spaces when copy-pasting commands in a terminal."""
    if s is None:
        return s
    # Strip embedded newlines and trim surrounding whitespace
    s2 = s.replace("\r", "").replace("\n", "").strip()
    return s2


def load_class_map(model_path: str):
    """Read the class_map required by the pipeline: same stem + _class_map.csv"""
    stem = os.path.splitext(model_path)[0]
    map_path = stem + "_class_map.csv"
    if os.path.exists(map_path):
        df = pd.read_csv(map_path)
        inv = {int(r["class_id"]): str(r["label"]) for _, r in df.iterrows()}
        labels = [inv[i] for i in sorted(inv)]
        return inv, labels, map_path
    # fallback
    inv = {0: "A", 1: "B", 2: "C"}
    labels = ["A", "B", "C"]
    return inv, labels, None


def load_model(model_path: str):
    # Model was trained with preprocess_input (EfficientNetV2); register it as custom_objects.
    custom = {"preprocess_input": tf.keras.applications.efficientnet_v2.preprocess_input}
    model = tf.keras.models.load_model(model_path, compile=False, safe_mode=False, custom_objects=custom)
    return model


def predict_probs(model, img_paths, batch_size: int):
    """Predict probabilities (softmax) for a list of PNG paths."""
    img_size = int(model.input_shape[1])

    def load_png(p):
        raw = tf.io.read_file(p)
        img = tf.io.decode_png(raw, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        img = tf.image.resize(img, (img_size, img_size), antialias=True)
        return img

    ds = (
        tf.data.Dataset.from_tensor_slices(img_paths)
        .map(load_png, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
    )
    probs = model.predict(ds, verbose=0)
    return np.asarray(probs)


def safe_props(a: int, b: int, c: int):
    tot = int(a) + int(b) + int(c)
    if tot <= 0:
        return 0, 0.0, 0.0, 0.0
    return tot, a / tot, b / tot, c / tot


def atomic_to_csv(df: pd.DataFrame, out_csv: Path):
    """Write CSV robustly. If the file is open (Excel), fall back to an alternate name."""
    out_csv = Path(out_csv)
    tmp = out_csv.with_suffix(out_csv.suffix + ".tmp")
    try:
        df.to_csv(tmp, index=False)
        tmp.replace(out_csv)
        return str(out_csv)
    except PermissionError:
        # Excel or another process is locking it
        alt = out_csv.with_name(out_csv.stem + "_ALT" + out_csv.suffix)
        df.to_csv(alt, index=False)
        # cleanup
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return str(alt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lif_file", required=True, help="Ruta a .lif")
    ap.add_argument("--model", required=True, help="Ruta a .keras (modelo entrenado)")
    ap.add_argument("--out_csv", required=True, help="CSV de salida (dual: sin filtro + filtrado)")
    default_pipeline = Path(__file__).with_name("vacuolar_pipeline.py")
    ap.add_argument("--pipeline_py", default=str(default_pipeline),
                    help="Ruta al pipeline que tiene el subcomando extract")

    # detection parameters (same as your pipeline)
    ap.add_argument("--crop_size", type=int, default=96)
    ap.add_argument("--threshold", type=float, default=0.06)
    ap.add_argument("--sigma_min", type=float, default=3.0)
    ap.add_argument("--sigma_max", type=float, default=12.0)

    # inference
    ap.add_argument("--batch_size", type=int, default=128)

    # confidence filtering
    ap.add_argument("--min_conf", type=float, default=0.70, help="Conf mínima para aceptar crop")
    ap.add_argument("--min_margin", type=float, default=0.15, help="Margen mínimo (top1-top2) para aceptar crop")
    ap.add_argument("--disable_filter", action="store_true",
                    help="Desactiva el filtrado por confianza: los campos *_f serán iguales a los brutos (sin descartar crops).")

    # behavior
    ap.add_argument("--keep_workdir", action="store_true",
                    help="Si se activa, NO borra la carpeta temporal con crops (útil para inspección).")

    args = ap.parse_args()

    lif_file = _sanitize_path_str(args.lif_file)
    model_path = _sanitize_path_str(args.model)
    out_csv = Path(_sanitize_path_str(args.out_csv))

    # Create output folder if missing
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Temp folder (avoid invalid characters; do not rely on out_csv if it contains newlines)
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", out_csv.stem)
    work_dir = out_csv.parent / f"{safe_stem}_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1) Extract crops with your pipeline
        cmd = [
            sys.executable, args.pipeline_py, "extract",
            "--lif_file", lif_file,
            "--out_dir", str(work_dir),
            "--crop_size", str(args.crop_size),
            "--threshold", str(args.threshold),
            "--sigma_min", str(args.sigma_min),
            "--sigma_max", str(args.sigma_max),
        ]
        print("\n[1/3] Extract crops:")
        print("  " + " ".join(cmd))
        subprocess.check_call(cmd)

        labels_tmpl = work_dir / "labels_template.csv"
        crops_dir = work_dir / "crops"
        if not labels_tmpl.exists():
            raise FileNotFoundError(f"No se encontró {labels_tmpl}. ¿Falló el extract?")
        if not crops_dir.exists():
            raise FileNotFoundError(f"No se encontró {crops_dir}. ¿Falló el extract?")

        lab = pd.read_csv(labels_tmpl)
        if "crop_file" not in lab.columns or "image_key" not in lab.columns:
            raise ValueError("labels_template.csv debe contener columnas: crop_file, image_key")

        # 2) Load model + class_map
        print("\n[2/3] Predict probabilities (softmax) for crops:")
        inv, class_labels, map_path = load_class_map(model_path)
        if map_path:
            print(f"  class_map: {map_path} -> {class_labels}")
        else:
            print(f"  class_map no encontrado, usando fallback: {class_labels}")

        model = load_model(model_path)

        crop_paths = [str(crops_dir / f) for f in lab["crop_file"].astype(str).tolist()]
        probs = predict_probs(model, crop_paths, batch_size=args.batch_size)

        top1 = probs.argmax(axis=1)
        conf = probs.max(axis=1)
        ps = np.sort(probs, axis=1)
        margin = ps[:, -1] - ps[:, -2]

        lab["pred_label"] = [inv.get(int(i), str(i)) for i in top1]
        lab["conf"] = conf
        lab["margin"] = margin
        if args.disable_filter:
            lab["keep"] = True
        else:
            lab["keep"] = (lab["conf"] >= float(args.min_conf)) & (lab["margin"] >= float(args.min_margin))

        # 3) Aggregate per image: unfiltered vs filtered counts
        print("\n[3/3] Aggregate per image and write CSV:")
        rows = []
        for image_key, g in lab.groupby("image_key", sort=False):
            # unfiltered: all detected crops (top-1 classification)
            a = int((g["pred_label"] == "A").sum())
            b = int((g["pred_label"] == "B").sum())
            c = int((g["pred_label"] == "C").sum())
            n_cells = int(len(g))
            tot, pA, pB, pC = safe_props(a, b, c)

            # filtered: keep==True only
            gf = g[g["keep"]].copy()
            a_f = int((gf["pred_label"] == "A").sum())
            b_f = int((gf["pred_label"] == "B").sum())
            c_f = int((gf["pred_label"] == "C").sum())
            n_cells_f = int(len(gf))
            tot_f, pA_f, pB_f, pC_f = safe_props(a_f, b_f, c_f)

            kept_frac = (n_cells_f / n_cells) if n_cells > 0 else 0.0

            rows.append({
                "image_key": image_key,

                # requested order
                "A": a, "B": b, "C": c,
                "n_cells": n_cells,
                "pA": pA, "pB": pB, "pC": pC,

                # filtered columns to the right
                "A_filt": a_f, "B_filt": b_f, "C_filt": c_f,
                "n_cells_filt": n_cells_f,
                "pA_filt": pA_f, "pB_filt": pB_f, "pC_filt": pC_f,

                # extra metrics (delete if you do not want them)
                "kept_frac": kept_frac,
                "min_conf": float(args.min_conf),
                "min_margin": float(args.min_margin),
            })

        out_df = pd.DataFrame(rows)

        # Ensure column order (in case any are missing)
        col_order = [
            "image_key",
            "A", "B", "C", "n_cells", "pA", "pB", "pC",
            "A_filt", "B_filt", "C_filt", "n_cells_filt", "pA_filt", "pB_filt", "pC_filt",
            "kept_frac", "min_conf", "min_margin",
        ]
        col_order = [c for c in col_order if c in out_df.columns] + [c for c in out_df.columns if c not in col_order]
        out_df = out_df[col_order]

        saved = atomic_to_csv(out_df, out_csv)
        print(f"\n✅ Listo. Guardado: {saved}")
        print("   (Si aparece _ALT, es porque el CSV original estaba abierto/bloqueado.)")

    finally:
        if not args.keep_workdir:
            try:
                shutil.rmtree(work_dir, ignore_errors=True)
            except Exception:
                pass


if __name__ == "__main__":
    main()
