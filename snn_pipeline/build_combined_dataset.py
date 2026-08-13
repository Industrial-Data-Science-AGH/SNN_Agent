# -*- coding: utf-8 -*-
"""
Buduje połączony (combined) manifest datasetu SNN z wielu niezależnych źródeł.

Źródła (wszystkie muszą już istnieć lokalnie pod --repo-root, patrz --help):
  - ESC-50            (data/ESC-50-master/audio)         klasa 38 = glass_breaking
  - DataSEC (Zenodo)  (dataset/datasec/PT_DATASET_250314) folder "Glass breaking"
  - notebooks/dataset (glass/ vs negative/)
  - dataset/clean     (VOICe) — pliki to wielogodzinne polifoniczne miksy z
    adnotacjami czasowymi (glassbreak/gunshot/babycry); skrypt wycina z nich
    pojedyncze klipy: glassbreak -> positive, gunshot/babycry -> hard negative
    (pomijając segmenty nakładające się z jakimkolwiek glassbreak w tym samym pliku,
    żeby nie zanieczyścić negatywów fragmentem szkła).

Nie dotyka snn_pipeline/data_pipeline.py, config.py ani żadnego pliku używanego
przez trwający trening — to samodzielny, jednorazowo uruchamiany skrypt.

Użycie:
    python snn_pipeline/build_combined_dataset.py --repo-root /path/do/repo
"""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import pandas as pd
import soundfile as sf
from sklearn.model_selection import GroupShuffleSplit

ESC50_GLASS_BREAKING_CLASS = 38  # zgodne z snn_pipeline/config.py
VOICE_CROP_PADDING_S = 0.1
VOICE_MIN_SEGMENT_S = 0.15
RANDOM_SEED = 42


@dataclass
class Entry:
  filepath: str  # ścieżka względna do repo_root
  label: str  # "positive" | "negative"
  source: str
  subclass: str
  group_id: str  # unikalne id grupy (zapobiega leakage w splitach)


def parse_esc50_target(filename: str) -> int:
  # format: {fold}-{clip_id}-{take}-{target}.wav
  stem = filename.rsplit(".", 1)[0]
  return int(stem.split("-")[-1])


def collect_esc50(repo_root: Path) -> List[Entry]:
  audio_dir = repo_root / "data" / "ESC-50-master" / "audio"
  if not audio_dir.exists():
    print(f"[WARN] ESC-50 nie znaleziono w {audio_dir} — pomijam.")
    return []
  entries = []
  for wav in sorted(audio_dir.glob("*.wav")):
    target = parse_esc50_target(wav.name)
    label = "positive" if target == ESC50_GLASS_BREAKING_CLASS else "negative"
    subclass = (
        "glass_breaking" if label == "positive" else f"esc50_class_{target}"
    )
    parts = wav.name.rsplit(".", 1)[0].split("-")
    group_id = (
        f"esc50_{parts[0]}_{parts[1]}"
        if len(parts) >= 4
        else f"esc50_{wav.stem}"
    )
    entries.append(
        Entry(
            str(wav.relative_to(repo_root)),
            label,
            "esc50",
            subclass,
            group_id,
        )
    )
  print(f"[INFO] ESC-50: {len(entries)} plików")
  return entries


def collect_datasec(repo_root: Path) -> List[Entry]:
  base = repo_root / "dataset" / "datasec" / "PT_DATASET_250314"
  if not base.exists():
    print(f"[WARN] DataSEC nie znaleziono w {base} — pomijam.")
    return []
  entries = []
  for class_dir in sorted(base.iterdir()):
    if not class_dir.is_dir():
      continue
    label = "positive" if class_dir.name == "Glass breaking" else "negative"
    for wav in sorted(class_dir.rglob("*.wav")):
      subclass = (
          wav.parent.name if wav.parent != class_dir else class_dir.name
      )
      group_id = f"datasec_{wav.stem}"
      entries.append(
          Entry(
              str(wav.relative_to(repo_root)),
              label,
              "datasec",
              subclass,
              group_id,
          )
      )
  print(f"[INFO] DataSEC: {len(entries)} plików")
  return entries


def collect_notebooks(repo_root: Path) -> List[Entry]:
  base = repo_root / "notebooks" / "dataset"
  if not base.exists():
    print(f"[WARN] notebooks/dataset nie znaleziono w {base} — pomijam.")
    return []
  entries = []
  for folder, label, subclass in [
      ("glass", "positive", "glass"),
      ("negative", "negative", "negative"),
  ]:
    d = base / folder
    if not d.exists():
      continue
    for wav in sorted(d.glob("*.wav")):
      group_id = f"notebooks_{wav.stem}"
      entries.append(
          Entry(
              str(wav.relative_to(repo_root)),
              label,
              "notebooks",
              subclass,
              group_id,
          )
      )
  print(f"[INFO] notebooks/dataset: {len(entries)} plików")
  return entries


def parse_voice_annotation(path: Path) -> List[Tuple[float, float, str]]:
  segments = []
  for line in path.read_text().splitlines():
    parts = line.strip().split("\t")
    if len(parts) != 3:
      continue
    start, end, label = parts
    segments.append((float(start), float(end), label))
  return segments


def overlaps_any(
    start: float, end: float, others: List[Tuple[float, float]]
) -> bool:
  return any(
      min(end, o_end) > max(start, o_start) for o_start, o_end in others
  )


def crop_voice(repo_root: Path, out_dir: Path) -> List[Entry]:
  annot_dir = repo_root / "dataset" / "clean" / "annotation"
  audio_dir = repo_root / "dataset" / "clean" / "audio"
  if not annot_dir.exists() or not audio_dir.exists():
    print("[WARN] dataset/clean nie znaleziono — pomijam wycinanie VOICe.")
    return []

  pos_dir = out_dir / "voice_crops" / "positive" / "glassbreak"
  neg_gunshot_dir = out_dir / "voice_crops" / "negative" / "gunshot"
  neg_babycry_dir = out_dir / "voice_crops" / "negative" / "babycry"
  for d in (pos_dir, neg_gunshot_dir, neg_babycry_dir):
    d.mkdir(parents=True, exist_ok=True)

  entries: List[Entry] = []
  annot_files = sorted(annot_dir.glob("*.txt"))
  for i, annot_path in enumerate(annot_files):
    wav_path = audio_dir / (annot_path.stem + ".wav")
    if not wav_path.exists():
      continue

    group_id = f"voice_{annot_path.stem}"
    segments = parse_voice_annotation(annot_path)
    glass_intervals = [(s, e) for s, e, lbl in segments if lbl == "glassbreak"]

    with sf.SoundFile(str(wav_path)) as f:
      sr = f.samplerate
      n_frames = len(f)
      duration = n_frames / sr

      def read_crop(start: float, end: float) -> "any":
        start_p = max(0.0, start - VOICE_CROP_PADDING_S)
        end_p = min(duration, end + VOICE_CROP_PADDING_S)
        if end_p - start_p < VOICE_MIN_SEGMENT_S:
          return None
        start_frame = int(start_p * sr)
        num_frames = int((end_p - start_p) * sr)
        f.seek(start_frame)
        return f.read(num_frames)

      for j, (s, e, lbl) in enumerate(segments):
        if lbl == "glassbreak":
          data = read_crop(s, e)
          if data is None:
            continue
          out_path = pos_dir / f"{annot_path.stem}_glass_{j:03d}.wav"
          sf.write(str(out_path), data, sr)
          entries.append(
              Entry(
                  filepath=str(out_path.relative_to(repo_root)),
                  label="positive",
                  source="voice",
                  subclass="glassbreak",
                  group_id=group_id,
              )
          )
        elif lbl in ("gunshot", "babycry"):
          if overlaps_any(s, e, glass_intervals):
            continue  # nie zanieczyszczaj negatywu fragmentem szkła
          data = read_crop(s, e)
          if data is None:
            continue
          target_dir = (
              neg_gunshot_dir if lbl == "gunshot" else neg_babycry_dir
          )
          out_path = target_dir / f"{annot_path.stem}_{lbl}_{j:03d}.wav"
          sf.write(str(out_path), data, sr)
          entries.append(
              Entry(
                  filepath=str(out_path.relative_to(repo_root)),
                  label="negative",
                  source="voice",
                  subclass=lbl,
                  group_id=group_id,
              )
          )

    if (i + 1) % 50 == 0:
      print(
          f"[INFO] VOICe: przetworzono {i + 1}/{len(annot_files)} plików"
          " źródłowych..."
      )

  print(f"[INFO] VOICe: wycięto {len(entries)} klipów")
  return entries


def index_existing_voice_crops(repo_root: Path, out_dir: Path) -> List[Entry]:
  """Indeksuje klipy VOICe wycięte we wcześniejszym uruchomieniu, bez ponownego wycinania."""
  entries: List[Entry] = []
  crops_root = out_dir / "voice_crops"
  for label in ("positive", "negative"):
    label_dir = crops_root / label
    if not label_dir.exists():
      continue
    for subclass_dir in sorted(label_dir.iterdir()):
      if not subclass_dir.is_dir():
        continue
      for wav in sorted(subclass_dir.glob("*.wav")):
        parts = wav.stem.rsplit("_", 2)
        source_stem = parts[0] if len(parts) >= 3 else wav.stem
        group_id = f"voice_{source_stem}"
        entries.append(
            Entry(
                filepath=str(wav.relative_to(repo_root)),
                label=label,
                source="voice",
                subclass=subclass_dir.name,
                group_id=group_id,
            )
        )
  print(f"[INFO] VOICe (z istniejących wycinków): {len(entries)} klipów")
  return entries


def group_based_split(
    entries: List[Entry], val_frac: float, test_frac: float, seed: int
) -> Dict[str, str]:
  """Splits dataset strictly by group_id to prevent any recording-level leakage."""
  df = pd.DataFrame([e.__dict__ for e in entries])

  # 1. Separate Test Set by group_id
  gss_test = GroupShuffleSplit(
      n_splits=1, test_size=test_frac, random_state=seed
  )
  train_val_idx, test_idx = next(gss_test.split(df, groups=df["group_id"]))

  train_val_df = df.iloc[train_val_idx].copy()
  test_df = df.iloc[test_idx].copy()

  # 2. Separate Validation Set from remaining Train/Val by group_id
  relative_val_frac = val_frac / (1.0 - test_frac)
  gss_val = GroupShuffleSplit(
      n_splits=1, test_size=relative_val_frac, random_state=seed
  )
  train_idx, val_idx = next(
      gss_val.split(train_val_df, groups=train_val_df["group_id"])
  )

  train_df = train_val_df.iloc[train_idx].copy()
  val_df = train_val_df.iloc[val_idx].copy()

  split_map = {}
  for fp in train_df["filepath"]:
    split_map[fp] = "train"
  for fp in val_df["filepath"]:
    split_map[fp] = "val"
  for fp in test_df["filepath"]:
    split_map[fp] = "test"

  return split_map


def print_summary(entries: List[Entry], split_map: Dict[str, str]) -> None:
  from collections import Counter

  by_label = Counter(e.label for e in entries)
  by_source_label = Counter((e.source, e.label) for e in entries)
  by_split_label = Counter((split_map[e.filepath], e.label) for e in entries)

  print("\n=== PODSUMOWANIE COMBINED DATASET ===")
  print(
      f"Razem: {len(entries)}  (positive={by_label['positive']},"
      f" negative={by_label['negative']}, stosunek neg:pos ="
      f" {by_label['negative'] / max(1, by_label['positive']):.1f}:1)"
  )
  print("\nWg źródła:")
  for (source, label), cnt in sorted(by_source_label.items()):
    print(f"  {source:10s} {label:9s} {cnt}")
  print("\nWg splitu:")
  for split in ("train", "val", "test"):
    pos = by_split_label.get((split, "positive"), 0)
    neg = by_split_label.get((split, "negative"), 0)
    print(f"  {split:6s} positive={pos:5d}  negative={neg:5d}")


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--repo-root",
      type=Path,
      default=Path(__file__).resolve().parent.parent,
  )
  parser.add_argument("--val-frac", type=float, default=0.1)
  parser.add_argument("--test-frac", type=float, default=0.1)
  parser.add_argument(
      "--skip-voice-crop",
      action="store_true",
      help="Pomiń wycinanie z VOICe (jeśli już wycięte wcześniej)",
  )
  args = parser.parse_args()

  repo_root = args.repo_root.resolve()
  out_dir = repo_root / "dataset" / "combined"
  out_dir.mkdir(parents=True, exist_ok=True)

  entries: List[Entry] = []
  entries += collect_esc50(repo_root)
  entries += collect_datasec(repo_root)
  entries += collect_notebooks(repo_root)
  if not args.skip_voice_crop:
    entries += crop_voice(repo_root, out_dir)
  else:
    entries += index_existing_voice_crops(repo_root, out_dir)

  split_map = group_based_split(
      entries, args.val_frac, args.test_frac, RANDOM_SEED
  )

  manifest_path = out_dir / "manifest.csv"
  with open(manifest_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["filepath", "label", "source", "subclass", "group_id", "split"]
    )
    for e in entries:
      writer.writerow([
          e.filepath,
          e.label,
          e.source,
          e.subclass,
          e.group_id,
          split_map[e.filepath],
      ])

  print(
      f"\n[INFO] Zapisano manifest: {manifest_path} ({len(entries)} wpisów)"
  )
  print_summary(entries, split_map)


if __name__ == "__main__":
  main()