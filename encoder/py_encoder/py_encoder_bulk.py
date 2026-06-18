import csv
import glob
import os
import wave

import numpy as np
import algorithms as algo

INPUT_DIR_POSITIVE = "snn_input/positive"
INPUT_DIR_NEGATIVE = "snn_input/negative"
BASE_OUTPUT_DIR = "encoder_output"


# ============================================================
#  PROCES PRZETWARZANIA POJEDYNCZEGO PLIKU AUDIO
# ============================================================
def process_single_wav(file_path, algorithm_func, csv_writer):
    with wave.open(file_path, "rb") as w:
        fs = w.getframerate()
        n_channels = w.getnchannels()
        n_samples = w.getnframes()
        sampwidth = w.getsampwidth()
        raw_bytes = w.readframes(n_samples)

        if sampwidth == 2:
            data = (
                np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)
                / 32768.0
            )
        elif sampwidth == 4:
            data = (
                np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
                / 2147483648.0
            )
        else:
            data = (
                np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
                - 128.0
            ) / 128.0

        if n_channels > 1:
            data = data[::n_channels]

    file_name = os.path.basename(file_path)

    # Wywołanie algorytmu przekazanego jako parametr funkcji
    frames = algorithm_func(
        data=data,
        fs=fs,
        frame_window_ms=20,
        hpf_enabled=False,
        alpha=0.99,
        gain=1.0,
    )

    final_rows = []
    for frame in frames:
        final_rows.append([file_name] + frame)

    if final_rows:
        csv_writer.writerows(final_rows)


# ============================================================
#  GŁÓWNA FUNKCJA PROCESUJĄCA ZBIORCZO DLA DANUGO ALGORYTMU
# ============================================================
def run_batch_audio_logger(input_dir, output_file, algorithm_func):
    search_path = os.path.join(input_dir, "*.wav")
    wav_files = glob.glob(search_path)

    if not wav_files:
        print(f"[BŁĄD] Brak plików .wav w katalogu '{input_dir}'!")
        return

    print(f" -> Przetwarzanie datasetu: {input_dir} ({len(wav_files)} plików)")

    with open(output_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Generyczny nagłówek kolumn
        writer.writerow(["File_Name", "Timestamp_ms", "F1", "F2", "F3"])

        for i, wav_file in enumerate(sorted(wav_files)):
            process_single_wav(wav_file, algorithm_func, writer)
            if i % 500 == 0 and i > 0:
                print(f"    ...przemieliłem {i} plików")


# ============================================================
#  GŁÓWNA PĘTLA WYKONAWCZA PO WSZYSTKICH ALGORYTMACH
# ============================================================
if __name__ == "__main__":
    algorithms_map = {
        "welford_v2": algo.calculate_features_welford_v2,
        "zcr": algo.calculate_features_zcr,
        "delta_energy": algo.calculate_features_delta_energy,
        "crest_factor": algo.calculate_features_crest_factor,
        "peak_counting": algo.calculate_features_peak_counting,
    }

    print("=== SNN FEATURE ENCODER: BATCH PROCESSING ALL ALGORITHMS ===")
    print("-" * 60)

    # Przechodzimy pętlą po każdym algorytmie z mapy
    for algo_name, algo_function in algorithms_map.items():
        print(f"\n[URUCHAMIAM ALGORYTM]: {algo_name.upper()}")

        algo_output_dir = os.path.join(BASE_OUTPUT_DIR, algo_name)
        os.makedirs(algo_output_dir, exist_ok=True)

        positive_csv = os.path.join(algo_output_dir, "positive.csv")
        negative_csv = os.path.join(algo_output_dir, "negative.csv")

        run_batch_audio_logger(INPUT_DIR_POSITIVE, positive_csv, algo_function)
        run_batch_audio_logger(INPUT_DIR_NEGATIVE, negative_csv, algo_function)

        print(f"[SUKCES] Dane dla '{algo_name}' zapisane w: {algo_output_dir}")

    print("\n" + "-" * 60)
    print("=== WSZYSTKIE ALGORYTMY ZOSTAŁY PRZETWORZONE POMYŚLNIE ===")