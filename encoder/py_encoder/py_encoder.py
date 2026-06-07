import csv
import glob
import math
import os
import wave

import numpy as np

# ============================================================
#  KONFIGURACJA ENKODERA
# ============================================================
FRAME_WINDOW_MS = 20
ALPHA = 0.99  # Współczynnik odcięcia DC dla HPF

# ---- PARAMETRY PLIKÓW ----
INPUT_DIR = "encoder/snn_input"
OUTPUT_DIR = "encoder/output"

# Współczynnik wzmocnienia (Gain) - dopasuj, jeśli wartości są za niskie
INPUT_GAIN = 1.0


# ============================================================
#  PROCES PRZETWARZANIA POJEDYNCZEGO PLIKU AUDIO
# ============================================================
def process_single_wav(file_path):
    # 1. Wczytanie pliku WAV przez moduł wave
    with wave.open(file_path, "rb") as w:
        fs = w.getframerate()
        n_channels = w.getnchannels()
        n_samples = w.getnframes()
        sampwidth = w.getsampwidth()
        raw_bytes = w.readframes(n_samples)

        # Konwersja bajtów na tablicę numpy w zależności od formatu
        if sampwidth == 2:
            data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = (
                np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32)
                / 2147483648.0
            )
        else:
            data = (
                np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0
            ) / 128.0

        # Jeśli stereo, bierzemy tylko pierwszy kanał (mono)
        if n_channels > 1:
            data = data[::n_channels]

    print(f"Przetwarzanie: {os.path.basename(file_path)} | Próbkowanie: {fs} Hz")

    # ---- INICJALIZACJA STANU GLOBALNEGO (Dokładnie jak w Arduino) ----
    hp_filtered = 0.0
    prev_raw = 512.0  # Spodziewana wartość spoczynkowa ADC

    frame_start_ms = 0.0
    maxAc = 0.0
    sumAc = 0.0
    sumSq = 0.0
    sample_cnt = 0

    features_log = []  # Lista na zebrane statystyki dla każdego timestampu

    dt_ms = 1000.0 / fs

    # ---- PĘTLA GŁÓWNA (Próbka po próbce) ----
    for idx, sample in enumerate(data):
        curr_time_ms = idx * dt_ms

        # Mapowanie sygnału na wirtualne ADC Arduino (0 - 1023)
        raw = 512.0 + (sample * 512.0 * INPUT_GAIN)
        raw = max(0.0, min(1023.0, raw))

        # Filtr HPF usuwający składową stałą
        hp_filtered = ALPHA * (hp_filtered + raw - prev_raw)
        prev_raw = raw
        val = abs(hp_filtered)

        # Akumulacja danych wewnątrz ramki czasowej
        maxAc = max(maxAc, val)
        sumAc += val
        sumSq += val * val
        sample_cnt += 1

        # Czy upłynął czas ramki (np. 20ms)?
        if (curr_time_ms - frame_start_ms) >= FRAME_WINDOW_MS:
            if sample_cnt == 0:
                sample_cnt = 1

            # Wyliczanie cech (dokładnie jak w processAudio na Arduino)
            peak_val = maxAc
            mean_val = sumAc / sample_cnt
            variance = sumSq / sample_cnt
            std_val = math.sqrt(variance)

            # Zapis do pamięci podręcznej (zaokrąglony timestamp)
            features_log.append(
                [
                    round(curr_time_ms, 2),
                    round(peak_val, 4),
                    round(mean_val, 4),
                    round(std_val, 4),
                ]
            )

            # Reset parametrów okna dla kolejnej ramki
            maxAc = 0.0
            sumAc = 0.0
            sumSq = 0.0
            sample_cnt = 0
            frame_start_ms = curr_time_ms

    # ---- ZAPIS CAŁOŚCI DO DEDYkowanego PLIKU CSV ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_csv = os.path.join(OUTPUT_DIR, f"{base_name}_features.csv")

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp_ms", "Peak", "Mean", "Std"])
        writer.writerows(features_log)

    print(f" -> Zapisano punktów: {len(features_log)} do {output_csv}")


# ============================================================
#  GŁÓWNA FUNKCJA URUCHAMIAJĄCA DLA KATALOGU
# ============================================================
def run_batch_audio_logger():
    # Szukanie wszystkich plików .wav w katalogu wejściowym
    search_path = os.path.join(INPUT_DIR, "*.wav")
    wav_files = glob.glob(search_path)

    if not wav_files:
        print(f"[BŁĄD] Brak plików .wav w katalogu '{INPUT_DIR}'!")
        return

    print(f"=== Audio Feature Batch Logger START ===")
    print(f"Znaleziono plików do przetworzenia: {len(wav_files)}")
    print("-" * 50)

    for wav_file in sorted(wav_files):
        process_single_wav(wav_file)

    print("-" * 50)
    print("=== Wszystkie pliki zostały przetworzone pomyślnie ===")


if __name__ == "__main__":
    run_batch_audio_logger()
