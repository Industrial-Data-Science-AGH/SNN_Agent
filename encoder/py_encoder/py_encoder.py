import csv
import glob
import math
import os
import wave

import numpy as np

# ============================================================
#  KONFIGURACJA ENKODERA (Zgodna z Wersją 2 Arduino)
# ============================================================
FRAME_WINDOW_MS = 20
HPF_ENABLED = False  # W nowej wersji Arduino domyślnie wyłączony
ALPHA = 0.99  # Współczynnik odcięcia DC dla HPF (jeśli włączony)

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

    # ---- INICJALIZACJA STANU GLOBALNEGO (Dokładnie jak w Arduino v2) ----
    hp_filtered = 0.0
    prev_raw = 450.0  # Zmieniono z 512.0 na 450.0 zgodnie z nowym kodem

    frame_start_ms = 0.0

    # Stan algorytmu Welforda Online dla średniej i wariancji
    wf_mean = 0.0
    wf_M2 = 0.0
    wf_n = 0
    frame_max = 0.0

    features_log = []  # Lista na zebrane statystyki (Peak, Mean, CV)
    dt_ms = 1000.0 / fs

    # ---- PĘTLA GŁÓWNA (Próbka po próbce) ----
    for idx, sample in enumerate(data):
        curr_time_ms = idx * dt_ms

        # Mapowanie sygnału na wirtualne ADC Arduino (0 - 1023)
        raw = 512.0 + (sample * 512.0 * INPUT_GAIN)
        raw = max(0.0, min(1023.0, raw))

        # Filtr HPF (zależny od konfiguracji HPF_ENABLED)
        if HPF_ENABLED:
            hp_filtered = ALPHA * (hp_filtered + raw - prev_raw)
            prev_raw = raw
            val = abs(hp_filtered)
        else:
            val = raw

        # Algorytm Welforda Online — inkrementacja "w locie"
        wf_n += 1
        delta = val - wf_mean
        wf_mean += delta / wf_n
        delta2 = val - wf_mean
        wf_M2 += delta * delta2

        if val > frame_max:
            frame_max = val

        # Czy upłynął czas ramki (np. 20ms)?
        if (curr_time_ms - frame_start_ms) >= FRAME_WINDOW_MS:
            # Guard (zabezpieczenie przed dzieleniem przez zero/ujemną próbą)
            if wf_n < 2:
                wf_n = 2
                wf_M2 = 0.001

            mean_val = wf_mean
            std_val = math.sqrt(wf_M2 / (wf_n - 1))  # Próbkowe odchylenie standardowe

            # Współczynnik zmienności (CV = std / mean) uniezależniający od gainu mikrofonu
            cv_val = (std_val / mean_val) if mean_val > 1.0 else 0.0

            # Zapis do pamięci podręcznej (Zastąpiono kolumnę Std przez CV)
            features_log.append(
                [
                    round(curr_time_ms, 2),
                    round(frame_max, 4),
                    round(mean_val, 4),
                    round(cv_val, 4),
                ]
            )

            # Reset parametrów okna Welforda dla kolejnej ramki
            wf_mean = 0.0
            wf_M2 = 0.0
            wf_n = 0
            frame_max = 0.0
            frame_start_ms = curr_time_ms

    # ---- ZAPIS CAŁOŚCI DO DEDYKOWANEGO PLIKU CSV ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    output_csv = os.path.join(OUTPUT_DIR, f"{base_name}_features.csv")

    with open(output_csv, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Timestamp_ms", "Peak", "Mean", "CV"]
        )  # Zmiana nagłówka ze Std na CV
        writer.writerows(features_log)

    print(f" -> Zapisano punktów: {len(features_log)} do {output_csv}")


# ============================================================
#  GŁÓWNA FUNKCJA URUCHAMIAJĄCA DLA KATALOGU
# ============================================================
def run_batch_audio_logger():
    search_path = os.path.join(INPUT_DIR, "*.wav")
    wav_files = glob.glob(search_path)

    if not wav_files:
        print(f"[BŁĄD] Brak plików .wav w katalogu '{INPUT_DIR}'!")
        return

    print("=== Audio Feature Batch Logger (Wersja v2 - CV & Welford) START ===")
    print(f"Znaleziono plików do przetworzenia: {len(wav_files)}")
    print("-" * 50)

    for wav_file in sorted(wav_files):
        process_single_wav(wav_file)

    print("-" * 50)
    print("=== Wszystkie pliki zostały przetworzone pomyślnie ===")


if __name__ == "__main__":
    run_batch_audio_logger()