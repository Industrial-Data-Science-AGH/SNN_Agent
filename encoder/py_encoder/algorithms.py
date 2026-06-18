import math


# --- ALGORYTM 0: AKTUALNY (Welforda v2: Peak, Mean, CV) ---
def calculate_features_welford_v2(
    data, fs, frame_window_ms=20, hpf_enabled=False, alpha=0.99, gain=1.0
):
    """Obecny algorytm: Wylicza Peak, Mean oraz Współczynnik Zmienności (CV)."""
    hp_filtered = 0.0
    prev_raw = 450.0
    frame_start_ms = 0.0
    dt_ms = 1000.0 / fs

    wf_mean, wf_M2, wf_n, frame_max = 0.0, 0.0, 0, 0.0
    calculated_frames = []

    for idx, sample in enumerate(data):
        curr_time_ms = idx * dt_ms
        raw = max(0.0, min(1023.0, 512.0 + (sample * 512.0 * gain)))

        if hpf_enabled:
            hp_filtered = alpha * (hp_filtered + raw - prev_raw)
            prev_raw = raw
            val = abs(hp_filtered)
        else:
            val = raw

        wf_n += 1
        delta = val - wf_mean
        wf_mean += delta / wf_n
        wf_M2 += delta * (val - wf_mean)

        if val > frame_max:
            frame_max = val

        if (curr_time_ms - frame_start_ms) >= frame_window_ms:
            if wf_n < 2:
                wf_n, wf_M2 = 2, 0.001

            mean_val = wf_mean
            std_val = math.sqrt(wf_M2 / (wf_n - 1))
            cv_val = (std_val / mean_val) if mean_val > 1.0 else 0.0

            calculated_frames.append(
                [
                    round(frame_start_ms, 2),
                    round(frame_max, 4),
                    round(mean_val, 4),
                    round(cv_val, 4),
                ]
            )

            wf_mean, wf_M2, wf_n, frame_max = 0.0, 0.0, 0, 0.0
            frame_start_ms = curr_time_ms

    return calculated_frames


# --- ALGORYTM 1: ZERO CROSSING RATE (ZCR) ---
def calculate_features_zcr(
    data, fs, frame_window_ms=20, hpf_enabled=False, alpha=0.99, gain=1.0
):
    """Wylicza częstotliwość przejść przez zero (ZCR) odniesioną do składowej stałej."""
    hp_filtered = 0.0
    prev_raw = 450.0
    frame_start_ms = 0.0
    dt_ms = 1000.0 / fs

    # Do ZCR potrzebujemy znać znak poprzedniej próbki względem punktu pracy (512)
    prev_val_centered = 0.0
    zcr_count = 0
    calculated_frames = []

    for idx, sample in enumerate(data):
        curr_time_ms = idx * dt_ms
        raw = max(0.0, min(1023.0, 512.0 + (sample * 512.0 * gain)))

        if hpf_enabled:
            hp_filtered = alpha * (hp_filtered + raw - prev_raw)
            prev_raw = raw
            val_centered = hp_filtered
        else:
            val_centered = raw - 512.0  # Środkowanie sygnału wokół zera

        # Detekcja zmiany znaku (przejścia przez zero)
        if idx > 0 and (
            (val_centered > 0.0 >= prev_val_centered)
            or (val_centered < 0.0 <= prev_val_centered)
        ):
            zcr_count += 1

        prev_val_centered = val_centered

        if (curr_time_ms - frame_start_ms) >= frame_window_ms:
            # Zwracamy zliczoną liczbę przejść jako cechę dominującej częstotliwości
            calculated_frames.append(
                [
                    round(frame_start_ms, 2),
                    zcr_count,
                    0.0,  # Wypełniacze, aby struktura kolumn CSV się zgadzała
                    0.0,
                ]
            )
            zcr_count = 0
            frame_start_ms = curr_time_ms

    return calculated_frames


# --- ALGORYTM 2: DELTA ENERGII (Temporal Derivative) ---
def calculate_features_delta_energy(
    data, fs, frame_window_ms=20, hpf_enabled=False, alpha=0.99, gain=1.0
):
    """Liczy różnicę średniej energii między obecną a poprzednią ramką czasową."""
    hp_filtered = 0.0
    prev_raw = 450.0
    frame_start_ms = 0.0
    dt_ms = 1000.0 / fs

    sum_val = 0.0
    sample_cnt = 0
    prev_frame_mean = 512.0  # Domyślny punkt startowy sygnału
    calculated_frames = []

    for idx, sample in enumerate(data):
        curr_time_ms = idx * dt_ms
        raw = max(0.0, min(1023.0, 512.0 + (sample * 512.0 * gain)))

        if hpf_enabled:
            hp_filtered = alpha * (hp_filtered + raw - prev_raw)
            prev_raw = raw
            val = abs(hp_filtered)
        else:
            val = raw

        sum_val += val
        sample_cnt += 1

        if (curr_time_ms - frame_start_ms) >= frame_window_ms:
            current_mean = sum_val / (sample_cnt if sample_cnt > 0 else 1)
            # Pochodna czasowa (Różnica)
            delta_mean = current_mean - prev_frame_mean

            calculated_frames.append(
                [
                    round(frame_start_ms, 2),
                    round(current_mean, 4),
                    round(delta_mean, 4),
                    0.0,
                ]
            )

            prev_frame_mean = current_mean
            sum_val = 0.0
            sample_cnt = 0
            frame_start_ms = curr_time_ms

    return calculated_frames


# --- ALGORYTM 3: WSPÓŁCZYNNIK SZCZYTU (Crest Factor) ---
def calculate_features_crest_factor(
    data, fs, frame_window_ms=20, hpf_enabled=False, alpha=0.99, gain=1.0
):
    """Mierzy 'szpiczastość' sygnału poprzez relację Peak / Mean."""
    hp_filtered = 0.0
    prev_raw = 450.0
    frame_start_ms = 0.0
    dt_ms = 1000.0 / fs

    sum_val = 0.0
    frame_max = 0.0
    sample_cnt = 0
    calculated_frames = []

    for idx, sample in enumerate(data):
        curr_time_ms = idx * dt_ms
        raw = max(0.0, min(1023.0, 512.0 + (sample * 512.0 * gain)))

        if hpf_enabled:
            hp_filtered = alpha * (hp_filtered + raw - prev_raw)
            prev_raw = raw
            val = abs(hp_filtered)
        else:
            val = raw

        sum_val += val
        sample_cnt += 1
        if val > frame_max:
            frame_max = val

        if (curr_time_ms - frame_start_ms) >= frame_window_ms:
            mean_val = sum_val / (sample_cnt if sample_cnt > 0 else 1)
            # Crest Factor = Peak / Mean
            crest_factor = (frame_max / mean_val) if mean_val > 1.0 else 1.0

            calculated_frames.append(
                [
                    round(frame_start_ms, 2),
                    round(frame_max, 4),
                    round(mean_val, 4),
                    round(crest_factor, 4),
                ]
            )

            sum_val = 0.0
            frame_max = 0.0
            sample_cnt = 0
            frame_start_ms = curr_time_ms

    return calculated_frames


# --- ALGORYTM 4: LICZNIK LOKALNYCH MAKSIMÓW (Peak Counting Rate) ---
def calculate_features_peak_counting(
    data, fs, frame_window_ms=20, hpf_enabled=False, alpha=0.99, gain=1.0
):
    """Zlicza mikro-szpilki (lokalne ekstrema) wewnątrz okna analizy."""
    hp_filtered = 0.0
    prev_raw = 450.0
    frame_start_ms = 0.0
    dt_ms = 1000.0 / fs

    # Bufory próbek do śledzenia trendu (potrzebujemy 3 kolejnych stanów)
    p_val, pp_val = 0.0, 0.0
    local_peaks_count = 0
    calculated_frames = []

    for idx, sample in enumerate(data):
        curr_time_ms = idx * dt_ms
        raw = max(0.0, min(1023.0, 512.0 + (sample * 512.0 * gain)))

        if hpf_enabled:
            hp_filtered = alpha * (hp_filtered + raw - prev_raw)
            prev_raw = raw
            val = abs(hp_filtered)
        else:
            val = raw

        # Szukamy punktu zwrotnego: gdy poprzednia próbka była większa od swoich sąsiadów
        if idx > 2 and (p_val > pp_val and p_val > val):
            local_peaks_count += 1

        # Przesunięcie rejestru próbek
        pp_val = p_val
        p_val = val

        if (curr_time_ms - frame_start_ms) >= frame_window_ms:
            calculated_frames.append(
                [
                    round(frame_start_ms, 2),
                    local_peaks_count,
                    0.0,
                    0.0,
                ]
            )
            local_peaks_count = 0
            frame_start_ms = curr_time_ms

    return calculated_frames
