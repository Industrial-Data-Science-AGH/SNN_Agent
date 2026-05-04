import serial
import wave
import numpy as np
import struct

# Konfiguracja
ser = serial.Serial('COM3', 250000)

with wave.open("sounds/glass1.wav", 'rb') as wav:
    audio_data = np.frombuffer(wav.readframes(-1), dtype=np.int16)
    # Skalowanie do 0-1023
    scaled_data = np.interp(audio_data, (-32768, 32767), (0, 1023)).astype(np.uint16)

print("Start binarnej transmisji...")

# Pakujemy dane w paczki bajtów
for val in scaled_data:
    # Wysyłamy 2 bajty (big-endian)
    ser.write(struct.pack('>H', val))
    
    # Sprawdzamy czy Arduino coś odesłało (raport co 32 próbki)
    if ser.in_waiting >= 8:
        data = ser.read(8)
        peak, mean, std, spikes = struct.unpack('HHHH', data)
        # Tutaj zapis do pliku jest opcjonalny - w RT lepiej tylko zbierać dane do RAM