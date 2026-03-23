# Protokół SPI dla Triggera SNN

Dokument ten opisuje układ przesyłu danych między Arduino Mega / układem analogowym a Raspberry Pi.

## Parametry fizyczne magistrali
- **Prędkość zegara SCK:** 1 MHz (umożliwia bezbłędny odczyt logiki 5V / 3.3V przy użyciu konwertera, długość kabla do 20cm)
- **SPI Mode:** 0 (CPOL=0, CPHA=0)
- **Bit Order:** MSB First
- **CS polarity:** Active LOW

## Zalecany format pakietu - v2 (Faza 1-2)

Pakiet ma rozmiar 2 bajtów i wysyłany jest co 10 ms (ok. 100 Hz). Taka częstotliwość wystarcza przy czasie zdarzenia (tłuczenie szkła) 50-200ms.

### Struktura

**Bajt 0:** DANE STATUSOWE
- `Bit 7..5` : Magic Number = `0b101` (0xA0 na najstarszych bitach). Zapobiega odczytowi śmieci na łączu w stanach nieustalonych. Weryfikacja: `(byte0 >> 5) == 0b101`.
- `Bit 4` : **TRIGGER**. 1 = aktywacja alarmu (wykryto rozbite szkło), 0 = idle.
- `Bit 3` : N1 Fired (1 = tak, 0 = nie)
- `Bit 2` : N2 Fired
- `Bit 1` : N3 Fired
- `Bit 0` : Zarezerwowane (np. status inhibitora)

**Bajt 1:** METADANE / TIMESTAMP
- `Bit 7..0`: Timestamp LSB (8 najmłodszych bitów aktualnego czasu w ms).

### Przykłady
- **Cisza, brak alarmu:** `0xA0 0xXX`
- **Alarm, wyzwolony N1, timestamp LSB 63:** `0xA8 0x3F`
