# Pochodzenie W0 i granica migracji

Źródło: `feat/dashboard` @ `bb1dbe0209f5cb645546d87637d6ab51c09828f7`.

| Plik / obszar | Decyzja W0 |
|---|---|
| agent/gpio.py | Zaimportowany guard backendu GPIO, jedynie formatowanie Ruff. Lazy import gpiozero; brak uruchomienia sprzętu przy imporcie. Testowany z atrapą. |
| agent/ports.py | Nowe lekkie interfejsy do JPEG, serial i alarmu; implementuje W1. |
| runtime/ports.py | Nowy interfejs Patryka; brak symulacji w mocku. |
| cloud/app/mock_api.py, mock_store.py | Nowy lokalny mock oparty o contracts; brak Azure/storage/auth SDK. |
| camera.py | W1 wykorzysta doświadczenia USB/CSI, ale stara wersja wymaga NumPy/OpenCV i klipów MP4. Nie jest lekkim adapterem JPEG dla Zero. |
| actuators.py | W1 przeniesie pod ograniczony czasowo adapter; stare alarm_on/blink nie są gotową obsługą TTL i idempotencji. |
| sync_queue.py / cloud_sync.py | W2 wykorzysta wzorzec kolejki po dodaniu trwałości, stanu luk i nowego API; stare zdarzenia nie pasują do SpikeBatch. |
| config.py / types.py / machine.py / prefilter.py / vision.py | Nie importowane: stara konfiguracja/ML/halt/Gemini i polityka fail-open są niezgodne z nowym zakresem. |
| cloud auth/storage/routes/infra, deploy | W2/W4: wymagają nowego auth, Key Vault, Foundry i sesji; nie kopiujemy niezgodnych endpointów i domyślnych danych logowania. |
| .env*, wyniki, stary dashboard i stare ADR | Nie importowane jako obowiązująca implementacja. Nowy UI ma zatwierdzony wzorzec. |

Źródło mapy kanałów: `feat/encoder-features` @
`c9ff2346abac54b07468d28e52e908788255569d`,
`encoder/features-improvement/WNIOSKI.md` i twin. To dowód różnicy base/swap;
nie zastępuje K1 i testu parity na aktualnym wybranym firmware.

Nie przeniesiono całej historii dev ani dependency lock starego agenta.
Selekcja pozwala rozpocząć implementację bez uruchamiania starych efektów ubocznych.
