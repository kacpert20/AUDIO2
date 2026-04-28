# Audio Frequency Analyser

> Aplikacja desktopowa do analizy częstotliwościowej sygnałów audio (.wav) napisana w Pythonie z GUI opartym na PyQt6 i pyqtgraph.

Projekt 2 z przedmiotu **Analiza i Przetwarzanie Dźwięku** (AiPD 2026).

---

## Funkcje

- **Wizualizacja sygnału** w dziedzinie czasu z interaktywnym suwakiem wyboru ramki
- **Funkcje okienkowe** — prostokątna, trójkątna, Hamminga, Hanna, Blackmana — z podglądem ramki przed i po zastosowaniu okna
- **FFT** — widmo amplitudowe w dB dla wybranej ramki, porównanie z oknem i bez
- **Spektrogram** — własna implementacja z konfigurowalnymi parametrami:
  - wybór okna
  - długość ramki (5–500 ms)
  - nakładanie ramek (overlap 0–90%)
- **Parametry częstotliwościowe na poziomie ramki (Frame-Level)**:
  - Volume (z widma)
  - Frequency Centroid (FC)
  - Effective Bandwidth (BW)
  - Band Energy Ratio (ERSB1 / ERSB2 / ERSB3)
  - Spectral Flatness Measure (SFM)
  - Spectral Crest Factor (SCF)
- **Cepstrum** — rzeczywiste cepstrum wybranej ramki z detekcją F0
- **F0(t)** — wykres tonu podstawowego dla całego sygnału metodą cepstrum

---

## Wymagania

| Biblioteka | Wersja | Opis |
|---|---|---|
| Python | ≥ 3.10 | — |
| `numpy` | ≥ 1.24 | obliczenia numeryczne, FFT |
| `PyQt6` | ≥ 6.4 | framework GUI |
| `pyqtgraph` | ≥ 0.13 | wykresy i wizualizacja |

Biblioteki `wave` i `struct` są częścią biblioteki standardowej Pythona — nie wymagają instalacji.

---

## Instalacja

```bash
# 1. Sklonuj repozytorium
git clone https://github.com/kacpert20/AUDIO2.git
cd AUDIO2

# 2. Utwórz i aktywuj wirtualne środowisko (zalecane)
python -m venv .venv

# Windows
.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate

# 3. Zainstaluj zależności
pip install numpy PyQt6 pyqtgraph
```

---

## Uruchomienie

```bash
python freq_gui.py
```

---

## Struktura projektu

```
.
├── audio_io.py       # wczytywanie plików .wav (wave + struct)
├── freq_engine.py    # cały silnik matematyczny:
│                     #   okna, FFT, spektrogram, parametry frame-level,
│                     #   cepstrum, estymacja F0
├── freq_gui.py       # GUI (PyQt6 + pyqtgraph), zakładki
└── README.md
```

---

## Obsługa

1. Kliknij **Wczytaj plik WAV** i wybierz plik `.wav` (mono lub stereo — stereo jest automatycznie konwertowane do mono).
2. Ustaw **globalną długość ramki** (spinbox w prawym górnym rogu) — wpływa na zakładki FFT i Cepstrum.
3. Użyj **suwaka** w zakładce *Sygnał + Okno* aby wybrać ramkę do analizy.
4. Wybierz **funkcję okienkową** z listy rozwijanej.
5. Przechodzij między zakładkami i klikaj przyciski obliczeniowe (`Przelicz FFT`, `Rysuj spektrogram`, `Oblicz parametry`, `Oblicz F0(t)`).

> **Wskazówka:** W zakładce *Spektrogram* użyj overlap 50–75% i ramki 50–100 ms dla czytelnej wizualizacji harmonicznych instrumentów.

---

## Szczegóły implementacji

### Spektrogram
Implementacja własna (bez `scipy`). Dla każdej ramki obliczana jest FFT z wybranym oknem, wynik zapisywany w macierzy `(n_ramek × n_freq)` i wyświetlany jako obraz z mapą kolorów *inferno*.

### Cepstrum
Rzeczywiste cepstrum obliczane według wzoru:

```
C(τ) = |IFFT(log|FFT(s(t))|)|
```

F0 estymowane jest jako `fs / τ_max`, gdzie `τ_max` to pozycja dominującego piku cepstrum w przedziale odpowiadającym zakresowi `[f_min, f_max]` Hz. Próg detekcji dźwięczności: pik musi przekraczać 2.5× średnią wartość cepstrum w przeszukiwanym przedziale.

### Parametry ERSB
Pasma częstotliwości zgodne z modelem słuchu człowieka (cochlear filters):
- **ERSB1**: 0–630 Hz
- **ERSB2**: 630–1720 Hz
- **ERSB3**: 1720–4400 Hz

---

## Znane ograniczenia

- Cepstrum może dawać błąd oktawy dla sygnałów quasi-sinusoidalnych (flety, flażolety gitarowe), gdzie fundamentalna harmoniczna jest tłumiona.
- Estymacja F0 dla akordów (wiele fundamentów jednocześnie) jest niezdefiniowana — algorytm wskazuje jeden z możliwych fundamentów lub szum.
- Obsługiwane są wyłącznie pliki `.wav` z całkowitą liczbą bitów próbki = 16 bit.

---

## Autorzy

- **Kacper Tomczyk** — implementacja, analiza, sprawozdanie

Projekt realizowany w ramach kursu *Analiza i Przetwarzanie Dźwięku*, semestr letni 2025/2026.
