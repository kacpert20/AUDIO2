import numpy as np

# =========================================================================
# 1. FUNKCJE OKIENKOWE
# =========================================================================

OKNA = ["Prostokątne", "Trójkątne", "Hamminga", "Hanna", "Blackmana"]


def okno_prostokatne(N):
    """w(n) = 1 dla n = 0, 1, ..., N-1"""
    return np.ones(N)


def okno_trojkatne(N):
    """
    Okno trójkątne (Bartlett).
    w(n) = 1 - |2n - (N-1)| / (N-1)
    """
    n = np.arange(N)
    return 1.0 - np.abs(2.0 * n - (N - 1)) / (N - 1)


def okno_hamminga(N):
    """
    Okno Hamminga.
    w(n) = 0.54 - 0.46 * cos(2*pi*n / (N-1))
    """
    n = np.arange(N)
    return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / (N - 1))


def okno_hanna(N):
    """
    Okno Hanna (von Hann).
    w(n) = 0.5 * (1 - cos(2*pi*n / (N-1)))
    """
    n = np.arange(N)
    return 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (N - 1)))


def okno_blackmana(N):
    """
    Okno Blackmana.
    w(n) = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
    """
    n = np.arange(N)
    return (0.42
            - 0.5  * np.cos(2.0 * np.pi * n / (N - 1))
            + 0.08 * np.cos(4.0 * np.pi * n / (N - 1)))


def pobierz_okno(nazwa, N):
    """
    Zwraca tablicę okna o długości N dla wybranej nazwy.
    Wszystkie okna zaimplementowane ręcznie według wzorów.
    """
    if nazwa == "Prostokątne":
        return okno_prostokatne(N)
    elif nazwa == "Trójkątne":
        return okno_trojkatne(N)
    elif nazwa == "Hamminga":
        return okno_hamminga(N)
    elif nazwa == "Hanna":
        return okno_hanna(N)
    elif nazwa == "Blackmana":
        return okno_blackmana(N)
    else:
        return okno_prostokatne(N)


# =========================================================================
# 2. PRZYGOTOWANIE SYGNAŁU I RAMKOWANIE
# =========================================================================

def przygotuj_sygnal(amplitudy):
    """Konwertuje listę amplitud na tablicę numpy float64."""
    return np.array(amplitudy, dtype=np.float64)


def wytnij_ramke(sygnal, fs, czas_s, dlugosc_ramki_ms=20):
    """
    Wycina ramkę sygnału zaczynającą się w czas_s.
    Zwraca ramkę (numpy array).
    """
    rozmiar = int((dlugosc_ramki_ms / 1000.0) * fs)
    idx_start = int(czas_s * fs)
    idx_start = max(0, min(idx_start, len(sygnal) - rozmiar))
    return sygnal[idx_start: idx_start + rozmiar]


def zastosuj_okno(ramka, nazwa_okna):
    """
    Aplikuje funkcję okienkową na ramkę.
    Zwraca (ramka_oryginalna, ramka_z_oknem, okno).
    """
    okno = pobierz_okno(nazwa_okna, len(ramka))
    return ramka, ramka * okno, okno


# =========================================================================
# 3. FFT
# =========================================================================

def oblicz_fft(ramka, fs):
    """
    Oblicza jednostronną FFT ramki.
    Zwraca (freqs, magnitude_dB).
    """
    N = len(ramka)
    widmo = np.fft.rfft(ramka)
    magnitude = np.abs(widmo)
    # Normalizacja + skala dB
    magnitude_db = 20 * np.log10(magnitude / N + 1e-12)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, magnitude_db


def oblicz_fft_liniowa(ramka, fs):
    """
    Oblicza FFT i zwraca magnitudę liniową (do obliczeń parametrów).
    """
    N = len(ramka)
    widmo = np.fft.rfft(ramka)
    magnitude = np.abs(widmo) / N
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, magnitude


# =========================================================================
# 4. SPEKTROGRAM (własna implementacja)
# =========================================================================

def oblicz_spektrogram(sygnal, fs, dlugosc_ramki_ms=20, overlap_procent=50, nazwa_okna="Hamminga"):
    """
    Własna implementacja spektrogramu.

    Parametry:
        sygnal          - sygnał numpy float64
        fs              - częstotliwość próbkowania
        dlugosc_ramki_ms - długość ramki w ms
        overlap_procent  - nakładanie ramek w % (0-90)
        nazwa_okna      - nazwa funkcji okienkowej

    Zwraca:
        times   - czasy środków ramek [s]
        freqs   - częstotliwości [Hz]
        Sdb     - macierz (n_ramek, n_freq) w dB
    """
    rozmiar_ramki = int((dlugosc_ramki_ms / 1000.0) * fs)
    krok = int(rozmiar_ramki * (1.0 - overlap_procent / 100.0))
    krok = max(1, krok)

    okno = pobierz_okno(nazwa_okna, rozmiar_ramki)

    # Zbieramy ramki
    pozycje = np.arange(0, len(sygnal) - rozmiar_ramki + 1, krok)
    n_ramek = len(pozycje)
    n_freq = rozmiar_ramki // 2 + 1

    Sdb = np.zeros((n_ramek, n_freq))

    for i, start in enumerate(pozycje):
        ramka = sygnal[start: start + rozmiar_ramki] * okno
        widmo = np.abs(np.fft.rfft(ramka))
        Sdb[i, :] = 20 * np.log10(widmo / rozmiar_ramki + 1e-12)

    times = (pozycje + rozmiar_ramki / 2) / fs
    freqs = np.fft.rfftfreq(rozmiar_ramki, d=1.0 / fs)

    return times, freqs, Sdb


# =========================================================================
# 5. PARAMETRY CZĘSTOTLIWOŚCIOWE (FRAME-LEVEL)
# =========================================================================

def oblicz_wszystkie_parametry_czest(sygnal, fs, dlugosc_ramki_ms=20,
                                     overlap_procent=0, nazwa_okna="Hamminga"):
    """
    Oblicza wszystkie parametry częstotliwościowe dla każdej ramki sygnału.

    Zwraca słownik:
        times   - czasy środków ramek [s]
        vol_freq - Volume z widma
        fc      - Frequency Centroid [Hz]
        bw      - Effective Bandwidth [Hz]
        ersb1   - Band Energy Ratio subband 1 (0-630 Hz)
        ersb2   - Band Energy Ratio subband 2 (630-1720 Hz)
        ersb3   - Band Energy Ratio subband 3 (1720-4400 Hz)
        sfm     - Spectral Flatness Measure (całe widmo)
        scf     - Spectral Crest Factor (całe widmo)
    """
    rozmiar_ramki = int((dlugosc_ramki_ms / 1000.0) * fs)
    krok = int(rozmiar_ramki * (1.0 - overlap_procent / 100.0))
    krok = max(1, krok)
    okno = pobierz_okno(nazwa_okna, rozmiar_ramki)

    pozycje = np.arange(0, len(sygnal) - rozmiar_ramki + 1, krok)
    n_ramek = len(pozycje)

    freqs = np.fft.rfftfreq(rozmiar_ramki, d=1.0 / fs)
    delta_f = freqs[1] - freqs[0]  # rozdzielczość częstotliwościowa

    # Granice pasm ERSB (Hz) dla fs=22050; skalują się z fs
    # Używamy indeksów bin FFT
    pasma = [
        (0, 630),
        (630, 1720),
        (1720, 4400),
    ]

    vol_freq = np.zeros(n_ramek)
    fc_arr   = np.zeros(n_ramek)
    bw_arr   = np.zeros(n_ramek)
    ersb1    = np.zeros(n_ramek)
    ersb2    = np.zeros(n_ramek)
    ersb3    = np.zeros(n_ramek)
    sfm_arr  = np.zeros(n_ramek)
    scf_arr  = np.zeros(n_ramek)

    for i, start in enumerate(pozycje):
        ramka = sygnal[start: start + rozmiar_ramki] * okno
        S = np.abs(np.fft.rfft(ramka)) / rozmiar_ramki   # |S_n(k)|
        S2 = S ** 2                                        # moc

        N = len(S)
        total_energy = np.sum(S2)

        if total_energy < 1e-20:
            continue

        # --- Volume (z widma) ---
        vol_freq[i] = np.sqrt(np.mean(S2))

        # --- Frequency Centroid ---
        fc = np.sum(freqs * S2) / total_energy
        fc_arr[i] = fc

        # --- Effective Bandwidth ---
        bw_arr[i] = np.sqrt(np.sum(((freqs - fc) ** 2) * S2) / total_energy)

        # --- ERSB ---
        for j, (f_lo, f_hi) in enumerate(pasma):
            mask = (freqs >= f_lo) & (freqs < f_hi)
            band_e = np.sum(S2[mask])
            ratio = band_e / total_energy
            if j == 0: ersb1[i] = ratio
            elif j == 1: ersb2[i] = ratio
            elif j == 2: ersb3[i] = ratio

        # --- SFM (całe widmo) ---
        S2_nz = S2[S2 > 1e-30]
        if len(S2_nz) > 0:
            geo_mean = np.exp(np.mean(np.log(S2_nz)))
            arith_mean = np.mean(S2_nz)
            sfm_arr[i] = geo_mean / arith_mean if arith_mean > 0 else 0

        # --- SCF (całe widmo) ---
        arith_mean_all = np.mean(S2)
        if arith_mean_all > 0:
            scf_arr[i] = np.max(S2) / arith_mean_all

    times = (pozycje + rozmiar_ramki / 2) / fs

    return {
        "times":    times,
        "vol_freq": vol_freq,
        "fc":       fc_arr,
        "bw":       bw_arr,
        "ersb1":    ersb1,
        "ersb2":    ersb2,
        "ersb3":    ersb3,
        "sfm":      sfm_arr,
        "scf":      scf_arr,
    }


# =========================================================================
# 6. CEPSTRUM I ESTYMACJA F0
# =========================================================================

def oblicz_cepstrum(ramka):
    """
    Oblicza rzeczywiste cepstrum ramki.
    C(tau) = |IFFT(log|FFT(s(t))|)|

    Zwraca: cepstrum (numpy array, długość = len(ramka))
    """
    N = len(ramka)
    widmo = np.fft.fft(ramka, n=N)
    log_widmo = np.log(np.abs(widmo) + 1e-12)
    cepstrum = np.real(np.fft.ifft(log_widmo))
    return np.abs(cepstrum)


def estymuj_f0_cepstrum(ramka, fs, f_min=50, f_max=400):
    """
    Estymuje F0 z cepstrum ramki w zakresie [f_min, f_max] Hz.
    Zwraca F0 w Hz lub 0 jeśli ramka bezdźwięczna.
    """
    if np.max(np.abs(ramka)) < 200:
        return 0.0

    C = oblicz_cepstrum(ramka)
    N = len(C)

    # Przeliczamy Hz → indeksy próbek cepstrum
    lag_min = int(fs / f_max)
    lag_max = int(fs / f_min)
    lag_max = min(lag_max, N // 2 - 1)

    if lag_min >= lag_max:
        return 0.0

    fragment = C[lag_min: lag_max]
    idx_peak = np.argmax(fragment)
    wartosc_piku = fragment[idx_peak]

    # Próg: pik musi być wyraźnie silniejszy niż ŚREDNIA w przedziale poszukiwań
    # (nie globalny max — który może być zaburzony przez składowe niskoquefrency)
    srednia = np.mean(fragment)
    if srednia <= 0 or wartosc_piku < 2.5 * srednia:
        return 0.0

    tau_max = idx_peak + lag_min
    f0 = fs / tau_max
    return f0


def oblicz_f0_cepstrum_dla_sygnalu(sygnal, fs, dlugosc_ramki_ms=20,
                                    overlap_procent=0, nazwa_okna="Hamminga",
                                    f_min=50, f_max=400):
    """
    Oblicza F0 metodą cepstrum dla każdej ramki sygnału.

    Zwraca:
        times  - czasy środków ramek [s]
        f0_tab - F0 w Hz (0 = bezdźwięczna)
    """
    rozmiar_ramki = int((dlugosc_ramki_ms / 1000.0) * fs)
    krok = int(rozmiar_ramki * (1.0 - overlap_procent / 100.0))
    krok = max(1, krok)
    okno = pobierz_okno(nazwa_okna, rozmiar_ramki)

    pozycje = np.arange(0, len(sygnal) - rozmiar_ramki + 1, krok)
    f0_tab = np.zeros(len(pozycje))

    for i, start in enumerate(pozycje):
        ramka = sygnal[start: start + rozmiar_ramki] * okno
        f0_tab[i] = estymuj_f0_cepstrum(ramka, fs, f_min, f_max)

    times = (pozycje + rozmiar_ramki / 2) / fs
    return times, f0_tab
