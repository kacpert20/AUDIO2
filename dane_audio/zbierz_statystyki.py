import os
import csv
from audio_io import wczytaj_plik_wav
from math_engine import (przygotuj_sygnal, podziel_na_ramki, oblicz_ste,
                         oblicz_glosnosc, oblicz_zcr, oblicz_vstd, oblicz_vdr,
                         oblicz_vu, oblicz_lster, oblicz_energy_entropy,
                         oblicz_zstd, oblicz_hzcrr)


def analizuj_folder(baza_sciezka, folder_nazwa, etykieta):
    wyniki = []
    pelen_folder = os.path.join(baza_sciezka, folder_nazwa)

    if not os.path.exists(pelen_folder):
        print(f"BŁĄD: Folder {pelen_folder} nie istnieje! Sprawdź ścieżkę.")
        return wyniki

    print(f"Analizuję folder: {folder_nazwa} (Oznaczam jako: {etykieta})...")

    for plik in os.listdir(pelen_folder):
        if plik.endswith('.wav'):
            sciezka = os.path.join(pelen_folder, plik)
            fs, amplitudy = wczytaj_plik_wav(sciezka)

            if amplitudy is None:
                continue

            # Matematyka
            sygnal_np = przygotuj_sygnal(amplitudy)
            ramki = podziel_na_ramki(sygnal_np, fs, dlugosc_ramki_ms=20)
            ste = oblicz_ste(ramki)
            glosnosc = oblicz_glosnosc(ramki)
            zcr = oblicz_zcr(ramki)

            # Pakujemy wyniki do słownika
            wyniki.append({
                "Plik": plik,
                "Klasa": etykieta,
                "VSTD": round(oblicz_vstd(glosnosc), 4),
                "VDR": round(oblicz_vdr(glosnosc), 4),
                "VU": round(oblicz_vu(glosnosc), 2),
                "LSTER": round(oblicz_lster(ste), 4),
                "Entropy": round(oblicz_energy_entropy(ramki), 2),
                "ZSTD": round(oblicz_zstd(zcr), 4),
                "HZCRR": round(oblicz_hzcrr(zcr), 4)
            })

    return wyniki


if __name__ == "__main__":
    # Upewnij się, że nazwa folderu głównego się zgadza!
    folder_glowny = "znormalizowane"

    dane_do_csv = []

    # Zbieramy mowę (załóżmy, że folder nazywa się "16")
    dane_do_csv.extend(analizuj_folder(folder_glowny, "16", "MOWA"))

    # Zbieramy instrumenty
    dane_do_csv.extend(analizuj_folder(folder_glowny, "instrumenty", "INSTRUMENT"))

    # Zapis
    sciezka_csv = "baza_statystyk.csv"
    if dane_do_csv:
        with open(sciezka_csv, mode='w', newline='', encoding='utf-8') as f:
            pola = ["Plik", "Klasa", "VSTD", "VDR", "VU", "LSTER", "Entropy", "ZSTD", "HZCRR"]
            writer = csv.DictWriter(f, fieldnames=pola)
            writer.writeheader()
            writer.writerows(dane_do_csv)
        print(f"\nSUKCES! Zakończono analizę. Zapisano {len(dane_do_csv)} plików do {sciezka_csv}.")
    else:
        print("\nNie udało się wygenerować statystyk. Brak plików do analizy.")