import wave
import struct


def wczytaj_plik_wav(sciezka_do_pliku):
    """
    Wczytuje plik .wav i zwraca częstotliwość próbkowania oraz listę amplitud.
    """
    try:
        with wave.open(sciezka_do_pliku, 'rb') as plik_wav:
            liczba_kanalow = plik_wav.getnchannels()
            czestotliwosc_probkowania = plik_wav.getframerate()
            liczba_probek = plik_wav.getnframes()

            surowe_bajty = plik_wav.readframes(liczba_probek)

            amplitudy = struct.unpack(f"<{liczba_probek * liczba_kanalow}h", surowe_bajty)

            # Jeśli stereo - bierzemy tylko lewy kanał
            if liczba_kanalow == 2:
                amplitudy = amplitudy[::2]

            return czestotliwosc_probkowania, list(amplitudy)

    except Exception as e:
        print(f"Błąd podczas wczytywania pliku: {e}")
        return None, None
