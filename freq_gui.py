import sys
import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTabWidget, QComboBox,
    QSlider, QGroupBox, QSpinBox, QDoubleSpinBox, QSplitter
)
from PyQt6.QtCore import Qt

from audio_io import wczytaj_plik_wav
from freq_engine import (
    OKNA, przygotuj_sygnal, zastosuj_okno, wytnij_ramke,
    oblicz_fft, oblicz_spektrogram,
    oblicz_wszystkie_parametry_czest,
    oblicz_cepstrum, oblicz_f0_cepstrum_dla_sygnalu
)

# ---------------------------------------------------------------------------
# Pomocnicze: tworzenie wykresu pyqtgraph ze wspólnym stylem
# ---------------------------------------------------------------------------

def _nowy_wykres(tytul, lewy, dolny):
    p = pg.PlotWidget(title=tytul)
    p.showGrid(x=True, y=True)
    p.setLabel('left', lewy)
    p.setLabel('bottom', dolny)
    return p


# ===========================================================================
# ZAKŁADKA 0 – Sygnał + Wybór ramki + Okno
# ===========================================================================

class TabSygnal(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout(self)

        # Wykres sygnału
        self.plot_sygnal = _nowy_wykres("Przebieg czasowy sygnału", "Amplituda", "Czas [s]")
        self.krzywa_sygnal = self.plot_sygnal.plot(pen='y')

        # Linia wskazująca wybraną ramkę
        self.linia_ramki = pg.InfiniteLine(pos=0, angle=90, pen=pg.mkPen('r', width=2))
        self.plot_sygnal.addItem(self.linia_ramki)

        layout.addWidget(self.plot_sygnal, stretch=2)

        # Suwak wyboru ramki
        ramka_box = QGroupBox("Wybór ramki do analizy")
        ramka_lay = QHBoxLayout(ramka_box)
        self.lbl_czas = QLabel("Czas ramki: 0.000 s")
        self.slider_ramka = QSlider(Qt.Orientation.Horizontal)
        self.slider_ramka.setMinimum(0)
        self.slider_ramka.setMaximum(1000)
        self.slider_ramka.setValue(0)
        self.slider_ramka.valueChanged.connect(self.on_slider_zmiana)
        ramka_lay.addWidget(QLabel("Początek ramki:"))
        ramka_lay.addWidget(self.slider_ramka)
        ramka_lay.addWidget(self.lbl_czas)
        layout.addWidget(ramka_box)

        # Panel okna
        okno_box = QGroupBox("Funkcja okienkowa dla wybranej ramki")
        okno_lay = QVBoxLayout(okno_box)

        ctrl_lay = QHBoxLayout()
        ctrl_lay.addWidget(QLabel("Okno:"))
        self.combo_okno = QComboBox()
        self.combo_okno.addItems(OKNA)
        self.combo_okno.setCurrentText("Hamminga")
        self.combo_okno.currentTextChanged.connect(self.rysuj_okno)
        ctrl_lay.addWidget(self.combo_okno)
        ctrl_lay.addStretch()
        okno_lay.addLayout(ctrl_lay)

        plots_okno = QHBoxLayout()
        self.plot_ramka_orig = _nowy_wykres("Ramka – oryginał", "Amplituda", "Próbki")
        self.plot_ramka_okno = _nowy_wykres("Ramka – po oknie", "Amplituda", "Próbki")
        self.krzywa_ramka_o = self.plot_ramka_orig.plot(pen='y')
        self.krzywa_ramka_w = self.plot_ramka_okno.plot(pen='c')
        plots_okno.addWidget(self.plot_ramka_orig)
        plots_okno.addWidget(self.plot_ramka_okno)
        okno_lay.addLayout(plots_okno)
        layout.addWidget(okno_box, stretch=2)

    def zaladuj(self):
        """Wywoływane po wczytaniu pliku."""
        sig = self.parent.sygnal
        fs  = self.parent.fs
        os_x = np.arange(len(sig)) / fs
        self.krzywa_sygnal.setData(x=os_x, y=sig)
        self.plot_sygnal.autoRange()
        self.slider_ramka.setValue(0)
        self.rysuj_okno()

    def on_slider_zmiana(self, val):
        sig = self.parent.sygnal
        if sig is None:
            return
        fs  = self.parent.fs
        czas_s = (val / 1000.0) * (len(sig) / fs)
        self.lbl_czas.setText(f"Czas ramki: {czas_s:.3f} s")
        self.linia_ramki.setPos(czas_s)
        self.parent.aktualny_czas_ramki = czas_s
        self.rysuj_okno()
        # Odśwież inne zakładki jeśli otwarte
        self.parent.odswierz_zalezne()

    def rysuj_okno(self):
        sig = self.parent.sygnal
        if sig is None:
            return
        czas_s = self.parent.aktualny_czas_ramki
        nazwa  = self.combo_okno.currentText()
        self.parent.aktualne_okno = nazwa

        ramka = wytnij_ramke(sig, self.parent.fs, czas_s,
                             self.parent.dlugosc_ramki_ms)
        orig, z_oknem, _ = zastosuj_okno(ramka, nazwa)
        xs = np.arange(len(orig))
        self.krzywa_ramka_o.setData(x=xs, y=orig)
        self.krzywa_ramka_w.setData(x=xs, y=z_oknem)


# ===========================================================================
# ZAKŁADKA 1 – FFT
# ===========================================================================

class TabFFT(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout(self)

        ctrl = QHBoxLayout()
        self.btn_przelicz = QPushButton("Przelicz FFT (aktualna ramka + okno)")
        self.btn_przelicz.clicked.connect(self.przelicz)
        ctrl.addWidget(self.btn_przelicz)
        ctrl.addStretch()
        layout.addLayout(ctrl)

        self.plot_fft = _nowy_wykres("Widmo FFT", "Amplituda [dB]", "Częstotliwość [Hz]")
        self.krzywa_fft_orig = self.plot_fft.plot(pen='y', name="Bez okna")
        self.krzywa_fft_okno = self.plot_fft.plot(pen='c', name="Z oknem")
        self.plot_fft.addLegend()
        layout.addWidget(self.plot_fft)

        self.lbl_info = QLabel("Wczytaj plik i kliknij Przelicz.")
        layout.addWidget(self.lbl_info)

    def przelicz(self):
        sig = self.parent.sygnal
        if sig is None:
            return
        czas_s = self.parent.aktualny_czas_ramki
        nazwa  = self.parent.aktualne_okno
        fs     = self.parent.fs

        ramka = wytnij_ramke(sig, fs, czas_s, self.parent.dlugosc_ramki_ms)
        _, z_oknem, _ = zastosuj_okno(ramka, nazwa)

        freqs_o, db_o = oblicz_fft(ramka, fs)
        freqs_w, db_w = oblicz_fft(z_oknem, fs)

        self.krzywa_fft_orig.setData(x=freqs_o, y=db_o)
        self.krzywa_fft_okno.setData(x=freqs_w, y=db_w)
        self.plot_fft.autoRange()

        idx_peak = np.argmax(db_w)
        self.lbl_info.setText(
            f"Ramka: {czas_s:.3f} s | Okno: {nazwa} | "
            f"Peak: {freqs_w[idx_peak]:.1f} Hz ({db_w[idx_peak]:.1f} dB)"
        )

    def odswierz(self):
        self.przelicz()


# ===========================================================================
# ZAKŁADKA 2 – Spektrogram
# ===========================================================================

class TabSpektrogram(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout(self)

        # Kontrolki
        ctrl_box = QGroupBox("Ustawienia spektrogramu")
        ctrl_lay = QHBoxLayout(ctrl_box)

        ctrl_lay.addWidget(QLabel("Okno:"))
        self.combo_okno = QComboBox()
        self.combo_okno.addItems(OKNA)
        self.combo_okno.setCurrentText("Hamminga")
        ctrl_lay.addWidget(self.combo_okno)

        ctrl_lay.addWidget(QLabel("  Dł. ramki [ms]:"))
        self.spin_ramka = QSpinBox()
        self.spin_ramka.setRange(5, 500)
        self.spin_ramka.setValue(100)
        self.spin_ramka.setSuffix(" ms")
        ctrl_lay.addWidget(self.spin_ramka)

        ctrl_lay.addWidget(QLabel("  Overlap:"))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 90)
        self.spin_overlap.setValue(50)
        self.spin_overlap.setSuffix(" %")
        ctrl_lay.addWidget(self.spin_overlap)

        self.btn_przelicz = QPushButton("Rysuj spektrogram")
        self.btn_przelicz.clicked.connect(self.przelicz)
        ctrl_lay.addWidget(self.btn_przelicz)
        ctrl_lay.addStretch()
        layout.addWidget(ctrl_box)

        # Wykres
        self.plot_spec = _nowy_wykres("Spektrogram", "Częstotliwość [Hz]", "Czas [s]")
        self.img_spec = pg.ImageItem()
        self.img_spec.setLookupTable(pg.colormap.get('inferno').getLookupTable())
        self.plot_spec.addItem(self.img_spec)
        layout.addWidget(self.plot_spec)



        self.lbl_info = QLabel("")
        layout.addWidget(self.lbl_info)

    def przelicz(self):
        sig = self.parent.sygnal
        if sig is None:
            return

        nazwa    = self.combo_okno.currentText()
        dl_ms    = self.spin_ramka.value()
        overlap  = self.spin_overlap.value()
        fs       = self.parent.fs

        times, freqs, Sdb = oblicz_spektrogram(sig, fs, dl_ms, overlap, nazwa)

        # pyqtgraph ImageItem: setImage(img) — img[x, y] → x = czas, y = freq
        img = Sdb  # shape (n_ramek, n_freq)
        self.img_spec.setImage(img, autoLevels=False,
                               levels=(np.percentile(Sdb, 5), np.max(Sdb)))

        # Skalowanie osi: 1 kolumna = dt sekund, 1 wiersz = df Hz
        dt = times[1] - times[0] if len(times) > 1 else 0.02
        df = freqs[1] - freqs[0]  if len(freqs) > 1 else fs / 2 / img.shape[1]

        tr = pg.QtGui.QTransform()
        tr.scale(dt, df)
        self.img_spec.setTransform(tr)
        self.plot_spec.getViewBox().invertY(False)
        self.plot_spec.autoRange()

        self.lbl_info.setText(
            f"Okno: {nazwa} | Ramka: {dl_ms} ms | Overlap: {overlap}% | "
            f"Ramek: {len(times)} | Rozdzielczość: {df:.1f} Hz"
        )


# ===========================================================================
# ZAKŁADKA 3 – Parametry częstotliwościowe (frame-level)
# ===========================================================================

class TabParametry(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout(self)

        ctrl_box = QGroupBox("Ustawienia")
        ctrl_lay = QHBoxLayout(ctrl_box)
        ctrl_lay.addWidget(QLabel("Okno:"))
        self.combo_okno = QComboBox()
        self.combo_okno.addItems(OKNA)
        self.combo_okno.setCurrentText("Hamminga")
        ctrl_lay.addWidget(self.combo_okno)
        ctrl_lay.addWidget(QLabel("  Overlap:"))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 90)
        self.spin_overlap.setValue(0)
        self.spin_overlap.setSuffix(" %")
        ctrl_lay.addWidget(self.spin_overlap)
        self.btn = QPushButton("Oblicz parametry")
        self.btn.clicked.connect(self.przelicz)
        ctrl_lay.addWidget(self.btn)
        ctrl_lay.addStretch()
        layout.addWidget(ctrl_box)

        # 8 wykresów
        splitter = QSplitter(Qt.Orientation.Vertical)

        def mw(t, l): return _nowy_wykres(t, l, "Czas [s]")

        self.p_vol  = mw("Volume (z widma)", "V")
        self.p_fc   = mw("Frequency Centroid (FC)", "Hz")
        self.p_bw   = mw("Effective Bandwidth (BW)", "Hz")
        self.p_ersb = mw("Band Energy Ratio (ERSB 1/2/3)", "Udział")
        self.p_sfm  = mw("Spectral Flatness Measure (SFM)", "SFM")
        self.p_scf  = mw("Spectral Crest Factor (SCF)", "SCF")

        self.p_ersb.addLegend()
        self.k_vol  = self.p_vol.plot(pen='y')
        self.k_fc   = self.p_fc.plot(pen='c')
        self.k_bw   = self.p_bw.plot(pen='m')
        self.k_e1   = self.p_ersb.plot(pen='r',  name="ERSB1 0-630 Hz")
        self.k_e2   = self.p_ersb.plot(pen='g',  name="ERSB2 630-1720 Hz")
        self.k_e3   = self.p_ersb.plot(pen='b',  name="ERSB3 1720-4400 Hz")
        self.k_sfm  = self.p_sfm.plot(pen=(255, 165, 0))
        self.k_scf  = self.p_scf.plot(pen=(180, 255, 100))

        for p in [self.p_vol, self.p_fc, self.p_bw, self.p_ersb, self.p_sfm, self.p_scf]:
            splitter.addWidget(p)

        layout.addWidget(splitter)

    def przelicz(self):
        sig = self.parent.sygnal
        if sig is None:
            return
        wyniki = oblicz_wszystkie_parametry_czest(
            sig, self.parent.fs,
            dlugosc_ramki_ms=self.parent.dlugosc_ramki_ms,
            overlap_procent=self.spin_overlap.value(),
            nazwa_okna=self.combo_okno.currentText()
        )
        t = wyniki["times"]
        self.k_vol.setData(x=t, y=wyniki["vol_freq"])
        self.k_fc.setData(x=t,  y=wyniki["fc"])
        self.k_bw.setData(x=t,  y=wyniki["bw"])
        self.k_e1.setData(x=t,  y=wyniki["ersb1"])
        self.k_e2.setData(x=t,  y=wyniki["ersb2"])
        self.k_e3.setData(x=t,  y=wyniki["ersb3"])
        self.k_sfm.setData(x=t, y=wyniki["sfm"])
        self.k_scf.setData(x=t, y=wyniki["scf"])


# ===========================================================================
# ZAKŁADKA 4 – Cepstrum + F0
# ===========================================================================

class TabCepstrum(QWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent = parent
        layout = QVBoxLayout(self)

        # Górny panel: cepstrum wybranej ramki
        cep_box = QGroupBox("Cepstrum wybranej ramki")
        cep_lay = QVBoxLayout(cep_box)

        ctrl = QHBoxLayout()
        self.btn_cep = QPushButton("Pokaż cepstrum aktualnej ramki")
        self.btn_cep.clicked.connect(self.rysuj_cepstrum_ramki)
        ctrl.addWidget(self.btn_cep)
        ctrl.addStretch()
        cep_lay.addLayout(ctrl)

        self.plot_cep = _nowy_wykres(
            "Cepstrum rzeczywiste (zakres 50–400 Hz)",
            "C(τ)", "τ [próbki]"
        )
        self.krzywa_cep = self.plot_cep.plot(pen='c')
        self.linia_f0   = pg.InfiniteLine(angle=90, pen=pg.mkPen('r', width=2, style=Qt.PenStyle.DashLine))
        self.plot_cep.addItem(self.linia_f0)
        self.lbl_cep = QLabel("F0 z cepstrum: -")
        cep_lay.addWidget(self.plot_cep)
        cep_lay.addWidget(self.lbl_cep)
        layout.addWidget(cep_box, stretch=2)

        # Dolny panel: F0(t) dla całego sygnału
        f0_box = QGroupBox("F0(t) – cepstrum dla całego sygnału")
        f0_lay = QVBoxLayout(f0_box)

        ctrl2 = QHBoxLayout()
        ctrl2.addWidget(QLabel("Okno:"))
        self.combo_okno = QComboBox()
        self.combo_okno.addItems(OKNA)
        self.combo_okno.setCurrentText("Hamminga")
        ctrl2.addWidget(self.combo_okno)
        ctrl2.addWidget(QLabel("  Overlap:"))
        self.spin_overlap = QSpinBox()
        self.spin_overlap.setRange(0, 90)
        self.spin_overlap.setValue(50)
        self.spin_overlap.setSuffix(" %")
        ctrl2.addWidget(self.spin_overlap)
        ctrl2.addWidget(QLabel("  F_min [Hz]:"))
        self.spin_fmin = QSpinBox()
        self.spin_fmin.setRange(20, 200)
        self.spin_fmin.setValue(50)
        ctrl2.addWidget(self.spin_fmin)
        ctrl2.addWidget(QLabel("  F_max [Hz]:"))
        self.spin_fmax = QSpinBox()
        self.spin_fmax.setRange(100, 1000)
        self.spin_fmax.setValue(400)
        ctrl2.addWidget(self.spin_fmax)
        self.btn_f0 = QPushButton("Oblicz F0(t)")
        self.btn_f0.clicked.connect(self.oblicz_f0)
        ctrl2.addWidget(self.btn_f0)
        ctrl2.addStretch()
        f0_lay.addLayout(ctrl2)

        self.plot_f0 = _nowy_wykres("F0(t) – Cepstrum", "F0 [Hz]", "Czas [s]")
        self.krzywa_f0 = self.plot_f0.plot(
            pen=None, symbol='o', symbolSize=3,
            symbolBrush='g', symbolPen=None
        )
        f0_lay.addWidget(self.plot_f0)
        layout.addWidget(f0_box, stretch=2)

    def rysuj_cepstrum_ramki(self):
        sig = self.parent.sygnal
        if sig is None:
            return
        czas_s = self.parent.aktualny_czas_ramki
        nazwa  = self.parent.aktualne_okno
        fs     = self.parent.fs

        ramka = wytnij_ramke(sig, fs, czas_s, self.parent.dlugosc_ramki_ms)
        _, z_oknem, _ = zastosuj_okno(ramka, nazwa)
        C = oblicz_cepstrum(z_oknem)

        N = len(C)
        lag_min = int(fs / 400)
        lag_max = int(fs / 50)
        lag_max = min(lag_max, N // 2)

        xs = np.arange(lag_min, lag_max)
        self.krzywa_cep.setData(x=xs, y=C[lag_min:lag_max])
        self.plot_cep.autoRange()

        # F0 dla tej ramki
        from freq_engine import estymuj_f0_cepstrum
        f0 = estymuj_f0_cepstrum(z_oknem, fs)
        if f0 > 0:
            tau = fs / f0
            self.linia_f0.setPos(tau)
            self.lbl_cep.setText(f"F0 z cepstrum: {f0:.1f} Hz  (τ = {tau:.1f} próbek)")
        else:
            self.lbl_cep.setText("F0: ramka bezdźwięczna")

    def oblicz_f0(self):
        sig = self.parent.sygnal
        if sig is None:
            return
        times, f0_tab = oblicz_f0_cepstrum_dla_sygnalu(
            sig, self.parent.fs,
            dlugosc_ramki_ms=self.parent.dlugosc_ramki_ms,
            overlap_procent=self.spin_overlap.value(),
            nazwa_okna=self.combo_okno.currentText(),
            f_min=self.spin_fmin.value(),
            f_max=self.spin_fmax.value()
        )
        mask = f0_tab > 0
        xs = times[mask]
        ys = f0_tab[mask]
        self.krzywa_f0.setData(x=xs, y=ys)
        if len(xs) > 0:
            self.plot_f0.setXRange(float(times[0]), float(times[-1]))
            self.plot_f0.setYRange(float(max(0, np.min(ys) * 0.9)), float(np.max(ys) * 1.1))

    def odswierz(self):
        self.rysuj_cepstrum_ramki()


# ===========================================================================
# GŁÓWNE OKNO
# ===========================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Analiza Częstotliwościowa Dźwięku")
        self.setGeometry(80, 80, 1400, 900)

        # Stan globalny
        self.sygnal = None
        self.fs = None
        self.aktualny_czas_ramki = 0.0
        self.aktualne_okno = "Hamminga"
        self.dlugosc_ramki_ms = 20

        # Widget centralny
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # ── Górny pasek ──────────────────────────────────────────────────
        top = QHBoxLayout()
        self.btn_wczytaj = QPushButton("📂  Wczytaj plik WAV")
        self.btn_wczytaj.setFixedHeight(38)
        self.btn_wczytaj.setStyleSheet("font-weight: bold; font-size: 13px;")
        self.btn_wczytaj.clicked.connect(self.wczytaj)
        top.addWidget(self.btn_wczytaj)

        self.lbl_plik = QLabel("Brak wczytanego pliku")
        self.lbl_plik.setStyleSheet("color: #aaaaff; font-size: 12px;")
        top.addWidget(self.lbl_plik)

        top.addStretch()

        top.addWidget(QLabel("Dł. ramki globalnej:"))
        self.spin_dl_ramki = QSpinBox()
        self.spin_dl_ramki.setRange(5, 200)
        self.spin_dl_ramki.setValue(20)
        self.spin_dl_ramki.setSuffix(" ms")
        self.spin_dl_ramki.setToolTip("Globalna długość ramki używana we wszystkich zakładkach")
        self.spin_dl_ramki.valueChanged.connect(self._zmien_dl_ramki)
        top.addWidget(self.spin_dl_ramki)

        root.addLayout(top)

        # ── Zakładki ─────────────────────────────────────────────────────
        self.tabs = QTabWidget()

        self.tab_sygnal    = TabSygnal(self)
        self.tab_fft       = TabFFT(self)
        self.tab_spektr    = TabSpektrogram(self)
        self.tab_params    = TabParametry(self)
        self.tab_cepstrum  = TabCepstrum(self)

        self.tabs.addTab(self.tab_sygnal,   "🔊 Sygnał + Okno")
        self.tabs.addTab(self.tab_fft,      "📈 FFT")
        self.tabs.addTab(self.tab_spektr,   "🌈 Spektrogram")
        self.tabs.addTab(self.tab_params,   "📊 Parametry (Frame)")
        self.tabs.addTab(self.tab_cepstrum, "🎙️ Cepstrum + F0")

        root.addWidget(self.tabs)

    # ── Akcje ────────────────────────────────────────────────────────────

    def _zmien_dl_ramki(self, val):
        self.dlugosc_ramki_ms = val

    def wczytaj(self):
        sciezka, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik WAV", "", "Pliki WAV (*.wav)"
        )
        if not sciezka:
            return
        fs, amplitudy = wczytaj_plik_wav(sciezka)
        if amplitudy is None:
            return

        self.fs = fs
        self.sygnal = przygotuj_sygnal(amplitudy)
        self.aktualny_czas_ramki = 0.0
        self.lbl_plik.setText(
            f"{sciezka.split('/')[-1]}   |   "
            f"fs={fs} Hz   |   "
            f"Czas: {len(self.sygnal)/fs:.2f} s"
        )

        # Załaduj zakładkę sygnału (reszta na żądanie)
        self.tab_sygnal.zaladuj()
        self.tabs.setCurrentIndex(0)

    def odswierz_zalezne(self):
        """Wywoływane gdy zmienia się ramka – odświeża aktywną zakładkę jeśli obsługuje."""
        idx = self.tabs.currentIndex()
        if idx == 1:
            self.tab_fft.odswierz()
        elif idx == 4:
            self.tab_cepstrum.odswierz()


# ===========================================================================

def main():
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
