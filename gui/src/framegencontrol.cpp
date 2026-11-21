// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#include "framegencontrol.h"
#include "qmlsettings.h"

#include <QCheckBox>
#include <QComboBox>
#include <QElapsedTimer>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QPushButton>
#include <QShowEvent>
#include <QSlider>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>

namespace {

QElapsedTimer &monotonicClock()
{
	static QElapsedTimer timer;
	if (!timer.isValid())
		timer.start();
	return timer;
}

uint64_t delta(uint64_t now, uint64_t before)
{
	return now >= before ? now - before : 0;
}

QLabel *valueLabel()
{
	QLabel *label = new QLabel(QStringLiteral("–"));
	label->setTextInteractionFlags(Qt::TextSelectableByMouse);
	QFont font = label->font();
	font.setBold(true);
	label->setFont(font);
	return label;
}

} // namespace

FrameGenControlWindow::FrameGenControlWindow(QmlSettings *settings, QmlMainWindow *window, QWidget *parent)
	: QWidget(parent), settings(settings), window(window)
{
	setWindowTitle(tr("chiaki-ng Frame Generation"));
	setAttribute(Qt::WA_ShowWithoutActivating);
	setMinimumWidth(360);

	auto *root = new QVBoxLayout(this);
	// Two tabs: frame generation and picture. The live values stay below both.
	auto *tabs = new QTabWidget;
	auto *fg_page = new QWidget;
	auto *fg_page_layout = new QVBoxLayout(fg_page);
	fg_page_layout->setContentsMargins(4, 4, 4, 4);
	auto *pic_page = new QWidget;
	auto *pic_page_layout = new QVBoxLayout(pic_page);
	pic_page_layout->setContentsMargins(4, 4, 4, 4);
	tabs->addTab(fg_page, tr("Frame Generation"));
	tabs->addTab(pic_page, tr("Bild"));

	// Mode
	auto *mode_group = new QGroupBox(tr("Modus"));
	auto *mode_layout = new QVBoxLayout(mode_group);
	mode_box = new QComboBox;
	mode_box->addItem(tr("Aus"));
	mode_box->addItem(tr("60 fps (doppelte Bilder ersetzen)"));
	mode_box->addItem(tr("Bis 120 fps"));
	mode_box->addItem(tr("Bildschirmrate"));
	mode_box->setToolTip(tr("Bei „Bis 120 fps“ und „Bildschirmrate“ entstehen zwischen zwei echten Bildern "
	                        "mehrere berechnete. Das Bild wird flüssiger, die Steuerung reagiert "
	                        "aber weiter im Takt des Spiels."));
	mode_layout->addWidget(mode_box);
	method_box = new QComboBox;
	method_box->addItem(tr("Zwischenbilder – sauberstes Bild"));
	method_box->addItem(tr("Gemischt"));
	method_box->addItem(tr("Vorhersage – keine Verzögerung"));
	method_box->setToolTip(tr("Zwischenbilder werden zwischen zwei echten Bildern berechnet: auch bei "
	                          "schnellen Drehungen und ruckartigen Animationen sauber, dafür erscheint das "
	                          "Bild knapp ein Spielbild später. Die Vorhersage rät das nächste Bild ohne "
	                          "Verzögerung, verzerrt aber bei schnellen oder plötzlichen Bewegungen. "
	                          "Gemischt liegt dazwischen."));
	mode_layout->addWidget(method_box);
	fg_page_layout->addWidget(mode_group);

	// Tuning
	auto *tune_group = new QGroupBox(tr("Feineinstellung"));
	auto *tune_layout = new QFormLayout(tune_group);

	prediction_slider = new QSlider(Qt::Horizontal);
	prediction_slider->setRange(50, 95);
	prediction_slider->setSingleStep(5);
	prediction_slider->setPageStep(5);
	prediction_value = new QLabel;
	prediction_value->setMinimumWidth(48);
	auto *prediction_row = new QHBoxLayout;
	prediction_row->addWidget(prediction_slider);
	prediction_row->addWidget(prediction_value);
	auto *prediction_label = new QLabel(tr("Max. Vorhersage:"));
	prediction_label->setToolTip(tr("Nur für „Vorhersage“ und „Gemischt“: wie weit ein Bild höchstens in die "
	                                "Zukunft rät (in % eines Spielbilds). Weniger = weniger Fehler bei plötzlichen "
	                                "Richtungswechseln, dafür steht das Bild kurz vor dem nächsten echten Bild still."));
	prediction_slider->setToolTip(prediction_label->toolTip());
	tune_layout->addRow(prediction_label, prediction_row);

	damping_slider = new QSlider(Qt::Horizontal);
	damping_slider->setRange(0, 100);
	damping_slider->setSingleStep(5);
	damping_slider->setPageStep(10);
	damping_value = new QLabel;
	damping_value->setMinimumWidth(48);
	auto *damping_row = new QHBoxLayout;
	damping_row->addWidget(damping_slider);
	damping_row->addWidget(damping_value);
	auto *damping_label = new QLabel(tr("Beschleunigung:"));
	damping_label->setToolTip(tr("Wie stark die Vorhersage einrechnet, ob die Kamera gerade schneller oder "
	                             "langsamer wird. Bremst die Kamera vor einem Richtungswechsel, bremst die "
	                             "Vorhersage mit, statt über das Ziel hinauszuschießen. 0 % = konstante Geschwindigkeit."));
	damping_slider->setToolTip(damping_label->toolTip());
	tune_layout->addRow(damping_label, damping_row);

	threshold_slider = new QSlider(Qt::Horizontal);
	threshold_slider->setRange(5, 60);
	threshold_slider->setSingleStep(1);
	threshold_slider->setPageStep(5);
	threshold_value = new QLabel;
	threshold_value->setMinimumWidth(48);
	auto *threshold_row = new QHBoxLayout;
	threshold_row->addWidget(threshold_slider);
	threshold_row->addWidget(threshold_value);
	auto *threshold_label = new QLabel(tr("Doppelbild-Schwelle:"));
	threshold_label->setToolTip(tr("Bis zu welchem Unterschied ein Bild als Wiederholung gilt. Höher, wenn "
	                               "Spiel-fps in ruhigen Szenen springen; niedriger, wenn langsame "
	                               "Bewegungen ruckeln."));
	threshold_slider->setToolTip(threshold_label->toolTip());
	tune_layout->addRow(threshold_label, threshold_row);

	marker_box = new QCheckBox(tr("Berechnete Bilder markieren (Quadrat oben links)"));
	tune_layout->addRow(marker_box);
	on_top_box = new QCheckBox(tr("Fenster immer im Vordergrund"));
	tune_layout->addRow(on_top_box);
	fg_page_layout->addWidget(tune_group);
	fg_page_layout->addStretch(1);

	// Picture quality: scaler preset (the same setting as in the video
	// settings) and sharpening.
	auto *quality_group = new QGroupBox(tr("Bildqualität"));
	auto *quality_layout = new QFormLayout(quality_group);
	scaler_box = new QComboBox;
	scaler_box->addItem(tr("Schnell"));
	scaler_box->addItem(tr("Standard"));
	scaler_box->addItem(tr("Hohe Qualität"));
	scaler_box->addItem(tr("KI-Hochskalierung (FSRCNNX)"));
	scaler_box->addItem(tr("KI-Hochskalierung stark (FSRCNNX 16)"));
	scaler_box->addItem(tr("Eigene Einstellungen"));
	auto *scaler_label = new QLabel(tr("Hochskalierung:"));
	scaler_label->setToolTip(tr("Wie das 1080p-Bild auf deinen Bildschirm vergrößert wird. Die "
	                            "KI-Hochskalierung rekonstruiert feine Details und wirkt deutlich schärfer, "
	                            "greift aber nur, wenn das Fenster größer als der Stream ist (z. B. 1440p- "
	                            "oder 4K-Monitor). „Stark“ braucht viel mehr Grafikleistung."));
	scaler_box->setToolTip(scaler_label->toolTip());
	quality_layout->addRow(scaler_label, scaler_box);

	sharpen_slider = new QSlider(Qt::Horizontal);
	sharpen_slider->setRange(0, 100);
	sharpen_slider->setSingleStep(5);
	sharpen_slider->setPageStep(10);
	sharpen_value = new QLabel;
	sharpen_value->setMinimumWidth(48);
	auto *sharpen_row = new QHBoxLayout;
	sharpen_row->addWidget(sharpen_slider);
	sharpen_row->addWidget(sharpen_value);
	auto *sharpen_label = new QLabel(tr("Schärfe:"));
	sharpen_label->setToolTip(tr("Kontrast-adaptive Schärfung nach dem Vergrößern: holt feine Details "
	                             "aus dem Stream, ohne helle Ränder an Kanten. Wirkt auch bei 1080p. "
	                             "Zu viel macht Kompressionsartefakte sichtbarer. 0 % = aus."));
	sharpen_slider->setToolTip(sharpen_label->toolTip());
	quality_layout->addRow(sharpen_label, sharpen_row);

	picture_label = new QLabel(QStringLiteral("–"));
	picture_label->setWordWrap(true);
	quality_layout->addRow(tr("Bild:"), picture_label);
	pic_page_layout->addWidget(quality_group);

	// Picture enhancement: style presets and the individual controls.
	auto *enh_group = new QGroupBox(tr("Bildverbesserung"));
	auto *enh_layout = new QFormLayout(enh_group);
	style_box = new QComboBox;
	style_box->addItem(tr("Eigene Einstellung"));
	style_box->addItem(tr("Aus (Originalbild)"));
	style_box->addItem(tr("Natürlich"));
	style_box->addItem(tr("Lebendig"));
	style_box->addItem(tr("Kino"));
	style_box->addItem(tr("Dunkle Szenen"));
	style_box->setToolTip(tr("Fertige Kombinationen aller Regler darunter. Verstellst du danach einen Regler, "
	                         "wird daraus „Eigene Einstellung“."));
	enh_layout->addRow(tr("Bildstil:"), style_box);

	struct SliderSpec { PictureSetting s; const char *label; const char *tip; };
	static const SliderSpec specs[] = {
		{ PictureSetting::Cleanup, QT_TR_NOOP("Glättung:"),
		  QT_TR_NOOP("Glättet Blöckchen und Flimmern vom Streaming (Kompression) in flachen und dunklen Flächen, "
		             "echte Kanten bleiben scharf. Zu viel lässt feine Strukturen wie Laub weicher wirken.") },
		{ PictureSetting::Clarity, QT_TR_NOOP("Klarheit:"),
		  QT_TR_NOOP("Mehr Tiefe und Struktur: hebt den Kontrast zwischen größeren Bildbereichen an (wie "
		             "„Klarheit“ in Fotoprogrammen), ohne feines Rauschen zu verstärken.") },
		{ PictureSetting::Vibrance, QT_TR_NOOP("Lebendigkeit:"),
		  QT_TR_NOOP("Kräftigere Farben, aber nur dort, wo sie blass sind – Hauttöne und schon kräftige Farben "
		             "bleiben natürlich.") },
		{ PictureSetting::Saturation, QT_TR_NOOP("Sättigung:"),
		  QT_TR_NOOP("Alle Farben gleichmäßig kräftiger oder blasser.") },
		{ PictureSetting::Contrast, QT_TR_NOOP("Kontrast:"),
		  QT_TR_NOOP("Sanfte S-Kurve: Schwarz und Weiß bleiben, dazwischen wird es knackiger (oder flacher).") },
		{ PictureSetting::Brightness, QT_TR_NOOP("Helligkeit:"),
		  QT_TR_NOOP("Das ganze Bild heller oder dunkler.") },
		{ PictureSetting::Shadows, QT_TR_NOOP("Schatten aufhellen:"),
		  QT_TR_NOOP("Macht dunkle Bereiche sichtbar (Nacht, Innenräume, Clubs), ohne helle Bereiche zu "
		             "verändern. Viel davon lässt Schwarz etwas grauer wirken.") },
		{ PictureSetting::Temperature, QT_TR_NOOP("Farbtemperatur:"),
		  QT_TR_NOOP("Links kühler (bläulich), rechts wärmer (orange).") },
	};
	for (const SliderSpec &spec : specs) {
		const size_t i = static_cast<size_t>(spec.s);
		auto *slider = new QSlider(Qt::Horizontal);
		slider->setRange(Settings::PictureSettingMin(spec.s), Settings::PictureSettingMax(spec.s));
		slider->setSingleStep(5);
		slider->setPageStep(10);
		auto *value = new QLabel;
		value->setMinimumWidth(48);
		auto *row = new QHBoxLayout;
		row->addWidget(slider);
		row->addWidget(value);
		auto *label = new QLabel(tr(spec.label));
		label->setToolTip(tr(spec.tip));
		slider->setToolTip(label->toolTip());
		enh_layout->addRow(label, row);
		pic_sliders[i] = slider;
		pic_values[i] = value;
		const PictureSetting s = spec.s;
		connect(slider, &QSlider::valueChanged, this, [this, s](int v) {
			updatePictureValueLabel(s);
			if (syncing)
				return;
			this->settings->setPictureSetting(static_cast<int>(s), v);
			this->settings->setPictureSetting(static_cast<int>(PictureSetting::Style), 0);
		});
	}
	deband_box = new QCheckBox(tr("Farbstufen glätten (Debanding)"));
	deband_box->setToolTip(tr("Entfernt sichtbare Stufen in Farbverläufen wie Himmel, Nebel oder dunklen Wänden."));
	enh_layout->addRow(deband_box);
	compare_box = new QCheckBox(tr("Vorher/Nachher: links Original, rechts verbessert"));
	compare_box->setToolTip(tr("Zum Vergleichen: Die linke Bildhälfte bleibt ohne Bildverbesserung und Schärfung."));
	enh_layout->addRow(compare_box);
	picture_note = new QLabel;
	picture_note->setWordWrap(true);
	picture_note->setVisible(false);
	enh_layout->addRow(picture_note);
	pic_page_layout->addWidget(enh_group);
	pic_page_layout->addStretch(1);
	root->addWidget(tabs);

	connect(style_box, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
		if (!syncing)
			applyPictureStyle(index);
	});
	connect(deband_box, &QCheckBox::toggled, this, [this](bool on) {
		if (syncing)
			return;
		this->settings->setPictureSetting(static_cast<int>(PictureSetting::Deband), on ? 1 : 0);
		this->settings->setPictureSetting(static_cast<int>(PictureSetting::Style), 0);
	});
	connect(compare_box, &QCheckBox::toggled, this, [this](bool on) {
		if (this->window)
			this->window->setPictureCompare(on);
	});

	// Live values
	auto *live_group = new QGroupBox(tr("Live"));
	auto *live_layout = new QFormLayout(live_group);
	status_label = valueLabel();
	status_label->setWordWrap(true);
	game_fps_label = valueLabel();
	stream_fps_label = valueLabel();
	bitrate_label = valueLabel();
	bitrate_label->setWordWrap(true);
	output_fps_label = valueLabel();
	generated_label = valueLabel();
	gpu_label = valueLabel();
	metric_label = valueLabel();
	display_label = valueLabel();
	pacer_label = valueLabel();
	pacer_label->setWordWrap(true);
	live_layout->addRow(tr("Status:"), status_label);
	live_layout->addRow(tr("Bildschirm:"), display_label);
	live_layout->addRow(tr("Spiel-fps:"), game_fps_label);
	live_layout->addRow(tr("Stream-fps:"), stream_fps_label);
	live_layout->addRow(tr("Bitrate:"), bitrate_label);
	live_layout->addRow(tr("Ausgabe-fps:"), output_fps_label);
	live_layout->addRow(tr("Davon berechnet:"), generated_label);
	live_layout->addRow(tr("GPU-Zeit:"), gpu_label);
	live_layout->addRow(tr("Bildunterschied:"), metric_label);
	live_layout->addRow(tr("Taktgeber:"), pacer_label);
	root->addWidget(live_group);

	hint_label = new QLabel;
	hint_label->setWordWrap(true);
	hint_label->setStyleSheet(QStringLiteral("color: #e65100;"));
	hint_label->hide();
	root->addWidget(hint_label);
	root->addStretch(1);

	connect(mode_box, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
		if (!syncing)
			this->settings->setFrameGenerationMode(index);
	});
	connect(method_box, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
		static const int delays[] = { 100, 50, 0 };
		if (!syncing && index >= 0 && index < 3)
			this->settings->setFrameGenerationDelay(delays[index]);
	});
	connect(prediction_slider, &QSlider::valueChanged, this, [this](int value) {
		prediction_value->setText(tr("%1 %").arg(value));
		if (!syncing)
			this->settings->setFrameGenerationMaxPrediction(value);
	});
	connect(damping_slider, &QSlider::valueChanged, this, [this](int value) {
		damping_value->setText(tr("%1 %").arg(value));
		if (!syncing)
			this->settings->setFrameGenerationDamping(value);
	});
	connect(threshold_slider, &QSlider::valueChanged, this, [this](int value) {
		threshold_value->setText(QString::number(value / 10.0, 'f', 1));
		if (!syncing)
			this->settings->setFrameGenerationThreshold(value);
	});
	connect(scaler_box, QOverload<int>::of(&QComboBox::activated), this, [this](int index) {
		if (syncing || index < 0 || index > 5)
			return;
		this->settings->setVideoPreset(index);
		if (this->window)
			this->window->setVideoPreset(static_cast<QmlMainWindow::VideoPreset>(index));
	});
	connect(sharpen_slider, &QSlider::valueChanged, this, [this](int value) {
		sharpen_value->setText(value > 0 ? tr("%1 %").arg(value) : tr("aus"));
		if (!syncing) {
			this->settings->setSharpening(value);
			this->settings->setPictureSetting(static_cast<int>(PictureSetting::Style), 0);
		}
	});
	connect(settings, &QmlSettings::videoPresetChanged, this, &FrameGenControlWindow::syncFromSettings);
	connect(marker_box, &QCheckBox::toggled, this, [this](bool on) {
		if (!syncing)
			this->settings->setFrameGenerationMarker(on);
	});
	connect(on_top_box, &QCheckBox::toggled, this, &FrameGenControlWindow::setStayOnTop);
	connect(settings, &QmlSettings::frameGenerationChanged, this, &FrameGenControlWindow::syncFromSettings);

	timer = new QTimer(this);
	timer->setInterval(500);
	connect(timer, &QTimer::timeout, this, &FrameGenControlWindow::refreshStats);

	syncFromSettings();
	resize(sizeHint());
}

void FrameGenControlWindow::showEvent(QShowEvent *event)
{
	QWidget::showEvent(event);
	last_sample_us = 0;
	refreshStats();
	timer->start();
}

void FrameGenControlWindow::hideEvent(QHideEvent *event)
{
	QWidget::hideEvent(event);
	timer->stop();
}

void FrameGenControlWindow::updatePictureValueLabel(PictureSetting s)
{
	const size_t i = static_cast<size_t>(s);
	if (!pic_sliders[i] || !pic_values[i])
		return;
	const int v = pic_sliders[i]->value();
	if (v == 0)
		pic_values[i]->setText(tr("aus"));
	else if (Settings::PictureSettingMin(s) < 0)
		pic_values[i]->setText(v > 0 ? QStringLiteral("+%1").arg(v) : QStringLiteral("−%1").arg(-v));
	else
		pic_values[i]->setText(QStringLiteral("%1 %").arg(v));
}

void FrameGenControlWindow::applyPictureStyle(int style)
{
	// cleanup, clarity, vibrance, saturation, contrast, brightness, shadows,
	// temperature, deband, sharpening
	static const int presets[][10] = {
		{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },          // 1 off
		{ 20, 15, 15, 0, 0, 0, 0, 0, 1, 30 },      // 2 natural
		{ 25, 35, 40, 5, 15, 0, 0, 0, 1, 45 },     // 3 vivid
		{ 25, 20, 10, -5, 20, -5, 0, 15, 1, 25 },  // 4 cinema
		{ 35, 25, 15, 0, 0, 5, 55, 0, 1, 30 },     // 5 dark scenes
	};
	if (style < 1 || style > 5) {
		settings->setPictureSetting(static_cast<int>(PictureSetting::Style), 0);
		return;
	}
	const int *p = presets[style - 1];
	const PictureSetting order[] = { PictureSetting::Cleanup, PictureSetting::Clarity, PictureSetting::Vibrance,
	                                 PictureSetting::Saturation, PictureSetting::Contrast, PictureSetting::Brightness,
	                                 PictureSetting::Shadows, PictureSetting::Temperature, PictureSetting::Deband };
	for (int i = 0; i < 9; i++)
		settings->setPictureSetting(static_cast<int>(order[i]), p[i]);
	settings->setSharpening(p[9]);
	settings->setPictureSetting(static_cast<int>(PictureSetting::Style), style);
	syncFromSettings();
}

void FrameGenControlWindow::syncFromSettings()
{
	syncing = true;
	mode_box->setCurrentIndex(settings->frameGenerationMode());
	const int delay = settings->frameGenerationDelay();
	method_box->setCurrentIndex(delay >= 75 ? 0 : (delay >= 25 ? 1 : 2));
	prediction_slider->setValue(settings->frameGenerationMaxPrediction());
	prediction_value->setText(tr("%1 %").arg(prediction_slider->value()));
	damping_slider->setValue(settings->frameGenerationDamping());
	damping_value->setText(tr("%1 %").arg(damping_slider->value()));
	threshold_slider->setValue(settings->frameGenerationThreshold());
	threshold_value->setText(QString::number(threshold_slider->value() / 10.0, 'f', 1));
	marker_box->setChecked(settings->frameGenerationMarker());
	scaler_box->setCurrentIndex(qBound(0, settings->videoPreset(), 5));
	sharpen_slider->setValue(settings->sharpening());
	for (size_t i = 0; i < pic_sliders.size(); i++) {
		if (!pic_sliders[i])
			continue;
		pic_sliders[i]->setValue(settings->pictureSetting(static_cast<int>(i)));
		updatePictureValueLabel(static_cast<PictureSetting>(i));
	}
	style_box->setCurrentIndex(settings->pictureSetting(static_cast<int>(PictureSetting::Style)));
	deband_box->setChecked(settings->pictureSetting(static_cast<int>(PictureSetting::Deband)) != 0);
	sharpen_value->setText(sharpen_slider->value() > 0 ? tr("%1 %").arg(sharpen_slider->value()) : tr("aus"));
	syncing = false;
}

void FrameGenControlWindow::setStayOnTop(bool on)
{
	const bool was_visible = isVisible();
	setWindowFlag(Qt::WindowStaysOnTopHint, on);
	if (was_visible)
		show();
}

void FrameGenControlWindow::refreshStats()
{
	const FrameGenerationInfo info = window->frameGenerationInfo();
	const qint64 now_us = monotonicClock().nsecsElapsed() / 1000;
	const double dt = last_sample_us > 0 ? (now_us - last_sample_us) / 1e6 : 0.0;
	const FrameGeneratorStats &st = info.stats;
	const FrameGeneratorStats &ls = last_info.stats;
	auto rate = [dt](uint64_t now, uint64_t before) {
		return dt > 0.0 ? delta(now, before) / dt : 0.0;
	};

	const double game_fps = rate(st.unique, ls.unique);
	const double stream_fps = rate(st.frames, ls.frames);
	const double output_fps = rate(info.video_presents, last_info.video_presents);
	const double generated_fps = rate(st.generated, ls.generated);
	const uint64_t generated = delta(st.generated, ls.generated);
	const uint64_t analysed = delta(st.frames, ls.frames);
	const double predict_ms = generated ? delta(st.gpu_generate_ns, ls.gpu_generate_ns) / 1e6 / generated : 0.0;
	const double analyse_ms = analysed ? delta(st.gpu_analysis_ns, ls.gpu_analysis_ns) / 1e6 / analysed : 0.0;
	const uint64_t fires = delta(info.pacer_fires, last_info.pacer_fires);
	const uint64_t pace_renders = delta(info.pacer_renders, last_info.pacer_renders);
	const double fires_per_s = dt > 0.0 ? fires / dt : 0.0;
	const double late_ms = fires ? delta(info.pacer_late_us, last_info.pacer_late_us) / 1000.0 / fires : 0.0;
	const double dispatch_ms = pace_renders ? delta(info.pacer_dispatch_us, last_info.pacer_dispatch_us) / 1000.0 / pace_renders : 0.0;

	mode_box->setItemText(3, tr("Bildschirmrate (%1 Hz)").arg(qRound(info.display_hz)));

	picture_note->setVisible(info.picture_hdr);
	if (info.picture_hdr)
		picture_note->setText(tr("HDR-Stream: Klarheit, Farbe und Ton sind für HDR nicht ausgelegt und deshalb aus. "
		                         "Glättung, Debanding und Schärfe wirken weiter."));
	if (info.src_w > 0 && info.dst_w > 0) {
		static const char *const scalers[] = {
			QT_TR_NOOP("Standard-Skalierung"), QT_TR_NOOP("FSR"),
			QT_TR_NOOP("KI-Hochskalierung (FSRCNNX)"), QT_TR_NOOP("KI-Hochskalierung stark (FSRCNNX 16)")
		};
		QString text;
		if (info.dst_w > info.src_w + 8 || info.dst_h > info.src_h + 8)
			text = tr("%1×%2 → %3×%4, %5").arg(info.src_w).arg(info.src_h).arg(info.dst_w).arg(info.dst_h)
			       .arg(tr(scalers[qBound(0, info.upscaler, 3)]));
		else
			text = tr("%1×%2 ohne Vergrößerung (Fenster nicht größer als der Stream)").arg(info.src_w).arg(info.src_h);
		if (info.sharpening > 0)
			text += tr(", Schärfe %1 %").arg(info.sharpening);
		picture_label->setText(text);
	}

	const bool sampled = dt > 0.0;
	QString status;
	if (info.mode == 0)
		status = tr("Aus");
	else if (!info.supported)
		status = tr("Von dieser Grafikkarte bzw. diesem Renderer nicht unterstützt");
	else if (!sampled)
		status = tr("Messe …");
	else if (stream_fps < 1.0)
		status = tr("Wartet auf Videobilder");
	else if (generated_fps >= 1.0)
		status = tr("Aktiv: %1 → %2 fps").arg(qRound(game_fps)).arg(qRound(output_fps));
	else
		status = tr("Bereit, gerade nichts zu tun (Spiel läuft mit voller Stream-Rate oder steht still)");
	status_label->setText(status);

	auto fps = [sampled](double v) {
		return sampled ? QString::number(v, 'f', 1) : QStringLiteral("–");
	};
	game_fps_label->setText(info.mode ? fps(game_fps) : QStringLiteral("–"));
	stream_fps_label->setText(info.mode ? fps(stream_fps) : QStringLiteral("–"));
	if (info.bitrate_mbps > 0.05) {
		QString text = tr("%1 Mbit/s").arg(info.bitrate_mbps, 0, 'f', 1);
		if (info.bitrate_mbps < 20.0)
			text += tr(" – mehr Bitrate in den Stream-Einstellungen = mehr Details");
		bitrate_label->setText(text);
	} else {
		bitrate_label->setText(QStringLiteral("–"));
	}
	output_fps_label->setText(fps(output_fps));
	generated_label->setText(info.mode ? tr("%1 /s").arg(fps(generated_fps)) : QStringLiteral("–"));
	if (!info.mode || !st.gpu_timers || !sampled)
		gpu_label->setText(QStringLiteral("–"));
	else
		gpu_label->setText(tr("%1 ms je berechnetem Bild, %2 ms Analyse je Bild")
		                   .arg(predict_ms, 0, 'f', 2).arg(analyse_ms, 0, 'f', 2));
	method_box->setEnabled(info.mode >= 2);
	if (info.paced && info.delay_ms >= 0.5)
		display_label->setText(tr("%1 Hz, Ziel %2 fps, Bild %3 ms später")
		                       .arg(qRound(info.display_hz)).arg(qRound(info.target_hz)).arg(qRound(info.delay_ms)));
	else if (info.paced)
		display_label->setText(tr("%1 Hz, Ziel %2 fps").arg(qRound(info.display_hz)).arg(qRound(info.target_hz)));
	else
		display_label->setText(tr("%1 Hz").arg(qRound(info.display_hz)));
	if (!info.paced || !sampled)
		pacer_label->setText(QStringLiteral("–"));
	else
		pacer_label->setText(tr("%1 Weckrufe/s, %2 ms zu spät, %3 ms bis zum Bild")
		                     .arg(fires_per_s, 0, 'f', 0).arg(late_ms, 0, 'f', 2).arg(dispatch_ms, 0, 'f', 2));
	metric_label->setText(info.mode && info.created
	                      ? tr("%1 (Schwelle %2)").arg(st.last_metric, 0, 'f', 2)
	                                               .arg(threshold_slider->value() / 10.0, 0, 'f', 1)
	                      : QStringLiteral("–"));

	QString hint;
	if (info.mode >= 2 && !info.direct_mapping)
		hint = tr("Für mehr als 60 fps in chiaki-ng unter Settings → Video „Frame Delivery: Direct Mapping“ "
		          "wählen (startet chiaki-ng neu). Bis dahin läuft der 60-fps-Modus.");
	else if (info.mode >= 2 && info.display_hz <= 61.0)
		hint = tr("Dieser Bildschirm meldet %1 Hz, mehr als 60 fps sind darauf nicht möglich. "
		          "Liegt das Fenster auf dem richtigen Monitor?").arg(qRound(info.display_hz));
	else if (info.mode == 1 && sampled && stream_fps >= 1.0 && stream_fps < 50.0)
		hint = tr("Der Stream kommt nur mit %1 fps an. Der 60-fps-Modus braucht einen 60-fps-Stream "
		          "(Settings → Stream „FPS“) – oder nimm „Bis 120 fps“ / „Bildschirmrate“.")
		       .arg(qRound(stream_fps));
	hint_label->setText(hint);
	hint_label->setVisible(!hint.isEmpty());

	last_info = info;
	last_sample_us = now_us;
}
