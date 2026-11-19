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
#include <QShowEvent>
#include <QSlider>
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

	// Mode
	auto *mode_group = new QGroupBox(tr("Modus"));
	auto *mode_layout = new QVBoxLayout(mode_group);
	mode_box = new QComboBox;
	mode_box->addItem(tr("Aus"));
	mode_box->addItem(tr("60 fps (doppelte Bilder ersetzen)"));
	mode_box->addItem(tr("Bis 120 fps"));
	mode_box->addItem(tr("Bildschirmrate"));
	mode_box->setToolTip(tr("Bei 120 fps und Bildschirmrate entstehen zwischen zwei echten Bildern "
	                        "mehrere vorhergesagte. Das Bild wird flüssiger, die Steuerung reagiert "
	                        "aber weiter im Takt des Spiels."));
	mode_layout->addWidget(mode_box);
	root->addWidget(mode_group);

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
	prediction_label->setToolTip(tr("Wie weit ein Zwischenbild höchstens in die Zukunft rät (in % eines "
	                                "Spielbilds). Weniger = weniger Fehler bei plötzlichen Richtungswechseln, "
	                                "dafür steht das Bild kurz vor dem nächsten echten Bild still."));
	prediction_slider->setToolTip(prediction_label->toolTip());
	tune_layout->addRow(prediction_label, prediction_row);

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

	marker_box = new QCheckBox(tr("Vorhergesagte Bilder markieren (Quadrat oben links)"));
	tune_layout->addRow(marker_box);
	on_top_box = new QCheckBox(tr("Fenster immer im Vordergrund"));
	tune_layout->addRow(on_top_box);
	root->addWidget(tune_group);

	// Live values
	auto *live_group = new QGroupBox(tr("Live"));
	auto *live_layout = new QFormLayout(live_group);
	status_label = valueLabel();
	status_label->setWordWrap(true);
	game_fps_label = valueLabel();
	stream_fps_label = valueLabel();
	output_fps_label = valueLabel();
	generated_label = valueLabel();
	gpu_label = valueLabel();
	metric_label = valueLabel();
	live_layout->addRow(tr("Status:"), status_label);
	live_layout->addRow(tr("Spiel-fps:"), game_fps_label);
	live_layout->addRow(tr("Stream-fps:"), stream_fps_label);
	live_layout->addRow(tr("Ausgabe-fps:"), output_fps_label);
	live_layout->addRow(tr("Davon vorhergesagt:"), generated_label);
	live_layout->addRow(tr("GPU-Zeit:"), gpu_label);
	live_layout->addRow(tr("Bildunterschied:"), metric_label);
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
	connect(prediction_slider, &QSlider::valueChanged, this, [this](int value) {
		prediction_value->setText(tr("%1 %").arg(value));
		if (!syncing)
			this->settings->setFrameGenerationMaxPrediction(value);
	});
	connect(threshold_slider, &QSlider::valueChanged, this, [this](int value) {
		threshold_value->setText(QString::number(value / 10.0, 'f', 1));
		if (!syncing)
			this->settings->setFrameGenerationThreshold(value);
	});
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

void FrameGenControlWindow::syncFromSettings()
{
	syncing = true;
	mode_box->setCurrentIndex(settings->frameGenerationMode());
	prediction_slider->setValue(settings->frameGenerationMaxPrediction());
	prediction_value->setText(tr("%1 %").arg(prediction_slider->value()));
	threshold_slider->setValue(settings->frameGenerationThreshold());
	threshold_value->setText(QString::number(threshold_slider->value() / 10.0, 'f', 1));
	marker_box->setChecked(settings->frameGenerationMarker());
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

	mode_box->setItemText(3, tr("Bildschirmrate (%1 Hz)").arg(qRound(info.display_hz)));

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
	output_fps_label->setText(fps(output_fps));
	generated_label->setText(info.mode ? tr("%1 /s").arg(fps(generated_fps)) : QStringLiteral("–"));
	if (!info.mode || !st.gpu_timers || !sampled)
		gpu_label->setText(QStringLiteral("–"));
	else
		gpu_label->setText(tr("%1 ms je Vorhersage, %2 ms Analyse je Bild")
		                   .arg(predict_ms, 0, 'f', 2).arg(analyse_ms, 0, 'f', 2));
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
	else if (info.mode && sampled && stream_fps >= 1.0 && stream_fps < 50.0)
		hint = tr("Der Stream kommt nur mit %1 fps an. Unter Settings → Stream „FPS“ auf 60 stellen.")
		       .arg(qRound(stream_fps));
	hint_label->setText(hint);
	hint_label->setVisible(!hint.isEmpty());

	last_info = info;
	last_sample_us = now_us;
}
