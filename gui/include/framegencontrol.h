// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#ifndef CHIAKI_FRAMEGENCONTROL_H
#define CHIAKI_FRAMEGENCONTROL_H

#include "qmlmainwindow.h"

#include <QWidget>

class QCheckBox;
class QComboBox;
class QLabel;
class QSlider;
class QTimer;
class QmlSettings;

// Small stand-alone window to place next to the stream: switches the frame
// generation mode, tunes it live and shows what it is doing.
class FrameGenControlWindow : public QWidget
{
	Q_OBJECT

public:
	FrameGenControlWindow(QmlSettings *settings, QmlMainWindow *window, QWidget *parent = nullptr);

protected:
	void showEvent(QShowEvent *event) override;
	void hideEvent(QHideEvent *event) override;

private:
	void syncFromSettings();
	void refreshStats();
	void setStayOnTop(bool on);

	QmlSettings *settings;
	QmlMainWindow *window;

	QComboBox *mode_box;
	QSlider *prediction_slider;
	QLabel *prediction_value;
	QSlider *threshold_slider;
	QLabel *threshold_value;
	QCheckBox *marker_box;
	QCheckBox *on_top_box;

	QLabel *status_label;
	QLabel *display_label;
	QLabel *pacer_label;
	QLabel *game_fps_label;
	QLabel *stream_fps_label;
	QLabel *output_fps_label;
	QLabel *generated_label;
	QLabel *gpu_label;
	QLabel *metric_label;
	QLabel *hint_label;

	QTimer *timer;
	FrameGenerationInfo last_info;
	qint64 last_sample_us = 0;
	bool syncing = false;
};

#endif // CHIAKI_FRAMEGENCONTROL_H
