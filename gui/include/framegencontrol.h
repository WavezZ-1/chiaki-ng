// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#ifndef CHIAKI_FRAMEGENCONTROL_H
#define CHIAKI_FRAMEGENCONTROL_H

#include "qmlmainwindow.h"

#include <QWidget>

#include <array>

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
	QComboBox *method_box;
	QComboBox *scaler_box;
	QSlider *sharpen_slider;
	QLabel *sharpen_value;
	QLabel *picture_label;
	// Picture enhancement ("Bild" tab), indexed by PictureSetting.
	std::array<QSlider *, static_cast<size_t>(PictureSetting::Count)> pic_sliders{};
	std::array<QLabel *, static_cast<size_t>(PictureSetting::Count)> pic_values{};
	QComboBox *style_box = nullptr;
	QCheckBox *deband_box = nullptr;
	QCheckBox *compare_box = nullptr;
	QLabel *picture_note = nullptr;
	void applyPictureStyle(int style);
	void updatePictureValueLabel(PictureSetting s);
	QSlider *prediction_slider;
	QLabel *prediction_value;
	QSlider *damping_slider;
	QLabel *damping_value;
	QSlider *threshold_slider;
	QLabel *threshold_value;
	QCheckBox *marker_box;
	QCheckBox *on_top_box;

	QLabel *status_label;
	QLabel *display_label;
	QLabel *pacer_label;
	QLabel *game_fps_label;
	QLabel *stream_fps_label;
	QLabel *bitrate_label;
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
