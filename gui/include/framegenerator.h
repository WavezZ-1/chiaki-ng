// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL
//
// Low-latency frame generation for 30 fps games streamed at 60 fps.
//
// When a console game renders at 30 fps but Remote Play streams at 60 fps,
// every game frame arrives twice. FrameGenerator detects those repeats on the
// GPU and replaces each one with a motion-extrapolated frame that predicts
// where the picture will be half a game frame later. It never waits for a
// future frame, so real frames are shown exactly as early as before; only the
// repeated slots are swapped for predicted ones.
//
// For display-rate output it can also interpolate between the two newest real
// frames instead (interpolate()), which is far cleaner on fast or sudden
// motion but has to be shown later than the newest frame arrived.
//
// The generator works on the decoded YCbCr planes, so the output goes through
// exactly the same rendering path (scaler, upscaling shaders, tone mapping) as
// the real frames.

#ifndef CHIAKI_FRAMEGENERATOR_H
#define CHIAKI_FRAMEGENERATOR_H

#include <libplacebo/dispatch.h>
#include <libplacebo/gpu.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <vector>

struct FrameGeneratorStats
{
	uint64_t frames = 0;        // frames inspected
	uint64_t unique = 0;        // frames with new content
	uint64_t duplicates = 0;    // repeated frames detected
	uint64_t generated = 0;     // predicted frames produced
	double cadence = 1.0;       // average stream frames per game frame (2.0 = 30 fps game at 60 fps)
	double frame_interval = 0.0; // average seconds between game frames (0 = unknown)
	bool active = false;        // a low-fps cadence (repeated frames) was detected
	float last_metric = 0.0f;   // last frame-difference metric (8-bit luma levels)
	float process_ms = 0.0f;    // time spent analysing the last frame (CPU side, includes the readback wait)
	bool gpu_timers = false;    // GPU timings below are available
	uint64_t gpu_analysis_ns = 0; // total GPU time of the per-frame analysis passes
	uint64_t gpu_generate_ns = 0; // total GPU time spent predicting frames
};

class FrameGenerator
{
public:
	FrameGenerator(pl_log log, pl_gpu gpu);
	~FrameGenerator();

	FrameGenerator(const FrameGenerator &) = delete;
	FrameGenerator &operator=(const FrameGenerator &) = delete;

	// False if the GPU lacks the required features or the shaders kept failing.
	bool valid() const { return dispatch != nullptr && !disabled; }

	// Inspect a freshly mapped source frame (as produced by pl_map_avframe).
	//
	// `pts` must be in seconds and increase for new frames; a frame with the same
	// pts as the previous one is treated as a re-submission and left untouched,
	// and a pts that goes backwards resets the internal history.
	//
	// Returns true if `src` repeats the previous game frame and was replaced by
	// a predicted frame. In that case `out` describes the predicted frame, whose
	// planes live in `out_tex` (PL_MAX_PLANES texture slots owned by the caller,
	// created or resized as needed). `out` holds no reference to `src`, so the
	// caller may unmap `src` right away. Returns false if `src` should be shown
	// unchanged.
	bool process(const struct pl_frame *src, double pts, pl_tex out_tex[PL_MAX_PLANES], struct pl_frame *out);

	// The two halves of process(), for callers that pace output themselves
	// (e.g. at the display refresh rate):
	//
	// analyze() inspects a freshly mapped frame and updates the history.
	// `*is_new` tells whether it carries new content or repeats the previous
	// game frame. Returns false if the frame could not be analysed (show it as
	// is and do not predict from it).
	bool analyze(const struct pl_frame *src, double pts, bool *is_new);

	// True once two distinct game frames are known and recent enough to
	// extrapolate from.
	bool canPredict() const;

	// Predicts the picture `factor` game frames after the newest real frame
	// (0 < factor < 1, capped internally). `meta` supplies the frame layout and
	// colour metadata (the last mapped frame); its textures are not used.
	bool predict(const struct pl_frame *meta, float factor, pl_tex out_tex[PL_MAX_PLANES], struct pl_frame *out);

	// Interpolates between the two newest real frames: t = 0 is the previous
	// game frame, t = 1 the newest one. Unlike predict() this needs no guess
	// about the future, so fast or sudden motion comes out much cleaner, but
	// the caller has to show the result later than the newest frame arrived.
	// Available whenever canPredict() is true.
	bool interpolate(const struct pl_frame *meta, float t, pl_tex out_tex[PL_MAX_PLANES], struct pl_frame *out);

	// Estimates the motion between the two newest frames right away (otherwise
	// it runs with the first generated picture). With `backward` also the
	// backward field that interpolate() needs. Returns false if no picture
	// can be generated for this pair.
	bool prepareMotion(bool backward);

	// Average time between two game frames in seconds (from the pts of new
	// frames), or 0 if not known yet.
	double gameFrameInterval() const { return frame_interval; }

	// Forget all history, e.g. after a seek, a reconnect or a format change.
	void reset();

	FrameGeneratorStats stats() const;

	// Frame-difference threshold below which a frame counts as a repeat,
	// in 8-bit luma levels (mean absolute difference over a 32x32 block).
	void setDuplicateThreshold(float levels) { dup_threshold_levels = levels; }

	// How much of the change in speed between the last two pairs of game
	// frames predictions take into account (0 = constant speed, 1 = full
	// constant-acceleration model). Makes predictions brake when the camera
	// slows down before a reversal.
	void setDamping(float strength) { damping = std::min(1.0f, std::max(0.0f, strength)); }

	// Draw a small white square in the top-left corner of predicted frames.
	void setMarker(bool enabled) { show_marker = enabled; }

	// Receives warnings (resource or shader failures). Defaults to stderr.
	void setLogger(std::function<void(const char *)> fn) { logger = std::move(fn); }

private:
	static constexpr int kSlots = 3;
	static constexpr int kLevels = 5; // pyramid levels 1..5 (1/2 .. 1/32 of luma resolution)

	// One GPU timer per distinct pass (libplacebo timers must only be reused
	// for identical operations). Passes before kTimerSearch run for every
	// frame, the rest only when a frame is predicted.
	enum {
		kTimerLuma = 0,
		kTimerDown = kTimerLuma + 1,               // + level (1..4)
		kTimerDiff = kTimerDown + kLevels,
		kTimerCopy = kTimerDiff + 1,               // + plane
		kTimerSearch = kTimerCopy + PL_MAX_PLANES, // + level
		kTimerSearchBack = kTimerSearch + kLevels, // + level
		kTimerCheck = kTimerSearchBack + kLevels,
		kTimerFill,
		kTimerSmooth,
		kTimerProject,
		kTimerWarp,                                // + plane
		kTimerCheckBack = kTimerWarp + PL_MAX_PLANES,
		kTimerFillBack,
		kTimerSmoothBack,
		kTimerInterp,
		kTimerBlend,                               // + plane
		kTimerTiles = kTimerBlend + PL_MAX_PLANES,
		kTimerGlobal,
		kTimerTilesBack,
		kTimerGlobalBack,
		kTimerSupport,
		kTimerSupportBack,
		kTimerAccel,
		kTimerInterpSmooth,
		kTimerConf,
		kTimerCount
	};

	struct Slot
	{
		std::array<pl_tex, PL_MAX_PLANES> planes{};
		std::array<pl_tex, kLevels> pyramid{};
		double pts = 0.0;
		bool valid = false;
	};

	struct Layout
	{
		int num_planes = 0;
		int luma_plane = -1;
		int luma_component = 0;
		std::array<int, PL_MAX_PLANES> w{}, h{};
		std::array<pl_fmt, PL_MAX_PLANES> fmt{};
		bool operator==(const Layout &o) const;
		bool operator!=(const Layout &o) const { return !(*this == o); }
	};

	void warn(const char *msg) const;
	bool ensureResources(const struct pl_frame *src);
	void releaseResources();
	void resetState();

	bool lumaPass(pl_tex src, int component, float scale, pl_tex dst);
	bool downPass(pl_tex src, pl_tex dst, int level);
	bool copyPass(pl_tex src, pl_tex dst, int plane);
	bool diffPass(pl_tex a, pl_tex b, pl_tex dst);
	bool searchPass(pl_tex cur, pl_tex prev, pl_tex parent, pl_tex dst, int level, int radius, int patch, bool subpixel,
	                pl_tex temporal = nullptr, float temporal_ratio = 1.0f);
	bool smoothPass(pl_tex src, pl_tex dst, int timer = kTimerSmooth);
	bool checkPass(pl_tex fwd, pl_tex bwd, pl_tex dst, int timer = kTimerCheck);
	bool fillPass(pl_tex src, pl_tex dst, int timer = kTimerFill);
	bool interpPass(float t, pl_tex dst);
	bool accelPass(float t, pl_tex mv, pl_tex dst);
	bool blendPass(pl_tex prev, pl_tex cur, pl_tex mv, pl_tex dst, float fx, float fy, float t, bool bicubic, int plane);
	bool projectPass(pl_tex mv, pl_tex dst, float factor);
	bool warpPass(pl_tex src, pl_tex mv, pl_tex dst, float fx, float fy, bool bicubic, int plane);
	void collectTimers();

	bool readDiffMetric(float *metric);
	void recordFailure();
	bool estimateMotion();
	bool estimateBackward();
	void finishOutput(const struct pl_frame *src, pl_tex out_tex[PL_MAX_PLANES], struct pl_frame *out) const;
	bool generate(const struct pl_frame *src, float factor, pl_tex out_tex[PL_MAX_PLANES], struct pl_frame *out);

	std::function<void(const char *)> logger;
	pl_log log = nullptr;
	pl_gpu gpu = nullptr;
	pl_dispatch dispatch = nullptr;

	pl_fmt fmt_r = nullptr;     // single channel float, filterable
	pl_fmt fmt_mv = nullptr;    // four channel float, filterable
	pl_fmt fmt_diff = nullptr;  // single channel, host readable
	int diff_bytes = 0;

	Layout layout;
	std::array<Slot, kSlots> history;
	std::array<pl_tex, kLevels> mv_level{};
	std::array<pl_tex, kLevels> mvb_level{}; // backward motion (previous -> current)
	pl_tex mv_final = nullptr;
	pl_tex mvb_final = nullptr;     // backward motion at full detail, for interpolation
	pl_tex tile_tex = nullptr;      // dominant motion per tile (16x9)
	pl_tex support_tex = nullptr;   // support of each tile's vector (16x9)
	pl_tex global_tex = nullptr;    // dominant motion of the whole picture (1x1)
	pl_tex global_back_tex = nullptr; // the same for the backward field
	pl_tex path_tex = nullptr;      // interpolation: offset into the previous frame
	pl_tex conf_tex = nullptr;      // interpolation: cross-fade weight per region (32x18)
	pl_tex mv_proj = nullptr;
	pl_tex mv_prev_final = nullptr; // motion of the previous pair of game frames
	pl_tex diff_tex = nullptr;
	std::vector<uint8_t> diff_host;

	Layout failed_layout;      // layout whose texture allocation failed
	int consecutive_failures = 0;
	bool disabled = false;

	int cur_slot = -1;
	int prev_slot = -1;
	int scratch_slot = 0;

	double last_pts = 0.0;
	bool have_last_pts = false;
	int dups_since_unique = 0;
	double cadence = 1.0;
	double frame_interval = 0.0;
	int long_gaps = 0;
	int pair_frames = 1;       // game frames between the two newest real frames
	int prev_pair_frames = 1;  // the same for the pair before
	bool motion_pending = false;
	bool motion_valid = false;
	bool backward_valid = false;
	float dup_threshold_levels = 2.0f;
	bool show_marker = false;
	float damping = 1.0f;
	bool prev_motion_valid = false;
	bool have_motion_pts = false;
	double motion_cur_pts = 0.0;

	std::atomic<uint64_t> stat_frames{0};
	std::atomic<uint64_t> stat_unique{0};
	std::atomic<uint64_t> stat_dups{0};
	std::atomic<uint64_t> stat_generated{0};
	std::atomic<double> stat_cadence{1.0};
	std::atomic<double> stat_interval{0.0};
	std::atomic<bool> stat_active{false};
	std::atomic<float> stat_metric{0.0f};
	std::atomic<float> stat_ms{0.0f};
	std::atomic<uint64_t> stat_gpu_analysis_ns{0};
	std::atomic<uint64_t> stat_gpu_generate_ns{0};
	std::array<pl_timer, kTimerCount> timers{};
	bool gpu_timers = false;
};

#endif // CHIAKI_FRAMEGENERATOR_H
