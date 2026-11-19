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
// The generator works on the decoded YCbCr planes, so the output goes through
// exactly the same rendering path (scaler, upscaling shaders, tone mapping) as
// the real frames.

#ifndef CHIAKI_FRAMEGENERATOR_H
#define CHIAKI_FRAMEGENERATOR_H

#include <libplacebo/dispatch.h>
#include <libplacebo/gpu.h>
#include <libplacebo/log.h>
#include <libplacebo/renderer.h>

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
	uint64_t generated = 0;     // repeated frames replaced with predictions
	double cadence = 1.0;       // average stream frames per game frame (2.0 = 30 fps game at 60 fps)
	bool active = false;        // a low-fps cadence was detected and frames are being generated
	float last_metric = 0.0f;   // last frame-difference metric (8-bit luma levels)
	float process_ms = 0.0f;    // time spent inside process() for the last frame (CPU side, includes the readback wait)
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

	// Forget all history, e.g. after a seek, a reconnect or a format change.
	void reset();

	FrameGeneratorStats stats() const;

	// Frame-difference threshold below which a frame counts as a repeat,
	// in 8-bit luma levels (mean absolute difference over a 32x32 block).
	void setDuplicateThreshold(float levels) { dup_threshold_levels = levels; }

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
		kTimerCount = kTimerWarp + PL_MAX_PLANES
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
	bool searchPass(pl_tex cur, pl_tex prev, pl_tex parent, pl_tex dst, int level, int radius, int patch, bool subpixel);
	bool smoothPass(pl_tex src, pl_tex dst);
	bool checkPass(pl_tex fwd, pl_tex bwd, pl_tex dst);
	bool fillPass(pl_tex src, pl_tex dst);
	bool projectPass(pl_tex mv, pl_tex dst, float factor);
	bool warpPass(pl_tex src, pl_tex mv, pl_tex dst, float fx, float fy, bool bicubic, int plane);
	void collectTimers();

	bool readDiffMetric(float *metric);
	bool estimateMotion();
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
	std::array<pl_tex, kLevels> mvb_level{}; // backward motion, levels 1..4
	pl_tex mv_final = nullptr;
	pl_tex mv_proj = nullptr;
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
	bool motion_pending = false;
	bool motion_valid = false;
	float dup_threshold_levels = 2.0f;

	std::atomic<uint64_t> stat_frames{0};
	std::atomic<uint64_t> stat_unique{0};
	std::atomic<uint64_t> stat_dups{0};
	std::atomic<uint64_t> stat_generated{0};
	std::atomic<double> stat_cadence{1.0};
	std::atomic<bool> stat_active{false};
	std::atomic<float> stat_metric{0.0f};
	std::atomic<float> stat_ms{0.0f};
	std::atomic<uint64_t> stat_gpu_analysis_ns{0};
	std::atomic<uint64_t> stat_gpu_generate_ns{0};
	std::array<pl_timer, kTimerCount> timers{};
	bool gpu_timers = false;
};

#endif // CHIAKI_FRAMEGENERATOR_H
