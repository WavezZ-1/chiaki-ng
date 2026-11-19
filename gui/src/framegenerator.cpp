// SPDX-License-Identifier: LicenseRef-AGPL-3.0-only-OpenSSL

#include "framegenerator.h"

#include <libplacebo/colorspace.h>
#include <libplacebo/shaders/custom.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <string>

namespace {

// Pyramid level used for repeat detection (index 1 = 1/4 of the luma resolution).
constexpr int kDiffLevel = 1;
// Repeat detection works on blocks of kDiffBlock x kDiffBlock pixels of that
// level, i.e. 32x32 pixels of the source.
constexpr int kDiffBlock = 8;

// The game must run at a fraction of the stream rate for frames to be generated:
// 1.6 .. 4.5 stream frames per game frame (roughly 13 .. 37 fps at 60 fps).
constexpr double kMinCadence = 1.6;
constexpr double kMaxCadence = 4.5;

// Never extrapolate across a gap this long (stream stalls, packet loss).
constexpr double kMaxMotionAge = 0.2;
constexpr double kMaxFrameGap = 0.25;

// Do not extrapolate all the way to the next game frame; mispredictions there
// are the most visible.
constexpr float kMaxFactor = 0.95f;

// Motion confidence: absolute match cost (mean absolute luma difference) and
// match cost relative to "no motion".
constexpr float kConfAbsLow = 0.03f;
constexpr float kConfAbsHigh = 0.08f;
constexpr float kConfRelLow = 0.45f;
constexpr float kConfRelHigh = 0.85f;

// A match is "unreliable" (e.g. newly uncovered background) above these costs.
constexpr float kBadAbs = 0.03f;
constexpr float kBadRel = 0.6f;

struct TexArg
{
	const char *name;
	pl_tex tex;
	bool linear;
};

struct VarArg
{
	struct pl_var var;
	const void *data;
};

float halfToFloat(uint16_t h)
{
	const uint32_t sign = (h >> 15) & 1u;
	const int exponent = (h >> 10) & 0x1f;
	const uint32_t mantissa = h & 0x3ffu;
	float v;
	if (exponent == 0)
		v = std::ldexp(static_cast<float>(mantissa), -24);
	else if (exponent == 31)
		v = mantissa ? NAN : INFINITY;
	else
		v = std::ldexp(static_cast<float>(mantissa | 0x400u), exponent - 25);
	return sign ? -v : v;
}

int halfSize(int v)
{
	return std::max(1, (v + 1) / 2);
}

// --- GLSL ------------------------------------------------------------------
//
// All passes are plain fragment shaders writing `color` for the pixel at
// gl_FragCoord, so they run on every libplacebo backend.

// Source luma plane -> half resolution luma (2x2 box filter). Uses texelFetch
// because decoder surfaces are not guaranteed to support linear filtering.
const char *kLumaBody = R"(
    ivec2 p = ivec2(gl_FragCoord.xy) * 2;
    ivec2 mx = textureSize(fg_src, 0) - 1;
    vec4 s = texelFetch(fg_src, min(p, mx), 0)
           + texelFetch(fg_src, min(p + ivec2(1, 0), mx), 0)
           + texelFetch(fg_src, min(p + ivec2(0, 1), mx), 0)
           + texelFetch(fg_src, min(p + ivec2(1, 1), mx), 0);
    color = vec4(dot(s, fg_mask) * (0.25 * fg_scale), 0.0, 0.0, 1.0);
)";

// Pyramid level -> next level (2x2 box filter via one bilinear tap).
const char *kDownBody = R"(
    vec2 p = floor(gl_FragCoord.xy) * 2.0 + 1.0;
    color = vec4(texture(fg_src, p * fg_src_pt).r, 0.0, 0.0, 1.0);
)";

const char *kCopyBody = R"(
    color = texelFetch(fg_src, ivec2(gl_FragCoord.xy), 0);
)";

// Mean absolute difference of two luma images over 8x8 blocks.
const char *kDiffBody = R"(
    ivec2 o = ivec2(gl_FragCoord.xy) * 8;
    ivec2 mx = textureSize(fg_a, 0) - 1;
    float acc = 0.0;
    float n = 0.0;
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            ivec2 q = o + ivec2(x, y);
            if (q.x <= mx.x && q.y <= mx.y) {
                acc += abs(texelFetch(fg_a, q, 0).r - texelFetch(fg_b, q, 0).r);
                n += 1.0;
            }
        }
    }
    color = vec4(n > 0.0 ? acc / n : 0.0, 0.0, 0.0, 1.0);
)";

// Block matching: for every pixel of `cur`, find the displacement d such that
// cur(x) ~= prev(x - d). Candidates come from the coarser level (scaled by 2)
// and its four neighbours, plus the zero vector on coarse levels; each is
// refined in a (2R+1)^2 window. Output: (dx, dy, cost of best match, cost of
// zero motion).
const char *kSearchHeader = R"(
float fg_sad(sampler2D c, sampler2D p, ivec2 pos, ivec2 d, ivec2 mx)
{
    float acc = 0.0;
    for (int y = -FG_P; y <= FG_P; y++) {
        for (int x = -FG_P; x <= FG_P; x++) {
            ivec2 q = clamp(pos + ivec2(x, y), ivec2(0), mx);
            ivec2 r = clamp(q - d, ivec2(0), mx);
            acc += abs(texelFetch(c, q, 0).r - texelFetch(p, r, 0).r);
        }
    }
    return acc * (1.0 / float((2 * FG_P + 1) * (2 * FG_P + 1)));
}
)";

const char *kSearchBody = R"(
    ivec2 pos = ivec2(gl_FragCoord.xy);
    ivec2 mx = textureSize(fg_cur, 0) - 1;
    vec2 cand[6];
    int nc = 0;
#if FG_USE_PARENT
    ivec2 pmx = textureSize(fg_parent, 0) - 1;
    ivec2 pp = clamp(pos / 2, ivec2(0), pmx);
    cand[0] = texelFetch(fg_parent, pp, 0).xy * 2.0;
    cand[1] = texelFetch(fg_parent, clamp(pp + ivec2( 1,  0), ivec2(0), pmx), 0).xy * 2.0;
    cand[2] = texelFetch(fg_parent, clamp(pp + ivec2(-1,  0), ivec2(0), pmx), 0).xy * 2.0;
    cand[3] = texelFetch(fg_parent, clamp(pp + ivec2( 0,  1), ivec2(0), pmx), 0).xy * 2.0;
    cand[4] = texelFetch(fg_parent, clamp(pp + ivec2( 0, -1), ivec2(0), pmx), 0).xy * 2.0;
    nc = 5;
#endif
#if FG_ZERO
    cand[nc] = vec2(0.0);
    nc++;
#endif
    float best = 1e9;
    vec2 bestv = vec2(0.0);
    for (int c = 0; c < nc; c++) {
        vec2 base = floor(cand[c] + 0.5);
        // Neighbouring candidates are usually identical; search each window once.
        bool seen = false;
        for (int k = 0; k < c; k++)
            seen = seen || (floor(cand[k] + 0.5) == base);
        if (seen)
            continue;
        for (int dy = -FG_R; dy <= FG_R; dy++) {
            for (int dx = -FG_R; dx <= FG_R; dx++) {
                vec2 d = base + vec2(float(dx), float(dy));
                float cost = fg_sad(fg_cur, fg_prev, pos, ivec2(d), mx) + fg_lambda * length(d);
                if (cost < best) {
                    best = cost;
                    bestv = d;
                }
            }
        }
    }
    ivec2 bi = ivec2(bestv);
    float c0 = fg_sad(fg_cur, fg_prev, pos, bi, mx);
#if FG_SUBPIXEL
    float cxm = fg_sad(fg_cur, fg_prev, pos, bi - ivec2(1, 0), mx);
    float cxp = fg_sad(fg_cur, fg_prev, pos, bi + ivec2(1, 0), mx);
    float cym = fg_sad(fg_cur, fg_prev, pos, bi - ivec2(0, 1), mx);
    float cyp = fg_sad(fg_cur, fg_prev, pos, bi + ivec2(0, 1), mx);
    float dxn = cxm - 2.0 * c0 + cxp;
    float dyn = cym - 2.0 * c0 + cyp;
    vec2 sub = vec2(dxn > 1e-5 ? 0.5 * (cxm - cxp) / dxn : 0.0,
                    dyn > 1e-5 ? 0.5 * (cym - cyp) / dyn : 0.0);
    bestv += clamp(sub, vec2(-0.5), vec2(0.5));
#endif
    float cz = fg_sad(fg_cur, fg_prev, pos, ivec2(0), mx);
    color = vec4(bestv, c0, cz);
)";

// 3x3 vector median: removes isolated outliers from the motion field.
const char *kSmoothBody = R"(
    ivec2 pos = ivec2(gl_FragCoord.xy);
    ivec2 mx = textureSize(fg_src, 0) - 1;
    vec4 v[9];
    for (int y = 0; y < 3; y++) {
        for (int x = 0; x < 3; x++)
            v[y * 3 + x] = texelFetch(fg_src, clamp(pos + ivec2(x - 1, y - 1), ivec2(0), mx), 0);
    }
    float best = 1e9;
    vec4 bestv = v[4];
    for (int i = 0; i < 9; i++) {
        float s = 0.0;
        for (int j = 0; j < 9; j++)
            s += abs(v[i].x - v[j].x) + abs(v[i].y - v[j].y);
        if (s < best) {
            best = s;
            bestv = v[i];
        }
    }
    color = bestv;
)";

// Motion field texels hold (dx, dy, match cost, zero-motion cost). A vector is
// unreliable if even the best match is poor, both in absolute terms and
// compared to assuming no motion, or if the occlusion check failed (cost 1.0).
// Unreliable vectors that were replaced by a neighbour's vector carry a cost
// of -1.0: their vector is usable, but they were not actually matched.
const char *kReliabilityHeader = R"(
bool fg_bad(vec4 m)
{
    return m.z < 0.0 || (m.z > FG_BAD_ABS && m.z > FG_BAD_REL * m.w);
}

float fg_cost(vec4 m)
{
    return m.z < 0.0 ? 2.0 : m.z;
}
)";

// Occlusion check: a pixel of the current frame is only matched if the motion
// estimated in the opposite direction (previous -> current, one level coarser)
// leads back to it. Background that was hidden in the previous frame fails this
// check even when block matching found a plausible-looking wrong match.
const char *kCheckBody = R"(
    ivec2 pos = ivec2(gl_FragCoord.xy);
    vec4 f = texelFetch(fg_fwd, pos, 0);
    ivec2 bmx = textureSize(fg_bwd, 0) - 1;
    vec2 src = (vec2(pos) + 0.5 - f.xy) * 0.5;
    vec2 b = texelFetch(fg_bwd, clamp(ivec2(floor(src)), ivec2(0), bmx), 0).xy * 2.0;
    float e = length(f.xy + b);
    bool outside = src.x < 0.0 || src.y < 0.0 || src.x > float(bmx.x + 1) || src.y > float(bmx.y + 1);
    color = (outside || e > fg_tol + 0.15 * length(f.xy)) ? vec4(f.xy, 1.0, f.w) : f;
)";

// Unreliable vectors take the vector of the nearest reliable neighbour (rings
// of 2, 4, 8 and 16 pixels) and are marked as filled (cost -1.0), so later
// passes still know they were not matched.
const char *kFillBody = R"(
    ivec2 pos = ivec2(gl_FragCoord.xy);
    ivec2 mx = textureSize(fg_mv, 0) - 1;
    vec4 c = texelFetch(fg_mv, pos, 0);
    if (!fg_bad(c)) {
        color = c;
    } else {
        ivec2 dirs[8];
        dirs[0] = ivec2(1, 0);   dirs[1] = ivec2(-1, 0);
        dirs[2] = ivec2(0, 1);   dirs[3] = ivec2(0, -1);
        dirs[4] = ivec2(1, 1);   dirs[5] = ivec2(-1, 1);
        dirs[6] = ivec2(1, -1);  dirs[7] = ivec2(-1, -1);
        vec4 best = c;
        float bestz = 1e9;
        for (int r = 1; r <= 4; r++) {
            int dist = 1 << r;
            for (int k = 0; k < 8; k++) {
                vec4 s = texelFetch(fg_mv, clamp(pos + dirs[k] * dist, ivec2(0), mx), 0);
                if (!fg_bad(s) && s.z < bestz) {
                    bestz = s.z;
                    best = s;
                }
            }
            if (bestz < 1e8)
                break;
        }
        color = bestz < 1e8 ? vec4(best.xy, -1.0, c.w) : c;
    }
)";

// Projects the motion field to the predicted frame. Backward warping needs the
// motion of the pixel that *arrives* at each output position, which differs
// from the motion at that position around moving objects. Candidate vectors
// come from the neighbourhood and from the source positions they point at; a
// candidate is consistent if the pixel it fetches really moves with it.
// Where nothing arrives, background is being uncovered behind a moving object.
// There the candidate whose source pixel was itself just uncovered wins: it
// fills the gap with the neighbouring background instead of smearing the
// object across it.
const char *kProjectBody = R"(
    ivec2 pos = ivec2(gl_FragCoord.xy);
    ivec2 mx = textureSize(fg_mv, 0) - 1;
    vec2 y = vec2(pos);
    vec4 cands[11];
    for (int j = 0; j < 3; j++) {
        for (int i = 0; i < 3; i++)
            cands[j * 3 + i] = texelFetch(fg_mv, clamp(pos + ivec2(i - 1, j - 1) * 2, ivec2(0), mx), 0);
    }
    vec4 center = cands[4];
    vec4 s1 = texelFetch(fg_mv, clamp(ivec2(floor(y - fg_f * center.xy + 0.5)), ivec2(0), mx), 0);
    cands[9] = s1;
    cands[10] = texelFetch(fg_mv, clamp(ivec2(floor(y - fg_f * s1.xy + 0.5)), ivec2(0), mx), 0);
    float best_e = 1e9;
    vec4 best = center;
    float hole_e = 1e9;
    vec4 hole = center;
    bool hole_found = false;
    vec4 low = center;
    for (int k = 0; k < 11; k++) {
        vec4 c = cands[k];
        vec4 sv = texelFetch(fg_mv, clamp(ivec2(floor(y - fg_f * c.xy + 0.5)), ivec2(0), mx), 0);
        float e = length(sv.xy - c.xy);
        if (e < best_e - 0.05 || (e < best_e + 0.05 && fg_cost(c) < fg_cost(best))) {
            best_e = min(e, best_e);
            best = c;
        }
        if (fg_bad(sv) && !fg_bad(c) && e < hole_e) {
            hole_e = e;
            hole = c;
            hole_found = true;
        }
        if (fg_cost(c) < fg_cost(low))
            low = c;
    }
    vec4 r = best_e <= fg_tol ? best : (hole_found ? hole : low);
    // Second-order prediction: compare with how the same content moved one
    // game frame earlier and continue the change in speed (braking before a
    // reversal, speeding up when a pan starts). Clamped so that it never more
    // than doubles the motion and never turns it around on a guess.
    float sp = length(r.xy);
    if (fg_damp > 0.001 && sp > 0.5) {
        vec2 q = y - fg_f * r.xy - r.xy;
        vec2 pv = texelFetch(fg_prevmv, clamp(ivec2(floor(q + 0.5)), ivec2(0), mx), 0).xy;
        vec2 v = r.xy + (fg_damp * 0.5 * fg_f) * (r.xy - pv);
        float lv = length(v);
        if (lv > 2.0 * sp)
            v *= 2.0 * sp / lv;
        if (dot(v, r.xy) < 0.0)
            v = vec2(0.0);
        r.xy = v;
    }
    color = r;
)";

// Catmull-Rom resampling with 9 bilinear taps (keeps generated frames as sharp
// as the real ones for sub-pixel shifts).
const char *kWarpHeader = R"(
vec4 fg_catmull(sampler2D t, vec2 uv, vec2 size)
{
    vec2 sp = uv * size;
    vec2 t1 = floor(sp - 0.5) + 0.5;
    vec2 f = sp - t1;
    vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    vec2 w12 = w1 + w2;
    vec2 t0 = (t1 - 1.0) / size;
    vec2 t3 = (t1 + 2.0) / size;
    vec2 t12 = (t1 + w2 / w12) / size;
    vec4 r = vec4(0.0);
    r += texture(t, vec2(t0.x,  t0.y))  * (w0.x  * w0.y);
    r += texture(t, vec2(t12.x, t0.y))  * (w12.x * w0.y);
    r += texture(t, vec2(t3.x,  t0.y))  * (w3.x  * w0.y);
    r += texture(t, vec2(t0.x,  t12.y)) * (w0.x  * w12.y);
    r += texture(t, vec2(t12.x, t12.y)) * (w12.x * w12.y);
    r += texture(t, vec2(t3.x,  t12.y)) * (w3.x  * w12.y);
    r += texture(t, vec2(t0.x,  t3.y))  * (w0.x  * w3.y);
    r += texture(t, vec2(t12.x, t3.y))  * (w12.x * w3.y);
    r += texture(t, vec2(t3.x,  t3.y))  * (w3.x  * w3.y);
    return r;
}
)";

// Forward extrapolation: out(x) = cur(x - f * v(x)), with v faded out where
// the motion estimate is unreliable.
const char *kWarpBody = R"(
    vec2 pc = gl_FragCoord.xy;
    vec4 mv = texture(fg_mv, pc / fg_size);
    float ca = 1.0 - smoothstep(fg_conf.x, fg_conf.y, mv.z);
    float cr = 1.0 - smoothstep(fg_conf.z, fg_conf.w, mv.z / max(mv.w, 1e-4));
    float conf = mv.z < 0.0 ? 1.0 : max(ca, cr);
    vec2 d = mv.xy * fg_mv_scale * conf;
    vec2 suv = (pc - d) / fg_size;
#if FG_BICUBIC
    color = clamp(fg_catmull(fg_src, suv, fg_size), 0.0, 1.0);
#else
    color = texture(fg_src, suv);
#endif
    // Optional marker (top-left square) to make predicted frames visible.
    if (pc.x < fg_marker.x && pc.y < fg_marker.y)
        color = vec4(fg_marker.z);
)";

} // namespace

void FrameGenerator::warn(const char *msg) const
{
	if (logger)
		logger(msg);
	else
		std::fprintf(stderr, "FrameGenerator: %s\n", msg);
}

bool FrameGenerator::Layout::operator==(const Layout &o) const
{
	if (num_planes != o.num_planes || luma_plane != o.luma_plane || luma_component != o.luma_component)
		return false;
	for (int i = 0; i < num_planes; i++) {
		if (w[i] != o.w[i] || h[i] != o.h[i] || fmt[i] != o.fmt[i])
			return false;
	}
	return true;
}

FrameGenerator::FrameGenerator(pl_log log_, pl_gpu gpu_)
	: log(log_), gpu(gpu_)
{
	if (!gpu)
		return;
	if (gpu->glsl.version > 0 && gpu->glsl.version < 130)
		return;

	const enum pl_fmt_caps tex_caps = static_cast<enum pl_fmt_caps>(
		PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_LINEAR);
	fmt_r = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 0, tex_caps);
	if (!fmt_r)
		fmt_r = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 32, 0, tex_caps);
	if (!fmt_r)
		fmt_r = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 16, 0, tex_caps);
	fmt_mv = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 0, tex_caps);
	if (!fmt_mv)
		fmt_mv = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 0, tex_caps);

	const enum pl_fmt_caps diff_caps = static_cast<enum pl_fmt_caps>(
		PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_HOST_READABLE);
	fmt_diff = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 32, 32, diff_caps);
	diff_bytes = 4;
	if (!fmt_diff) {
		fmt_diff = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 16, diff_caps);
		diff_bytes = 2;
	}

	if (!fmt_r || !fmt_mv || !fmt_diff) {
		warn("required texture formats unavailable, frame generation disabled");
		return;
	}

	dispatch = pl_dispatch_create(log, gpu);
	for (auto &t : timers)
		t = pl_timer_create(gpu);
	gpu_timers = timers[0] != nullptr;
}

FrameGenerator::~FrameGenerator()
{
	releaseResources();
	if (gpu) {
		for (auto &t : timers)
			pl_timer_destroy(gpu, &t);
	}
	if (dispatch)
		pl_dispatch_destroy(&dispatch);
}

void FrameGenerator::releaseResources()
{
	if (!gpu)
		return;
	for (auto &slot : history) {
		for (auto &t : slot.planes)
			pl_tex_destroy(gpu, &t);
		for (auto &t : slot.pyramid)
			pl_tex_destroy(gpu, &t);
		slot.valid = false;
	}
	for (auto &t : mv_level)
		pl_tex_destroy(gpu, &t);
	for (auto &t : mvb_level)
		pl_tex_destroy(gpu, &t);
	pl_tex_destroy(gpu, &mv_final);
	pl_tex_destroy(gpu, &mv_prev_final);
	pl_tex_destroy(gpu, &mv_proj);
	pl_tex_destroy(gpu, &diff_tex);
	layout = Layout();
}

void FrameGenerator::resetState()
{
	for (auto &slot : history)
		slot.valid = false;
	cur_slot = -1;
	prev_slot = -1;
	scratch_slot = 0;
	have_last_pts = false;
	last_pts = 0.0;
	dups_since_unique = 0;
	cadence = 1.0;
	frame_interval = 0.0;
	motion_pending = false;
	motion_valid = false;
	prev_motion_valid = false;
	have_motion_pts = false;
	stat_cadence.store(cadence);
	stat_interval.store(0.0);
	stat_active.store(false);
}

void FrameGenerator::reset()
{
	resetState();
}

FrameGeneratorStats FrameGenerator::stats() const
{
	FrameGeneratorStats s;
	s.frames = stat_frames.load();
	s.unique = stat_unique.load();
	s.duplicates = stat_dups.load();
	s.generated = stat_generated.load();
	s.cadence = stat_cadence.load();
	s.frame_interval = stat_interval.load();
	s.active = stat_active.load();
	s.last_metric = stat_metric.load();
	s.process_ms = stat_ms.load();
	s.gpu_timers = gpu_timers;
	s.gpu_analysis_ns = stat_gpu_analysis_ns.load();
	s.gpu_generate_ns = stat_gpu_generate_ns.load();
	return s;
}

bool FrameGenerator::ensureResources(const struct pl_frame *src)
{
	Layout nl;
	nl.num_planes = src->num_planes;
	for (int i = 0; i < src->num_planes; i++) {
		const struct pl_plane &plane = src->planes[i];
		if (!plane.texture)
			return false;
		nl.w[i] = plane.texture->params.w;
		nl.h[i] = plane.texture->params.h;
		nl.fmt[i] = plane.texture->params.format;
		for (int c = 0; c < plane.components && c < 4; c++) {
			if (plane.component_mapping[c] == PL_CHANNEL_Y && nl.luma_plane < 0) {
				nl.luma_plane = i;
				nl.luma_component = c;
			}
		}
	}
	if (nl.luma_plane < 0)
		return false;

	if (nl == layout && diff_tex)
		return true;
	if (nl == failed_layout)
		return false; // already failed for this video format, don't retry every frame

	releaseResources();
	resetState();

	const enum pl_fmt_caps copy_caps = static_cast<enum pl_fmt_caps>(
		PL_FMT_CAP_SAMPLEABLE | PL_FMT_CAP_RENDERABLE | PL_FMT_CAP_LINEAR);
	std::array<pl_fmt, PL_MAX_PLANES> copy_fmt{};
	for (int i = 0; i < nl.num_planes; i++) {
		pl_fmt f = nl.fmt[i];
		if ((f->caps & copy_caps) != copy_caps || f->type == PL_FMT_UINT || f->type == PL_FMT_SINT) {
			const int depth = f->component_depth[0] > 8 ? 16 : 8;
			pl_fmt alt = pl_find_fmt(gpu, PL_FMT_UNORM, f->num_components, depth, 0, copy_caps);
			if (!alt)
				return false;
			f = alt;
		}
		copy_fmt[i] = f;
	}

	auto make = [this](int w, int h, pl_fmt fmt, bool readable) -> pl_tex {
		struct pl_tex_params params;
		std::memset(&params, 0, sizeof(params));
		params.w = w;
		params.h = h;
		params.format = fmt;
		params.sampleable = (fmt->caps & PL_FMT_CAP_SAMPLEABLE) != 0;
		params.renderable = true;
		params.host_readable = readable;
		return pl_tex_create(gpu, &params);
	};

	std::array<int, kLevels> lw{}, lh{};
	int w = nl.w[nl.luma_plane];
	int h = nl.h[nl.luma_plane];
	for (int l = 0; l < kLevels; l++) {
		w = halfSize(w);
		h = halfSize(h);
		lw[l] = w;
		lh[l] = h;
	}

	bool ok = true;
	for (auto &slot : history) {
		for (int i = 0; i < nl.num_planes; i++)
			ok = ok && (slot.planes[i] = make(nl.w[i], nl.h[i], copy_fmt[i], false));
		for (int l = 0; l < kLevels; l++)
			ok = ok && (slot.pyramid[l] = make(lw[l], lh[l], fmt_r, false));
	}
	for (int l = 0; l < kLevels; l++)
		ok = ok && (mv_level[l] = make(lw[l], lh[l], fmt_mv, false));
	for (int l = 1; l < kLevels; l++)
		ok = ok && (mvb_level[l] = make(lw[l], lh[l], fmt_mv, false));
	ok = ok && (mv_final = make(lw[0], lh[0], fmt_mv, false));
	ok = ok && (mv_proj = make(lw[0], lh[0], fmt_mv, false));
	ok = ok && (mv_prev_final = make(lw[0], lh[0], fmt_mv, false));
	const int dw = (lw[kDiffLevel] + kDiffBlock - 1) / kDiffBlock;
	const int dh = (lh[kDiffLevel] + kDiffBlock - 1) / kDiffBlock;
	ok = ok && (diff_tex = make(dw, dh, fmt_diff, true));
	if (!ok) {
		warn("failed to allocate textures, frame generation paused for this video format");
		releaseResources();
		failed_layout = nl;
		return false;
	}
	diff_host.assign(static_cast<size_t>(dw) * dh * diff_bytes, 0);
	layout = nl;
	return true;
}

static bool runPass(pl_dispatch dp, const char *description, const std::string &header, const char *body,
                    std::initializer_list<TexArg> texs, std::initializer_list<VarArg> vars, pl_tex target,
                    pl_timer timer)
{
	std::vector<struct pl_shader_desc> descs;
	descs.reserve(texs.size());
	for (const TexArg &t : texs) {
		struct pl_shader_desc d;
		std::memset(&d, 0, sizeof(d));
		d.desc.name = t.name;
		d.desc.type = PL_DESC_SAMPLED_TEX;
		d.binding.object = t.tex;
		d.binding.address_mode = PL_TEX_ADDRESS_CLAMP;
		d.binding.sample_mode = t.linear ? PL_TEX_SAMPLE_LINEAR : PL_TEX_SAMPLE_NEAREST;
		descs.push_back(d);
	}
	std::vector<struct pl_shader_var> svars;
	svars.reserve(vars.size());
	for (const VarArg &v : vars) {
		struct pl_shader_var sv;
		std::memset(&sv, 0, sizeof(sv));
		sv.var = v.var;
		sv.data = v.data;
		sv.dynamic = true;
		svars.push_back(sv);
	}

	struct pl_custom_shader cs;
	std::memset(&cs, 0, sizeof(cs));
	cs.description = description;
	cs.header = header.empty() ? nullptr : header.c_str();
	cs.body = body;
	cs.input = PL_SHADER_SIG_NONE;
	cs.output = PL_SHADER_SIG_COLOR;
	cs.descriptors = descs.data();
	cs.num_descriptors = static_cast<int>(descs.size());
	cs.variables = svars.data();
	cs.num_variables = static_cast<int>(svars.size());

	pl_shader sh = pl_dispatch_begin(dp);
	if (!pl_shader_custom(sh, &cs)) {
		pl_dispatch_abort(dp, &sh);
		return false;
	}
	struct pl_dispatch_params params;
	std::memset(&params, 0, sizeof(params));
	params.shader = &sh;
	params.target = target;
	params.timer = timer;
	return pl_dispatch_finish(dp, &params);
}

bool FrameGenerator::lumaPass(pl_tex src, int component, float scale, pl_tex dst)
{
	float mask[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
	mask[std::min(std::max(component, 0), 3)] = 1.0f;
	return runPass(dispatch, "framegen luma", std::string(), kLumaBody,
	               { { "fg_src", src, false } },
	               { { pl_var_vec4("fg_mask"), mask }, { pl_var_float("fg_scale"), &scale } },
	               dst, timers[kTimerLuma]);
}

bool FrameGenerator::downPass(pl_tex src, pl_tex dst, int level)
{
	float pt[2] = { 1.0f / src->params.w, 1.0f / src->params.h };
	return runPass(dispatch, "framegen down", std::string(), kDownBody,
	               { { "fg_src", src, true } },
	               { { pl_var_vec2("fg_src_pt"), pt } },
	               dst, timers[kTimerDown + level]);
}

bool FrameGenerator::copyPass(pl_tex src, pl_tex dst, int plane)
{
	return runPass(dispatch, "framegen copy", std::string(), kCopyBody,
	               { { "fg_src", src, false } }, {}, dst, timers[kTimerCopy + plane]);
}

bool FrameGenerator::diffPass(pl_tex a, pl_tex b, pl_tex dst)
{
	return runPass(dispatch, "framegen diff", std::string(), kDiffBody,
	               { { "fg_a", a, false }, { "fg_b", b, false } }, {}, dst, timers[kTimerDiff]);
}

bool FrameGenerator::searchPass(pl_tex cur, pl_tex prev, pl_tex parent, pl_tex dst, int level, int radius, int patch, bool subpixel)
{
	// The zero vector is offered on the coarse levels, where it lets static
	// overlays (HUD, subtitles) snap to "no motion"; finer levels inherit it.
	const bool zero = !parent || level >= 2;
	char defs[192];
	std::snprintf(defs, sizeof(defs), "#define FG_R %d\n#define FG_P %d\n#define FG_USE_PARENT %d\n#define FG_SUBPIXEL %d\n#define FG_ZERO %d\n",
	              radius, patch, parent ? 1 : 0, subpixel ? 1 : 0, zero ? 1 : 0);
	const std::string header = std::string(defs) + kSearchHeader;

	// Small bias towards short vectors so that flat or repetitive areas prefer
	// the smallest consistent motion. Scaled so it stays a tie-breaker at
	// every level (vectors double in length from one level to the next).
	float lambda = 0.0005f * std::pow(0.5f, static_cast<float>(kLevels - 1 - level));
	pl_timer timer = timers[(dst == mvb_level[level] ? kTimerSearchBack : kTimerSearch) + level];

	if (parent) {
		return runPass(dispatch, "framegen search", header, kSearchBody,
		               { { "fg_cur", cur, false }, { "fg_prev", prev, false }, { "fg_parent", parent, false } },
		               { { pl_var_float("fg_lambda"), &lambda } },
		               dst, timer);
	}
	return runPass(dispatch, "framegen search", header, kSearchBody,
	               { { "fg_cur", cur, false }, { "fg_prev", prev, false } },
	               { { pl_var_float("fg_lambda"), &lambda } },
	               dst, timer);
}

bool FrameGenerator::smoothPass(pl_tex src, pl_tex dst)
{
	return runPass(dispatch, "framegen smooth", std::string(), kSmoothBody,
	               { { "fg_src", src, false } }, {}, dst, timers[kTimerSmooth]);
}

static std::string reliabilityHeader()
{
	char defs[96];
	std::snprintf(defs, sizeof(defs), "#define FG_BAD_ABS %.4f\n#define FG_BAD_REL %.4f\n", kBadAbs, kBadRel);
	return std::string(defs) + kReliabilityHeader;
}

bool FrameGenerator::checkPass(pl_tex fwd, pl_tex bwd, pl_tex dst)
{
	float tol = 2.5f;
	return runPass(dispatch, "framegen check", std::string(), kCheckBody,
	               { { "fg_fwd", fwd, false }, { "fg_bwd", bwd, false } },
	               { { pl_var_float("fg_tol"), &tol } },
	               dst, timers[kTimerCheck]);
}

bool FrameGenerator::fillPass(pl_tex src, pl_tex dst)
{
	return runPass(dispatch, "framegen fill", reliabilityHeader(), kFillBody,
	               { { "fg_mv", src, false } }, {}, dst, timers[kTimerFill]);
}

bool FrameGenerator::projectPass(pl_tex mv, pl_tex dst, float factor)
{
	float tol = 1.0f;
	// Strength of the second-order (acceleration) term, 0 = constant speed.
	float damp = prev_motion_valid ? damping : 0.0f;
	return runPass(dispatch, "framegen project", reliabilityHeader(), kProjectBody,
	               { { "fg_mv", mv, false }, { "fg_prevmv", mv_prev_final, false } },
	               { { pl_var_float("fg_f"), &factor }, { pl_var_float("fg_tol"), &tol },
	                 { pl_var_float("fg_damp"), &damp } },
	               dst, timers[kTimerProject]);
}

bool FrameGenerator::warpPass(pl_tex src, pl_tex mv, pl_tex dst, float fx, float fy, bool bicubic, int plane)
{
	const std::string header = std::string(bicubic ? "#define FG_BICUBIC 1\n" : "#define FG_BICUBIC 0\n") + kWarpHeader;
	float size[2] = { static_cast<float>(src->params.w), static_cast<float>(src->params.h) };
	float scale[2] = { fx, fy };
	float conf[4] = { kConfAbsLow, kConfAbsHigh, kConfRelLow, kConfRelHigh };
	// Marker: a 1/24-of-the-height square, white in luma and neutral in chroma.
	float marker[3] = { 0.0f, 0.0f, 0.0f };
	if (show_marker) {
		const float box = std::max(8.0f, layout.h[layout.luma_plane] / 24.0f);
		marker[0] = box * src->params.w / layout.w[layout.luma_plane];
		marker[1] = box * src->params.h / layout.h[layout.luma_plane];
		marker[2] = bicubic ? 0.92f : 0.5f;
	}
	return runPass(dispatch, "framegen warp", header, kWarpBody,
	               { { "fg_src", src, true }, { "fg_mv", mv, false } },
	               { { pl_var_vec2("fg_size"), size }, { pl_var_vec2("fg_mv_scale"), scale }, { pl_var_vec4("fg_conf"), conf },
	                 { pl_var_vec3("fg_marker"), marker } },
	               dst, timers[kTimerWarp + plane]);
}

void FrameGenerator::collectTimers()
{
	if (!gpu_timers)
		return;
	uint64_t analysis = 0, generation = 0;
	for (int i = 0; i < kTimerCount; i++) {
		uint64_t ns;
		while ((ns = pl_timer_query(gpu, timers[i])) != 0) {
			if (i < kTimerSearch)
				analysis += ns;
			else
				generation += ns;
		}
	}
	stat_gpu_analysis_ns.fetch_add(analysis);
	stat_gpu_generate_ns.fetch_add(generation);
}

bool FrameGenerator::readDiffMetric(float *metric)
{
	struct pl_tex_transfer_params tp;
	std::memset(&tp, 0, sizeof(tp));
	tp.tex = diff_tex;
	tp.ptr = diff_host.data();
	tp.row_pitch = static_cast<size_t>(diff_tex->params.w) * diff_bytes;
	if (!pl_tex_download(gpu, &tp))
		return false;

	const size_t count = static_cast<size_t>(diff_tex->params.w) * diff_tex->params.h;
	float mx = 0.0f;
	for (size_t i = 0; i < count; i++) {
		float v;
		if (diff_bytes == 4) {
			std::memcpy(&v, diff_host.data() + i * 4, 4);
		} else {
			uint16_t hv;
			std::memcpy(&hv, diff_host.data() + i * 2, 2);
			v = halfToFloat(hv);
		}
		if (std::isfinite(v))
			mx = std::max(mx, v);
	}
	*metric = mx;
	return true;
}

bool FrameGenerator::estimateMotion()
{
	const Slot &c = history[cur_slot];
	const Slot &p = history[prev_slot];
	// Current -> previous, coarse to fine, down to level 0 (half resolution).
	for (int l = kLevels - 1; l >= 0; l--) {
		const bool top = l == kLevels - 1;
		const int radius = top ? 6 : 1;   // +-192 px at the top level, +-30 px from refinements
		const int patch = l >= 3 ? 2 : 1; // 5x5 on the two coarsest levels, 3x3 below
		if (!searchPass(c.pyramid[l], p.pyramid[l], top ? nullptr : mv_level[l + 1], mv_level[l], l, radius, patch, l == 0))
			return false;
	}
	// Previous -> current, down to level 1 only; used for the occlusion check.
	for (int l = kLevels - 1; l >= 1; l--) {
		const bool top = l == kLevels - 1;
		if (!searchPass(p.pyramid[l], c.pyramid[l], top ? nullptr : mvb_level[l + 1], mvb_level[l], l, top ? 6 : 1, l >= 3 ? 2 : 1, false))
			return false;
	}
	// Keep the previous field for motion damping. It describes the previous
	// pair of game frames only if it was estimated for exactly that pair.
	std::swap(mv_final, mv_prev_final);
	prev_motion_valid = have_motion_pts && std::fabs(motion_cur_pts - p.pts) < 1e-6;
	// mv_proj is only needed later in generate(), so it serves as scratch here.
	const bool ok = checkPass(mv_level[0], mvb_level[1], mv_proj)
	             && fillPass(mv_proj, mv_level[0])
	             && smoothPass(mv_level[0], mv_final);
	have_motion_pts = ok;
	motion_cur_pts = c.pts;
	if (!ok)
		prev_motion_valid = false;
	return ok;
}

bool FrameGenerator::generate(const struct pl_frame *src, float factor, pl_tex out_tex[PL_MAX_PLANES], struct pl_frame *out)
{
	const Slot &c = history[cur_slot];
	if (!projectPass(mv_final, mv_proj, factor))
		return false;
	const float lw = static_cast<float>(layout.w[layout.luma_plane]);
	const float lh = static_cast<float>(layout.h[layout.luma_plane]);
	for (int i = 0; i < layout.num_planes; i++) {
		struct pl_tex_params params;
		std::memset(&params, 0, sizeof(params));
		params.w = layout.w[i];
		params.h = layout.h[i];
		params.format = c.planes[i]->params.format;
		params.sampleable = true;
		params.renderable = true;
		if (!pl_tex_recreate(gpu, &out_tex[i], &params))
			return false;
		// Motion vectors are in pyramid level 1 pixels (half luma resolution).
		const float fx = factor * 2.0f * (layout.w[i] / lw);
		const float fy = factor * 2.0f * (layout.h[i] / lh);
		if (!warpPass(c.planes[i], mv_proj, out_tex[i], fx, fy, i == layout.luma_plane, i))
			return false;
	}

	*out = *src;
	for (int i = 0; i < layout.num_planes; i++)
		out->planes[i].texture = out_tex[i];
	out->acquire = nullptr;
	out->release = nullptr;
	out->user_data = nullptr;
	out->prev = nullptr;
	out->next = nullptr;
	out->icc = nullptr;
	std::memset(&out->profile, 0, sizeof(out->profile));
	out->lut = nullptr;
	out->overlays = nullptr;
	out->num_overlays = 0;
	std::memset(&out->film_grain, 0, sizeof(out->film_grain));
	out->repr.dovi = nullptr;
	return true;
}

void FrameGenerator::recordFailure()
{
	resetState();
	if (++consecutive_failures >= 3) {
		warn("GPU passes keep failing, frame generation disabled");
		disabled = true;
	} else {
		warn("GPU pass failed, resetting history");
	}
}

bool FrameGenerator::analyze(const struct pl_frame *src, double pts, bool *is_new)
{
	const auto t_begin = std::chrono::steady_clock::now();
	*is_new = true;
	if (!valid() || !src)
		return false;
	if (src->num_planes <= 0 || src->num_planes > PL_MAX_PLANES)
		return false;
	if (src->repr.sys == PL_COLOR_SYSTEM_RGB || src->repr.sys == PL_COLOR_SYSTEM_XYZ)
		return false;

	if (have_last_pts) {
		if (std::fabs(pts - last_pts) < 1e-6)
			return false; // the same frame submitted again
		if (pts < last_pts || pts - last_pts > kMaxFrameGap)
			resetState();
	}

	if (!ensureResources(src))
		return false;

	stat_frames.fetch_add(1);
	collectTimers();

	struct pl_frame frame = *src;
	if (frame.acquire && !frame.acquire(gpu, &frame)) {
		if (frame.release)
			frame.release(gpu, &frame);
		return false;
	}

	Slot &in = history[scratch_slot];
	struct pl_color_repr repr = frame.repr;
	const float scale = pl_color_repr_normalize(&repr);
	const bool have_cur = cur_slot >= 0 && history[cur_slot].valid;

	bool ok = lumaPass(frame.planes[layout.luma_plane].texture, layout.luma_component, scale, in.pyramid[0]);
	for (int l = 1; ok && l <= kDiffLevel; l++)
		ok = downPass(in.pyramid[l - 1], in.pyramid[l], l);

	float metric = 1.0f;
	if (ok && have_cur)
		ok = diffPass(in.pyramid[kDiffLevel], history[cur_slot].pyramid[kDiffLevel], diff_tex) && readDiffMetric(&metric);

	const float levels = metric * 255.0f;
	const bool cadence_locked = cadence >= kMinCadence && cadence <= kMaxCadence;
	const int expected_dups = std::max(1, static_cast<int>(std::lround(cadence)) - 1);
	float threshold = dup_threshold_levels;
	if (cadence_locked)
		threshold *= dups_since_unique < expected_dups ? 1.5f : 0.75f;
	else if (cadence < 1.3)
		threshold *= 0.25f; // stream without repeats: only near-identical frames count as repeats
	const bool is_dup = ok && have_cur && levels < threshold;

	if (ok && !is_dup) {
		for (int i = 0; ok && i < layout.num_planes; i++)
			ok = copyPass(frame.planes[i].texture, in.planes[i], i);
		for (int l = kDiffLevel + 1; ok && l < kLevels; l++)
			ok = downPass(in.pyramid[l - 1], in.pyramid[l], l);
		if (ok) {
			in.valid = true;
			in.pts = pts;
			if (have_cur) {
				const int interval = dups_since_unique + 1;
				cadence = std::min(8.0, std::max(1.0, cadence * 0.85 + interval * 0.15));
				const double dt = pts - history[cur_slot].pts;
				if (dt > 0.0 && dt < kMaxMotionAge)
					frame_interval = frame_interval > 0.0 ? frame_interval * 0.8 + dt * 0.2 : dt;
			}
			const int old_prev = prev_slot;
			prev_slot = have_cur ? cur_slot : -1;
			cur_slot = scratch_slot;
			if (old_prev >= 0 && old_prev != cur_slot && old_prev != prev_slot) {
				scratch_slot = old_prev;
			} else {
				for (int i = 0; i < kSlots; i++) {
					if (i != cur_slot && i != prev_slot) {
						scratch_slot = i;
						break;
					}
				}
			}
			history[scratch_slot].valid = false;
			dups_since_unique = 0;
			motion_pending = prev_slot >= 0 && pts - history[prev_slot].pts < kMaxMotionAge;
			motion_valid = false;
			stat_unique.fetch_add(1);
		}
	} else if (ok) {
		dups_since_unique++;
		stat_dups.fetch_add(1);
	}

	if (frame.release)
		frame.release(gpu, &frame);

	if (!ok) {
		recordFailure();
		return false;
	}
	consecutive_failures = 0;

	*is_new = !is_dup;
	last_pts = pts;
	have_last_pts = true;
	if (have_cur)
		stat_metric.store(levels);
	stat_cadence.store(cadence);
	stat_interval.store(frame_interval);
	stat_active.store(cadence >= kMinCadence && cadence <= kMaxCadence);
	const auto t_end = std::chrono::steady_clock::now();
	stat_ms.store(std::chrono::duration<float, std::milli>(t_end - t_begin).count());
	return true;
}

bool FrameGenerator::canPredict() const
{
	return valid() && cur_slot >= 0 && prev_slot >= 0 && history[cur_slot].valid && history[prev_slot].valid
	       && (motion_pending || motion_valid);
}

bool FrameGenerator::predict(const struct pl_frame *meta, float factor, pl_tex out_tex[PL_MAX_PLANES], struct pl_frame *out)
{
	if (!meta || !out_tex || !out || !canPredict())
		return false;
	if (meta->num_planes != layout.num_planes)
		return false;
	if (motion_pending) {
		motion_valid = estimateMotion();
		motion_pending = false;
		if (!motion_valid) {
			recordFailure();
			return false;
		}
	}
	if (!motion_valid)
		return false;
	factor = std::min(kMaxFactor, std::max(0.0f, factor));
	if (!generate(meta, factor, out_tex, out)) {
		recordFailure();
		return false;
	}
	stat_generated.fetch_add(1);
	return true;
}

bool FrameGenerator::process(const struct pl_frame *src, double pts, pl_tex out_tex[PL_MAX_PLANES], struct pl_frame *out)
{
	if (!out_tex || !out)
		return false;
	bool is_new = true;
	if (!analyze(src, pts, &is_new) || is_new)
		return false;
	const bool cadence_locked = cadence >= kMinCadence && cadence <= kMaxCadence;
	const int expected_dups = std::max(1, static_cast<int>(std::lround(cadence)) - 1);
	if (!cadence_locked || dups_since_unique > expected_dups)
		return false;
	return predict(src, static_cast<float>(dups_since_unique / cadence), out_tex, out);
}
