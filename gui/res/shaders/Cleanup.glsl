// Compression cleanup for chiaki-ng.
//
// Streaming at a few Mbit/s leaves small steps and speckles in the picture:
// block edges in flat areas, "mosquito" noise around edges, crawling noise in
// dark scenes. They are low in contrast compared to real edges. This filter
// averages each pixel with its 5x5 neighbourhood, but only with neighbours
// that differ from it by little (an edge-preserving bilateral filter), so
// those small steps are smoothed while real edges and strong detail stay.
// Runs on the decoded planes before anything else.

//!PARAM cleanup
//!DESC Strength (0 = off, 1 = strong)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM compare
//!DESC 1 = leave the left half untouched (before/after view)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!HOOK LUMA
//!HOOK CHROMA
//!BIND HOOKED
//!WHEN cleanup 0 >
//!DESC chiaki-ng compression cleanup
vec4 hook()
{
    vec4 center = HOOKED_texOff(vec2(0.0, 0.0));
    if (compare > 0.5 && HOOKED_pos.x < 0.5)
        return center;
    // Differences below about 1.5 % (light) to 4.5 % (strong) of the full
    // range count as compression noise.
    float sigma_r = mix(0.015, 0.045, cleanup);
    float inv_r = 1.0 / (2.0 * sigma_r * sigma_r);
    vec4 sum = vec4(0.0);
    vec4 wsum = vec4(0.0);
    for (int y = -2; y <= 2; y++) {
        for (int x = -2; x <= 2; x++) {
            vec4 s = HOOKED_texOff(vec2(float(x), float(y)));
            float ws = exp(-float(x * x + y * y) / 4.5);
            vec4 d = s - center;
            vec4 w = ws * exp(-d * d * inv_r);
            sum += s * w;
            wsum += w;
        }
    }
    vec4 smooth_v = sum / max(wsum, vec4(1e-6));
    return mix(center, smooth_v, mix(0.5, 1.0, cleanup));
}
