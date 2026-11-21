// Contrast adaptive sharpening for chiaki-ng.
//
// Sharpens each pixel against its four direct neighbours. The amount adapts to
// the local contrast: flat areas and fine detail get the full effect, edges
// that are already hard get less, so there are no bright halos and no
// amplified noise in dark areas. Runs after scaling, at the final resolution.

//!PARAM sharpness
//!DESC Strength (0 = light, 1 = strong)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.5

//!PARAM compare
//!DESC 1 = leave the left half untouched (before/after view)
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!HOOK SCALED
//!BIND HOOKED
//!DESC chiaki-ng sharpening
vec4 hook()
{
    vec4 center = HOOKED_texOff(vec2(0.0, 0.0));
    if (compare > 0.5 && HOOKED_pos.x < 0.5)
        return center;
    vec3 e = center.rgb;
    vec3 b = HOOKED_texOff(vec2(0.0, -1.0)).rgb;
    vec3 d = HOOKED_texOff(vec2(-1.0, 0.0)).rgb;
    vec3 f = HOOKED_texOff(vec2(1.0, 0.0)).rgb;
    vec3 h = HOOKED_texOff(vec2(0.0, 1.0)).rgb;
    vec3 a = HOOKED_texOff(vec2(-1.0, -1.0)).rgb;
    vec3 c = HOOKED_texOff(vec2(1.0, -1.0)).rgb;
    vec3 g = HOOKED_texOff(vec2(-1.0, 1.0)).rgb;
    vec3 i = HOOKED_texOff(vec2(1.0, 1.0)).rgb;

    // Soft minimum and maximum: the cross plus the full 3x3 neighbourhood,
    // giving a range of 0..2.
    vec3 cross_min = min(min(min(b, d), min(e, f)), h);
    vec3 cross_max = max(max(max(b, d), max(e, f)), h);
    vec3 lo = cross_min + min(cross_min, min(min(a, c), min(g, i)));
    vec3 hi = cross_max + max(cross_max, max(max(a, c), max(g, i)));

    // Headroom to black and to white, relative to the local maximum.
    vec3 amount = sqrt(clamp(min(lo, 2.0 - hi) / max(hi, vec3(1e-5)), 0.0, 1.0));

    // Negative weight for the four neighbours: -1/8 (light) to -1/5 (strong).
    vec3 w = amount * (-1.0 / mix(8.0, 5.0, sharpness));
    vec3 result = ((b + d + f + h) * w + e) / (1.0 + 4.0 * w);
    return vec4(max(result, vec3(0.0)), center.a);
}
