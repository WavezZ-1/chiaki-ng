// Clarity (local contrast) for chiaki-ng.
//
// Brings out structure and depth the way a "clarity" slider in photo editors
// does: each pixel is pushed away from the average brightness of its wide
// surroundings (about 30 pixels), so shapes, textures and distance read more
// clearly without sharpening fine noise. Near black and near white the effect
// fades out, so nothing clips. Works on the brightness only, so colours do not
// shift. The wide average is computed at a quarter of the resolution.

//!PARAM clarity
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
//!BIND HOOKED
//!SAVE CLARITY_SMALL
//!WIDTH HOOKED.w 4 /
//!HEIGHT HOOKED.h 4 /
//!COMPONENTS 1
//!WHEN clarity 0 >
//!DESC chiaki-ng clarity (reduce)
vec4 hook()
{
    // Four bilinear taps, each averaging 2x2 pixels: a 4x4 box.
    float v = HOOKED_texOff(vec2(-1.0, -1.0)).x + HOOKED_texOff(vec2(1.0, -1.0)).x
            + HOOKED_texOff(vec2(-1.0, 1.0)).x + HOOKED_texOff(vec2(1.0, 1.0)).x;
    return vec4(v * 0.25, 0.0, 0.0, 0.0);
}

//!HOOK LUMA
//!BIND CLARITY_SMALL
//!SAVE CLARITY_BLUR_H
//!WIDTH CLARITY_SMALL.w
//!HEIGHT CLARITY_SMALL.h
//!COMPONENTS 1
//!WHEN clarity 0 >
//!DESC chiaki-ng clarity (blur horizontal)
vec4 hook()
{
    float sum = 0.0;
    float wsum = 0.0;
    for (int i = -6; i <= 6; i++) {
        float w = exp(-float(i * i) / 18.0);
        sum += CLARITY_SMALL_texOff(vec2(float(i), 0.0)).x * w;
        wsum += w;
    }
    return vec4(sum / wsum, 0.0, 0.0, 0.0);
}

//!HOOK LUMA
//!BIND CLARITY_BLUR_H
//!SAVE CLARITY_BLUR
//!WIDTH CLARITY_BLUR_H.w
//!HEIGHT CLARITY_BLUR_H.h
//!COMPONENTS 1
//!WHEN clarity 0 >
//!DESC chiaki-ng clarity (blur vertical)
vec4 hook()
{
    float sum = 0.0;
    float wsum = 0.0;
    for (int i = -6; i <= 6; i++) {
        float w = exp(-float(i * i) / 18.0);
        sum += CLARITY_BLUR_H_texOff(vec2(0.0, float(i))).x * w;
        wsum += w;
    }
    return vec4(sum / wsum, 0.0, 0.0, 0.0);
}

//!HOOK LUMA
//!BIND HOOKED
//!BIND CLARITY_BLUR
//!WHEN clarity 0 >
//!DESC chiaki-ng clarity (apply)
vec4 hook()
{
    vec4 c = HOOKED_texOff(vec2(0.0, 0.0));
    if (compare > 0.5 && HOOKED_pos.x < 0.5)
        return c;
    float y = c.x;
    float detail = y - CLARITY_BLUR_tex(HOOKED_pos).x;
    // Fade out towards black and white (and keep the lift small in the
    // darkest tones, where streaming noise lives).
    float guard = clamp(4.0 * y * (1.0 - y), 0.0, 1.0);
    c.x = clamp(y + clarity * 0.8 * detail * guard, 0.0, 1.0);
    return c;
}
