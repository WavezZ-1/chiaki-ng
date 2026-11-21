// Colour and tone for chiaki-ng.
//
// Runs on the RGB picture before scaling. All controls are 0 = unchanged.
//   vibrance     raises the saturation of dull colours more than that of
//                colours that are already strong (no neon skin)
//   saturation   all colours, -1 = grey, 1 = double
//   contrast     a soft S-curve that keeps black and white where they are
//   brightness   lifts or lowers the whole picture
//   shadows      brightens dark areas without greying out black
//   temperature  -1 = cooler (blue), 1 = warmer (orange)

//!PARAM vibrance
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM saturation
//!TYPE float
//!MINIMUM -1.0
//!MAXIMUM 1.0
0.0

//!PARAM contrast
//!TYPE float
//!MINIMUM -1.0
//!MAXIMUM 1.0
0.0

//!PARAM brightness
//!TYPE float
//!MINIMUM -1.0
//!MAXIMUM 1.0
0.0

//!PARAM shadows
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!PARAM temperature
//!TYPE float
//!MINIMUM -1.0
//!MAXIMUM 1.0
0.0

//!PARAM compare
//!TYPE float
//!MINIMUM 0.0
//!MAXIMUM 1.0
0.0

//!HOOK MAIN
//!BIND HOOKED
//!DESC chiaki-ng colour and tone
vec4 hook()
{
    vec4 c = HOOKED_texOff(vec2(0.0, 0.0));
    if (compare > 0.5 && HOOKED_pos.x < 0.5)
        return c;
    vec3 rgb = max(c.rgb, vec3(0.0));
    const vec3 luma_w = vec3(0.2126, 0.7152, 0.0722);

    // Temperature: shift red against blue, keep the brightness.
    vec3 tint = vec3(1.0 + 0.08 * temperature, 1.0, 1.0 - 0.08 * temperature);
    float y0 = dot(rgb, luma_w);
    rgb *= tint;
    rgb *= y0 / max(dot(rgb, luma_w), 1e-5);

    // Brightness, shadows and contrast act on the brightness; the colour
    // offsets ride along so hue and saturation stay.
    float y = dot(rgb, luma_w);
    float ny = y + 0.1 * brightness;
    ny = clamp(ny, 0.0, 1.0);
    // Shadows: a gamma lift that only acts on the dark part of the range.
    ny = mix(ny, pow(ny, 1.0 / (1.0 + 0.7 * shadows)), (1.0 - ny) * (1.0 - ny));
    ny += contrast * 0.5 * ny * (1.0 - ny) * (2.0 * ny - 1.0) * 2.0;
    rgb += ny - y;
    y = ny;

    // Saturation and vibrance.
    float sat = max(max(rgb.r, rgb.g), rgb.b) - min(min(rgb.r, rgb.g), rgb.b);
    float gain = (1.0 + saturation) * (1.0 + vibrance * 0.8 * (1.0 - clamp(sat * 2.0, 0.0, 1.0)));
    rgb = y + (rgb - y) * gain;

    c.rgb = clamp(rgb, 0.0, 1.0);
    return c;
}
