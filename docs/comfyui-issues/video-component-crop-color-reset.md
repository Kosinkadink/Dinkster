# Component video cropping resets encoding color space

Status: reproduced at ComfyUI `bc5f880dbdd43e426077f42ccec53bb577ffee9f`;
not reported upstream.

`VideoInput.as_cropped` reconstructs a cropped component video with
`VideoFromComponents(..., bit_depth=self.get_bit_depth())` but does not pass
the source color space. The constructor defaults to sRGB. HDR pixels are not
converted, so later encoding labels their unchanged values as sRGB. Empty or
full-frame rectangles return the original value and retain its color space.

CPU reproduction: construct two 6x8 float64 RGB frames with HDR PQ and depth
10, then crop x=3, y=1, width=5, height=5. The even-aligned result has exact
copied 4x4 pixels, depth 10, the original rational frame rate and complete
audio, but `get_color_space()` returns sRGB instead of HDR PQ.

An upstream correction would pass `color_space=self.get_color_space()` when
constructing the cropped components. Dinkster's native crop node preserves
the fixed-reference reset rather than changing this observable contract.
