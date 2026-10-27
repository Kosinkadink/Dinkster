# InfiniteTalk requires an initial image or previous frames

At ComfyUI reference `bc5f880dbdd43e426077f42ccec53bb577ffee9f`,
`WanInfiniteTalkToVideo` declares both `start_image` and `previous_frames`
optional, but execution without either raises `UnboundLocalError`.

`comfy_extras/nodes_wan.py` initializes `concat_latent_image` only when
`start_image` is present. Without previous frames, it subsequently reads
`concat_latent_image[:, :, :1]` for the motion-context wrapper.

Reproduction: execute the single-speaker node with valid model, model patch,
VAE, conditioning and audio encoder output, omitting both image inputs. A
CPU comparison executing the unchanged reference body reproduces the error;
the native body preserves the failure rather than inventing motion context.

A source fix would need to define initialization without either image, or
reject that combination explicitly. Providing an initial image or sufficient
previous frames avoids this failure.

Status: reproduced at the pinned reference; not reported upstream.
