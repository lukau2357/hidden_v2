## Misc
Implementation of bilinear image upsampling:
```python
import numpy as np

def bilinear_resize(image, scale):
    """
    Bilinear upsampling of an image by a given scale factor.

    Args:
        image: numpy array, shape (H, W) or (H, W, C)
        scale: float > 1, upsampling factor

    Returns:
        Resized image as numpy array.
    """
    if scale <= 1:
        raise ValueError("Only upsampling supported. Scale must be > 1.")

    # Original dimensions
    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]

    # New dimensions
    new_h = int(h * scale)
    new_w = int(w * scale)

    # Create output grid
    # Values in original coordinate system
    y = np.linspace(0, h - 1, new_h)
    x = np.linspace(0, w - 1, new_w)
    x_grid, y_grid = np.meshgrid(x, y)

    # Indices of the 4 neighbors
    x0 = np.floor(x_grid).astype(int)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y0 = np.floor(y_grid).astype(int)
    y1 = np.clip(y0 + 1, 0, h - 1)

    # Fractional parts
    dx = x_grid - x0
    dy = y_grid - y0

    # Get the four neighbors
    if c == 1:
        Ia = image[y0, x0]
        Ib = image[y1, x0]
        Ic = image[y0, x1]
        Id = image[y1, x1]
    else:
        Ia = image[y0, x0, :]
        Ib = image[y1, x0, :]
        Ic = image[y0, x1, :]
        Id = image[y1, x1, :]

    # Bilinear interpolation formula
    out = (
        Ia * (1 - dx) * (1 - dy)
        + Ic * dx * (1 - dy)
        + Ib * (1 - dx) * dy
        + Id * dx * dy
    )

    return out.astype(image.dtype)
```

* Will use val2014 for training, CBA. 40k images, do an 80/20 split, or 90/10. Full training on the other hand has 81k images, out of scope for the time given for the implementation. COCO val2014 resolution stats:
```
Iteration time: 147.18680095672607 s.
Min resolution: (120, 120, 3)
Max resolution: (640, 640, 3)
```

* Try data loader with pin_memory = True initially.
* Code is not reproducible, make it reproducible!!!
* Mention the idea of augmentation sapmling, both horizontally (accross augmentations) and vertically (strength parameters for a chosen augmentation). The idea is that in initial iterations the model should with high probability onlt sample identity, as training progresses it should vary different augmentations but with relatively small strength, and as training further progresses the augmentation strengths should increase as well.
* Also suggest applying multiple augmentations, not just a single one. Combine augmentation for example is an augmentation which combines Perspective Wrap + Brightness + JPEG.
## TODO (Completeness)
* Investigate previous implementation.
* Convince yourself why DCT implementation works.