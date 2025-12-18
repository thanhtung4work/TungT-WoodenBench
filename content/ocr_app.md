Title: Turning Images into Searchable PDFs
Date: 2025-12-13 11:00
Category: Machine Learning
Tags: machine learning
Slug: autoocr
Authors: Tung Thanh Tran
Summary: How to turn images into searchable, extractable PDF files

If you’ve ever tried to run OCR on a poorly taken photo — tilted angles, bad lighting, too much noise — you know the pain. That’s exactly why I built **[Autocrop-OCR](https://github.com/thanhtung4work/Autocrop-OCR)**: a lightweight, developer-friendly toolkit to preprocess images *before* sending them to an OCR engine.

This post walks you through the core features and gives you a high-level look at how they work.

## What Autocrop-OCR does?
Autocrop-OCR acts as the smart “prep cook” — cleaning, trimming, straightening, and optimizing your input image so OCR models can work with cleaner text regions.

Key functionalities include:

- **Perspective Cropping** - straighten skewed documents  
- **Color Quantization** - simplify colors
- **Edge Detection** - find text boundaries or document contours  
- (Plus optional cropping, thresholding, and preprocessing utilities)


## Edge Detection  
Before you can crop or isolate text, you need to know *where* the edges are.

```python
def detect_corners(image: np.ndarray) -> np.ndarray:
    """
    Detects the 4 corners of the largest rectangular contour in an image.
    Returns corner points ordered as: top-left, top-right, bottom-right, bottom-left.
    """

    # 1. Resize image (optional step for consistency)
    scale_percent = 100
    h, w = image.shape[:2]
    new_w = int(w * scale_percent / 100)
    new_h = int(h * scale_percent / 100)
    image_resized = cv2.resize(image, (new_w, new_h))

    # 2. Preprocessing: grayscale, blur, threshold
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 80, 180)

    # 3. Find external contours
    contours, _ = cv2.findContours(
        edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # 4. Detect the largest contour with 4 edges (the “page”)
    page_contour = None
    largest_area = -np.inf
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        if len(approx) == 4 and area > largest_area:
            page_contour = approx
            largest_area = area

    # Fallback: if no 4-sided contour found, return whole image rectangle
    if page_contour is None:
        print("Could not detect page corners.")
        return np.array([
            [0, 0],
            [new_w, 0],
            [new_w, new_h],
            [0, new_h]
        ], dtype=np.float32)

    # 5. Extract corner points
    corner_pts = page_contour[:, 0, :].astype(np.float32)

    # 6. Sort corners clockwise using angle from center
    center = np.mean(corner_pts, axis=0)

    # Sort by angle relative to center (atan2 gives counterclockwise ordering)
    ordered = sorted(
        corner_pts,
        key=lambda pt: np.arctan2(pt[1] - center[1], pt[0] - center[0])
    )

    # Format: TL, TR, BR, BL
    ordered = np.array(ordered, dtype=np.float32)

    top_left, top_right, bottom_right, bottom_left = ordered

    return np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.float32)
```


## Perspective Cropping  
Autocrop-OCR detects the document boundaries and applies a **perspective transform** so the image becomes a clean, top-down rectangle.

```python
def perspective_crop(image: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Applies a perspective transform to crop a quadrilateral region from the image.
    Expects `points` in the order:
        top-left, top-right, bottom-right, bottom-left.
    Saves the warped image and also returns it.
    """

    # Ensure points are float32
    pts = points.astype("float32")

    # Compute output dimensions by measuring opposite sides
    width_top = np.linalg.norm(pts[0] - pts[1])
    width_bottom = np.linalg.norm(pts[3] - pts[2])
    height_left = np.linalg.norm(pts[0] - pts[3])
    height_right = np.linalg.norm(pts[1] - pts[2])

    width = int(max(width_top, width_bottom))
    height = int(max(height_left, height_right))

    # Destination rectangle coordinates
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype="float32")

    # Compute perspective transform
    matrix = cv2.getPerspectiveTransform(pts, dst)
    warped = cv2.warpPerspective(image, matrix, (width, height))

    return warped
```

## Color Quantization  
High-resolution images often contain more colors than OCR models actually need.

```python
def quantize_image(image, k=8):
    """
    Reduce the number of colors in the image using K-means clustering.
    k = number of colors to keep.
    """
    # Convert to float32 and reshape to (num_pixels, 3)
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)

    # KMeans criteria: stop after 10 iterations or accuracy < 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Apply KMeans
    _, labels, centers = cv2.kmeans(
        Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert colors back to uint8
    centers = np.uint8(centers)

    # Apply clustered colors to pixels
    quantized = centers[labels.flatten()]
    quantized = quantized.reshape(image.shape)

    return quantized
```

## How It All Fits Together  
A typical preprocessing flow might look like this:

1. Run edge detection → find document or text boundaries  
2. Apply perspective cropping → straighten the view  
3. Apply color quantization → simplify and denoise  
4. Send the cleaned output to your favorite OCR engine (Tesseract, EasyOCR, PaddleOCR, etc.)

This pipeline transforms a messy smartphone photo into a clean, accurate input while keeping performance lightweight.

---

## Try It Out  
You can explore the code or clone the project here:

**Autocrop-OCR on GitHub**  
https://github.com/thanhtung4work/Autocrop-OCR

The repo includes examples, source code, and utilities to help you integrate it directly into your project.