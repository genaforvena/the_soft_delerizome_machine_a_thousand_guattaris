# Text Pattern Transformer — Auto-run & Example

## Usage Guide (markdown version)

### Overview
This script:
- Checks if a trained pattern detection model exists.
- If missing, it trains one on synthetic raster images.
- Converts *Finnegans Wake* to a raster.
- Analyzes it with the trained model.
- Generates Grad-CAM heatmaps.

## Steps
1. Place `finnegans_wake.txt` in the project folder.
2. Run the script.
3. The script will:
   - Train the model if `pattern_detector.pth` doesn’t exist.
   - Convert the text to a raster image.
   - Analyze the raster.
   - Save outputs: `raster.png`, `gradcam_overlay.png`, and print pattern score.
