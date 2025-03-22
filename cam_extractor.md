
### How to get from training to analyzing *Finnegans Wake* — over-explained version:

1. **Train the model**  
- The `train_model()` function teaches the model what patterns look like.
- The model looks at images (raster versions of text) and tries to learn if they contain order or randomness.
- Training happens by showing examples and adjusting the model until it can guess correctly.

2. **Prepare Finnegans Wake**  
- Save the text as `finnegans_wake.txt`.
- Run `convert_text_to_all_forms('finnegans_wake.txt', 'output_visualisations')`.
- This creates `raster.png`, a black-and-white image representing the structure of the text.

3. **Load your trained model**  
- You saved the trained model’s brain (weights) to a file.
- Now, you load it back into the same model shape:
```python
model = PatternDetector()
model.load_state_dict(torch.load('path_to_trained_model/pattern_detector.pth'))
model.eval()  # sets it to analysis mode
```

4. **Run analysis on Finnegans Wake**  
- Open the raster image:
```python
from PIL import Image
img = Image.open('output_visualisations/raster.png').convert('RGB')
```
- Convert it into a tensor (what PyTorch understands):
```python
img_tensor = transform(img).unsqueeze(0)  # Add batch dimension for prediction
```
- Pass it through the model:
```python
output = model(img_tensor)
print(f"Detected pattern score: {output.item()}")
```
- The number you get is the model’s guess: near 1 = strong pattern, near 0 = little or none.

5. **See where the model is looking (Grad-CAM visualisation)**  
- Grad-CAM shows what part of the image influenced the model’s decision.
- Install `torchcam` or use built-in hooks.
- Example with `torchcam`:
```python
from torchcam.methods import GradCAM
cam_extractor = GradCAM(model, target_layer='features.layer4')
activation_map = cam_extractor(output.squeeze(0).argmax(), img_tensor)
```
- Overlay the activation map onto the raster image to see hotspots.

6. **Interpretation in plain language**  
- The model might highlight areas with strange repetition, symmetry, or voids.
- These could be Joyce’s hidden structures, rhythmic layering, or text spirals.
- Compare with other texts to see if Joyce’s raster looks unusually patterned.

7. **Expansion ideas**  
- Cut *Finnegans Wake* into chunks, run analysis per chapter.
- Compare results with known cryptic texts.
- See if the model finds patterns Joyce didn’t intend — this becomes part of the fun.
