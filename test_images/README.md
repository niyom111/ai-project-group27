# Test images for lighting enhancement (Person 1, Step 6)

Add 2–3 photos here, e.g.:
- one normal lighting
- one dark / low light
- one with strong shadow

Then from **project root** (`ai-project-group27/`) run:

```bash
python -c "from enhancement import enhance_image; p = enhance_image('test_images/your_photo.jpg'); print('Saved to', p)"
```

Replace `your_photo.jpg` with your filename. Or run:

```bash
python scripts/test_enhancement.py
```

Enhanced images are saved under `lighting_enhancement/output/`. Check by eye that dark/shadow areas look better.
