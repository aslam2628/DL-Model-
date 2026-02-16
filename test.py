import sys
import traceback
from pathlib import Path

try:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import load_model
except Exception as e:
    print("Error importing TensorFlow/Keras:", e)
    print("Ensure TensorFlow is installed in the active Python environment (pip install tensorflow).")
    raise

print("Loading model...")
BASE_DIR = Path(__file__).resolve().parent
model_path = BASE_DIR / "models" / "emotion_model.h5"
if not model_path.exists():
    print(f"Model not found: {model_path}")
    sys.exit(1)
model = load_model(str(model_path))

test_dir = BASE_DIR / "Datasets" / "test"

def check_dir(path: Path, name: str):
    if not path.exists():
        print(f"{name} directory does not exist: {path}")
        if (BASE_DIR / 'Datasets').exists():
            print("Available classes:", [p.name for p in (BASE_DIR / 'Datasets').iterdir() if p.is_dir()])
        sys.exit(1)
    if not any(path.iterdir()):
        print(f"{name} directory is empty: {path}")
        sys.exit(1)

check_dir(test_dir, "Test")

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    str(test_dir),
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical",
    shuffle=False
)

print("Evaluating on test dataset...")

loss, acc = model.evaluate(test_gen)

print("\nTest Accuracy:", round(acc*100,2), "%")
print("Test Loss:", round(loss,4))