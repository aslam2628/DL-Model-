import os
import sys
import traceback
from pathlib import Path

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
except Exception as e:
    print("Error importing TensorFlow or Keras APIs:", e)
    print("Ensure TensorFlow is installed in the active Python environment (pip install tensorflow).")
    raise

print("Preparing dataset...")

# Use project-relative dataset paths so the script works regardless of the current CWD
BASE_DIR = Path(__file__).resolve().parent
train_dir = BASE_DIR / "Datasets" / "train"
test_dir = BASE_DIR / "Datasets" / "test"

# Validate dataset directories early with helpful messages
def check_dir(path: Path, name: str):
    if not path.exists():
        print(f"{name} directory does not exist: {path}")
        print(f"Looking in {BASE_DIR / 'Datasets'} for available classes:")
        if (BASE_DIR / 'Datasets').exists():
            print([p.name for p in (BASE_DIR / 'Datasets').iterdir() if p.is_dir()])
        sys.exit(1)
    if not any(path.iterdir()):
        print(f"{name} directory is empty: {path}")
        sys.exit(1)

check_dir(train_dir, "Train")
check_dir(test_dir, "Test")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    zoom_range=0.1,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    str(train_dir),
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    str(test_dir),
    target_size=(48, 48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

print("Building CNN model...")

model = Sequential([
    Conv2D(32,(3,3),activation="relu",input_shape=(48,48,1)),
    MaxPooling2D(),
    Conv2D(64,(3,3),activation="relu"),
    MaxPooling2D(),
    Conv2D(128,(3,3),activation="relu"),
    MaxPooling2D(),
    Flatten(),
    Dense(128,activation="relu"),
    Dropout(0.5),
    Dense(7,activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Training started...")

try:
    model.fit(
        train_gen,
        validation_data=test_gen,
        epochs=10
    )
except Exception:
    print("An error occurred during training:")
    traceback.print_exc()
    sys.exit(1)

# Ensure models directory exists relative to project
models_dir = BASE_DIR / "models"
models_dir.mkdir(parents=True, exist_ok=True)
model_path = models_dir / "emotion_model.h5"

try:
    model.save(str(model_path))
    print(f"Model saved successfully: {model_path}")
except Exception:
    print("Failed to save the model:")
    traceback.print_exc()
    sys.exit(1)