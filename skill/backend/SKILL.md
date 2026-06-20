# Coding Style Guide

Coding style guidelines for this project.

## 🎯 Core Principles

### 1. Simplicity First
- **Write code that is easy to read and not overly complex.**
- Avoid unnecessary abstraction.
- Use clear and meaningful variable and function names.
- Do not add features that are not being used.

```python
# ✅ Good - Clear and straightforward
def save_image(image: np.ndarray, path: str):
    img_pil = Image.fromarray(image)
    img_pil.save(path, quality=95)

# ❌ Bad - Overly complex
class ImageSaveStrategy:
    def execute(self, image, path, **kwargs):
        ...
```

### 2. Clear Module Structure
Organize code by responsibility:

```text
project/
├── schemas/          # Data structures (dataclasses)
├── utils_module/     # Utility functions
├── models_module/    # Model wrappers
├── processors/       # Business logic
└── main.py           # Entry point
```

### 3. Use Dataclass for Data Structures

```python
from dataclasses import dataclass, field
from typing import List

@dataclass
class Detection:
    class_id: int
    confidence: float
    box: tuple  # (cx, cy, w, h) normalized
    class_name: str = ""

@dataclass
class ImageResult:
    image_id: int
    file_name: str
    width: int
    height: int
    detections: List[Detection] = field(default_factory=list)
    masks: List = field(default_factory=list)
```

### 4. Use Type Hints Everywhere

```python
def process_image(
    self,
    image_path: Path,
    image_id: int
) -> List[Tuple[ImageResult, np.ndarray]]:
    """Process single image with optional augmentation"""
    ...
```

### 5. Separate Configuration
- Use YAML for configuration.
- Do not hardcode values in the code.
- Create a Configuration class to manage config.

```yaml
# configuration.yaml
models:
  grounding_dino: "GroundingDINO/weights/groundingdino_swint_ogc.pth"
  mobile_sam: "MobileSAM/weights/mobile_sam.pt"

detection:
  box_threshold: 0.35
  text_threshold: 0.25

classes:
  - "person"
  - "car"
  - "bicycle"
```

### 6. Error Handling and Logging

```python
import logging

logger = logging.getLogger(__name__)

def process_image(self, image_path: Path, image_id: int):
    try:
        # Process logic
        ...
    except Exception as e:
        logger.error(f"Failed to process {image_path.name}: {e}")
        return None
    finally:
        # Cleanup temp files
        if temp_file and os.path.exists(temp_file):
            os.unlink(temp_file)
```

### 7. Reduce Unnecessary Print Statements

```python
# ❌ Bad - Too many print statements
print("Loading model...")
print("Model loaded successfully!")
print(f"Processing image 1/10...")
print(f"Found {len(boxes)} boxes")
print("Done!")

# ✅ Good - Short summary of important information only
logger.info("✓ Complete!")
logger.info(f"  Images: {stats.processed_images}/{total_images}")
logger.info(f"  Annotations: {stats.total_detections}")
```

### 8. Use Meaningful Names

```python
# ✅ Good
def detect_multiclass(
    self,
    image,
    class_names: List[str]
) -> List[Detection]:
    """Detect multiple classes in single pass"""
    ...

# ❌ Bad
def process(self, img, cn):
    ...
```

### 9. Separate Concerns
Keep responsibilities separated:

```python
# File operations
def get_image_files(folder: Path) -> List[Path]:
    ...

# Annotation formatting
def save_yolo_annotation(detections, image_shape, output_path):
    ...

# Model operations
class GroundingDINOModel:
    def detect(self, image, text):
        ...

# Business logic
class ImageProcessor:
    def process_image(self, image_path, image_id):
        ...
```

### 10. Resource Management

```python
def _process_single_version(
    self,
    img_source,
    image_path,
    image_id,
    suffix
):
    temp_path = None
    try:
        # Create temp file
        temp_path = self._save_to_temp(img_source)

        # Process
        ...
    finally:
        # Always cleanup
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass
```

### 11. Concise Docstrings

```python
def process_image(
    self,
    image_path: Path,
    image_id: int
) -> List[Tuple[ImageResult, np.ndarray]]:
    """Process single image with optional augmentation

    Args:
        image_path: Path to image file
        image_id: Unique image ID

    Returns:
        List of (result, image_source) tuples
    """
```

### 12. Private Methods (Underscore Convention)

```python
class ImageProcessor:
    # Public API
    def process_image(self, image_path, image_id):
        ...

    # Internal helpers
    def _load_image(self, image_path):
        ...

    def _segment_objects(self, result, image_source):
        ...

    def _create_visualization(
        self,
        image_source,
        detections,
        masks
    ):
        ...
```

## 🚫 Things to Avoid

1. ❌ **Abstract factories and complex patterns** (unless necessary)
2. ❌ **Unnecessary nested classes**
3. ❌ **Magic numbers** - use config instead
4. ❌ **Global variables** - pass parameters instead
5. ❌ **Print statements in production code** - use logging
6. ❌ **Hardcoded paths** - use config
7. ❌ **Functions longer than 50 lines** - break them into smaller functions

## ✅ Best Practices

1. ✅ **Dataclasses** for data structures
2. ✅ **Type hints** wherever possible
3. ✅ **YAML configuration** for settings
4. ✅ **Logging module** instead of print
5. ✅ **Try-finally** for resource cleanup
6. ✅ **List comprehension** when appropriate
7. ✅ **Pathlib** instead of string paths
8. ✅ **Descriptive naming**, even if the names are slightly longer

## 📝 Example: Good Code Structure

```python
# schemas/detection.py
from dataclasses import dataclass

@dataclass
class Detection:
    class_id: int
    confidence: float
    box: tuple
    class_name: str = ""

# utils_module/file_utils.py
from pathlib import Path
from typing import List

def get_image_files(folder: Path) -> List[Path]:
    """Get all image files from folder"""
    extensions = {'.jpg', '.jpeg', '.png'}
    return [
        f for f in folder.iterdir()
        if f.suffix.lower() in extensions
    ]

# processors/image_processor.py
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Process images for auto-labeling"""

    def __init__(
        self,
        config: Config,
        augment: bool = False
    ):
        self.config = config
        self.dino = GroundingDINOModel(config)
        self.sam = MobileSAMModel(config)
        self.augment = augment

    def process_image(
        self,
        image_path: Path,
        image_id: int
    ) -> List[Tuple[ImageResult, np.ndarray]]:
        """Process single image with optional augmentation"""
        try:
            results = self._do_processing(
                image_path,
                image_id
            )
            return results
        except Exception as e:
            logger.error(
                f"Failed to process {image_path.name}: {e}"
            )
            return [None]

    def _do_processing(self, image_path, image_id):
        """Internal processing logic"""
        ...
```

---

**Summary**: Focus on simplicity, readability, maintainability, and practical code that works. 🚀