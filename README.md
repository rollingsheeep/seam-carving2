# Seam Carving Implementation

This is a C++ implementation of the seam carving algorithm for content-aware image resizing, based on the paper "Seam Carving for Content-Aware Image Resizing" by Shai Avidan and Ariel Shamir. The implementation uses OpenCV for image processing and supports both image resizing and object removal.

## Features

- Content-aware image resizing (both enlarging and reducing)
- Object removal using seam carving
- Forward energy calculation for better results
- Support for protective masking
- Visualization of seam removal process
- Both vertical and horizontal seam operations

## Prerequisites

- OpenCV 4.x
- C++ compiler with C++17 support
- CMake 3.x (optional, for build configuration)

## Project Structure

```
.
├── seam_carving.cpp    # Main implementation file
├── CMakeLists.txt     # CMake build configuration
├── images/            # Test images directory
└── output/           # Output directory for processed images
```

## Building

### Using CMake

```bash
mkdir build
cd build
cmake ..
cmake --build .
```

### Manual Build using UCRT64

```bash
# Windows (MinGW)
cd d:/project/seam-carving2
## for seam_carving(self)
g++ seam_carving.cpp -o seam_carving -I/ucrt64/include/opencv4 -L/ucrt64/lib -lopencv_imgcodecs -lopencv_imgproc -lopencv_highgui -lopencv_core -std=c++17
## for main(github)
g++ main.cpp -o main.exe -std=c++17
```

## Usage

The program supports two main modes: resize and object removal.

### Command Line Arguments

```
seam_carving [options]
Options:
  -resize              Resize image (requires -dx and/or -dy)
  -remove              Remove object (requires -mask)
  -im <filename>       Input image file
  -out <filename>      Output image file
  -dx <value>          Width change (positive to enlarge, negative to reduce)
  -dy <value>          Height change (positive to enlarge, negative to reduce)
  -target_width <value> Target width in pixels (alternative to -dx)
  -target_height <value> Target height in pixels (alternative to -dy)
  -mask <filename>     Mask file for object removal or protection
  -vis                 Visualize seams during processing
  -help                Show help message
```

### Examples

1. Reduce width by 100 pixels:
```bash
./seam_carving -resize -im images/example.jpg -out output/resized.jpg -dx -100
```

2. Increase height by 50 pixels:
```bash
./seam_carving -resize -im images/example.jpg -out output/enlarged.jpg -dy 50
```

3. Resize to specific dimensions:
```bash
./seam_carving -resize -im images/example.jpg -out output/resized.jpg -target_width 800 -target_height 600
./seam_carving -resize -im images/Lena_512.png -out output/Lena_162.png -target_width 162 -targetheight 512
```

4. Remove object using a mask:
```bash
./seam_carving -remove -im images/example.jpg -out output/removed.jpg -mask images/mask.png
```

5. Resize with protective mask:
```bash
./seam_carving -resize -im images/example.jpg -out output/protected.jpg -dx -100 -mask images/protect.png
```

### Examples

5. Resize image:
```bash
./main.exe images/Lena_512.png output/Lena_162.png --energy forward
```

## Implementation Details

The implementation uses forward energy by default, which typically produces better results than backward energy. The algorithm processes seams individually rather than in batches to maintain image quality.

Key features of the implementation:

- Forward energy calculation for better seam selection
- Efficient seam tracking and removal
- Improved blending for seam insertion
- Batch processing for large size changes
- Progress visualization support

## Performance Considerations

- Large images are processed in batches to maintain memory efficiency
- Forward energy calculation is used by default for better quality
- Visualization may slow down processing but is useful for debugging

## Known Limitations

- Very large size changes might affect image quality
- Processing time increases with image size and number of seams
- Memory usage scales with image dimensions

## License

This project is licensed under the MIT License - see the LICENSE file for details.
