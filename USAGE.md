# Seam Carving Program Usage Guide

This guide explains how to compile and run the Seam Carving program using MSYS2 UCRT64 terminal.

## Prerequisites

1. Install MSYS2 from https://www.msys2.org/
2. Open MSYS2 UCRT64 terminal (not PowerShell or Command Prompt)

## Compilation Steps

1. Navigate to your project directory:
```bash
cd /path/to/seam-carving2
```

2. Create build directory:
```bash
mkdir -p build
```

3. Create implementation files for stb_image libraries:
```bash
# Create stb_image.c
echo '#define STB_IMAGE_IMPLEMENTATION
#include "../stb_image.h"' > build/stb_image.c

# Create stb_image_write.c
echo '#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "../stb_image_write.h"' > build/stb_image_write.c
```

4. Compile the libraries and program:
```bash
# Compile stb_image
gcc -c build/stb_image.c -o build/stb_image.o

# Compile stb_image_write
gcc -c build/stb_image_write.c -o build/stb_image_write.o

# Compile the main program
gcc -o nob main.c build/stb_image.o build/stb_image_write.o -lm
```

## Running the Program

The basic syntax to run the program is:
```bash
./nob.exe <input_image> <output_image>
```

### Example with Default Image
```bash
./nob.exe ./images/Lena_512.png output.png
```

### Using Your Own Images

1. Place your image in the `images` directory (or use full path)
2. Run the program with your image:
```bash
./nob.exe ./images/your_image.jpg output.jpg
```

Supported image formats:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- GIF (.gif)
- TGA (.tga)

## Notes

- The program will create a resized version of your input image
- The output image will be saved in the current directory
- Make sure you have write permissions in the current directory
- The program works best with images that have clear areas of low importance that can be removed

## Troubleshooting

If you get compilation errors:
1. Make sure you're using MSYS2 UCRT64 terminal
2. Verify all files are in the correct locations
3. Try cleaning the build directory and recompiling:
```bash
rm -f build/*.o
```
Then repeat the compilation steps. 