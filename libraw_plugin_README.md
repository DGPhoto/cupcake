# LibRaw Plugin for Cupcake

This plugin provides advanced RAW file processing capabilities for Cupcake by leveraging the native LibRaw library installed on your system.

## Features

- Advanced RAW file processing with high quality demosaicing algorithms
- Support for all major camera manufacturers' RAW formats
- Custom processing options for fine-tuning RAW conversion
- Automatic detection of installed LibRaw libraries

## Installation

### 1. Install LibRaw on your system

#### Windows
- Download the pre-built LibRaw binaries from https://www.libraw.org/download
- Extract and place the DLL files in your system PATH or in the Cupcake application directory

#### macOS
```bash
brew install libraw
```

#### Linux (Debian/Ubuntu)
```bash
sudo apt update
sudo apt install libraw-dev
```

#### Linux (Red Hat/Fedora)
```bash
sudo yum install libraw-devel
```

### 2. Verify the plugin

The plugin will be automatically loaded by Cupcake when it starts. You can verify that it's working by checking the logs for a message like:

```
INFO - Loaded LibRaw from: libraw.dll
INFO - LibRaw version: 0.21.1
INFO - Successfully initialized LibRaw Plugin
```

## Configuration

You can customize the LibRaw plugin behavior through the Cupcake settings. Available options include:

- **libraw_path**: Custom path to LibRaw library (optional)
- **processing_options**:
  - **use_camera_wb**: Use camera white balance if available (0=no, 1=yes)
  - **half_size**: Half-size processing for faster results (0=no, 1=yes)
  - **output_color**: Output color profile (0=raw, 1=sRGB, 2=Adobe, 3=Wide, 4=ProPhoto, 5=XYZ)
  - **output_bps**: Bits per sample (8 or 16)
  - **highlight_mode**: Highlight recovery mode (0=clip, 1=unclip, 2=blend, 3-9=rebuild)
  - **brightness**: Brightness multiplier
  - **user_qual**: Demosaicing algorithm (0=linear, 1=VNG, 2=PPG, 3=AHD, 4=DCB)
  - **auto_bright**: Auto brightness adjustment (0=disable, 1=enable)
  - **fbdd_noise_reduction**: FBDD noise reduction (0=off, 1=light, 2=full)

## Troubleshooting

If the plugin fails to load, check the following:

1. Ensure LibRaw is correctly installed on your system
2. Make sure the LibRaw library files are in your system's PATH
3. Check the Cupcake logs for specific error messages
4. Try setting a custom path to the LibRaw library in the plugin configuration

## Advanced Usage

If you have multiple versions of LibRaw installed or want to use a custom build, you can specify the path in the Cupcake settings:

```python
from src import get_settings

settings = get_settings()
settings.update_settings({
    "libraw_plugin": {
        "libraw_path": "/path/to/your/custom/libraw.so"
    }
})
```

## Supported RAW Formats

This plugin supports all major RAW formats including:

- Canon (CR2, CR3, CRW)
- Nikon (NEF, NRW)
- Sony (ARW, SRF, SR2)
- Fujifilm (RAF)
- Olympus (ORF)
- Panasonic (RW2)
- Pentax (PEF)
- Adobe DNG
- Leica (RAW, RWL)
- Hasselblad (3FR, FFF)
- Phase One (IIQ)
- Sigma (X3F)
- And many more!