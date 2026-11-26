# Getting Started

## Portable Windows application

Download the Windows x64 ZIP from [Releases](https://github.com/CoronaEngine/LightFieldStudio/releases/tag/v0.1.0), verify its SHA-256, and extract the whole directory. Run `LightFieldStudio.exe`. Coffee, CBox, SpaceShip and the required application runtimes are included. The application does not download code or assets at startup.

Use Windows 10/11 x64, an RTX 20-series or newer NVIDIA GPU, and an NVIDIA driver compatible with the bundled CUDA runtime. Shader compilation uses NVRTC included in the package. First launch can take longer while the local shader cache is created.

The application and ZIP are unsigned. Windows may show an unknown-publisher warning. Download only from the project's release page and check the published checksum. No driver files are bundled; install or update the NVIDIA driver using NVIDIA's installer.

Studio opens CBox in Quality mode at 3840 × 2160. Each bundled scene offers 1536 × 2048 at 50 views, 3840 × 2160 at 50 views and 7680 × 4320 at 96 views. Select your physical display, apply its calibration, and use **Native fullscreen** for pixel-exact presentation. A normal monitor displays the interlaced image. The bundled calibration is a starting profile, not a universal calibration.

## Build from source

Install Git, CMake 4.0 or newer, Visual Studio with the Desktop development with C++ workload and Windows SDK, CUDA Toolkit 13.1, and Python 3 for the backend's build-time resource generation. A compatible NVIDIA driver is required to run the result. Python and the CUDA Toolkit are build prerequisites only; portable users do not need them.

```powershell
git clone --recurse-submodules https://github.com/CoronaEngine/LightFieldStudio.git
cd LightFieldStudio
.\tools\build.ps1
```

The build script downloads the pinned scene archive from Hugging Face, verifies its SHA-256 and individual files, builds the application, and places all three scenes beside the executable. It prints the executable location when complete. Dependency versions are pinned by the source checkout.

To choose a build directory:

```powershell
.\tools\build.ps1 -BuildDirectory C:\Build\LightFieldStudio
```

For offline assets, download `LightFieldStudio-scenes-v0.1.0.zip` from the [dataset](https://huggingface.co/datasets/Royalvice/LightFieldStudio-Scenes) once, then pass `-AssetArchive C:\Downloads\LightFieldStudio-scenes-v0.1.0.zip`. A valid existing asset directory is reused without downloading. Building dependencies offline also requires their pinned sources to have been fetched previously.

## Other scenes

Use **Open scene...** to load a Vision JSON scene. Keep referenced geometry and textures with the scene. The scene must contain a supported camera and valid light-field calibration. Camera movement and history reconstruction remain owned by LightFieldStudio; embedded backend denoisers are removed from the runtime scene.

## Troubleshooting

- **No NVIDIA device / incompatible driver:** install a compatible NVIDIA driver and select the high-performance NVIDIA GPU in Windows Graphics settings.
- **Missing DLL, native module, or scene:** extract the complete ZIP. Do not move the executable out of its directory.
- **First launch takes longer:** wait for shader compilation. Cache and settings are stored under `%LOCALAPPDATA%\LightFieldStudio`.
- **Out of device memory:** close other GPU applications or explicitly choose the lower resolution. The application does not silently lower quality or switch renderers.
- **Fullscreen unavailable:** set the chosen monitor to the selected native resolution, then retry.
- **Incorrect 3D image:** verify resolution, view count, pitch, slant in radians, offset, view angle, and focal distance against the physical display's calibration.

Runtime errors are shown in the control window. Closing either window exits the application.
