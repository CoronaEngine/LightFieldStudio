<div align="center">
  <img src="docs/media/teaser.gif" alt="SSAT light-field path tracing on 3D displays" width="100%" />

  <h1>LightFieldStudio</h1>
  <p><sub>Official implementation of <em>Real-Time Light-Field Path Tracing for 3D Displays via Sparse Spatial-Angular-Temporal Reconstruction</em></sub></p>
  <p>
    <a href="https://doi.org/10.1145/3799902.3811158"><img alt="ACM DOI" src="https://img.shields.io/badge/ACM_DOI-10.1145%2F3799902.3811158-0085CA?logo=acm&logoColor=white" /></a>
    <a href="https://coronaengine.github.io/ssat-page/"><img alt="Project page" src="https://img.shields.io/badge/Project-SSAT-5B5BD6?logo=githubpages&logoColor=white" /></a>
  </p>
  <p><strong>SIGGRAPH 2026 Conference Papers</strong></p>
  <p>
    <a href="https://github.com/CoronaEngine/LightFieldStudio/releases"><img alt="Latest release" src="https://img.shields.io/github/v/release/CoronaEngine/LightFieldStudio?display_name=tag" /></a>
    <a href="LICENSE"><img alt="AGPL-3.0-only" src="https://img.shields.io/badge/license-AGPL--3.0--only-blue.svg" /></a>
    <img alt="Windows 10/11 x64" src="https://img.shields.io/badge/platform-Windows_10%2F11_x64-0078D4" />
    <img alt="NVIDIA RTX" src="https://img.shields.io/badge/GPU-NVIDIA_RTX-76B900?logo=nvidia&logoColor=white" />
  </p>
</div>

LightFieldStudio is an interactive path-tracing application for calibrated lenticular light-field displays. It connects physical subpixel rays, spatial reconstruction (ES), and physical light-field temporal reconstruction (LT) in one GPU-resident pipeline.

## Features

- Calibration-driven, interlaced LFD rendering with one path sample per encoded target.
- Quality and Performance profiles of the same ES+LT pipeline.
- Interactive camera controls and support for Vision JSON scenes.
- Monitor selection and DPI-aware native fullscreen output.
- Coffee, CBox and SpaceShip in an offline Windows package, each with 2K, 4K and 8K display presets.
- Source-based [Vision renderer](https://github.com/CoronaEngine/Vision/tree/ssat-backend), pinned as a submodule.

## Quick start

**Just want to run it? [Download the portable Windows ZIP here.](https://github.com/CoronaEngine/LightFieldStudio/releases/tag/v0.1.0)**

1. Download `LightFieldStudio-v0.1.0-win-x64.zip` and extract the **entire** directory.
2. Run `LightFieldStudio.exe`. CBox opens in Quality mode at 3840 × 2160. Choose Coffee, CBox or SpaceShip from the Scene menu and select the display resolution.
3. Select your output display, enter its calibration, then choose **Native fullscreen**.

Requires Windows 10/11 x64, an NVIDIA RTX 20-series or newer GPU, and a compatible NVIDIA driver. The portable package includes the three scenes and the application runtimes. An installed compatible NVIDIA driver is sufficient. First launch compiles shaders and can take longer.

The included calibration belongs to the bundled display profile. Use your display's own calibration for correct 3D output. A regular monitor shows the interlaced image. The package is unsigned; Windows may show an unknown-publisher warning. Verify its SHA-256 against the release checksum.

## Documentation

- [Getting Started](docs/GETTING_STARTED.md): portable use, source builds, assets, and troubleshooting.
- [Architecture](docs/ARCHITECTURE.md): phase ownership and execution flow.
- [Controls and Calibration](docs/CONTROLS.md): camera, display, and scene controls.
- [Scene assets and attribution](https://huggingface.co/datasets/Royalvice/LightFieldStudio-Scenes)

## Build summary

```powershell
git clone --recurse-submodules https://github.com/CoronaEngine/LightFieldStudio.git
cd LightFieldStudio
.\tools\build.ps1
```

Source builds require the Windows C++ toolchain, CMake, CUDA Toolkit, and the dependency setup described in [Getting Started](docs/GETTING_STARTED.md). The build script downloads and verifies the three bundled scenes automatically. Binary users should use the portable package above.

## Roadmap

- [ ] Release benchmarks with one-command reproducible metrics.
- [ ] Achieve real-time rendering at 8K.
- [ ] Integrate the RGB subpixel reuse technique from [DirectLStudio](https://github.com/CoronaEngine/DirectLStudio).

## Acknowledgements

LightFieldStudio builds on [Vision](https://github.com/CoronaEngine/Vision/tree/ssat-backend) and [Horizon](https://github.com/CoronaEngine/Horizon). We thank the SSAT authors and contributors to the third-party libraries used by these projects. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the asset notices included with each scene.

## License

Self-authored source is licensed under the **GNU Affero General Public License v3.0 only** (`AGPL-3.0-only`); see [LICENSE](LICENSE). Third-party components retain their original licenses. Coffee Maker by cekuhnen uses CC BY 3.0. SpaceShip by thecali uses CC0. CBox includes self-authored assets under CC BY 4.0 and third-party assets under their individual terms.

## Citation

If SSAT or LightFieldStudio helps your work, please cite:

```bibtex
@inproceedings{yang2026ssat,
  title={Real-Time Light-Field Path Tracing for 3D Displays
         via Sparse Spatial-Angular-Temporal Reconstruction},
  author={Yang, Zongyuan and Zou, Liulei and Zhu, Ling and
          Xiong, Yongping and Fan, Honghui and Liu, Baolin and
          Song, Yingde and Zhu, Yu and Zeng, Chang and Ye, Wangping},
  booktitle={ACM SIGGRAPH 2026 Conference Papers},
  year={2026},
  doi={10.1145/3799902.3811158},
  publisher={ACM}
}
```
