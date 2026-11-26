# Architecture

LightFieldStudio contains the interactive Studio and one spatial/temporal light-field reconstruction pipeline. Vision remains a pinned source submodule. Quality and Performance configure this same pipeline.

## Phase ownership

- **Phase 1:** calibrated physical LF rays, interlacing/ViewId, layout tables, and dense encoded-target work submission. The renderer traces `3WH` targets at one sample per target and a maximum path depth of 16.
- **Phase 2:** current-frame demodulation, moments, outlier preprocessing, spatial reconstruction, and final reconstruction.
- **Phase 3:** physical LF history queries, geometry/material validation, early RGB temporal measurement, and history lifecycle.
- **Composer:** selects the proper encoded carrier and produces the canonical display image.

The folders express responsibility, not a strictly sequential three-pass schedule. The execution graph is:

```text
Calibrated 3WH path samples
  -> ES initialization / preprocessing / step1
  -> physical LF RGB temporal history
  -> spatial suffix / final reconstruction
  -> encoded carrier composition
  -> LFD output
```

Only the early RGB measurement feeds its recurrent history. Later spatial reconstruction does not recursively train that history. Quality uses the retained glossy signal transform and reflection correspondence; Performance uses its corresponding lower-cost profile.

An encoded neighbor is not an ordinary screen-space neighbor: spatially adjacent carriers can belong to different calibrated views. Spatial filtering and temporal queries preserve each target's LF geometry and carrier semantics.

## Runtime ownership

The application creates one Vision renderer session, one reconstruction pipeline, and one light-field processor. Module resources are prepared for the active layout and reused across frames. Layout, scene, and quality changes reset the relevant history; ordinary camera motion uses reprojection.

Native CUDA modules and runtime shader resources are loaded relative to the executable. User-writable caches and settings live outside the package directory. The three bundled scenes have a separate versioned asset archive on Hugging Face and ship in the portable ZIP.

## Source layout

```text
src/studio/                 Scene selection, camera input and native output window
src/ssat/phase1/             Calibrated rays and phase-space layout
src/ssat/phase2/             Spatial observation and reconstruction
src/ssat/phase3/             Physical LF correspondence and RGB history
src/ssat/native/             Shared GPU records and carrier composition
src/backend/                Renderer interface
backend/Vision/             Pinned renderer submodule
assets/scenes.json          Immutable asset revision and file checksums
tools/                      Asset download, build, and portable packaging
```

Only Quality and Performance are application profiles. GPU modules are built with
the host application and must be distributed together; kernel parameter records
are internal interfaces, not a stable plugin ABI. Scene inputs are copied from the
asset manifest so that a package cannot accidentally include unrelated local data.
