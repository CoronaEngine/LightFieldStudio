# Controls and Calibration

| Control | Action |
|---|---|
| Open scene | Load a supported Vision JSON scene |
| Scene | Load bundled Coffee, CBox or SpaceShip at the selected resolution |
| Quality | Choose Quality or Performance |
| Resolution | Select 2K (1536 × 2048), 4K (3840 × 2160) or 8K (7680 × 4320) |
| Reset camera | Restore the scene camera and reset history |
| Pause / Resume | Pause or resume rendering and camera input |
| Output display | Choose a connected display |
| Native fullscreen | Present at the selected native pixel resolution |
| F11 in output | Enter or leave fullscreen, including on a single display |
| Light-field calibration | Edit optical and LF geometry parameters; Apply resets history |
| WASD | Move forward, backward, left, and right |
| Q / E | Move down / up |
| Shift | Move faster |
| Mouse drag in output | Look around |
| Mouse wheel | Dolly |
| Escape in output | Exit |

Slant is expressed in **radians**; view angle is in **degrees**. View count, pitch, offset, focal distance, and view angle must match the physical LFD. A scaled window is for navigation; calibrated viewing requires the output resolution and native fullscreen to match the physical display.

Use `--scene cbox`, `--scene coffee` or `--scene spaceship` for bundled scenes, or pass a Vision JSON path. Select a display preset with `--resolution 2k`, `4k` or `8k`. Studio also accepts `--performance`, `--fullscreen` and `--help`. With no arguments, the application opens bundled CBox in Quality mode at 3840 x 2160.
