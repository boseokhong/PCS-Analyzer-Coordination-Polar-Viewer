# Coordination Polar Viewer

Coordination Polar Viewer is a standalone-capable plugin for **PCS Analyzer** that compares multiple XYZ coordination structures in a metal-centred 2D polar representation.

## Version

`0.1.2`

## Plugin ID

`coordination_polar_viewer`

## Features

- Load a reference XYZ structure and one or more comparison structures.
- Metal-centred polar visualization of coordination environments.
- Per-structure metal selection and optional manual metal coordinates.
- Coordination-sphere detection and axis alignment.
- Raw and Kabsch-aligned RMSD analysis.
- Metal-ligand distance and ligand-metal-ligand angle analysis.
- Coordination-geometry indices and trans-pair analysis.
- Shannon ionic-radius correlation tools.
- Session save/restore.
- Figure and numeric-data export.

## Requirements

Required Python packages:

- `numpy`
- `matplotlib`

Optional:

- `mplcursors` for richer hover tooltips. The plugin includes a fallback tooltip implementation when it is not installed.

PCS Analyzer compatibility:

- PCS Analyzer `>= 1.3.4`

## PCS Analyzer installation

The recommended installation method is through **PCS Analyzer > Module Manager** using the official plugin catalog.

The packaged plugin ZIP must contain `manifest.json` and `plugin.py` at the archive root.

## Standalone use

The plugin can also be launched directly:

```bash
python plugin.py
```

## Release packaging

For release `0.1.2`, create a release asset named:

```text
coordination_polar_viewer-0.1.2.zip
```

with this archive layout:

```text
manifest.json
plugin.py
```

Use the uploaded release asset itself—not GitHub's automatically generated source archive—as the Module Manager download target.

## License

BSD 3-Clause License. See [LICENSE](LICENSE).
