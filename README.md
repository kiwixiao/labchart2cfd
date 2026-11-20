# LabChart2CFD

Convert LabChart-exported .mat flow profile data to CSV format for CFD simulation boundary conditions.

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/kiwixiao/labchart2cfd.git

# Or clone and install locally (for development)
git clone https://github.com/kiwixiao/labchart2cfd.git
cd labchart2cfd
pip install -e .
```

Requires Python >= 3.10.

## Usage

### Process a single file

```bash
labchart2cfd process input.mat OSAMRI029 --start 12.97 --end 16.16
```

### With all options

```bash
labchart2cfd process input.mat OSAMRI029 \
    --row 2 --column 3 \
    --start 12.97 --end 16.16 \
    --workflow standard \
    --output-dir ./output
```

### Validate .mat structure

```bash
labchart2cfd validate input.mat
```

### Visualize data (helps find start/end times)

```bash
labchart2cfd visualize input.mat --row 2 --column 3
```

### Launch GUI

Running the bare command opens the GUI:

```bash
labchart2cfd
```

(`labchart2cfd gui` also works.)

The left sidebar is read from top to bottom:

1. **Recording.** Open a LabChart `.mat` export (or drop it on the window). Check that Flow, Trigger and Pressure name the right channels, then click a block on the overview to open it.
2. **Study type.** MRI, CPAP, Phase contrast or CT. Only the controls of that type are shown.
3. **Time window.** Click the plot once for Start and once for End, or type the times.
4. **CT only.** Select the scan step, then choose how the breath is built: reorder image parts, landmarks, or cycle frames.

Enter the Subject ID and press **Export**. The line above the button says what is still missing, and a panel lists the files after the export. The coloured line above the plot always says what a click on the plot will do.

## Workflow Types

- **MRI**: Standard OSAMRI processing (drift correction on full signal)
- **cpap**: CPAP variant (drift correction on windowed data)
- **phase-contrast**: Xenon phase contrast with voltage calibration

## Output

- `{subject}FlowProfile.csv`: Mass flow rate in kg/s
- `{subject}PressureProfile.csv`: Pressure in Pa (MRI/cpap only)
