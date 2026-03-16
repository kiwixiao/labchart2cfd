# LabChart2CFD

Convert LabChart-exported .mat flow profile data to Star-CCM+ compatible CSV format.

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

```bash
labchart2cfd gui
```

## Workflow Types

- **MRI**: Standard OSAMRI processing (drift correction on full signal)
- **cpap**: CPAP variant (drift correction on windowed data)
- **phase-contrast**: Xenon phase contrast with voltage calibration

## Output

- `{subject}FlowProfile.csv`: Mass flow rate in kg/s
- `{subject}PressureProfile.csv`: Pressure in Pa (MRI/cpap only)
