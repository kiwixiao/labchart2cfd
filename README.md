# LabChart2CFD

Turn LabChart flow recordings into CSV boundary conditions for CFD in Star-CCM+. The window reads a LabChart `.mat` export or the native `.adicht` recording. The command line processes `.mat` exports and converts native recordings to `.mat`.

## Installation

```bash
# Install from GitHub
pip install git+https://github.com/kiwixiao/labchart2cfd.git

# Or clone and install locally (for development)
git clone https://github.com/kiwixiao/labchart2cfd.git
cd labchart2cfd
pip install -e .
```

Requires Python 3.10 or newer.

## The window

Running the bare command opens the window:

```bash
labchart2cfd
```

(`labchart2cfd gui` also works.) Help, Quick guide shows the steps below inside the app.

The sidebar is read from top to bottom:

1. **Recording.** Press Open recording and pick a `.mat` export or a native `.adicht` file, or drop the file on the window. Check that Flow, Trigger and Pressure name the right channels, then click a block on the overview to open it.
2. **Study type.** MRI, CPAP, Phase contrast or CT. Only the controls of that type are shown.
3. **Time window.** Click the plot once for Start and once for End, or type the times. In CT, Select step sets them. CT also has the **Drift correction breath** here: one complete breath, typed as From and To or picked with two clicks (Pick on plot). Its mean flow is the drift taken out of the export. Without one, the drift is measured on the image grid, and the line above Export says so in amber.
4. **Triggers** (MRI, CPAP and Phase contrast). Optional. Pick a run of trigger pulses, and Mark triggers sets Start and End to it.
4. **CT scan** (CT). Detect steps finds the scan steps on the Trigger channel; it also runs when a block is opened. Select step sets Start and End to the step and draws the image grid. Temporal Res (ms) and Images in step are set separately, and a typed image count moves End at once. Start Shift (ms) moves the window and waits for Apply shift.
5. **Build the breath** (CT). Only the method on screen is exported.
   - **Reorder image parts.** Pick ranges of images and put them in breathing order.
   - **Inhale and exhale landmarks.** Mark where inhale starts and exhale ends. The profile is rotated to start at the inhale image when needed.
   - **Label cycle frames.** Click the images of a breath in order. You may go back to an earlier image. Each run of consecutive images is cut out, and the runs are joined in the order you clicked.

   With Reorder image parts and Label cycle frames, **Extra flow** keeps the recorded flow after a frame up to a time you set (Until, typed or picked on the plot), before the next frame starts. It is drawn hatched on the plot.

Enter the Subject ID and press **Export**. The line above the button says what is still missing, and Export stays held back until nothing is. After the export a panel lists the files and the numbers. The coloured line above the plot always says what a click on the plot will do. A control that would remove marks you placed asks first; only the Clear buttons do not.

### The composed breath (CT)

With Reorder image parts and Label cycle frames, Export first opens the Composed Breath Preview. It shows the flow that will be written, with the frame number above each image and the image number below, the extra flow shaded, and notes on the drift breath.

- Under **Complete half**, choose the half kept as recorded: above zero or below zero. The other half is multiplied by one factor so the volume in equals the volume out. No time moves.
- A factor below 0.9 or above 1.1 is cautioned in amber: the two halves differed that much before scaling, so check the extra flow and the drift correction breath.
- If one half holds no volume, or less than 1 percent of the other, nothing can be written. Check the drift correction breath and the frames, then export again.
- The written flow file is read back. The volumes above and below zero and the net volume go to the result panel and the details file. A net volume above 0.01 mL is shown as a red error, and that file must not be used.

The net volume is taken over the profile from its first sample to its last. When Star-CCM+ repeats the profile, the step from the last sample back to the first is not part of that check.

### Plot controls

Above the plot, **Flow Y** and **Trigger Y** enlarge or shrink each scale (the mouse wheel over the numbers of an axis does the same). Fit sets the flow scale to the flow in view. Full shows the whole range of the trigger channel in the block, which is also what the trigger scale starts with and keeps when the view zooms in time. The mouse wheel over a plot zooms in time about the pointer.

The comments typed in LabChart are drawn as dashed purple lines with their text, from a `.mat` export (`com`, `comtext`, `tickrate`) and from a native file alike. View, Show comments hides them.

## Command line

The `process` command reads a `.mat` export. Convert a native recording first with `convert-adicht` (below).

```bash
labchart2cfd process input.mat OSAMRI029 --start 12.97 --end 16.16

labchart2cfd process input.mat OSAMRI029 \
    --row 2 --column 3 \
    --start 12.97 --end 16.16 \
    --workflow standard \
    --output-dir ./output

# CT: the window comes from a scan step on the trigger channel
labchart2cfd process input.mat CT001 -w ct --step-number 1 --inhale-start 5.2 --temporal-resolution 200
```

`labchart2cfd process --help` lists every option, including `--trigger-row`, `--pressure-row`, `--no-pressure`, `--bag` and `--sample-rate`. The composed breath (Reorder image parts, Label cycle frames, the drift correction breath and extra flow) is built in the window only.

Other commands:

```bash
labchart2cfd validate input.mat                      # describe the .mat structure
labchart2cfd visualize input.mat --row 2 --column 3  # plot a block to find start and end times
labchart2cfd detect-steps input.mat                  # list the CT scan steps on the trigger channel
labchart2cfd version
```

### Study types

| In the window | `--workflow` | What differs |
| --- | --- | --- |
| MRI | `standard` | Drift correction on the full signal, then the window is cut. |
| CPAP | `cpap` | Drift correction on the windowed signal. |
| Phase contrast | `phase-contrast` | Xenon density (5.761 kg/m³) and voltage calibration. `--bag` (Bag1, Bag2, Bag4) is on the command line only. No pressure output. |
| CT | `ct` | The window comes from a scan step on the trigger channel and holds whole images. |

## Native LabChart files (.adicht)

A native LabChart recording can be used directly, with no manual export from LabChart first.

In the GUI, open or drop the `.adicht` file exactly as you would a `.mat` export. It takes longer to open, because the recording is decoded.

On the command line, convert it to a `.mat` with the same layout as LabChart's own "Export MATLAB":

```bash
labchart2cfd convert-adicht rec.adicht --info            # look inside, convert nothing
labchart2cfd convert-adicht rec.adicht --out converted   # whole recording
labchart2cfd convert-adicht rec.adicht --out converted --channels 1,3,5 --record 2 --start 950 --end 1150
```

The output folder must not be the recording's own folder. `python -m labchart2cfd.io.adicht` takes the same options.

What to expect:

- The format has no public description. The reader was worked out from real recordings and checked sample for sample against LabChart's own exports of them (LabChart 6.1.2 and 8.1.x). Where it converts, the samples, index tables, titles, units, comments and block times equal the export. The two display range variables (`rangemin`, `rangemax`) are not written, and `comtext` is left out when a recording has no comments (LabChart writes it empty). This program reads none of the three.
- A channel that LabChart computes is not stored in the file. The Integral calculation is recomputed with LabChart's arithmetic. A channel with any other calculation is left empty and named in the output.
- A file with anything the reader has not been checked on is refused with the reason (exit code 2). It does not guess. Export that file from LabChart by hand. Of 198 native files tried, 158 open and 40 are refused: files saved from a selection, files with records deleted in LabChart, imports from other systems, and a few with settings never seen in a checked file.
- One known gap. Only recordings whose computed channels are Integrals had an export to compare against. A channel with another calculation (a digital filter, arithmetic, a derivative) is left empty when its marker is found in the file. If LabChart stored such a calculation without that marker, the reader would return the channel's raw input with no warning. No such case was seen, and it cannot be ruled out. If a recording uses those calculations, compare it with a manual export once.
- When a new kind of recording comes along (another setup or LabChart version), export one file by hand once and compare:

```bash
LABCHART2CFD_ADICHT_FILE=rec.adicht LABCHART2CFD_ADICHT_MAT=rec.mat pytest tests/test_recording_loading.py tests/test_cli.py
```

## Output

The window writes to the folder shown above Export. By default it is `FlowResults_{subject}` next to the recording.

| File | When | Content |
| --- | --- | --- |
| `{subject}FlowProfile.csv` | Every export | Columns `"time (s)"` and `"Massflowrate (kg/s)"`. Mass flow is written in scientific notation (`3.001234560e-04`). |
| `{subject}PressureProfile.csv` | Every study type except phase contrast, when a pressure channel is chosen and has data | Columns `"time (s)"` and `"Pressure (Pa)"`. |
| `{subject}_trigger_info.txt` | Every export from the window | The window in the recording and in the exported profile. MRI, CPAP and Phase contrast add the trigger numbers. CT adds the scan step, the images and the drift: where it was measured and how much was removed. A composed breath also adds the frame order, the complete half, the volumes before and after scaling, the scale factor and the net volume read back. |
| `{subject}_sanity_check.png` | The window, except CT Reorder image parts | A picture of the exported data. |
| `{subject}ImageReorderMap.csv` | CT, Reorder image parts and Label cycle frames | One row per frame: `new_frame`, `original_image`, `original_time_s`, `profile_start_s`, `profile_end_s`, `extra_flow_s`, `extra_flow_until_s`. The last field is empty for a frame with no extra flow. |
| `{subject}_frame_order.png` | CT, Reorder image parts and Label cycle frames | The preview picture with its notes. |

The command line writes the two CSV files only.
