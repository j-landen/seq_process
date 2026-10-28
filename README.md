# FLIR SEQ Video Processing

## Overview
This repository contains a Python script (`Seq_process.py`) for batch processing FLIR `.seq` thermal video files. The workflow uses the `split_seqs` utility from [FLIRPy](https://github.com/LJMUAstroecology/flirpy) to extract individual frames, retrieves frame timestamps, generates colorized thermal images, and creates AVI videos for visualization.

## Requirements
- Python 3
- [FLIRPy](https://github.com/LJMUAstroecology/flirpy), including its `split_seqs` utility
- [ExifTool](https://exiftool.org/)
- [NumPy](https://numpy.org/)
- [ImageIO](https://imageio.readthedocs.io/)
- [OpenCV](https://opencv.org/)
- [tqdm](https://tqdm.github.io/)

## Usage
1. Install FLIRPy and the required Python dependencies.
2. Configure the input directory, output directory, and paths to ExifTool and the FLIRPy `split_seqs` utility in `Seq_process.py`.
3. Place the FLIR `.seq` files in the input directory.
4. Run the script:

   ```bash
   python Seq_process.py
   ```

The script processes all `.seq` files in the input directory.

## Outputs
For each input video, the script generates:
- **Radiometric frames:** TIFF images extracted by FLIRPy.
- **Timestamp lookup table:** `Timestamp_lut.csv`, containing video ID, frame number, and timestamp.
- **Colorized frames:** Rainbow-colored PNG images for visualization.
- **AVI video:** A colorized video assembled from the PNG frames at the estimated frame rate.

## Notes
- Directory paths must be configured for the local environment before execution.
- Frame timestamps are extracted from FLIR metadata using ExifTool.
- Colorized images are normalized individually for visualization and should not be interpreted as calibrated temperature maps.
- This repository supports research data processing and visualization.

## Acknowledgments
This workflow uses [FLIRPy](https://github.com/LJMUAstroecology/flirpy) for FLIR SEQ file extraction. Please consult the FLIRPy repository for its citation and licensing information.

## License
See the repository license for terms of use.


## License
Please refer to the repository license for terms of use.

## Citation

If this software is used in a publication, please cite the following manuscript:

Vandendoren Morgane, Landen Jason G, Rogers Joseph F, Killmer Samantha, Alamiri Baizar, Pohlman Celeste, Tattersall Glenn J, Bedford Nicole L, Nelson Adam C (2025) *Oxytocin neurons signal state-dependent transitions from rest to thermogenesis and behavioral arousal in social and non-social settings* eLife 14:RP108212

https://doi.org/10.7554/eLife.108212.3

## Contact

For questions about the software or its use in the associated research, contact the corresponding author listed in the manuscript.
