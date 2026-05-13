
# Smart Card-Edge Inference for Physiological Stress Detection On ESP32

A brief description of what this project does and who it's for


## Installation

```bash
1. download WESAD dataset from https://www.kaggle.com/datasets/orvile/wesad-wearable-stress-affect-detection-dataset/data
2. Extract it and make the following folder structure:
data/S2 | data/S3 | data/S4
3. install virtual env and install necessary packages using requirements.txt
4. run the python phase scripts in order (phase1 > phase2 > phase3) 
5. simulation folder needs to know what port the esp32 is connected to and fed into the command like python simulations/simulate_S4.py --port /dev/ttyUSB0
```
    