# transfuser-efficiency-study
This repository is a research oriented extension of the [TransFuser repository](https://github.com/autonomousvision/transfuser/tree/2022) aimed at analysing the computational efficiency of the encoders and the self attention mechanism.

## Setup

Follow the original TransFuser setup instructions:

```bash
git clone https://github.com/your-username/transfuser-efficiency.git
cd transfuser-efficiency
git checkout 2022
chmod +x setup_carla.sh
./setup_carla.sh
conda env create -f environment.yml
conda activate tfuse
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cu102.html
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.11.0/index.html
```

## Model Versions

This repository contains two implementations of **TransFuser**:

- **`transfuser.py`** – The original implementation from the base repository  
- **`transfusereff.py`** – An efficiency-focused version optimized for performance analysis

To switch between model implementations, simply modify the import statement in (`team_code_transfuser/model.py`):

```python
# For the original implementation:
from transfuser import TransFuser

# For the efficiency-focused implementation:
from transfusereff import TransFuserEff as TransFuser
```

## Dataset generation, training and evaluation
Follow the original TransFuser instructions for generating dataset, training and evaluation. The added modifications do not change these processes.

## Inference time analysis
The custom parser (`log_parser.py`) extracts and summarizes inference times from log files (if stored).  
It reports the following statistics:

- **Count of inference time entries**  
- **Total combined inference time**  
- **Average inference time**  
- **Median inference time**  
- **Minimum inference time**  
- **Maximum inference time**  
- **Second highest inference time**  
- **Standard deviation**  
- **90th percentile inference time**  

Usage:
```python
python log_parser.py <path_to_log_file>
```

## Results
The study evaluated three encoder variants of the TransFuser architecture (ResNet, RegNet, and EfficientNet-B0) on the CARLA simulator.

ResNet (baseline): Avg. Driving Score 0.309, Route Completion 0.614, Inference Time 60.5 ms (~16.5 Hz), Model Size 197 MB.
Unstable with frequent blockages (92.9%) and latency spikes up to 3.8 s.

RegNet: Driving Score 0.458, Route Completion 1.538, Inference Time 83.8 ms (~12 Hz), Model Size 643 MB.
Strong performance but too slow and memory-heavy for real-time use.

EfficientNet-B0 (40% data): Driving Score 0.381, Route Completion 1.079 (91% higher than ResNet), Inference Time 55.5 ms (~18 Hz, fastest and most stable), Model Size 76 MB.
Predictable latency with low variance.

EfficientNet-B0 (100% data): Driving Score 0.508, Route Completion 1.680, Inference Time 78.3 ms (~13 Hz), Model Size 76 MB.
Outperforms RegNet in driving ability while remaining far more efficient.

EfficientNet enables a practical balance between driving competency and real-time efficiency, making it the most viable candidate for real-world deployment.

## Citation

If you use this code in your research, please cite the original TransFuser papers:

```bibtex
@article{Chitta2023PAMI,
  author = {Chitta, Kashyap and
            Prakash, Aditya and
            Jaeger, Bernhard and
            Yu, Zehao and
            Renz, Katrin and
            Geiger, Andreas},
  title = {TransFuser: Imitation with Transformer-Based Sensor Fusion for Autonomous Driving},
  journal = {Pattern Analysis and Machine Intelligence (PAMI)},
  year = {2023},
}

@inproceedings{Prakash2021CVPR,
  author = {Prakash, Aditya and
            Chitta, Kashyap and
            Geiger, Andreas},
  title = {Multi-Modal Fusion Transformer for End-to-End Autonomous Driving},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2021}
}
