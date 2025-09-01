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
Follow the original TransFuser instructions for training and evaluation. The efficiency analysis tools can be used with both the original and efficiency-focused implementations.

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

## 📑 Citation

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
