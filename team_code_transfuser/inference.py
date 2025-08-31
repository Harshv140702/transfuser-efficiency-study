import torch
import time
import os
from model import LidarCenterNet
from config import GlobalConfig

def benchmark_model(model_path):
    config = GlobalConfig(setting='eval')
    device = torch.device('cuda')

    # Load model - specify the correct number of LiDAR channels (3 in this case)
    model = LidarCenterNet(config, device, 'transFuser', 'regnety_032', 'regnety_032', True)
    
    # Load pre-trained weights
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle distributed training prefix if present
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    
    model.eval()
    model.to(device)

    # Create dummy inputs with 3 LiDAR channels
    dummy_image = torch.randn(1, 3, 256, 1024).to(device)  # RGB image
    dummy_lidar = torch.randn(1, 3, 256, 256).to(device)    # LiDAR BEV (now 3 channels)
    dummy_velocity = torch.randn(1, 1).to(device)            # Ego velocity
    dummy_target_point = torch.randn(1, 2).to(device)        # Target point
    dummy_target_point_image = torch.randn(1, 1, 256, 256).to(device)  # Target point image

    # Warm-up (important for CUDA)
    print("Warming up...")
    for _ in range(10):
        with torch.no_grad():
            _ = model.forward_ego(dummy_image, dummy_lidar, dummy_target_point, 
                                 dummy_target_point_image, dummy_velocity)

    # Measure inference time
    num_runs = 100
    print(f"Running benchmark for {num_runs} iterations...")
    
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model.forward_ego(dummy_image, dummy_lidar, dummy_target_point, 
                                 dummy_target_point_image, dummy_velocity)
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / num_runs * 1000  # Convert to ms
    fps = num_runs / total_time  # Frames per second

    print("\nBenchmark Results:")
    print(f"Total time for {num_runs} runs: {total_time:.2f} seconds")
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"FPS: {fps:.2f}")

    # Memory usage
    print("\nMemory Usage:")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2):.2f} MB")
    
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated(device) / (1024**2):.2f} MB")
        print(f"GPU Memory cached: {torch.cuda.memory_reserved(device) / (1024**2):.2f} MB")

if __name__ == "__main__":
    # Path to your pre-trained model
    model_path = "/mnt/iusers01/fse-ugpgt01/compsci01/f15583hs/Dissertation/transfuser/team_code_transfuser/log_reg2/transfuser/model_20.pth"  # Update this path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    
    benchmark_model(model_path)