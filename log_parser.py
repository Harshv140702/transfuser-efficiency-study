import re
import sys
import statistics

def analyze_inference_time(file_path):
    """
    Analyzes a file for inference time statistics.
    Returns: average, count, total, max, second max, min, median, std_dev, p90
    """
    inference_time_regex = re.compile(r"Inference time: (\d+\.?\d*)")
    times = []

    try:
        with open(file_path, 'r') as f:
            for line in f:
                match = inference_time_regex.search(line)
                if match:
                    times.append(float(match.group(1)))

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    if not times:
        print("No inference time entries were found in the file.")
        return None

    times_sorted = sorted(times, reverse=True)
    count = len(times)
    total_time = sum(times)
    avg = statistics.mean(times)
    max_time = times_sorted[0]
    second_max_time = times_sorted[1] if count >= 2 else 0.0
    min_time = min(times)
    median_time = statistics.median(times)
    std_dev = statistics.stdev(times) if count > 1 else 0.0
    p90 = statistics.quantiles(times, n=100)[89]  # 90th percentile

    return {
        "count": count,
        "total": total_time,
        "average": avg,
        "max": max_time,
        "second_max": second_max_time,
        "min": min_time,
        "median": median_time,
        "std_dev": std_dev,
        "p90": p90
    }

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <path_to_log_or_out_file>")
        sys.exit(1)

    file_name = sys.argv[1]
    stats = analyze_inference_time(file_name)

    if stats:
        print(f"Analysis of '{file_name}':")
        print(f"---------------------------------")
        print(f"Found {stats['count']} inference time entries.")
        print(f"Total combined inference time: {stats['total']:.2f} ms")
        print(f"Average inference time: {stats['average']:.2f} ms")
        print(f"Median inference time: {stats['median']:.2f} ms")
        print(f"Minimum inference time: {stats['min']:.2f} ms")
        print(f"Maximum inference time: {stats['max']:.2f} ms")
        if stats['count'] >= 2:
            print(f"Second highest inference time: {stats['second_max']:.2f} ms")
        print(f"Standard deviation: {stats['std_dev']:.2f} ms")
        print(f"90th percentile inference time: {stats['p90']:.2f} ms")
