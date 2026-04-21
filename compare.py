
import numpy as np
import matplotlib.pyplot as plt


def smooth(data, window=None):
    """
    Moving average smoothing (adaptive)
    """
    data = np.array(data, dtype=float)

    if len(data) < 5:
        return data

    if window is None:
        window = max(5, len(data) // 20)

    if len(data) < window:
        return data

    kernel = np.ones(window) / window
    return np.convolve(data, kernel, mode='valid')


def plot_metric(histories, key, title, ylabel):
    """
    Plot a metric dynamically from histories
    """

    plt.figure()

    for algo, hist in histories.items():

        if key not in hist:
            print(f"[WARN] {algo} missing '{key}', skipping")
            continue

        values = np.array(hist[key], dtype=float)

        if len(values) == 0:
            continue

        smoothed = smooth(values)

        x = np.arange(len(smoothed))

        plt.plot(x, smoothed, label=algo)

    plt.title(title)
    plt.xlabel("Episodes")
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)


def compare_all(histories):
    """
    Automatically detect and plot all metrics
    """

    if not histories:
        print("[ERROR] Empty histories")
        return

    # detect all keys across all algorithms
    all_keys = set()
    for hist in histories.values():
        all_keys.update(hist.keys())

    print(f"[INFO] Metrics detected: {list(all_keys)}")

    titles = {
        "rewards": ("Reward Comparison", "Total Reward"),
        "actor_loss": ("Actor Loss", "Loss"),
        "critic_loss": ("Critic Loss", "Loss"),
        "energy": ("Energy Consumption", "Energy"),
        "delay": ("Delay", "Time"),
        "throughput": ("Throughput", "Rate"),
    }

    for key in all_keys:
        title, ylabel = titles.get(
            key, (f"{key} Comparison", key)
        )
        plot_metric(histories, key, title, ylabel)

    plt.show()


def print_summary(histories):
    """
    Print real computed statistics
    """

    print("\n" + "="*60)
    print("FINAL PERFORMANCE SUMMARY (Computed from Training Data)")
    print("="*60)

    for algo, hist in histories.items():
        print(f"\n--- {algo} ---")

        for key, values in hist.items():

            values = np.array(values, dtype=float)

            if len(values) == 0:
                continue

            mean_val = np.mean(values)
            std_val = np.std(values)
            max_val = np.max(values)
            min_val = np.min(values)
            last_val = values[-1]

            print(f"{key}:")
            print(f"  Mean : {mean_val:.4f}")
            print(f"  Std  : {std_val:.4f}")
            print(f"  Max  : {max_val:.4f}")
            print(f"  Min  : {min_val:.4f}")
            print(f"  Last : {last_val:.4f}")


def save_results(histories, filename="results.npy"):
    """
    Save real experiment results
    """
    np.save(filename, histories)
    print(f"[INFO] Results saved to {filename}")


def load_results(filename="results.npy"):
    """
    Load previously saved results
    """
    data = np.load(filename, allow_pickle=True).item()
    print(f"[INFO] Results loaded from {filename}")
    return data
