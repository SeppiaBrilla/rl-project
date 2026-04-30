import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob



def parse_args():
    parser = argparse.ArgumentParser(description="Learning curve TD3,SAC,PPO plot Script")
    parser.add_argument("--algo", type=str, default="PPO",nargs="+", choices=[ "SAC", "TD3", "PPO"], help="Algorithm's learning curves to plot")
    return parser.parse_args()




def plotAllEnvironments(algorithm_names):
    # 1. Configuration
    environments = ['carRacing', 'cartPole', 'acroBot']
    # Professional color palette for algorithms
    algo_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'] 
    
    # 2. Initialize the Figure
    # We create 1 row and 3 columns. This happens ONCE.
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # 3. Data Processing and Plotting
    for algo_idx, algorithm in enumerate(algorithm_names):
        # Pick a color for this specific algorithm
        current_color = algo_colors[algo_idx % len(algo_colors)]
        
        for env_idx, env in enumerate(environments):
            ax = axes[env_idx]
            
            # Find all seed files (e.g., PPO_cartPole-1.csv, PPO_cartPole-2.csv)
            pattern = f'report/training_results/{algorithm}_{env}-*.csv'
            all_files = glob.glob(pattern)
            
            if not all_files:
                print(f"Warning: No files found for {algorithm} in {env} (Pattern: {pattern})")
                continue

            # Load and aggregate all seeds
            try:
                dataframes = [pd.read_csv(f) for f in all_files]
                combined = pd.concat(dataframes)
                
                # Group by epoch to get the mean and standard deviation across seeds
                stats = combined.groupby('epoch').agg({
                    'eval_reward_mean': 'mean',
                    'eval_reward_std': 'mean'  
                }).reset_index()
                ax.plot(
                    stats['epoch'], 
                    stats['eval_reward_mean'], 
                    color=current_color, 
                    lw=2, 
                    label=f'{algorithm}'
                )
                
                # Plot the Standard Deviation Shading
                ax.fill_between(
                    stats['epoch'], 
                    stats['eval_reward_mean'] - stats['eval_reward_std'], 
                    stats['eval_reward_mean'] + stats['eval_reward_std'], 
                    color=current_color, 
                    alpha=0.15
                )
            except Exception as e:
                print(f"Error processing {algorithm} in {env}: {e}")

    # 4. Final Formatting (applied to each subplot)
    for i, env in enumerate(environments):
        ax = axes[i]
        ax.set_title(f'Environment: {env}', fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Mean Eval Reward', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # Consolidate the legend so it only shows unique algorithm entries
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='lower right', fontsize=10)

    # Global Title
    plt.suptitle(f'Cross-Environment Learning Curves Comparison', fontsize=18, y=0.96)
    
    # Adjust layout to prevent overlapping labels
    plt.tight_layout()
    
    # 5. Save and Show
    if not os.path.isdir("report/plots"):
        os.makedirs("report/plots")
    output_filename = f'report/plots/algorithms_{'_'.join(algorithm_names)}_comparison.pdf'
    plt.savefig(output_filename, bbox_inches='tight')
    print(f"Success: Plot saved to report/plots/{output_filename}")
    plt.show()

if __name__ == "__main__":
    args = parse_args()
    plotAllEnvironments(args.algo)
