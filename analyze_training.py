"""
Script to analyze TensorBoard training logs for PPO models.
Reads event files and extracts key metrics to understand training progress and plateaus.
"""

import os
import numpy as np
from collections import defaultdict

try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("Installing tensorboard...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'tensorboard'])
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_tensorboard_logs(log_dir):
    """Load all scalar data from a TensorBoard log directory."""
    event_acc = EventAccumulator(log_dir)
    event_acc.Reload()
    
    tags = event_acc.Tags()
    scalar_tags = tags.get('scalars', [])
    
    data = {}
    for tag in scalar_tags:
        events = event_acc.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]
        data[tag] = {'steps': steps, 'values': values}
    
    return data


def analyze_model(model_name, log_path):
    """Analyze a single model's training logs."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {model_name}")
    print(f"{'='*80}")
    
    if not os.path.exists(log_path):
        print(f"  ERROR: Path does not exist: {log_path}")
        return None
    
    try:
        data = load_tensorboard_logs(log_path)
    except Exception as e:
        print(f"  ERROR loading logs: {e}")
        return None
    
    if not data:
        print("  No scalar data found in logs.")
        return None
    
    print(f"\nAvailable metrics: {list(data.keys())}")
    
    analysis = {}
    
    # Key metrics to analyze
    key_metrics = [
        'rollout/ep_rew_mean',
        'rollout/ep_len_mean', 
        'train/loss',
        'train/policy_loss',
        'train/value_loss',
        'train/entropy_loss',
        'train/approx_kl',
        'train/clip_fraction',
        'train/explained_variance',
        'train/learning_rate',
    ]
    
    for metric in key_metrics:
        if metric in data:
            values = np.array(data[metric]['values'])
            steps = np.array(data[metric]['steps'])
            
            analysis[metric] = {
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'final': float(values[-1]) if len(values) > 0 else None,
                'first': float(values[0]) if len(values) > 0 else None,
                'count': len(values),
                'final_step': int(steps[-1]) if len(steps) > 0 else None,
            }
            
            # Calculate trend (last 20% vs first 20%)
            if len(values) >= 10:
                first_portion = values[:len(values)//5]
                last_portion = values[-len(values)//5:]
                analysis[metric]['trend'] = float(np.mean(last_portion) - np.mean(first_portion))
                
                # Check for plateau (low std in last 30%)
                last_30_pct = values[-len(values)*3//10:]
                analysis[metric]['last_30_std'] = float(np.std(last_30_pct))
                analysis[metric]['last_30_mean'] = float(np.mean(last_30_pct))
    
    # Print analysis
    print("\n--- REWARD ANALYSIS ---")
    if 'rollout/ep_rew_mean' in analysis:
        rew = analysis['rollout/ep_rew_mean']
        print(f"  Episode Reward Mean:")
        print(f"    First: {rew['first']:.2f}")
        print(f"    Final: {rew['final']:.2f}")
        print(f"    Max:   {rew['max']:.2f}")
        print(f"    Min:   {rew['min']:.2f}")
        print(f"    Trend (last-first 20%): {rew.get('trend', 'N/A'):.2f}")
        print(f"    Last 30% Mean: {rew.get('last_30_mean', 'N/A'):.2f}")
        print(f"    Last 30% Std:  {rew.get('last_30_std', 'N/A'):.2f}")
        print(f"    Total steps:   {rew['final_step']}")
        
        # Plateau detection
        if rew.get('last_30_std', float('inf')) < 50:
            print(f"    ⚠️  PLATEAU DETECTED: Low variance in last 30% of training")
    
    print("\n--- EPISODE LENGTH ---")
    if 'rollout/ep_len_mean' in analysis:
        ep_len = analysis['rollout/ep_len_mean']
        print(f"  Episode Length Mean:")
        print(f"    First: {ep_len['first']:.2f}")
        print(f"    Final: {ep_len['final']:.2f}")
        print(f"    Max:   {ep_len['max']:.2f}")
    
    print("\n--- TRAINING LOSSES ---")
    for loss_metric in ['train/loss', 'train/policy_loss', 'train/value_loss']:
        if loss_metric in analysis:
            loss = analysis[loss_metric]
            metric_name = loss_metric.split('/')[-1]
            print(f"  {metric_name}:")
            print(f"    First: {loss['first']:.4f} -> Final: {loss['final']:.4f}")
            print(f"    Trend: {loss.get('trend', 'N/A'):.4f}")
    
    print("\n--- POLICY HEALTH INDICATORS ---")
    if 'train/entropy_loss' in analysis:
        ent = analysis['train/entropy_loss']
        print(f"  Entropy Loss:")
        print(f"    First: {ent['first']:.4f} -> Final: {ent['final']:.4f}")
        if abs(ent['final']) < 0.1:
            print(f"    ⚠️  LOW ENTROPY: Policy may be too deterministic")
    
    if 'train/approx_kl' in analysis:
        kl = analysis['train/approx_kl']
        print(f"  Approx KL Divergence:")
        print(f"    Mean: {kl['mean']:.4f}, Max: {kl['max']:.4f}")
        if kl['max'] > 0.1:
            print(f"    ⚠️  HIGH KL: Policy updates may be too aggressive")
    
    if 'train/clip_fraction' in analysis:
        clip = analysis['train/clip_fraction']
        print(f"  Clip Fraction:")
        print(f"    Mean: {clip['mean']:.4f}, Final: {clip['final']:.4f}")
        if clip['mean'] > 0.3:
            print(f"    ⚠️  HIGH CLIPPING: Consider reducing learning rate")
    
    if 'train/explained_variance' in analysis:
        ev = analysis['train/explained_variance']
        print(f"  Explained Variance:")
        print(f"    Final: {ev['final']:.4f}")
        if ev['final'] < 0.5:
            print(f"    ⚠️  LOW EV: Value function may be poor")
    
    return analysis


def compare_models(all_analyses):
    """Compare metrics across all models."""
    print("\n" + "="*80)
    print("CROSS-MODEL COMPARISON")
    print("="*80)
    
    print("\n--- Reward Progression Across Training Runs ---")
    print(f"{'Model':<50} {'Start':>10} {'Final':>10} {'Max':>10} {'Plateau?':>10}")
    print("-" * 90)
    
    for model_name, analysis in all_analyses.items():
        if analysis and 'rollout/ep_rew_mean' in analysis:
            rew = analysis['rollout/ep_rew_mean']
            plateau = "Yes" if rew.get('last_30_std', float('inf')) < 50 else "No"
            print(f"{model_name:<50} {rew['first']:>10.1f} {rew['final']:>10.1f} {rew['max']:>10.1f} {plateau:>10}")
    
    print("\n--- Key Observations ---")
    
    # Check if reward is improving across retraining
    rewards = []
    for model_name, analysis in all_analyses.items():
        if analysis and 'rollout/ep_rew_mean' in analysis:
            rewards.append((model_name, analysis['rollout/ep_rew_mean']['final']))
    
    if len(rewards) > 1:
        improving = all(rewards[i][1] <= rewards[i+1][1] for i in range(len(rewards)-1))
        if not improving:
            print("  ⚠️  Retraining is not consistently improving rewards!")
    
    # Check entropy across models
    print("\n--- Entropy Progression (Policy Exploration) ---")
    for model_name, analysis in all_analyses.items():
        if analysis and 'train/entropy_loss' in analysis:
            ent = analysis['train/entropy_loss']
            print(f"  {model_name}: {ent['first']:.4f} -> {ent['final']:.4f}")


def provide_recommendations(all_analyses):
    """Provide actionable recommendations based on analysis."""
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    last_model = list(all_analyses.keys())[-1]
    last_analysis = all_analyses[last_model]
    
    if not last_analysis:
        print("  Unable to provide recommendations - no valid analysis.")
        return
    
    recommendations = []
    
    # Check reward plateau
    if 'rollout/ep_rew_mean' in last_analysis:
        rew = last_analysis['rollout/ep_rew_mean']
        if rew.get('last_30_std', float('inf')) < 50:
            recommendations.append(
                "🔄 REWARD PLATEAU DETECTED:\n"
                "   - Increase entropy coefficient (ent_coef) to encourage more exploration\n"
                "   - Try learning rate scheduling (reduce LR over time)\n"
                "   - Consider curriculum learning with easier scenarios first"
            )
        
        if rew['max'] > 850 and rew['final'] < 850:
            recommendations.append(
                "📉 MODEL ACHIEVED HIGH REWARD BUT REGRESSED:\n"
                "   - Consider early stopping when reward reaches threshold\n"
                "   - Save checkpoints at peak performance\n"
                "   - Reduce learning rate to stabilize high-reward policy"
            )
    
    # Check entropy
    if 'train/entropy_loss' in last_analysis:
        ent = last_analysis['train/entropy_loss']
        if abs(ent['final']) < 0.5:
            recommendations.append(
                "🎯 LOW ENTROPY (deterministic policy):\n"
                "   - Increase ent_coef from current value (try 0.02-0.05)\n"
                "   - Policy may be stuck in local optimum"
            )
    
    # Check explained variance
    if 'train/explained_variance' in last_analysis:
        ev = last_analysis['train/explained_variance']
        if ev['final'] < 0.5:
            recommendations.append(
                "📊 LOW EXPLAINED VARIANCE:\n"
                "   - Value function is not predicting returns well\n"
                "   - Try increasing value function network size\n"
                "   - Consider increasing n_epochs for more value function updates"
            )
    
    # Check clip fraction
    if 'train/clip_fraction' in last_analysis:
        clip = last_analysis['train/clip_fraction']
        if clip['mean'] > 0.2:
            recommendations.append(
                "✂️ HIGH CLIP FRACTION:\n"
                "   - Policy updates are being clipped frequently\n"
                "   - Reduce learning rate (current may be too high)\n"
                "   - Consider reducing clip_range below 0.2"
            )
    
    # Environment-specific recommendations
    recommendations.append(
        "\n🎮 ENVIRONMENT-SPECIFIC SUGGESTIONS (based on harderEnv):\n"
        "   - Your reward structure has: goal=200, death=-25, time=-0.05/step, distance*3\n"
        "   - Consider increasing goal reward if agent isn't motivated enough\n"
        "   - The action_arr_size=12 means 12 actions per step - ensure this is appropriate\n"
        "   - Try adjusting distancepen multiplier if agent isn't making progress toward goal"
    )
    
    # General recommendations for plateau
    recommendations.append(
        "\n🔧 GENERAL PLATEAU-BREAKING STRATEGIES:\n"
        "   1. Learning Rate Schedule: Use linear decay or cosine annealing\n"
        "   2. Increase batch_size or n_steps for more stable gradients\n"
        "   3. Try different network architectures (deeper/wider)\n"
        "   4. Add reward shaping for intermediate goals\n"
        "   5. Normalize observations if not already doing so\n"
        "   6. Increase training timesteps significantly (10x current)"
    )
    
    for rec in recommendations:
        print(f"\n{rec}")


def main():
    base_path = "/Users/ryanheffernan/Documents/Buffalo/CSE446/CSE-4-546-Final-Project-Team-49/ppo_2d_tensorboard"
    
    models = [
        ("PPO_2D_mlp_MLP_1", f"{base_path}/PPO_2D_mlp_MLP_1"),
        ("2D_ppo_2d_retrain_2", f"{base_path}/2D_ppo_2d_retrain_2"),
        ("2D_ppo_2d_retrained_retrain_1", f"{base_path}/2D_ppo_2d_retrained_retrain_1"),
        ("2D_ppo_2d_retrained_retrained_retrain_1", f"{base_path}/2D_ppo_2d_retrained_retrained_retrain_1"),
    ]
    
    all_analyses = {}
    
    for model_name, log_path in models:
        analysis = analyze_model(model_name, log_path)
        all_analyses[model_name] = analysis
    
    compare_models(all_analyses)
    provide_recommendations(all_analyses)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
