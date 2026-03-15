"""
Evaluation script for RawEuristicAgent.

This script creates a RawEuristicAgent, runs comprehensive metrics on it,
and prints all results in a structured format optimized for AI agent development.
"""
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path for imports
# Go up 4 levels: eval_raw_euristic_agent.py -> eval -> envs -> src -> project_root
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Also set PYTHONPATH environment variable for subprocess calls
os.environ['PYTHONPATH'] = project_root

from src.envs.agents.raw_euristic_agent import RawEuristicAgent
from src.envs.agent_metrics import Metrics
from src.game.template import EXTENDED_KUTULU_ACTIONS


def print_section_header(title):
    """Print a section header with clear delimiters."""
    print()
    print("=" * 70)
    print(f"=== {title} ===")
    print("=" * 70)


def print_behavior_metric(name, values):
    """
    Print a behavior metric with all 5 fields.
    
    Args:
        name: Metric name (e.g., 'check_exp')
        values: Tuple of (accuracy, std, top_action_diversity, max_logit, mean_logit)
    """
    if isinstance(values, tuple) and len(values) == 5:
        accuracy, std, top_action_diversity, max_logit, mean_logit = values
        print(f"\n{name}:")
        print(f"  accuracy: {accuracy:.6f}")
        print(f"  std: {std:.6f}")
        print(f"  top_action_diversity: {top_action_diversity}")
        print(f"  max_logit: {max_logit:.6f}")
        print(f"  mean_logit: {mean_logit:.6f}")
    else:
        print(f"\n{name}: {values}")


def analyze_metrics(results):
    """
    Analyze metrics and provide AI-actionable insights.
    
    Args:
        results: Dictionary of metric results
        
    Returns:
        Dictionary of insights
    """
    insights = {
        'failing_metrics': [],
        'warnings': [],
        'strengths': [],
        'recommendations': []
    }
    
    # Define thresholds
    ACCURACY_THRESHOLD = 0.70
    DIVERSITY_TARGET = 4  # Target value: agent should adapt to all 4 rotations
    
    # Analyze behavior metrics
    behavior_metrics = {
        'check_exp': 'explorer targeting',
        'check_wan': 'wanderer avoidance',
        'check_slsh': 'slasher avoidance',
        'check_exp_normal_plan1': 'explorer targeting with PLAN (close)',
        'check_exp_normal_plan0': 'explorer targeting with PLAN (far)'
    }
    
    for metric_name, description in behavior_metrics.items():
        if metric_name in results:
            values = results[metric_name]
            if isinstance(values, tuple) and len(values) >= 3:
                accuracy, std, top_action_diversity = values[0], values[1], values[2]
                
                if accuracy < ACCURACY_THRESHOLD:
                    insights['failing_metrics'].append(metric_name)
                    insights['recommendations'].append(
                        f"Improve {description}: accuracy is {accuracy:.2%} (target: >{ACCURACY_THRESHOLD:.0%})"
                    )
                else:
                    insights['strengths'].append(f"{description}: {accuracy:.2%} accuracy")
                
                # Check if agent adapts to all 4 directional rotations
                if top_action_diversity < DIVERSITY_TARGET:
                    insights['warnings'].append(
                        f"{metric_name}: not adapting to all rotations (diversity={top_action_diversity}, target={DIVERSITY_TARGET})"
                    )
                elif top_action_diversity == DIVERSITY_TARGET:
                    insights['strengths'].append(f"{description}: correctly adapts to all 4 rotations")
    
    # Analyze accuracy scores
    if 'acc_weighted' in results:
        acc = results['acc_weighted']
        if acc < 0.70:
            insights['recommendations'].append(
                f"Overall weighted accuracy is low ({acc:.2%}). Focus on wanderer avoidance and explorer targeting."
            )
    
    # Analyze challenge scores
    challenge_scores = []
    if 'winner_score_ppo' in results and results['winner_score_ppo'] is not None:
        score = results['winner_score_ppo']
        challenge_scores.append(score)
        if score < 0.40:
            insights['recommendations'].append(
                f"Low win rate vs PPO ({score:.2%}). Agent needs strategic improvements."
            )
    
    if 'winner_score_qdn_conv' in results and results['winner_score_qdn_conv'] is not None:
        score = results['winner_score_qdn_conv']
        challenge_scores.append(score)
        if score < 0.40:
            insights['recommendations'].append(
                f"Low win rate vs QDN Conv ({score:.2%}). Agent needs strategic improvements."
            )
    
    # Calculate overall grade
    if 'acc_weighted_full' in results:
        score = results['acc_weighted_full']
        if score >= 0.90:
            grade = 'A'
        elif score >= 0.80:
            grade = 'B'
        elif score >= 0.70:
            grade = 'C'
        elif score >= 0.60:
            grade = 'D'
        else:
            grade = 'F'
        insights['overall_grade'] = grade
        insights['overall_score'] = score
    
    return insights


def main():
    """
    Main evaluation function.
    
    Creates a RawEuristicAgent, runs metrics with challenge mode enabled,
    prints all metrics in AI-friendly format, and provides actionable insights.
    """
    print("=" * 70)
    print("=== RAWEURISTICAGENT EVALUATION ===")
    print("=" * 70)
    print()
    
    # Create RawEuristicAgent
    # Note: train=True is required for metrics to work properly
    # (non-training agents return zero metrics)
    np.random.seed(42)
    agent = RawEuristicAgent(
        state_type='raw',
        action_space_n=len(EXTENDED_KUTULU_ACTIONS),
        train=True,
        verbose=False
    )
    
    print(f"Agent Type: {type(agent).__name__}")
    print(f"State Type: {agent.state_type}")
    print(f"Action Space: {agent.action_space_n}")
    
    # Create Metrics instance with challenge mode enabled
    # This will load default competitors (ppo and qdn_conv)
    try:
        metrics = Metrics(use_challenge=True)
        
        if metrics.competitors is None:
            print("\nWARNING: Could not load default competitors.")
            print("Challenge metrics will not be available.")
    except Exception as e:
        print(f"\nERROR: Failed to create Metrics instance: {e}")
        sys.exit(1)
    
    # Run metrics calculation
    print("\nCalculating metrics (this may take a few minutes)...")
    
    try:
        results = metrics._calculate_metrics(agent, use_challenge=True)
    except Exception as e:
        print(f"\nERROR: Failed to calculate metrics: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print all metrics in organized sections
    print_section_header("BEHAVIOR VALIDATION METRICS")
    
    behavior_metrics = [
        'check_exp',
        'check_wan',
        'check_slsh',
        'check_exp_normal_plan1',
        'check_exp_normal_plan0'
    ]
    
    for metric_name in behavior_metrics:
        if metric_name in results:
            print_behavior_metric(metric_name, results[metric_name])
    
    print_section_header("WEIGHTED ACCURACY SCORES")
    print()
    
    accuracy_metrics = ['acc_weighted', 'acc_weighted_full']
    
    for metric_name in accuracy_metrics:
        if metric_name in results:
            value = results[metric_name]
            print(f"{metric_name}: {value:.6f}")
    
    print_section_header("CHALLENGE MATCH RESULTS")
    print()
    
    # Extract challenge scores
    winner_score_ppo = results.get('winner_score_ppo', None)
    winner_score_qdn_conv = results.get('winner_score_qdn_conv', None)
    
    if winner_score_ppo is not None:
        print(f"winner_score_ppo: {winner_score_ppo:.6f}")
    else:
        print("winner_score_ppo: N/A (competitor not available)")
    
    if winner_score_qdn_conv is not None:
        print(f"winner_score_qdn_conv: {winner_score_qdn_conv:.6f}")
    else:
        print("winner_score_qdn_conv: N/A (competitor not available)")
    
    # Calculate and print average
    if winner_score_ppo is not None and winner_score_qdn_conv is not None:
        average_score = (winner_score_ppo + winner_score_qdn_conv) / 2
        print(f"\naverage_challenge_score: {average_score:.6f}")
    else:
        print("\naverage_challenge_score: N/A (one or both competitors missing)")
    
    # Analyze metrics and provide insights
    print_section_header("AI INSIGHTS")
    
    insights = analyze_metrics(results)
    
    print()
    if insights.get('overall_grade'):
        print(f"OVERALL_GRADE: {insights['overall_grade']}")
        print(f"OVERALL_SCORE: {insights['overall_score']:.6f}")
    
    if insights['strengths']:
        print("\nSTRENGTHS:")
        for strength in insights['strengths']:
            print(f"  ✓ {strength}")
    
    if insights['failing_metrics']:
        print("\nFAILING_METRICS:")
        for metric in insights['failing_metrics']:
            print(f"  ✗ {metric}")
    
    if insights['warnings']:
        print("\nWARNINGS:")
        for warning in insights['warnings']:
            print(f"  ⚠ {warning}")
    
    if insights['recommendations']:
        print("\nRECOMMENDATIONS:")
        for i, rec in enumerate(insights['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print_section_header("EVALUATION COMPLETE")
    print()
    
    # Return appropriate exit code
    if insights['failing_metrics']:
        sys.exit(1)  # Some metrics failed
    else:
        sys.exit(0)  # All metrics passed


if __name__ == '__main__':
    main()
