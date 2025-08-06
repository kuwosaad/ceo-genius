"""
test_nrpa.py

Unit tests for NRPA implementation.
"""
import math
import sys
import os
# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(__file__))

from collections import defaultdict
from nrpa import code, logsumexp, run_nrpa, PolicyManager, rollout


def test_code_determinism():
    """Test that code() produces consistent results"""
    action = "refinement1"
    c1 = code(action)
    c2 = code(action)
    assert c1 == c2


def test_softmax_sample():
    """Test softmax sampling behavior"""
    actions = ["a", "b", "c"]
    policy_manager = PolicyManager()
    
    # Uniform sampling when all weights are 0
    samples = [policy_manager.softmax_sample(actions) for _ in range(100)]
    # Should see all actions
    assert set(samples) == set(actions)
    
    # Bias sampling when weights differ
    policy_manager.policy[code("a")] = 10.0  # High weight
    policy_manager.policy[code("b")] = 0.0
    policy_manager.policy[code("c")] = 0.0
    
    samples = [policy_manager.softmax_sample(actions) for _ in range(100)]
    # Should mostly pick "a"
    a_count = samples.count("a")
    assert a_count > 80  # Very likely given 10x weight advantage


def test_logsumexp():
    """Test numerical stability of logsumexp"""
    # Normal case
    xs = [1.0, 2.0, 3.0]
    result = logsumexp(xs)
    expected = math.log(sum(math.exp(x) for x in xs))
    assert abs(result - expected) < 1e-10
    
    # Empty list
    assert logsumexp([]) == float('-inf')
    
    # Large numbers (would overflow regular exp)
    xs = [1000.0, 1001.0, 1002.0]
    result = logsumexp(xs)
    # Should not be inf
    assert math.isfinite(result)
    # Check relative values preserved
    shift = 1000.0
    expected = shift + math.log(sum(math.exp(x - shift) for x in xs))
    assert abs(result - expected) < 1e-10


def test_clip_policy():
    """Test policy weight clipping"""
    policy_manager = PolicyManager(max_weight=100.0)
    
    # Set extreme values
    policy_manager.policy[code("a")] = 1000.0
    policy_manager.policy[code("b")] = -1000.0
    policy_manager.policy[code("c")] = 50.0
    
    policy_manager.clip_policy()
    
    assert policy_manager.policy[code("a")] == 100.0
    assert policy_manager.policy[code("b")] == -100.0
    assert policy_manager.policy[code("c")] == 50.0


def test_adapt_probability_shift():
    """Test that adapt() shifts policy weights toward a sequence"""
    policy_manager = PolicyManager()
    seq = ["action1", "action2", "action3"]
    
    def mock_children_provider(step, prefix):
        if step == 0:
            return ["action1", "other1"]
        elif step == 1:
            return ["action2", "other2"]
        elif step == 2:
            return ["action3", "other3"]
        return []
    
    # Initial policy weights (all zero = uniform)
    initial_weight_1 = policy_manager.policy.get(code("action1"), 0.0)
    initial_weight_2 = policy_manager.policy.get(code("action2"), 0.0)
    
    # Adapt toward the sequence
    policy_manager.adapt(seq, alpha=1.0, children_provider=mock_children_provider)
    
    # Weights for chosen actions should increase
    final_weight_1 = policy_manager.policy.get(code("action1"), 0.0)
    final_weight_2 = policy_manager.policy.get(code("action2"), 0.0)
    
    assert final_weight_1 > initial_weight_1
    assert final_weight_2 > initial_weight_2


def test_rollout_terminates():
    """Test that rollout respects max_depth"""
    policy_manager = PolicyManager()
    initial_candidates = ["start"]
    max_depth = 3
    
    def mock_children_provider(step, prefix):
        if step < max_depth:
            return [f"action_{step}_1", f"action_{step}_2"]
        return []
    
    def mock_score_fn(seq):
        return len(seq) / max_depth  # Score based on length
    
    cache = {}
    score, seq = rollout(policy_manager, initial_candidates, max_depth, mock_children_provider, mock_score_fn, cache)
    
    # Sequence should not exceed max_depth
    assert len(seq) <= max_depth
    # Score should be in [0,1]
    assert 0.0 <= score <= 1.0


def test_run_nrpa_improves_over_random():
    """Test that run_nrpa improves over random baseline"""
    initial_strategies = ["strategy1", "strategy2", "strategy3"]
    max_depth = 3
    levels = 2
    iterations = 10
    
    def mock_children_provider(step, prefix):
        if step < max_depth:
            # Return deterministic refinements
            base = len(prefix)
            return [f"refine_{base}_1", f"refine_{base}_2", f"refine_{base}_3"]
        return []
    
    # Score function that rewards longer sequences with specific actions
    def mock_score_fn(seq):
        if not seq:
            return 0.0
        # Reward sequences that contain "refine_0_1" and "refine_1_2"
        score = 0.0
        if "refine_0_1" in seq:
            score += 0.3
        if "refine_1_2" in seq:
            score += 0.3
        # Bonus for length
        score += len(seq) * 0.1
        return min(1.0, score)
    
    cache = {}
    
    # Run NRPA
    best_score, best_seq = run_nrpa(
        levels=levels,
        iterations=iterations,
        alpha=1.0,
        initial_strategies=initial_strategies,
        children_provider=mock_children_provider,
        score_fn=mock_score_fn,
        cache=cache
    )
    
    # Should find a decent score
    assert best_score > 0.5
    assert len(best_seq) > 0
