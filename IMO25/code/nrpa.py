"""
nrpa.py

Nested Rollout Policy Adaptation for strategy refinement search.

Implements the NRPA algorithm atop our strategy enumeration and refinement pipeline:
- States are (step, prefix) tuples representing a refinement path
- Actions are refinements of the current strategy prefix
- Policy adapts via gradient ascent on rollout sequences
- Rollouts use softmax policy to bias future trials

Key functions:
- run_nrpa: main entrypoint, returns best (score, seq)
- rollout: simulate a path using current policy
- adapt: shift policy weights toward a successful sequence
- generate_refinements: use LLM to refine strategy prefix
"""
import os
import json
import hashlib
import math
import time
import random
from typing import List, Tuple, Dict, Any, Callable, Optional
from collections import defaultdict

# --- CONFIGURATION ---
NRPA_LEVELS = int(os.getenv("NRPA_LEVELS", 2))
NRPA_ITER = int(os.getenv("NRPA_ITER", 60))
NRPA_ALPHA = float(os.getenv("NRPA_ALPHA", 1.0))
NRPA_MAX_DEPTH = int(os.getenv("NRPA_MAX_DEPTH", 4))

# --- POLICY UTILS ---
Code = str  # hashable representation of action
Policy = Dict[Code, float]

def logsumexp(xs: List[float]) -> float:
    """Numerically stable log(sum(exp(x)))"""
    if not xs:
        return float('-inf')
    max_x = max(xs)
    if max_x == float('-inf'):
        return float('-inf')
    return max_x + math.log(sum(math.exp(x - max_x) for x in xs))

class PolicyManager:
    """Manages the policy dictionary and related operations."""
    
    def __init__(self, max_weight: float = 100.0):
        self.policy: Policy = defaultdict(float)
        self.max_weight = max_weight
    
    def softmax_sample(self, actions: List[str]) -> str:
        """Sample action proportional to exp(pol[code(a)]) - state independent"""
        if not actions:
            raise ValueError("No actions to sample")
        codes = [code(a) for a in actions]
        weights = [math.exp(self.policy.get(c, 0.0)) for c in codes]
        total = sum(weights)
        if total <= 0:
            return random.choice(actions)
        probs = [w / total for w in weights]
        return random.choices(actions, weights=probs, k=1)[0]
    
    def adapt(self, seq: List[str], alpha: float, children_provider: Callable[[int, Tuple[str, ...]], List[str]]) -> None:
        """
        Adapt policy weights toward a successful sequence using gradient ascent.
        
        For each action a_t in the sequence:
        - Increase pol[code(a_t)] by alpha
        - Decrease all pol[code(a)] by alpha * P(a|s_t) to maintain normalization
        """
        prefix = ()
        for step, action in enumerate(seq):
            legal_actions = children_provider(step, prefix)
            if not legal_actions:
                break
                
            action_code = code(action)
            self.policy[action_code] = self.policy.get(action_code, 0.0) + alpha
            
            # Compute normalization constant z = sum_a exp(pol[code(a)])
            codes = [code(a) for a in legal_actions]
            log_z = logsumexp([self.policy.get(c, 0.0) for c in codes])
            
            # Subtract alpha * P(a|s) from each action
            for a, c in zip(legal_actions, codes):
                prob = math.exp(self.policy.get(c, 0.0) - log_z)
                self.policy[c] = self.policy.get(c, 0.0) - alpha * prob
                
            prefix = prefix + (action,)
            
        self.clip_policy()
    
    def clip_policy(self) -> None:
        """Clip policy weights to prevent overflow"""
        for k, v in self.policy.items():
            self.policy[k] = max(-self.max_weight, min(self.max_weight, v))

# --- STATE/ACTION CODES ---
def code(action: str) -> Code:
    """Generate a unique, deterministic code for an action"""
    return hashlib.md5(action.encode()).hexdigest()

# --- CORE NRPA ---

def rollout(policy_manager: PolicyManager, 
           initial_candidates: List[str], 
           max_depth: int, 
           children_provider: Callable[[int, Tuple[str, ...]], List[str]],
           score_fn: Callable[[List[str]], float],
           cache: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Perform a single rollout using the current policy.
    
    Returns:
        (score, sequence) where sequence is the path taken
    """
    seq = []
    prefix = ()
    
    # Start with initial candidates
    if initial_candidates:
        action = policy_manager.softmax_sample(initial_candidates)
        seq.append(action)
        prefix = (action,)
    
    # Refine up to max_depth
    for step in range(1, max_depth):
        legal_actions = children_provider(step, prefix)
        if not legal_actions:
            break
        action = policy_manager.softmax_sample(legal_actions)
        seq.append(action)
        prefix = prefix + (action,)
    
    # Score the final sequence
    score = score_fn(seq)
    return score, seq

def run_nrpa(levels: int,
            iterations: int,
            alpha: float,
            initial_strategies: List[str],
            children_provider: Callable[[int, Tuple[str, ...]], List[str]],
            score_fn: Callable[[List[str]], float],
            cache: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Run NRPA to find the best strategy sequence.
    
    Args:
        levels: Nesting depth (0 = base rollout)
        iterations: Number of iterations per level
        alpha: Policy update step size
        initial_strategies: Root-level strategy candidates
        children_provider: Function to generate refinements for a state
        score_fn: Function to score a complete sequence
        cache: Shared cache for LLM results
        
    Returns:
        (best_score, best_sequence)
    """
    if levels == 0:
        policy_manager = PolicyManager()
        return rollout(policy_manager, initial_strategies, NRPA_MAX_DEPTH, children_provider, score_fn, cache)
    
    best_score = float('-inf')
    best_seq = []
    policy_manager = PolicyManager()
    
    for _ in range(iterations):
        score, seq = run_nrpa(levels - 1, iterations, alpha, initial_strategies, children_provider, score_fn, cache)
        if score >= best_score:
            best_score = score
            best_seq = seq
            policy_manager.adapt(seq, alpha, children_provider)
            
    return best_score, best_seq
