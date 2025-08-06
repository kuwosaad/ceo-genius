#!/usr/bin/env python3

"""
run_parallel.py

Parallel orchestrator to launch multiple independent instances of agent.py.

Purpose
- Increase the chance of finding a correct solution by exploring multiple stochastic runs in parallel.
- Each worker process is a separate Python interpreter invoking agent.py with its own log.
- Summarize progress, detect when any agent finds a correct solution, and print a final report.

Key Features
- ProcessPoolExecutor-based fan-out with configurable concurrency.
- Per-agent log files (logs/agent_XX.log) and stdout/stderr capture.
- Heuristic solution detection by scanning both stdout and the log for the success marker.
- Graceful handling of timeouts and interrupts.
- Optional --other_prompts for providing hints/biases shared by all agents.

Typical usage
  python run_parallel.py problems/imo01.txt -n 8 -w 8 -t 1800 -d logs
  python run_parallel.py problems/imo05.txt --other_prompts "use invariants,try bounding"

Return code
- 0 if at least one agent found a correct solution
- 1 otherwise

Notes
- This script does not cancel other agents after one succeeds; all run to completion to keep logs useful.
- The success marker it searches is printed by agent.py:
    "Found a correct solution in run"
"""
"""
MIT License

Copyright (c) 2025 Lin Yang, Yichen Huang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import subprocess
import os
import sys
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
import threading
import json
import re

def run_agent(agent_id, problem_file, log_dir, timeout=None, other_prompts=[]):
    """
    Run a single agent instance with the specified parameters in a dedicated subprocess.

    Args:
        agent_id: Unique identifier for this agent instance.
        problem_file: Path to the problem statement file.
        log_dir: Directory to store log files.
        timeout: Timeout in seconds (None for no timeout).
        other_prompts: List of comma-separated hints passed to agent.py.

    Returns:
        tuple: (agent_id, return_code, stdout, stderr, solution_found)
    """
    log_file = os.path.join(log_dir, f"agent_{agent_id:02d}.log")
    
    cmd = [
        sys.executable, "agent.py", 
        problem_file,
        "--log", log_file,
        "--other_prompts", f'\"{",".join(other_prompts)}\"'
    ]
    
    try:
        # Run the agent with timeout
        if timeout:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
        else:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True,
                cwd=os.path.dirname(os.path.abspath(__file__))
            )
        
        # Check if a solution was found by looking for the success message
        solution_found = False
        if result.returncode == 0:
            # Look for the success message in stdout
            if "Found a correct solution in run" in result.stdout:
                solution_found = True
            # Also check the log file for the success message
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    if "Found a correct solution in run" in log_content:
                        solution_found = True
            except:
                pass
        
        return (agent_id, result.returncode, result.stdout, result.stderr, solution_found)
    
    except subprocess.TimeoutExpired:
        return (agent_id, -1, "", f"Agent {agent_id} timed out after {timeout} seconds", False)
    except Exception as e:
        return (agent_id, -1, "", f"Agent {agent_id} failed with error: {str(e)}", False)

def print_status(agent_id, status, stdout="", stderr=""):
    """Print status information for an agent (with trimmed stdout/stderr for readability)."""
    print(f"[Agent {agent_id:02d}] {status}")
    if stdout.strip():
        print(f"[Agent {agent_id:02d}] STDOUT: {stdout.strip()}")
    if stderr.strip():
        print(f"[Agent {agent_id:02d}] STDERR: {stderr.strip()}")

def main():
    """
    CLI entrypoint to launch N parallel agents.

    Arguments:
      problem_file: path to IMO problem text (string)
      --num-agents/-n: number of agent processes to start
      --log-dir/-d: directory for agent logs
      --timeout/-t: per-process timeout in seconds
      --max-workers/-w: size of the process pool (defaults to num-agents)
      --other_prompts/-o: optional comma-separated list of hints

    Behavior:
      - Submits all agent runs to a ProcessPoolExecutor.
      - Streams progress as agents finish.
      - After completion, prints a summary (timings, success rate, IDs).
      - Attempts to extract and print the found solution from the winning agent's log.
    """
    parser = argparse.ArgumentParser(description='Run multiple IMO agent instances in parallel')
    parser.add_argument('problem_file', help='Path to the problem statement file')
    parser.add_argument('--num-agents', '-n', type=int, default=10, 
                       help='Number of parallel agents to run (default: 10)')
    parser.add_argument('--log-dir', '-d', default='logs', 
                       help='Directory to store log files (default: logs)')
    parser.add_argument('--timeout', '-t', type=int, default=None,
                       help='Timeout in seconds for each agent (default: no timeout)')
    parser.add_argument('--max-workers', '-w', type=int, default=None,
                       help='Maximum number of worker processes (default: number of agents)')
    parser.add_argument('--other_prompts', '-o', type=str, help='Other prompts (optional)')
    
    args = parser.parse_args()
    
    # Create log directory if it doesn't exist
    os.makedirs(args.log_dir, exist_ok=True)
    
    print(f"Starting {args.num_agents} parallel agents...")
    print(f"Problem file: {args.problem_file}")
    print(f"Log directory: {args.log_dir}")
    if args.timeout:
        print(f"Timeout per agent: {args.timeout} seconds")
    print(f"Max workers: {args.max_workers or args.num_agents}")
    print("Note: All agents will run to completion regardless of solution found")
    print("-" * 50)
    
    # Track results
    completed_agents = []
    successful_agents = []
    failed_agents = []
    solution_found = False
    solution_agent_id = None

    other_prompts = []
    if args.other_prompts:
        other_prompts = args.other_prompts.split(',')
    
    start_time = time.time()
    
    try:
        with ProcessPoolExecutor(max_workers=args.max_workers or args.num_agents) as executor:
            # Submit all agent tasks
            future_to_agent = {
                executor.submit(run_agent, i, args.problem_file, args.log_dir, args.timeout, other_prompts): i 
                for i in range(args.num_agents)
            }
            
            # Process completed agents
            for future in as_completed(future_to_agent):
                agent_id, return_code, stdout, stderr, found_solution = future.result()
                completed_agents.append(agent_id)
                
                if found_solution:
                    solution_found = True
                    solution_agent_id = agent_id
                    status = "FOUND CORRECT SOLUTION!"
                    successful_agents.append(agent_id)
                    print(f"\n🎉 SOLUTION FOUND by Agent {agent_id:02d}! 🎉")
                    print(f"[Agent {agent_id:02d}] {status}")
                    print_status(agent_id, status, stdout, stderr)
                elif return_code == 0:
                    status = "COMPLETED SUCCESSFULLY (no solution found)"
                    successful_agents.append(agent_id)
                else:
                    status = f"FAILED (return code: {return_code})"
                    failed_agents.append(agent_id)
                
                print_status(agent_id, status, stdout, stderr)
                print(f"Progress: {len(completed_agents)}/{args.num_agents} agents completed")
                print("-" * 30)
    
    except KeyboardInterrupt:
        print("\nReceived interrupt signal. Shutting down gracefully...")
        # The ProcessPoolExecutor will handle cleanup automatically
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Print summary
    print("\n" + "=" * 50)
    print("FINAL SUMMARY")
    print("=" * 50)
    print(f"Total execution time: {total_time:.2f} seconds")
    print(f"Total agents: {args.num_agents}")
    print(f"Successful agents: {len(successful_agents)}")
    print(f"Failed agents: {len(failed_agents)}")
    print(f"Success rate: {len(successful_agents)/args.num_agents*100:.1f}%")
    
    if solution_found:
        print(f"\n🎉 SOLUTION FOUND by Agent {solution_agent_id:02d}! 🎉")
        print(f"Log file with solution: {os.path.join(args.log_dir, f'agent_{solution_agent_id:02d}.log')}")
        
        # Try to extract and display the solution
        solution_log_file = os.path.join(args.log_dir, f"agent_{solution_agent_id:02d}.log")
        try:
            with open(solution_log_file, 'r', encoding='utf-8') as f:
                log_content = f.read()
                # Look for the solution in the log
                if "Found a correct solution in run" in log_content:
                    # Extract the solution JSON
                    solution_match = re.search(r'Found a correct solution in run \d+\.\s*\n(.*?)(?=\n\n|\n>>>>>>>|\Z)', 
                                            log_content, re.DOTALL)
                    if solution_match:
                        solution_text = solution_match.group(1).strip()
                        print(f"\nSOLUTION FOUND:")
                        print("=" * 50)
                        print(solution_text)
                        print("=" * 50)
        except Exception as e:
            print(f"Could not extract solution from log file: {e}")
    
    if successful_agents:
        print(f"\nSuccessful agent IDs: {sorted(successful_agents)}")
    
    if failed_agents:
        print(f"Failed agent IDs: {sorted(failed_agents)}")
    
    print(f"\nLog files are available in: {os.path.abspath(args.log_dir)}")
    
    # List log files
    log_files = [f for f in os.listdir(args.log_dir) if f.endswith('.log')]
    if log_files:
        print(f"\nGenerated log files:")
        for log_file in sorted(log_files):
            file_path = os.path.join(args.log_dir, log_file)
            file_size = os.path.getsize(file_path)
            print(f"  {log_file} ({file_size} bytes)")
    
    return 0 if solution_found else 1

if __name__ == "__main__":
    sys.exit(main())
