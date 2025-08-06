# NRPA Re-architecting Scratchpad

## Task Overview
Fix the core issues in the NRPA system to build a real learning engine with proper state-independent action coding.

## Key Issues to Fix

### 1. State-Independent Action Coding ✅
- [x] Fix the `code()` function to be truly state-independent
- [x] Update PolicyManager methods to remove state dependency
- [x] Update function signatures throughout the system

### 2. Clean Up Feature Flag Approach
- [x] Replace clumsy `if ENABLE_NRPA:` with proper strategy pattern
- [x] Create cleaner abstraction for strategy selection

### 3. Extract API Client Logic
- [x] Move request handling from agent.py to reusable components
- [x] Improve error handling and retry mechanisms

### 4. Improve Dependency Injection
- [ ] Properly inject TelemetrySystem and BacktrackingManager
- [ ] Clean up the main agent loop

## Progress Tracking
Started: August 6, 2025
Core issue fixed: State-independent action coding