# pokAI

A self-play deep reinforcement learning system for heads-up No-Limit Texas Hold’em (HUNL), implementing a Trinal-Clip PPO loss over Siamese-style policy/value networks, with Elo-based opponent selection and a small fixed population of static strategies for diversity.

---

## Overview

pokAI trains a two-player (“heads-up”) No-Limit Hold’em poker agent via on-policy Proximal Policy Optimization with:

- **Trinal-Clip** policy loss (per Alpha Hold’em paper)  
- **Clipped value loss**  
- **Siamese-style** neural network shared across policy & value  
- **Elo**-based opponent selection from a small pool of historical selves  
- Three static baseline opponents (always-fold, always-all-in, random legal) to maintain policy diversity  

It simulates hands in-memory (no external poker server), collects Slumbot-style replays, builds experiences per street, and updates the policy & value networks via PPO.

---

## Features

- 🚀 **Self-Play Loop** with population-based opponent selection  
- 📊 **Elo Rating** to measure and drive improvement  
- ♟️ **Static Strategies** (fold-only, all-in, random) for exploration  
- 📈 **TensorBoard** logging for live metrics (policy loss, value loss, mBB/hand, Elo)  
- 🔄 **Experience Replay Builder** parses action histories into training examples  
- 🃏 **Fast, In-Memory** poker simulator compatible with Slumbot pipeline  

---

## Getting Started

### Prerequisites

- **Python 3.8+**  
- **PyTorch** (tested on 1.13+) with CUDA or MPS support  
- **NumPy**, **Matplotlib**, **TensorBoard**  
- (Optional) GPU recommended for faster training  

### Installation

1. Clone the repository  
   ```bash
   git clone https://github.com/acaen007/pokAI.git
   cd pokAI





Configuration
Key hyperparameters in the main loop:

| Variable            | Description                                   | Default    |
| ------------------- | --------------------------------------------- | ---------- |
| `num_cycles`        | Number of self-play + train cycles            | `100`      |
| `match_hands`       | Hands per match for Elo evaluation            | `5000`     |
| `matches_per_cycle` | Matches per cycle for Elo update              | `1`        |
| `BIG_BLIND`         | Big blind size (chips)                        | `100`      |
| `SMALL_BLIND`       | Small blind size (chips)                      | `50`       |
| `K`                 | Top-K historical agents to sample             | `3`        |
| PPO Clip (ε)        | Policy clip parameter                         | `0.1`      |
| Value Clip (δ₂, δ₃) | From Trinal-Clip PPO (via `deltas` from hand) | per-street |
| Trinal Clip (δ1)    | Avoids high variance with negative advantage  | `3`        |
