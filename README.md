# Modern LLM Post-Training & Alignment Framework (RLHF, DPO, PPO, GRPO)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![TRL 0.27.2](https://img.shields.io/badge/TRL-0.27.2-yellow.svg)](https://github.com/huggingface/trl)
[![Transformers 4.57.1](https://img.shields.io/badge/transformers-4.57.1-orange.svg)](https://huggingface.co/docs/transformers)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

A modular, high-performance research framework for modern Large Language Model (LLM) post-training, preference optimization, and reinforcement learning alignment. Implements Direct Preference Optimization (DPO), Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO / DeepSeek-style), LLM-as-a-Judge Safety Auditing, and comparative evaluation of Reinforcement Learning with Verifiable Rewards (RLVR) versus AI Feedback (RLAIF).

---

## 📌 Framework Highlights

- **Direct Preference Optimization (DPO)**:
  - Exact offline preference optimization eliminating the need for a separate reward model.
  - Parameterized $eta$ temperature ablations assessing policy drift and implicit reward scaling.
  - Systematic length-bias diagnostics isolating verbosity exploitation from true reasoning quality.
- **Proximal Policy Optimization (PPO)**:
  - Actor-critic policy gradient optimization with generalized advantage estimation (GAE).
  - PPO ratio clipping dynamics ($\epsilon=0.2$) and value function calibration diagnostics.
  - Dynamic adaptive versus fixed KL penalty ($eta_{	ext{KL}}$) ablations to control entropy collapse.
- **Group Relative Policy Optimization (GRPO)**:
  - Critic-free policy gradient optimization following DeepSeekMath & DeepSeek-R1 methodologies.
  - Computes advantage estimates normalized within candidate rollout groups ( \in \{4, 8, 16\}$).
  - Comparative analysis of standard score normalization versus reward-centering dynamics.
- **Automated Safety Calibration & LLM-as-a-Judge**:
  - Structured evaluation harness pairing response generation with automated rubric-guided critique.
  - Generates audit matrices and calibration sheets measuring alignment reliability under adversarial probes.
- **Reasoning Alignment: RLVR vs RLAIF**:
  - Compares ground-truth deterministic verifiers (RLVR) against pairwise LLM judges (RLAIF) on GSM8K and SVAMP challenge benchmarks.
  - Evaluates out-of-distribution robustness, resistance to reward hacking, and reasoning generalization.

---

## 📁 Repository Structure

```text
├── configs/                   # Experiment YAML configurations (DPO, PPO, GRPO, Safety)
│   ├── dpo.yaml               # DPO learning rates, beta, batch sizes
│   ├── ppo.yaml               # Actor-critic hyperparams, clipping epsilon, KL penalty
│   ├── grpo.yaml              # Group sizes, advantage normalization, rollout limits
│   └── feedback.yaml          # Safety judge prompts, RLVR/RLAIF verifier configurations
├── task1_dpo/                 # Task 1: Direct Preference Optimization Suite
│   ├── train.py               # DPO training pipeline
│   ├── evaluate.py            # Win-rate & log-likelihood evaluator
│   ├── ablate_beta.py         # Beta sensitivity grid search
│   └── analyze_length.py      # Response length correlation & bias analyzer
├── task2_ppo/                 # Task 2: Proximal Policy Optimization Suite
│   ├── continue_train.py      # PPO policy/value continuation trainer
│   ├── evaluate.py            # Benchmark accuracy & value loss monitor
│   ├── analyze_clipping.py    # Policy ratio clipping boundary audit
│   └── ablate_kl.py           # KL divergence penalty schedule ablations
├── task3_grpo/                # Task 3: Group Relative Policy Optimization Suite
│   ├── continue_train.py      # Critic-free GRPO training loop
│   ├── evaluate.py            # Group rollout evaluator
│   ├── analyze_group_size.py  # Advantage variance vs group size G in {4, 8, 16}
│   └── compare_normalization.py # Intra-group z-score vs min-max normalization
├── task4_safety/              # Task 4: Automated Safety Calibration
│   ├── generate_responses.py  # Multi-model probe generation
│   ├── judge_responses.py     # Pairwise and pointwise LLM judge harness
│   ├── make_audit_sheet.py    # Formatted audit sheets for manual/AI verification
│   └── evaluate_safety.py     # Refusal rates, harmlessness, and helpfulness metrics
├── task5_feedback/            # Task 5: RLVR vs RLAIF Benchmark
│   ├── evaluate_math.py       # GSM8K / SVAMP reasoning benchmark runner
│   ├── score_perturbations.py # Adversarial prompt perturbation sensitivity
│   └── compare_feedback.py    # Verifiable reward vs neural judge alignment comparison
├── scripts/                   # Environment validation and utility tools
│   ├── check_environment.py   # Dependency and CUDA driver validator
│   ├── download_assets.py     # Download required datasets and pretrained checkpoints
│   └── validate_assets.py     # Checksum verification for model checkpoints
└── requirements.txt           # Pinned dependencies (Transformers 4.57.1, TRL 0.27.2, PEFT 0.17.1)
```

---

## 🚀 Quick Start

### 1. Installation

```bash
git clone https://github.com/supus4562/llm-alignment-post-training.git
cd llm-alignment-post-training
pip install -r requirements.txt
python -m scripts.check_environment
```

### 2. Download Pretrained Assets & Datasets

```bash
python -m scripts.download_assets
python -m scripts.validate_assets
```

---

## 🔬 Running Post-Training & Ablations

### Direct Preference Optimization (DPO)
```bash
# Standard DPO training
python -m task1_dpo.train --config configs/dpo.yaml --run-name standard

# Evaluate trained adapter
python -m task1_dpo.evaluate --config configs/dpo.yaml --adapter outputs/task1_dpo/standard --name standard

# Beta temperature and length bias ablations
python -m task1_dpo.ablate_beta --config configs/dpo.yaml
python -m task1_dpo.analyze_length --config configs/dpo.yaml
```

### Proximal Policy Optimization (PPO)
```bash
# PPO Actor-Critic training
python -m task2_ppo.continue_train --config configs/ppo.yaml --run-name standard

# Evaluation and clipping/KL sensitivity analysis
python -m task2_ppo.evaluate --config configs/ppo.yaml --adapter outputs/task2_ppo/standard --name standard
python -m task2_ppo.analyze_clipping --config configs/ppo.yaml
python -m task2_ppo.ablate_kl --config configs/ppo.yaml
```

### Group Relative Policy Optimization (GRPO)
```bash
# DeepSeek-style GRPO training
python -m task3_grpo.continue_train --config configs/grpo.yaml --run-name standard

# Group size scaling (G = 4, 8, 16) and advantage normalization ablations
python -m task3_grpo.evaluate --config configs/grpo.yaml --adapter outputs/task3_grpo/standard --name standard
python -m task3_grpo.analyze_group_size --config configs/grpo.yaml
python -m task3_grpo.compare_normalization --config configs/grpo.yaml
```

### Safety Calibration & LLM-as-a-Judge
```bash
python -m task4_safety.generate_responses --config configs/feedback.yaml
python -m task4_safety.judge_responses --config configs/feedback.yaml
python -m task4_safety.make_audit_sheet --config configs/feedback.yaml
python -m task4_safety.evaluate_safety --config configs/feedback.yaml
```

### RLVR vs RLAIF Reasoning Comparison
```bash
# Benchmark mathematical reasoning on GSM8K
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset gsm

# Out-of-domain transfer evaluation on SVAMP challenge set
python -m task5_feedback.evaluate_math --config configs/feedback.yaml --dataset transfer
python -m task5_feedback.score_perturbations --config configs/feedback.yaml
python -m task5_feedback.compare_feedback --config configs/feedback.yaml
```

---

## 📜 Key References & Foundations

- Rafailov et al., *"Direct Preference Optimization: Your Language Model is Secretly a Reward Model"*, NeurIPS 2023.
- Schulman et al., *"Proximal Policy Optimization Algorithms"*, arXiv:1707.06347, 2017.
- Shao et al., *"DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models"*, 2024.
- DeepSeek-AI, *"DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"*, 2025.
- Uesato et al., *"Solving Math Word Problems with Process- and Outcome-Based Feedback"*, 2022.

---

## 📄 License
This project is licensed under the MIT License.
