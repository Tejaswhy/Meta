# AutoPilotEnv

AutoPilotEnv is a hackathon-ready, production-style OpenEnv environment for realistic autonomous driving decision-making. It simulates safety-critical policy choices (not game mechanics) and provides deterministic, reproducible rewards and grading.

## Motivation

Most RL environments simplify driving too aggressively. AutoPilotEnv models practical decision layers used in autonomous systems:
- lane centering and speed compliance,
- obstacle response under limited maneuver options,
- traffic signal and pedestrian safety handling.

The goal is to benchmark policy reliability under clear operational constraints.

## Architecture

- `env.py`: OpenEnv environment class implementing `reset()`, `step(action)`, `state()`, and `close()`.
- `models.py`: Pydantic `Observation`, `Action`, and `Reward` schemas.
- `grader.py`: deterministic scoring modules per task.
- `inference.py`: hackathon-compliant inference runner with strict stdout format.
- `app.py`: Hugging Face Space UI and OpenEnv entrypoint class.
- `openenv.yaml`: metadata and task configuration for validator tooling.

## Observation Space

Shared typed fields across tasks:
- lane/speed fields: `lane_position`, `speed`, `speed_limit`, `road_curvature`
- obstacle fields: `front_distance`, `left_lane_clear`, `right_lane_clear`
- signal safety fields: `traffic_light`, `pedestrian_crossing`, `pedestrian_distance`

Each task exposes the relevant subset while preserving a stable model contract.

## Action Space

`Action` model:
- `steering` in `[-1.0, 1.0]`
- `acceleration` in `[0.0, 1.0]`
- `brake` in `[0.0, 1.0]`
- `action_type` (e.g., `maintain`, `brake`, `change_lane`, `stop`, `emergency_stop`)
- `lane_change` (`left`, `right`, `none`)

## Task Difficulty

1. **Easy: Lane Keeping + Speed Control**
   - Reward for centered driving, smooth steering, legal speed
   - Penalty for lane drift and overspeed

2. **Medium: Obstacle Avoidance**
   - Reward for collision avoidance and safe lane changes
   - Penalty for crash-risk actions and unsafe maneuvers

3. **Hard: Traffic Signal + Pedestrian Safety**
   - Reward for signal obedience and safe pedestrian handling
   - Penalty for violations and collision-risk behavior

## Reward Design

Deterministic incremental feedback (formatted to 2 decimals):
- `safe_action -> +0.20`
- `partial progress -> +0.10`
- `collision -> -1.00`
- `loop repetition -> -0.20`
- `destructive action -> -0.50`

## Programmatic Graders

`grader.py` includes:
- `grade_lane_keeping()`
- `grade_obstacle_avoidance()`
- `grade_signal_handling()`

Each returns deterministic `0.0 <= score <= 1.0`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Inference

```bash
export HF_TOKEN=your_hf_router_token
python inference.py
```

Optional:
- `API_BASE_URL` (default: `https://api.openai.com/v1`)
- `MODEL_NAME` (default: `gpt-4.1-mini`)
- `TASK_NAME` (`lane_keeping`, `obstacle_avoidance`, `signal_safety`)

## Sample Inference Logs

```text
[START] task=lane_keeping env=autopilotenv model=gpt-4.1-mini
[STEP] step=1 action=maintain reward=0.20 done=false error=null
[STEP] step=2 action=maintain reward=0.10 done=false error=null
[STEP] step=3 action=maintain reward=0.20 done=true error=null
[END] success=true steps=3 rewards=0.20,0.10,0.20
```

## Baseline Scores

Deterministic fallback policy baseline (illustrative target):
- lane_keeping: `0.90`
- obstacle_avoidance: `1.00`
- signal_safety: `1.00`
- aggregate mean: `0.97`

## Docker Usage

```bash
docker build -t autopilotenv .
docker run --rm -e HF_TOKEN=$HF_TOKEN autopilotenv
```

## Hugging Face Space Deployment

1. Create a new Space (SDK: Docker)
2. Push this repository
3. Add secret `HF_TOKEN`
4. Launch Space (default app in `app.py`)

This project is lightweight and designed to run within the typical Space limits (2 vCPU / 8 GB RAM).

## OpenEnv Validation

```bash
openenv validate
```

`inference.py` is intentionally placed in the repository root to satisfy hackathon validation rules.
