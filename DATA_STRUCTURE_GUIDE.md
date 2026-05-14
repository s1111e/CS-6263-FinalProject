# AutoVLA Training Data Structure - Detailed Guide

## 1. Preprocessed JSON File Structure

**Location**: `data/nuscenes_processed/train/*.json` (one JSON per scene, ~19k training scenes)

### Example JSON File: `00014584c6de4789a69b4717fa3c0d22.json`

```json
{
  "token": "00014584c6de4789a69b4717fa3c0d22",
  "dataset_name": "nuscenes",
  
  // ============ CAMERA PATHS ============
  "front_camera_paths": [
    "/work/amd456/autovla/dataset/nuscenes/samples/CAM_FRONT/n015-2018-09-27-15-33-17+0800__CAM_FRONT__1538033992762460.jpg",
    "/work/amd456/autovla/dataset/nuscenes/samples/CAM_FRONT/n015-2018-09-27-15-33-17+0800__CAM_FRONT__1538033993262460.jpg",
    "/work/amd456/autovla/dataset/nuscenes/samples/CAM_FRONT/n015-2018-09-27-15-33-17+0800__CAM_FRONT__1538033993762460.jpg",
    "/work/amd456/autovla/dataset/nuscenes/samples/CAM_FRONT/n015-2018-09-27-15-33-17+0800__CAM_FRONT__1538033994262461.jpg"
  ],
  "front_left_camera_paths": [
    "/work/amd456/autovla/dataset/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-09-27-15-33-17+0800__CAM_FRONT_LEFT__1538033992754844.jpg",
    "/work/amd456/autovla/dataset/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-09-27-15-33-17+0800__CAM_FRONT_LEFT__1538033993254844.jpg",
    "/work/amd456/autovla/dataset/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-09-27-15-33-17+0800__CAM_FRONT_LEFT__1538033993754844.jpg",
    "/work/amd456/autovla/dataset/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-09-27-15-33-17+0800__CAM_FRONT_LEFT__1538033994254844.jpg"
  ],
  "front_right_camera_paths": [
    // ... 4 image paths
  ],
  
  // ============ VEHICLE STATE ============
  "velocity": 6.224178873023417,        // m/s - magnitude of (vx, vy)
  "acceleration": -0.37226212742084286, // m/s² - magnitude of (ax, ay)
  "instruction": "Turn Right",           // Discrete driving command
  
  // ============ GROUND TRUTH TRAJECTORY ============
  "gt_trajectory": [
    [3.051576614379883, -0.09214502573013306, -0.030176819565866618],      // t=0.5s
    [5.925003528594971, -0.28817954659461975, -0.06809407296384226],       // t=1.0s
    [8.664985656738281, -0.6648350954055786, -0.13656098829771834],        // t=1.5s
    [11.459773063659668, -1.2829723358154297, -0.21759540804659416],       // t=2.0s
    [14.264491081237793, -2.1411688327789307, -0.29683713418309704],       // t=2.5s
    [17.122028350830078, -3.121873617172241, -0.3304960738642098],         // t=3.0s
    [19.954378128051758, -4.147490978240967, -0.34730774586596463],        // t=3.5s
    [22.768848419189453, -5.08488130569458, -0.32139895821826225],         // t=4.0s
    [25.508102416992188, -5.917522430419922, -0.29499014487912106],        // t=4.5s
    [28.308042526245117, -6.6153082847595215, -0.24415551885282866]        // t=5.0s
  ],
  // Each waypoint: [x_meters, y_meters, heading_radians]
  // 10 waypoints = 5 second horizon at 0.5s intervals
  
  // ============ CHAIN-OF-THOUGHT REASONING (nuScenes) ============
  "cot_output": [
    // Part 0: Scene description
    "There are two pedestrians to the back left of the ego car. There is one car in front of the ego car. There is one pedestrian to the front left of the ego car. There is one car behind the ego car.",
    
    // Part 1: Critical objects
    "There is a white commercial vehicle to the front of the ego vehicle, a person wearing a blue shirt to the back left of the ego vehicle, a white and red truck to the back of the ego vehicle, and a letter AHEAD on the ground to the front of the ego vehicle",
    
    // Part 2: Object motion
    "The moving status of **White commercial vehicle** is stationary. The moving status of **Person wearing a blue shirt** is keep going straight. The moving status of **White and red truck** is stationary.",
    
    // Part 3: Reasoning
    "Firstly notice **A letter AHEAD on the ground**. It is a traffic sign, so the ego vehicle should continue at the same speed. Secondly notice **White commercial vehicle**. It is stationary, so the ego vehicle should continue at the same speed. Thirdly notice **White and red truck**. It is also stationary, so the ego vehicle should continue at the same speed.",
    
    // Part 4: Action (text description)
    "turn right with a deceleration"
  ]
}
```

### Field Type Summary

| Field | Type | Shape | Example Value | Notes |
|-------|------|-------|----------------|-------|
| `token` | str | - | `"00014584c6de..."` | Unique scene ID (32-char hex) |
| `dataset_name` | str | - | `"nuscenes"` | "nuplan", "waymo", or "nuscenes" |
| `front_camera_paths` | List[str] | [4] | image paths | 4 consecutive frames at 2 Hz (0.5s interval) |
| `front_left_camera_paths` | List[str] | [4] | image paths | Same temporal coverage |
| `front_right_camera_paths` | List[str] | [4] | image paths | Same temporal coverage |
| `velocity` | float | scalar | `6.22` | Magnitude in m/s |
| `acceleration` | float | scalar | `-0.372` | Magnitude in m/s² (negative = deceleration) |
| `instruction` | str | - | `"Turn Right"` | Discrete command from CARLA/nuPlan |
| `gt_trajectory` | List[List[float]] | [10, 3] | see above | Waypoints: x, y in meters; heading in radians |
| `cot_output` | List[str] | [5] | see above | Only for nuScenes (other datasets may have different structure) |

---

## 2. Training Batch Structure

### During SFTDataset.__getitem__() 

The dataset returns a dictionary with raw features **before** collate function:

```python
{
    # ============ TEXT & MESSAGES ============
    'text': "<|im_start|>system\nYou are an Advanced Driver...<|im_end|>\n...",
    # Processed chat template with special tokens and vision IDs
    
    # ============ VISION DATA (paths, not loaded yet) ============
    'video_inputs': [
        # List of video frame objects for Qwen processor
        [frame1, frame2, frame3, frame4],  # front camera
        [frame1, frame2, frame3, frame4],  # front_left camera
        [frame1, frame2, frame3, frame4],  # front_right camera
    ],
    'image_inputs': None,
    
    # ============ TARGETS FOR TRAINING ============
    'gt_trajectory': array([[3.05, -0.09, -0.03],    # [10, 3] - raw continuous trajectory
                            [5.93, -0.29, -0.07],
                            ...,
                            [28.31, -6.62, -0.24]]),
    
    'gt_action': array([[idx_0],                      # [1, 10] - action token indices for each waypoint
                        [idx_1],                      # Token indices 0-2047
                        ...,
                        [idx_9]]),
    
    # ============ METADATA ============
    'has_cot': True,                                  # Whether this scene has CoT text
    'data_path': Path('/work/amd456/autovla/data/nuscenes_processed/train/00014584c6de4789a69b4717fa3c0d22.json')
}
```

### After DataCollator.__call__() [batch_size=1]

```python
{
    # ============ TOKENIZED INPUTS ============
    'input_ids': tensor([
        # [1, seq_len] - tokenized chat with vision tokens embedded
        # Sequence: [system_tokens] [vision_tokens] [user_tokens] [assistant_tokens]
        # Example values: 151644, 77091 (assistant markers)
        [151644, 77091, ..., 151665, 151666, 151667, ...]
    ]),
    
    # ============ ATTENTION ============
    'attention_mask': tensor([[1, 1, 1, ..., 1]]),  # [1, seq_len] - all 1s (attend to all tokens)
    
    # ============ VISION TENSORS ============
    'pixel_values_videos': tensor of shape [1, 3, 6, 3, 224, 224]
    # Interpretation:
    # [batch=1, num_videos=3, num_frames=6, channels=3, height=224, width=224]
    # - 3 videos: front, front_left, front_right
    # - 6 frames per video (4 input + 2 padding/interpolation)
    # - 224×224 resolution (Qwen's default)
    
    'video_grid_thw': tensor([[6, 14, 14]])  # Time-Height-Width grid for vision transformer
    # [1, 3] - describes how video frames are arranged in vision token grid
    
    # ============ LABELS FOR LOSS ============
    'labels': tensor([
        # [1, seq_len] - same as input_ids but with IGNORE_INDEX (-100) for non-assistant tokens
        [-100, -100, ..., -100,           # System prompt: ignored
         -100, -100, ..., -100,           # User query with images: ignored  
         151644, 77091, 151665, 151666, 151667, ...]  # Assistant response: used for loss
    ]),
    # Only tokens after "</assistant>" marker are used for computing cross-entropy loss
    
    # ============ GROUND TRUTH ============
    'gt_trajectory': tensor([[
        [3.05, -0.09, -0.03],
        [5.93, -0.29, -0.07],
        ...,
        [28.31, -6.62, -0.24]
    ]])  # [1, 10, 3]
    
    'gt_action': tensor([[
        [idx_0],
        [idx_1],
        ...,
        [idx_9]
    ]])  # [1, 10, 1] - ground truth action token indices
    
    'has_cot': tensor([True])  # [1] - boolean flag
}
```

### Key Tensor Dimensions

| Tensor | Shape | Dtype | Purpose |
|--------|-------|-------|---------|
| `input_ids` | [batch, seq_len] | int64 | Token IDs for LLM |
| `attention_mask` | [batch, seq_len] | int64 | Attention pattern |
| `pixel_values_videos` | [batch, 3, 6, 3, 224, 224] | float32 | Vision input (3 cameras, 6 frames, RGB) |
| `video_grid_thw` | [batch, 3] | int64 | Vision token grid dimensions |
| `labels` | [batch, seq_len] | int64 | Training targets (-100 = ignored) |
| `gt_trajectory` | [batch, 10, 3] | float32 | Continuous trajectory for metrics |
| `gt_action` | [batch, 10, 1] | int64 | Discrete action tokens 0-2047 |

---

## 3. Text Prompt Construction

### System Prompt (with CoT enabled)

```
You are an Advanced Driver Assistance and Full Self-Driving System. 
You will receive visual observations from the ego vehicle's cameras and dynamic information about the vehicle's current state. 
Your task is to predict the optimal driving action for the next five seconds.

First, carefully analyze the surrounding environment by considering traffic lights, the movements of other vehicles and pedestrians, lane markings, and any other relevant factors.

If necessary, use step-by-step reasoning (Chain-of-Thought) to arrive at the best driving action. Otherwise, you may directly predict the final driving action.

Present the final action clearly after your reasoning steps.
```

### User Query (Multi-modal Input)

```
The autonomous vehicle is equipped with three cameras mounted at the front, left, and right, enabling a comprehensive perception of the surrounding environment.

The first video presents the front view of the vehicle, comprising four sequential frames sampled at 2 Hz.
<video>file:///path/to/front_camera_frame1.jpg</video>
<video>file:///path/to/front_camera_frame2.jpg</video>
<video>file:///path/to/front_camera_frame3.jpg</video>
<video>file:///path/to/front_camera_frame4.jpg</video>

The second video presents the front-left view of the vehicle, comprising four sequential frames sampled at 2 Hz.
<video>file:///path/to/front_left_camera_frame1.jpg</video>
<video>file:///path/to/front_left_camera_frame2.jpg</video>
<video>file:///path/to/front_left_camera_frame3.jpg</video>
<video>file:///path/to/front_left_camera_frame4.jpg</video>

The third video presents the front-right view of the vehicle, comprising four sequential frames sampled at 2 Hz.
<video>file:///path/to/front_right_camera_frame1.jpg</video>
<video>file:///path/to/front_right_camera_frame2.jpg</video>
<video>file:///path/to/front_right_camera_frame3.jpg</video>
<video>file:///path/to/front_right_camera_frame4.jpg</video>

The current velocity of the vehicle is 6.224 m/s, and the current acceleration is -0.372 m/s². 
The driving instruction is: Turn Right. 
Based on this information, plan the action trajectory for the autonomous vehicle over the next five seconds.
```

### Assistant Response with CoT (if `cot_output` available in JSON)

```
<think>
This is a complex scenario requiring additional reasoning.
There are two pedestrians to the back left of the ego car. There is one car in front of the ego car. There is one pedestrian to the front left of the ego car. There is one car behind the ego car.

There is a white commercial vehicle to the front of the ego vehicle, a person wearing a blue shirt to the back left of the ego vehicle, a white and red truck to the back of the ego vehicle, and a letter AHEAD on the ground to the front of the ego vehicle

The moving status of **White commercial vehicle** is stationary. The moving status of **Person wearing a blue shirt** is keep going straight. The moving status of **White and red truck** is stationary.

Firstly notice **A letter AHEAD on the ground**. It is a traffic sign, so the ego vehicle should continue at the same speed. Secondly notice **White commercial vehicle**. It is stationary, so the ego vehicle should continue at the same speed. Thirdly notice **White and red truck**. It is also stationary, so the ego vehicle should continue at the same speed.
</think>

<answer>
The final output action is: <action_1256><action_1257><action_1258><action_1259><action_1260><action_1261><action_1262><action_1263><action_1264><action_1265>
</answer>
```

### Assistant Response without CoT (if `cot_output` is None or empty)

```
<think>
This is a straightforward scenario, and a direct decision can be made.
</think>

<answer>
The final output action is: <action_1256><action_1257><action_1258><action_1259><action_1260><action_1261><action_1262><action_1263><action_1264><action_1265>
</answer>
```

### Key Points

- **Action tokens**: Each `<action_XXXX>` represents one discrete action token (0-2047) for one 0.5-second interval
- **Sequence length**: 10 tokens for 5-second horizon (0.5s per token)
- **Video format**: 4 consecutive 500ms frames at 2Hz = 2 seconds coverage per camera
- **Vehicle state**: Scalar velocity and acceleration magnitudes (not x,y components)
- **Chat template**: Qwen2.5-VL-specific format with special tokens

---

## 4. Action Tokenization

### Codebook Structure

**File**: `codebook_cache/agent_vocab.pkl`

```python
# Loaded data structure:
{
    'token_all': {
        'veh': numpy.ndarray  # Shape: (2048, 6, 4, 2)
    }
}

# Codebook shape breakdown:
# Dimension 0: 2048 discrete action tokens (indices 0-2047)
# Dimension 1: 6 waypoints (0.5s each = 3-second trajectory snippet)
# Dimension 2: 4 control points per waypoint (Bezier curve definition)
# Dimension 3: 2 coordinates (x, y in meters, relative to current pose)
```

### Example Action Token

**Token 0** - First waypoint:
```python
array([
    [ 2.4,  1.0],   # Control point 0: x=2.4m, y=1.0m
    [ 2.4, -1.0],   # Control point 1: x=2.4m, y=-1.0m
    [-2.4, -1.0],   # Control point 2: x=-2.4m, y=-1.0m
    [-2.4,  1.0]    # Control point 3: x=-2.4m, y=1.0m
])
# This forms a rectangular bounding box defining a primitive motion
```

### Text Representation

Action tokens are converted to text for LLM processing:

```python
# Python code from action_tokenizer.py
action_token_array = np.array([1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265])

text = ""
for token_id in action_token_array:
    text += f"<action_{token_id}>"

# Result:
# "<action_1256><action_1257><action_1258><action_1259><action_1260><action_1261><action_1262><action_1263><action_1264><action_1265>"
```

### Action Token Matching Process (Training Data Generation)

During preprocessing, continuous trajectories are matched to the nearest action tokens:

```python
# From TokenProcessor._match_agent_token()

# Input: continuous trajectory for current timestep
pos = tensor([11.46, -1.28])          # Global x, y position [m]
heading = tensor([-0.22])              # Global heading [rad]

# Transform codebook tokens to current pose
token_world = transform_to_global(
    pos_local=codebook[i, :],         # 4 control points in local coords
    head_local=None,
    pos_now=prev_pos,                 # Previous vehicle position
    head_now=prev_head,               # Previous vehicle heading
)

# Find best matching token via L2 distance
distances = torch.norm(token_world - current_pose_contour, dim=-1)
gt_idx = torch.argmin(distances)      # Best matching token index

# Result: gt_idx ≈ 1256 (the action token that best represents this waypoint)
```

### Token Usage in Training

```python
# What the model learns to predict:
input:  "You are a driving system... [vision tokens] ... velocity 6.22 m/s, acceleration -0.37 m/s²"
target: "<action_1256><action_1257><action_1258><action_1259><action_1260><action_1261><action_1262><action_1263><action_1264><action_1265>"

# Cross-entropy loss:
loss = CrossEntropyLoss(
    predictions=logits[:, :, :2048],   # [batch, seq_len, 2048] - logits for each action token
    targets=gt_action_indices,          # [batch, 10] - ground truth token IDs
)
```

---

## 5. Complete Data Flow: JSON → Batch → Model Input

### Step 1: JSON File on Disk
```
/work/amd456/autovla/data/nuscenes_processed/train/00014584c6de4789a69b4717fa3c0d22.json
  ├─ Scene metadata
  ├─ Camera image paths (relative to sensor_data_path)
  ├─ Vehicle state (velocity, acceleration)
  ├─ Driving command (instruction)
  ├─ Ground truth trajectory (10 waypoints × 3 dims)
  └─ Chain-of-thought reasoning (5 parts)
```

### Step 2: SFTDataset.__getitem__()

```python
# Load JSON
with open(json_path) as f:
    scene_data = json.load(f)  # Dict with all fields

# Extract features using builders
input_features = {}
for builder in agent.get_feature_builders():
    input_features.update(builder.compute_features(scene_data))
# Result: input_features['images'], ['velocity'], ['acceleration'], ['driving_command']

# Extract targets
target_trajectory = {}
for builder in agent.get_target_builders():
    target_trajectory.update(builder.compute_targets(scene_data))
# Result: target_trajectory['gt_idx'], ['gt_pos_raw'], ['gt_heading_raw']

# Build chat messages with video paths
messages = [
    {'role': 'system', 'content': [{'type': 'text', 'text': system_prompt}]},
    {'role': 'user', 'content': [
        {'type': 'text', 'text': camera_description},
        {'type': 'video', 'video': [file paths to 4 frames]},
        ...
        {'type': 'text', 'text': vehicle_state_and_instruction}
    ]},
    {'role': 'assistant', 'content': [{'type': 'text', 'text': cot + actions}]}
]

# Process with Qwen processor (not yet loading images)
image_inputs, video_inputs = process_vision_info(messages)
text = processor.apply_chat_template(messages, ...)

# Return dict
return {
    'text': text,                           # Processed chat string
    'video_inputs': video_inputs,           # Vision object references
    'image_inputs': image_inputs,           # Image object references
    'gt_trajectory': target_trajectory['gt_pos_raw'],  # [10, 3]
    'gt_action': target_trajectory['gt_idx'],          # [10, 1]
    'has_cot': (cot is not None),
    'data_path': json_path
}
```

### Step 3: DataCollator.__call__() [Batch Size 1]

```python
# Input: List with 1 dict from Step 2
features = [{..., 'text': text_str, 'video_inputs': [...], 'labels': [...]}]

# Process all vision data through Qwen processor
batch = processor(
    text=[text_str],
    videos=[video_inputs],
    images=[image_inputs],
    padding=True,
    return_tensors="pt"
)
# Returns: input_ids, attention_mask, pixel_values_videos, video_grid_thw

# Create labels (copy input_ids, then mask non-assistant tokens)
labels = batch["input_ids"].clone()
assistant_tokens = [151644, 77091]  # Qwen's assistant role markers
# Find where assistant response starts
for i in range(seq_len - len(assistant_tokens)):
    if input_ids[i:i+len(assistant_tokens)] == assistant_tokens:
        labels[:i] = IGNORE_INDEX  # Mask to -100
        break

# Add ground truth
batch['labels'] = labels
batch['gt_trajectory'] = tensor(gt_trajectory)  # [1, 10, 3]
batch['gt_action'] = tensor(gt_action)          # [1, 10, 1]
batch['has_cot'] = tensor([has_cot])

return batch
```

### Step 4: Model Forward Pass

```python
# Input batch from Step 3
batch = {
    'input_ids': [1, 384],               # Token IDs
    'attention_mask': [1, 384],          # All ones
    'pixel_values_videos': [1, 3, 6, 3, 224, 224],  # RGB video frames
    'video_grid_thw': [1, 3],            # Vision token grid
    'labels': [1, 384]                   # With -100 masking
}

# Forward through Qwen2.5-VL-3B
model = Qwen2_5_VLForConditionalGeneration.from_pretrained('Qwen2.5-VL-3B-Instruct')
outputs = model(
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    pixel_values_videos=batch['pixel_values_videos'],
    video_grid_thw=batch['video_grid_thw'],
    labels=batch['labels']
)

# Output:
# outputs.loss = scalar (cross-entropy on action tokens)
# outputs.logits = [1, 384, vocab_size] where vocab includes custom <action_*> tokens

# Training: backward(loss), optimizer step
loss.backward()
optimizer.step()
```

### Step 5: Inference (Generation)

```python
# After training, run model in generation mode
with torch.no_grad():
    # Prepare input (without labels)
    batch = {
        'input_ids': [1, 384],
        'attention_mask': [1, 384],
        'pixel_values_videos': [1, 3, 6, 3, 224, 224],
        'video_grid_thw': [1, 3]
    }
    
    # Generate action tokens
    output_ids = model.generate(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask'],
        pixel_values_videos=batch['pixel_values_videos'],
        video_grid_thw=batch['video_grid_thw'],
        max_new_tokens=20,
        temperature=0.7
    )
    
    # Decode: output_ids[0] = [150644, 77091, 151665, 151666, ..., 151674]
    # Where 151665-151674 are the action token special tokens
    
    # Extract action indices
    action_indices = [1256, 1257, 1258, 1259, 1260, 1261, 1262, 1263, 1264, 1265]
    
    # Decode back to continuous trajectory using codebook
    action_tokens = codebook[action_indices]  # [10, 6, 4, 2]
    trajectory = action_tokenizer.rollout(action_tokens)  # [1, 11, 3]
    # Result: [11, 3] continuous trajectory (10 predicted waypoints + current pose)
```

---

## Summary Table

| Stage | Input | Processing | Output |
|-------|-------|-----------|--------|
| **JSON** | Scene data on disk | Load JSON | Dict with metadata, paths, trajectory, CoT |
| **Dataset** | JSON dict | Feature/target builders | {'text', 'video_inputs', 'gt_trajectory', 'gt_action'} |
| **Collator** | List[1] of dicts | Processor (Qwen) | {'input_ids', 'pixel_values_videos', 'labels', 'gt_action'} |
| **Model** | Batch dict | LLM + Vision encoder | logits [1, 384, vocab], loss scalar |
| **Generation** | Batch dict | LLM generation | action_indices [10], trajectory [10, 3] |

---

## Configuration Parameters

**From** `config/training/qwen2.5-vl-3B-mix-sft.yaml`:

```yaml
model:
  codebook_cache_path: "codebook_cache/agent_vocab.pkl"  # 2048 action tokens
  trajectory:
    num_poses: 10           # 10 waypoints
    interval_length: 0.5    # 0.5 second per waypoint = 5 second horizon
    time_horizon: 5.0       # 5 seconds total
  tokens:
    action_start_id: 151665 # First custom action token ID (added to Qwen vocab)
    ignore_index: -100      # Mask for non-assistant tokens in loss
    assistant_id: [151644, 77091]  # Qwen's assistant role token IDs
```

**Effective sequence structure**:
- 4 video frames per camera × 3 cameras = 12 vision inputs
- Each frame tokenized to ~576 vision tokens (28 × 28 × 1 for Qwen's grid)
- ~3,456 vision tokens total
- ~1,000 text tokens (chat template + description + CoT)
- **~4,456 total tokens per sequence**
- Action sequence: 10 tokens (one per waypoint)

