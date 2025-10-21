import gym
import numpy as np

# gym-aloha uses Gymnasium, not the old Gym
try:
    import gymnasium
    import gym_aloha

    GYMNASIUM_AVAILABLE = True
except ImportError:
    GYMNASIUM_AVAILABLE = False
    gymnasium = None
    gym_aloha = None


class AlohaEnv:
    def __init__(self, task, action_repeat=1, size=(64, 64), seed=0):
        """
        Wrapper for gym-aloha environments.

        Args:
            task: Name of the Aloha task (e.g., 'AlohaInsertion-v0', 'TransferCube-v0')
            action_repeat: Number of times to repeat each action
            size: Size to resize images to (height, width)
            seed: Random seed
        """
        if not GYMNASIUM_AVAILABLE:
            raise ImportError(
                "gym-aloha requires gymnasium. Please install it with: pip install gym-aloha"
            )

        # Create the environment using gymnasium
        # gym-aloha registers environments with format: gym_aloha/TaskName-v0
        self._env = gymnasium.make(f"gym_aloha/{task}", render_mode="rgb_array")

        # Set seed (gymnasium uses reset with seed parameter, not env.seed())
        self._seed = seed
        self._action_repeat = action_repeat
        self._size = size

    @property
    def observation_space(self):
        """Return observation space."""
        spaces = {}

        # Get the original observation space
        orig_obs_space = self._env.observation_space

        # Handle different observation space types
        if isinstance(orig_obs_space, gymnasium.spaces.Dict):
            for key, space in orig_obs_space.spaces.items():
                if len(space.shape) == 3:  # Image observation
                    # Rename 'top' camera to 'image' for DreamerV3
                    if key == "top":
                        spaces["image"] = gym.spaces.Box(
                            0, 255, self._size + (space.shape[-1],), dtype=np.uint8
                        )
                    else:
                        # Keep other cameras with original names
                        spaces[key] = gym.spaces.Box(
                            0, 255, self._size + (space.shape[-1],), dtype=np.uint8
                        )
                else:  # Vector observation
                    spaces[key] = gym.spaces.Box(
                        space.low, space.high, space.shape, dtype=np.float32
                    )
        elif isinstance(orig_obs_space, gymnasium.spaces.Box):
            # If it's a single box space
            if len(orig_obs_space.shape) == 3:  # Image
                spaces["image"] = gym.spaces.Box(
                    0, 255, self._size + (orig_obs_space.shape[-1],), dtype=np.uint8
                )
            else:  # Vector
                spaces["state"] = gym.spaces.Box(
                    orig_obs_space.low,
                    orig_obs_space.high,
                    orig_obs_space.shape,
                    dtype=np.float32,
                )

        # Add is_first and is_last keys
        spaces["is_first"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_last"] = gym.spaces.Box(0, 1, (), dtype=bool)
        spaces["is_terminal"] = gym.spaces.Box(0, 1, (), dtype=bool)

        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        """Return action space."""
        return self._env.action_space

    def _process_observation(
        self, obs, is_first=False, is_last=False, is_terminal=False
    ):
        """Process observation to match expected format."""
        processed = {}

        if isinstance(obs, dict):
            for key, value in obs.items():
                if len(value.shape) == 3:  # Image
                    # Rename camera views to 'image' for DreamerV3
                    # Aloha uses 'top', 'front', etc. - we'll use 'top' as the main image
                    if key == "top":
                        processed["image"] = self._resize_image(value)
                    else:
                        # Keep other camera views with their original names
                        processed[key] = self._resize_image(value)
                else:  # Vector
                    processed[key] = value.astype(np.float32)
        else:
            # Single observation
            if len(obs.shape) == 3:  # Image
                processed["image"] = self._resize_image(obs)
            else:  # Vector
                processed["state"] = obs.astype(np.float32)

        # Add episode indicators
        processed["is_first"] = np.array(is_first, dtype=bool)
        processed["is_last"] = np.array(is_last, dtype=bool)
        processed["is_terminal"] = np.array(is_terminal, dtype=bool)

        return processed

    def _resize_image(self, image):
        """Resize image to target size."""
        import cv2

        if image.shape[:2] != self._size:
            image = cv2.resize(image, self._size[::-1], interpolation=cv2.INTER_AREA)
        return image

    def step(self, action):
        """Execute action and return observation, reward, done, info."""
        # Handle numpy arrays and clip actions to prevent physics instability
        if isinstance(action, np.ndarray):
            action = action.astype(np.float32)

        # Clip actions to valid range to prevent MuJoCo instability
        action = np.clip(action, -1.0, 1.0)

        # Repeat action
        total_reward = 0.0
        terminated = False
        truncated = False

        try:
            for _ in range(self._action_repeat):
                # Gymnasium returns 5 values: obs, reward, terminated, truncated, info
                obs, reward, terminated, truncated, info = self._env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
        except Exception as e:
            # If physics error occurs, reset the environment and mark as done
            # This prevents training from crashing due to unstable physics
            if "PhysicsError" in str(type(e).__name__) or "BADQACC" in str(e):
                print(
                    f"Warning: Physics instability detected, resetting environment. Error: {e}"
                )
                obs, _ = self._env.reset()
                total_reward = 0.0
                terminated = True
                truncated = False
                info = {}
            else:
                # Re-raise if it's not a physics error
                raise

        # Combine terminated and truncated into done for old gym API
        done = terminated or truncated

        # Process observation
        obs = self._process_observation(obs, is_last=done, is_terminal=terminated)

        # Add discount to info if not present
        if "discount" not in info:
            info["discount"] = np.array(0.0 if done else 1.0, dtype=np.float32)

        return obs, total_reward, done, info

    def reset(self):
        """Reset environment and return initial observation."""
        # Gymnasium reset returns (obs, info)
        obs, info = self._env.reset(seed=self._seed)
        obs = self._process_observation(obs, is_first=True)
        return obs

    def render(self, mode="rgb_array"):
        """Render environment."""
        # Gymnasium doesn't use mode parameter in render()
        return self._env.render()

    def close(self):
        """Close environment."""
        return self._env.close()
