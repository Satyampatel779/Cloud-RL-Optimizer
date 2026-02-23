import gymnasium as gym
from gymnasium import spaces
import numpy as np


class CloudDataCenterEnv(gym.Env):
    """
    Custom Gymnasium environment simulating a 3-server cloud data centre.

    The PPO agent routes incoming tasks across servers to minimise energy
    consumption (cpu² × temperature) while preventing CPU/RAM/thermal overload.

    Observation space (11-dim, all in [0, 1]):
        [cpu₀, cpu₁, cpu₂, ram₀, ram₁, ram₂, temp₀, temp₁, temp₂,
         queue_norm, next_task_size_norm]

    Action space: Discrete(3) — route the pending task to server 0, 1, or 2
    """

    metadata = {"render_modes": ["console"]}

    CPU_DECAY  = 0.86
    RAM_DECAY  = 0.94
    HEAT_GAIN  = 0.07
    HEAT_LOSS  = 0.08
    AMBIENT    = 0.15

    CRASH_CPU  = 0.95
    CRASH_RAM  = 0.95
    CRASH_TEMP = 0.98

    def __init__(self, max_steps: int = 200):
        super().__init__()

        self.num_servers = 3
        self.max_queue   = 50
        self.max_steps   = max_steps

        self.action_space = spaces.Discrete(self.num_servers)
        self.observation_space = spaces.Box(
            low=np.zeros(11, dtype=np.float32),
            high=np.ones(11,  dtype=np.float32),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.cpu_loads = self.np_random.uniform(0.05, 0.20, self.num_servers).astype(np.float32)
        self.ram_loads = self.np_random.uniform(0.05, 0.20, self.num_servers).astype(np.float32)
        self.temps     = self.np_random.uniform(0.20, 0.40, self.num_servers).astype(np.float32)

        self.queue        = float(self.np_random.integers(5, 20))
        self.current_step = 0
        self.next_task    = self._sample_task()

        return self._get_obs(), {}

    def step(self, action: int):
        self.current_step += 1
        task_cpu, task_ram = self.next_task

        self.cpu_loads[action] = np.clip(self.cpu_loads[action] + task_cpu, 0.0, 1.0)
        self.ram_loads[action] = np.clip(self.ram_loads[action] + task_ram, 0.0, 1.0)
        self.queue = max(0.0, self.queue - 1.0)

        self.cpu_loads = np.clip(self.cpu_loads * self.CPU_DECAY, 0.0, 1.0)
        self.ram_loads = np.clip(self.ram_loads * self.RAM_DECAY, 0.0, 1.0)
        heat_gain = self.cpu_loads * self.HEAT_GAIN
        heat_loss = (self.temps - self.AMBIENT) * self.HEAT_LOSS
        self.temps = np.clip(self.temps + heat_gain - heat_loss, 0.0, 1.0)

        n_arrivals = int(self.np_random.poisson(1.2))
        self.queue = min(self.max_queue, self.queue + n_arrivals)
        self.next_task = self._sample_task()

        energy_cost   = float(np.sum(self.cpu_loads ** 2 * (1.0 + self.temps)) * 4.0)
        latency_cost  = float(self.queue * 0.08)
        thermal_risk  = float(np.sum(np.maximum(0.0, self.temps - 0.70) * 15.0))
        balance_bonus = float(max(0.0, 2.0 - np.std(self.cpu_loads) * 6.0))
        reward        = 3.0 + balance_bonus - energy_cost - latency_cost - thermal_risk

        cpu_crash = bool(np.any(self.cpu_loads >= self.CRASH_CPU))
        ram_crash = bool(np.any(self.ram_loads >= self.CRASH_RAM))
        overheat  = bool(np.any(self.temps     >= self.CRASH_TEMP))

        terminated = cpu_crash or ram_crash or overheat
        if terminated:
            reward -= 150.0

        truncated = bool(self.current_step >= self.max_steps)

        info = {
            "energy_cost":   energy_cost,
            "latency_cost":  latency_cost,
            "thermal_risk":  thermal_risk,
            "balance_bonus": balance_bonus,
            "cpu_loads":     self.cpu_loads.copy(),
            "ram_loads":     self.ram_loads.copy(),
            "temps":         self.temps.copy(),
            "queue":         self.queue,
            "survived":      not terminated,
            "crash_reason":  ("cpu" if cpu_crash else "ram" if ram_crash
                              else "heat" if overheat else None),
        }

        return self._get_obs(), float(reward), terminated, truncated, info

    def _sample_task(self):
        """Return (cpu_demand, ram_demand) for a randomly sized task (50% small, 35% medium, 15% large)."""
        size = self.np_random.choice(['small', 'medium', 'large'], p=[0.50, 0.35, 0.15])
        if size == 'small':
            cpu = self.np_random.uniform(0.03, 0.08)
            ram = self.np_random.uniform(0.02, 0.06)
        elif size == 'medium':
            cpu = self.np_random.uniform(0.08, 0.16)
            ram = self.np_random.uniform(0.06, 0.12)
        else:  # large
            cpu = self.np_random.uniform(0.16, 0.28)
            ram = self.np_random.uniform(0.12, 0.22)
        return (float(cpu), float(ram))

    def _get_obs(self) -> np.ndarray:
        task_cpu, task_ram = self.next_task
        task_size_norm = np.clip((task_cpu + task_ram) / 0.50, 0.0, 1.0)
        queue_norm     = np.clip(self.queue / self.max_queue, 0.0, 1.0)
        return np.concatenate([
            self.cpu_loads,
            self.ram_loads,
            self.temps,
            [queue_norm, task_size_norm],
        ]).astype(np.float32)

    def render(self, mode='console'):
        print(f"\n  Step {self.current_step}")
        for i in range(self.num_servers):
            cpu_bar  = '█' * int(self.cpu_loads[i] * 20)
            temp_bar = '█' * int(self.temps[i] * 20)
            print(
                f"  Server {i+1}: "
                f"CPU {self.cpu_loads[i]*100:5.1f}% [{cpu_bar:<20}]  "
                f"RAM {self.ram_loads[i]*100:5.1f}%  "
                f"Temp {self.temps[i]*100:5.1f}% [{temp_bar:<20}]"
            )
        task_cpu, task_ram = self.next_task
        print(f"  Queue: {int(self.queue)} tasks | "
              f"Next task → CPU +{task_cpu*100:.0f}%  RAM +{task_ram*100:.0f}%")