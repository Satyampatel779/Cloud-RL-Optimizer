import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from cloud_env import CloudDataCenterEnv

TOTAL_TIMESTEPS = 300_000
N_ENVS          = 4
MODEL_PATH      = "ppo_cloud_optimizer"
BEST_MODEL_DIR  = "best_model"
LOG_DIR         = "tensorboard_logs"

os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR,        exist_ok=True)


def make_env():
    """Factory so each subprocess gets its own independent env instance."""
    def _init():
        return CloudDataCenterEnv()
    return _init


if __name__ == "__main__":
    print("Validating environment …")
    check_env(CloudDataCenterEnv(), warn=True)
    print("✔  Environment is Gym-compliant.\n")

    train_env = SubprocVecEnv([make_env() for _ in range(N_ENVS)])
    eval_env  = SubprocVecEnv([make_env()])

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        eval_freq=max(10_000 // N_ENVS, 1),
        n_eval_episodes=10,
        deterministic=True,
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=max(50_000 // N_ENVS, 1),
        save_path="./checkpoints/",
        name_prefix="ppo_cloud",
    )

    print("Initialising PPO agent …")
    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    print(f"Starting training — {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} parallel envs …")
    print("(Run  tensorboard --logdir tensorboard_logs  in another terminal to watch live)\n")
    try:
        import tqdm
        use_pbar = True
    except ImportError:
        use_pbar = False
        print("(Install tqdm for a live progress bar: pip install tqdm)\n")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[eval_callback, checkpoint_callback],
        progress_bar=use_pbar,
    )

    model.save(MODEL_PATH)
    print(f"\n✔  Training complete!")
    print(f"   Final model → {MODEL_PATH}.zip")
    print(f"   Best model  → {BEST_MODEL_DIR}/best_model.zip")
    print(f"   TensorBoard → {LOG_DIR}/")

    train_env.close()
    eval_env.close()