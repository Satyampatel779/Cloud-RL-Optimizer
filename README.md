☁️ Autonomous Cloud Resource Optimizer (Deep RL)
An end-to-end Machine Learning project that uses Deep Reinforcement Learning (PPO) to autonomously balance live web traffic across a simulated multi-node data center.

The agent dynamically learns to minimize a data center's energy consumption by balancing the base power cost of waking up sleeping servers against the exponential heat penalties of overloading active hardware.

🎥 Live Demo
Demo Video.mp4

🧠 The Engineering Challenge: The "True Cloud" MDP
Building a custom RL environment requires a carefully designed Markov Decision Process (MDP). If the reward function is flawed, the AI will exploit it.

During development, the agent discovered a hilarious "Reward Hacking" loophole: It realized that running the data center for 500 hours cost more "electricity points" than simply overloading a single server and crashing the entire simulation on day one. It chose corporate sabotage to save on the electric bill!

To fix this and achieve steady-state equilibrium, the environment (cloud_env.py) was re-engineered with the True Cloud Trade-off:

Base Power Cost: A flat penalty for every server turned on (encourages the AI to let servers sleep).

Exponential Heat Penalty: A squared penalty for high CPU loads (encourages the AI to spread traffic out).

Survival Bonus: A generous reward for keeping the system alive, paired with an apocalyptic penalty for crashing the servers.

Observation Normalization: Scaling the massive task queue down to a percentage so the neural network doesn't suffer from feature blindness.

🛠️ Tech Stack
Reinforcement Learning: stable-baselines3 (Proximal Policy Optimization)

Environment Architecture: gymnasium (Custom API)

Mathematical Operations: numpy

Frontend / Dashboard: streamlit, pandas

☁️ Autonomous Cloud Resource Optimizer (Deep RL)
An end-to-end Machine Learning project that uses Deep Reinforcement Learning (PPO) to autonomously balance live web traffic across a simulated multi-node data center.

The agent dynamically learns to minimize a data center's energy consumption by balancing the base power cost of waking up sleeping servers against the exponential heat penalties of overloading active hardware.

🎥 Live Demo
(Note: Upload the video you recorded to GitHub and replace this line with the video link or a .gif of your Streamlit dashboard in action!)

🧠 The Engineering Challenge: The "True Cloud" MDP
Building a custom RL environment requires a carefully designed Markov Decision Process (MDP). If the reward function is flawed, the AI will exploit it.

During development, the agent discovered a hilarious "Reward Hacking" loophole: It realized that running the data center for 500 hours cost more "electricity points" than simply overloading a single server and crashing the entire simulation on day one. It chose corporate sabotage to save on the electric bill!

To fix this and achieve steady-state equilibrium, the environment (cloud_env.py) was re-engineered with the True Cloud Trade-off:

Base Power Cost: A flat penalty for every server turned on (encourages the AI to let servers sleep).

Exponential Heat Penalty: A squared penalty for high CPU loads (encourages the AI to spread traffic out).

Survival Bonus: A generous reward for keeping the system alive, paired with an apocalyptic penalty for crashing the servers.

Observation Normalization: Scaling the massive task queue down to a percentage so the neural network doesn't suffer from feature blindness.

🛠️ Tech Stack
Reinforcement Learning: stable-baselines3 (Proximal Policy Optimization)

Environment Architecture: gymnasium (Custom API)

Mathematical Operations: numpy

Frontend / Dashboard: streamlit, pandas
