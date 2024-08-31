import pybullet as p
import pybullet_data
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import time

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
planeId = p.loadURDF("plane.urdf")

# Robot parameters
NUM_JOINTS = 4
LINK_MASSES = [1, 1, 1, 1]
LINK_LENGTHS = [0.5, 0.5, 0.5, 0.5]

# DQN parameters
BATCH_SIZE = 64
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MEMORY_SIZE = 10000
TARGET_UPDATE = 10

# Define the DQN
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Create the bipedal robot
def create_robot():
    robotId = p.createMultiBody(
        baseMass=1,
        basePosition=[0, 0, 1],
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.2, 0.2, 0.2], rgbaColor=[0.8, 0.8, 0.8, 1]),
        linkMasses=LINK_MASSES,
        linkCollisionShapeIndices=[
            p.createCollisionShape(p.GEOM_CYLINDER, radius=0.05, height=LINK_LENGTHS[i])
            for i in range(NUM_JOINTS)
        ],
        linkVisualShapeIndices=[
            p.createVisualShape(p.GEOM_CYLINDER, radius=0.05, length=LINK_LENGTHS[i], rgbaColor=[0.8, 0.6, 0.4, 1])
            for i in range(NUM_JOINTS)
        ],
        linkPositions=[
            [0, -0.2, -0.3], [0, 0, -LINK_LENGTHS[0]],
            [0, 0.2, -0.3], [0, 0, -LINK_LENGTHS[2]]
        ],
        linkOrientations=[[0, 0, 0, 1]] * NUM_JOINTS,
        linkInertialFramePositions=[[0, 0, 0]] * NUM_JOINTS,
        linkInertialFrameOrientations=[[0, 0, 0, 1]] * NUM_JOINTS,
        linkParentIndices=[0, 1, 0, 3],
        linkJointTypes=[p.JOINT_REVOLUTE] * NUM_JOINTS,
        linkJointAxis=[[0, 1, 0], [0, 1, 0], [0, 1, 0], [0, 1, 0]]
    )
    return robotId

# Get the state of the robot
def get_state(robot):
    pos, ori = p.getBasePositionAndOrientation(robot)
    linear_vel, angular_vel = p.getBaseVelocity(robot)
    joint_states = p.getJointStates(robot, range(NUM_JOINTS))
    joint_positions = [state[0] for state in joint_states]
    joint_velocities = [state[1] for state in joint_states]
    return np.concatenate([pos, ori, linear_vel, angular_vel, joint_positions, joint_velocities])

# Calculate the reward
def calculate_reward(robot, target_pos, prev_distance):
    pos, ori = p.getBasePositionAndOrientation(robot)
    current_distance = np.linalg.norm(np.array(pos[:2]) - np.array(target_pos[:2]))
    
    reward = 0
    reward += (prev_distance - current_distance) * 10  # Reward for moving towards the target
    
    # Penalize tilting
    _, pitch, _ = p.getEulerFromQuaternion(ori)
    reward -= abs(pitch) * 5
    
    if pos[2] < 0.5:  # Robot has fallen
        reward -= 100
    elif current_distance < 0.5:  # Robot reached the target
        reward += 1000
    
    # Penalize excessive joint movements
    joint_states = p.getJointStates(robot, range(NUM_JOINTS))
    joint_velocities = [abs(state[1]) for state in joint_states]
    reward -= sum(joint_velocities) * 0.1
    
    return reward, current_distance

# Discretize the action space
def get_discrete_actions():
    return [(-0.5, -0.5, -0.5, -0.5), (-0.5, -0.5, -0.5, 0.5), (-0.5, -0.5, 0.5, -0.5), (-0.5, -0.5, 0.5, 0.5),
            (-0.5, 0.5, -0.5, -0.5), (-0.5, 0.5, -0.5, 0.5), (-0.5, 0.5, 0.5, -0.5), (-0.5, 0.5, 0.5, 0.5),
            (0.5, -0.5, -0.5, -0.5), (0.5, -0.5, -0.5, 0.5), (0.5, -0.5, 0.5, -0.5), (0.5, -0.5, 0.5, 0.5),
            (0.5, 0.5, -0.5, -0.5), (0.5, 0.5, -0.5, 0.5), (0.5, 0.5, 0.5, -0.5), (0.5, 0.5, 0.5, 0.5)]

# Main training loop
def train():
    robot = create_robot()
    target_pos = [5, 0, 1]
    p.addUserDebugLine([0, 0, 0], target_pos, [1, 0, 0])

    state_size = len(get_state(robot))
    action_size = len(get_discrete_actions())
    
    policy_net = DQN(state_size, action_size)
    target_net = DQN(state_size, action_size)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = deque(maxlen=MEMORY_SIZE)

    epsilon = EPSILON_START
    discrete_actions = get_discrete_actions()

    num_episodes = 1000
    max_steps = 1000

    for episode in range(num_episodes):
        p.resetBasePositionAndOrientation(robot, [0, 0, 1], [0, 0, 0, 1])
        for joint in range(NUM_JOINTS):
            p.resetJointState(robot, joint, 0)

        state = get_state(robot)
        total_reward = 0
        prev_distance = np.linalg.norm(np.array([0, 0]) - np.array(target_pos[:2]))

        for step in range(max_steps):
            if random.random() < epsilon:
                action_idx = random.randrange(action_size)
            else:
                with torch.no_grad():
                    action_idx = policy_net(torch.FloatTensor(state)).max(0)[1].item()

            action = discrete_actions[action_idx]

            for joint, pos in enumerate(action):
                p.setJointMotorControl2(robot, joint, p.POSITION_CONTROL, targetPosition=pos, force=100)

            p.stepSimulation()
            time.sleep(1/240)  # To slow down the simulation for visualization

            next_state = get_state(robot)
            reward, current_distance = calculate_reward(robot, target_pos, prev_distance)
            total_reward += reward
            prev_distance = current_distance

            memory.append((state, action_idx, reward, next_state, reward < -99 or reward > 999))
            state = next_state

            if len(memory) >= BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

                state_batch = torch.FloatTensor(state_batch)
                action_batch = torch.LongTensor(action_batch)
                reward_batch = torch.FloatTensor(reward_batch)
                next_state_batch = torch.FloatTensor(next_state_batch)
                done_batch = torch.BoolTensor(done_batch)

                current_q_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
                next_q_values = target_net(next_state_batch).max(1)[0].detach()
                target_q_values = reward_batch + (GAMMA * next_q_values * (~done_batch))

                loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values.unsqueeze(1))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if reward < -99 or reward > 999:  # Robot has fallen or reached the target
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    p.disconnect()

if __name__ == "__main__":
    train()




