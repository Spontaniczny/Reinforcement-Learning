from typing import Optional
import os

import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm


class ActorCriticController:
    def __init__(self, environment, learning_rate: float, discount_factor: float) -> None:
        self.environment = environment
        self.discount_factor: float = discount_factor
        self.model: tf.keras.Model = self.create_actor_critic_model()
        self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)  
        self.log_action_probability: Optional[tf.Tensor] = None 
        self.tape: Optional[tf.GradientTape] = None  
        self.last_error_squared: float = 0.0  


    @staticmethod
    def create_actor_critic_model() -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(4,), name="state_input") 

        # Hidden Layer 1
        x = tf.keras.layers.Dense(1024, activation='relu')(inputs)  
        x = tf.keras.layers.LayerNormalization()(x)

        # Hidden Layer 2
        x = tf.keras.layers.Dense(256, activation='relu')(x)      
        x = tf.keras.layers.LayerNormalization()(x)

        # Actor Output (softmax over actions)
        actor_output = tf.keras.layers.Dense(2, activation='softmax', name="actor_output")(x) 

        # Critic Output (single value)
        critic_output = tf.keras.layers.Dense(1, activation='linear', name="critic_output")(x)

        model = tf.keras.Model(inputs=inputs, outputs=[actor_output, critic_output])
        return model

    def choose_action(self, state: np.ndarray) -> int:
        state = self.format_state(state)  

        self.tape = tf.GradientTape(persistent=True)
        with self.tape:
            actor_probs = self.model(state, training=True)[0] # pierwsza siec
            distribution = tfp.distributions.Categorical(probs=actor_probs)
            action = distribution.sample()

            self.log_action_probability = distribution.log_prob(action) 

        return int(action.numpy()[0])


    # noinspection PyTypeChecker
    def learn(self, state: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool) -> None:
        state = self.format_state(state)
        new_state = self.format_state(new_state)

        with self.tape: 

            curr_state_est = self.model(state, training=True)[1] # pierwsza
            curr_state_est = tf.squeeze(curr_state_est)

            error = reward
            if not terminal:
                new_state_est = self.model(new_state, training=True)[1]
                new_state_est = tf.squeeze(new_state_est)

                error += self.discount_factor * float(new_state_est)

            error -= curr_state_est

            critic_loss = error ** 2
            actor_loss = -float(error) * self.log_action_probability
            loss = actor_loss + critic_loss

            self.last_error_squared = float(critic_loss)


        gradients = self.tape.gradient(loss, self.model.trainable_weights)  
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_weights)) 
        del self.tape
        

    @staticmethod
    def format_state(state: np.ndarray) -> np.ndarray:
        return np.reshape(state, (1, state.size))


class TwoNetworksActorCriticController(ActorCriticController):
    def __init__(self, environment, learning_rate: float, discount_factor: float) -> None:
        self.environment = environment
        self.discount_factor: float = discount_factor
        self.optimizer: tf.keras.optimizers.Optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate) 
        self.log_action_probability: Optional[tf.Tensor] = None  
        self.tape: Optional[tf.GradientTape] = None  
        self.last_error_squared: float = 0.0  

        self.actor = self.create_actor_model()
        self.critic = self.create_critic_model()


    @staticmethod
    def create_actor_model() -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(4,), name="state_input")

        x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)

        actor_output = tf.keras.layers.Dense(2, activation='softmax', name="actor_output")(x)

        return tf.keras.Model(inputs=inputs, outputs=actor_output, name="Actor")


    @staticmethod
    def create_critic_model() -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(4,), name="state_input")

        x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)

        critic_output = tf.keras.layers.Dense(1, activation='linear', name="critic_output")(x)

        return tf.keras.Model(inputs=inputs, outputs=critic_output, name="Critic")



    def choose_action(self, state: np.ndarray) -> int:
        state = self.format_state(state)  

        self.tape = tf.GradientTape(persistent=True)
        with self.tape:
            actor_probs = self.actor(state, training=True) 
            distribution = tfp.distributions.Categorical(probs=actor_probs)
            action = distribution.sample()

            self.log_action_probability = distribution.log_prob(action)  

        return int(action.numpy()[0])


    # noinspection PyTypeChecker
    def learn(self, state: np.ndarray, reward: float, new_state: np.ndarray, terminal: bool) -> None:
        state = self.format_state(state)
        new_state = self.format_state(new_state)

        with self.tape:

            curr_state_est = self.critic(state, training=True)  
            curr_state_est = tf.squeeze(curr_state_est)

            error = reward
            if not terminal:
                new_state_est = self.critic(new_state, training=True)  

                new_state_est = tf.squeeze(new_state_est)
                error += self.discount_factor * float(new_state_est)
            error -= curr_state_est

            critic_loss = error ** 2
            actor_loss = - float(error) * self.log_action_probability
            loss = actor_loss + critic_loss

            self.last_error_squared = float(critic_loss)


        total_variables = self.actor.trainable_weights + self.critic.trainable_weights
        gradients = self.tape.gradient(loss, total_variables)
        self.optimizer.apply_gradients(zip(gradients, total_variables))
        del self.tape


class SmallTwoNetworksActorCriticController(TwoNetworksActorCriticController):

    @staticmethod
    def create_actor_model() -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(4,), name="state_input")

        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)

        actor_output = tf.keras.layers.Dense(2, activation='softmax', name="actor_output")(x)

        return tf.keras.Model(inputs=inputs, outputs=actor_output, name="Actor")


    @staticmethod
    def create_critic_model() -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(4,), name="state_input")

        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(32, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)

        critic_output = tf.keras.layers.Dense(1, activation='linear', name="critic_output")(x)

        return tf.keras.Model(inputs=inputs, outputs=critic_output, name="Critic")


class ShipTwoNetworkActorCriticController(TwoNetworksActorCriticController):

    @staticmethod
    def create_actor_model() -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(8,), name="state_input")

        x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)

        actor_output = tf.keras.layers.Dense(4, activation='softmax', name="actor_output")(x)

        return tf.keras.Model(inputs=inputs, outputs=actor_output, name="Actor")


    @staticmethod
    def create_critic_model() -> tf.keras.Model:
        inputs = tf.keras.Input(shape=(8,), name="state_input")

        x = tf.keras.layers.Dense(1024, activation='relu')(inputs)
        x = tf.keras.layers.LayerNormalization()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.LayerNormalization()(x)

        critic_output = tf.keras.layers.Dense(1, activation='linear', name="critic_output")(x)

        return tf.keras.Model(inputs=inputs, outputs=critic_output, name="Critic")


def dir_exists_and_not_empty(path: str) -> bool:
    return os.path.exists(path) and os.path.isdir(path) and bool(os.listdir(path))


def evaluate_critic_stick():
    critic_model = tf.keras.models.load_model("critic_stick_1450.h5")


    # test for stick angles and rotation speed
    # angle_range = np.linspace(-0.5, 0.5, 100)
    # ang_vel_range = np.linspace(-3, 3, 100) 

    # cart_position = 0.0
    # cart_velocity = 0.0

    # angle_grid, ang_vel_grid = np.meshgrid(angle_range, ang_vel_range)
    # state_grid = np.stack([np.full_like(angle_grid, cart_position),
    #                     np.full_like(angle_grid, cart_velocity),
    #                     angle_grid,
    #                     ang_vel_grid], axis=-1)

    # cart position and cart speed tests
    position_range = np.linspace(-3, 3, 100)    
    velocity_range = np.linspace(-3.0, 3.0, 100) 

    pole_angle = 0.0
    pole_ang_velocity = 0.0

    pos_grid, vel_grid = np.meshgrid(position_range, velocity_range)
    state_grid = np.stack([pos_grid,
                        vel_grid,
                        np.full_like(pos_grid, pole_angle),
                        np.full_like(pos_grid, pole_ang_velocity)], axis=-1)

    flat_states = state_grid.reshape(-1, 4) 

    values = critic_model.predict(flat_states, verbose=0)

    # values_grid = values.reshape(angle_grid.shape)
    values_grid = values.reshape(pos_grid.shape)

    # contour = plt.contourf(angle_grid, ang_vel_grid, values_grid, levels=50)

    title = "Critic Evaluation over Cart States (Pole Angle = 0)"
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(pos_grid, vel_grid, values_grid, levels=50)
    plt.xlabel("Cart Position")
    plt.ylabel("Cart Velocity")
    plt.title(title)
    plt.colorbar(contour, label="Value")
    plt.show()
    plt.savefig(f"{title}.png")          


def evaluate_critic_ship():
    critic_model = tf.keras.models.load_model("critic_ship_950.h5")

    state_names = [
        "X Position", "Y Position", "X Velocity", "Y Velocity",
        "Angle", "Angular Velocity", "Left Leg Contact", "Right Leg Contact"
    ]

    state_ranges = [
        np.linspace(-1.5, 1.5, 100),  # X Position
        np.linspace(0.0, 1.5, 100),   # Y Position
        np.linspace(-2.0, 2.0, 100),  # X Velocity
        np.linspace(-2.0, 2.0, 100),  # Y Velocity
        np.linspace(-1.5, 1.5, 100),  # Angle (radians)
        np.linspace(-2.0, 2.0, 100),  # Angular Velocity
        np.array([0.0, 1.0]),         # Left Leg Contact
        np.array([0.0, 1.0])          # Right Leg Contact
    ]

    fixed_state = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    index_pairs = [
        (0, 2),  # X Position vs X Velocity
        (1, 3),  # Y Position vs Y Velocity
        (4, 5),  # Angle vs Angular Velocity
        (0, 4),  # X Position vs Angle
        (1, 5),  # Y Position vs Angular Velocity
        (2, 3),  # X Velocity vs Y Velocity
    ]

    os.makedirs("plots", exist_ok=True)

    for idx1, idx2 in index_pairs:
        label1, range1 = state_names[idx1], state_ranges[idx1]
        label2, range2 = state_names[idx2], state_ranges[idx2]

        if len(range1) == 2:
            range1 = np.linspace(0, 1, 2)
        if len(range2) == 2:
            range2 = np.linspace(0, 1, 2)

        grid1, grid2 = np.meshgrid(range1, range2)

        state_grid = np.tile(fixed_state, (grid1.size, 1))
        state_grid[:, idx1] = grid1.ravel()
        state_grid[:, idx2] = grid2.ravel()

        values = critic_model.predict(state_grid, verbose=0)
        values_grid = values.reshape(grid1.shape)

        plt.figure(figsize=(8, 6))
        contour = plt.contourf(grid1, grid2, values_grid, levels=50)
        plt.xlabel(label1)
        plt.ylabel(label2)
        title = f"Critic Evaluation: {label1} vs {label2}"
        plt.title(title)
        plt.colorbar(contour, label="Value Estimate")
        plt.tight_layout()

        filename = title.replace(" ", "_") + ".png"
        plt.savefig(os.path.join("plots", filename), dpi=300)
        plt.close()


def main() -> None:
    test_name = "ship_long_XXX"

    base_path = os.path.join("outputs", test_name)
    plots_path = os.path.join(base_path, "plots")
    videos_path = os.path.join(base_path, "videos")
    checkpoints_path = os.path.join(base_path, "checkpoints")

    if dir_exists_and_not_empty(plots_path):
        raise RuntimeError(f"Directory '{plots_path}' already exists and is not empty.")

    os.makedirs(plots_path, exist_ok=True)
    os.makedirs(videos_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # what's brown and sticky 
    # raw_env = gym.make('CartPole-v1', render_mode='rgb_array')

    raw_env = gym.make('LunarLander-v2', render_mode='rgb_array')
    environment = RecordVideo(
        raw_env,
        video_folder=videos_path,
        episode_trigger=lambda i: i % 50 == 0,
        name_prefix=f"{test_name}_cartpole"
    )
    controller = ShipTwoNetworkActorCriticController(environment, 0.00001, 0.99)

    past_rewards = []
    past_errors = []
    for i_episode in tqdm(range(5001)):
        done = False
        state, info = environment.reset()
        reward_sum = 0.0
        errors_history = []
        iterations = 0

        while not done and iterations < 1000: # was 500 for sticky stick
            action = controller.choose_action(state)
            new_state, reward, done, truncated, info = environment.step(action)
            controller.learn(state, reward, new_state, done)
            state = new_state
            reward_sum += reward
            errors_history.append(controller.last_error_squared)
            iterations += 1

        past_rewards.append(reward_sum)
        past_errors.append(np.mean(errors_history))

        window_size = 50
        if i_episode % 50 == 0 and len(past_rewards) >= window_size:
            fig, axs = plt.subplots(2)
            axs[0].plot(
                [np.mean(past_errors[i:i + window_size]) for i in range(len(past_errors) - window_size)],
                'tab:red',
            )
            axs[0].set_title('mean squared error')
            axs[1].plot(
                [np.mean(past_rewards[i:i + window_size]) for i in range(len(past_rewards) - window_size)],
                'tab:green',
            )
            axs[1].set_title('sum of rewards')
            plt.savefig(os.path.join(plots_path, f'learning_{i_episode}.png'))
            plt.clf()

        if i_episode % 50 == 0:
            # for one network for both
            # controller.model.save(os.path.join(checkpoints_path, f"model_{i_episode}.h5"))

            # for two networks
            controller.actor.save(os.path.join(checkpoints_path, f"model_actor_{i_episode}.h5"))
            controller.critic.save(os.path.join(checkpoints_path, f"model_critic_{i_episode}.h5"))


    environment.close()

    # for one network for both
    # controller.model.save(os.path.join(base_path, "final.model"))

    # for two networks
    controller.actor.save(os.path.join(checkpoints_path, f"final_actor_{i_episode}.h5"))
    controller.critic.save(os.path.join(checkpoints_path, f"final_critic_{i_episode}.h5"))


if __name__ == '__main__':
    # evaluate_critic_stick()
    # evaluate_critic_ship()
    main()
