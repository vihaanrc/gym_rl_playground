from ddpg import DDPG
import gymnasium as gym



if __name__ == "__main__":
    
    env = gym.make("MountainCarContinuous-v0")
    state_dim = 2 #position of car, velocity of car
    print(state_dim)
    action_dim = 1 #directional force
   
    h1_dim = 20
    mountainCarDDPG = DDPG(state_dim=state_dim, action_dim = action_dim, h1_dim = h1_dim, env=env)
    mountainCarDDPG.train(episodes=150, env=env)
    env.close()

    env = env = gym.make("MountainCarContinuous-v0", render_mode="human")
    
    mountainCarDDPG.test(episodes=10, env=env)

    env.close()