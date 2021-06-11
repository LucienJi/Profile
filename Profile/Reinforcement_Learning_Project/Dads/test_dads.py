import gym
from algo.Dads.agent import SAC

env_name = 'BipedalWalker-v2'
env = gym.make(env_name)

"""
with open('{}_test.csv'.format(env_name), 'w+') as myfile:
    myfile.write('{0},{1},{2}\n'.format("Episode", "Reward", "Value_Loss"))
"""

print(env.action_space.shape)
print(env.observation_space.shape)
skill_dim =5

agent = SAC(env.observation_space,skill_dim,env.action_space)
total_episode = 5000
per_episode = 500

for i_episode in range(total_episode):
    data = {}
    observation = env.reset()
    data['obs'] = observation.reshape(1,-1)
    done = False
    ct = 0
    data['skill'] = agent.sd.get_prior(data['obs'])

    while ct < per_episode and not done:
        ct += 1
        env.render()

        # 网络在每一个step都会根据 obs 输出 action，qvalue，policy
        # action 用于更新agent动作
        # qvalue，policy用于放入更新阶段

        data['act'] = agent.act(data['obs'],data["skill"])

        data['next_obs'], external_reward, data['done'], info = env.step(data['act'][0])
        data['next_obs'] = data['next_obs'].reshape(1,-1)
        data['rew'] = agent.sd.get_reward(data['obs'],data['skill'],data['next_obs'])

        agent.push(data)
        data['obs'] = data['next_obs']
        done = data['done']


    agent.train()

agent.save()
