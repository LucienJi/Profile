import torch
import torch.nn as nn
import torch.nn.functional as F
from python import magent 
from torch.distributions.categorical import Categorical
import numpy as np

class SumTree(object):
    def __init__(self, r, v, alpha):

        """
        data = array of priority
        :param data: pos 0 no ele; ele from [1 to 2N[
        """

        td = ((r - v) * (r - v)) ** alpha
        td = td / np.sum(td)
        data = td
        l = len(data)
        if l % 2 == 0:
            self.N = l
        else:
            self.N = l - 1
        self.data = np.zeros(2 * self.N)
        self.data[self.N:2 * self.N] = data[0:self.N]

    def _sum(self, i):
        if 2 * i >= self.N:
            self.data[i] = self.data[2 * i] + self.data[2 * i + 1]
            return self.data[i]
        else:
            self.data[i] = self._sum(2 * i) + self._sum(2 * i + 1)
            return self.data[i]

    def build(self):
        self.total_p = self._sum(1)

    def find(self, p):
        idx = 1
        while idx < self.N:
            l = 2 * idx
            r = l + 1
            if self.data[l] >= p:
                idx = l
            else:
                idx = r
                p = p - self.data[l]
        return idx - self.N

    def sample(self, batchsize):
        real_index = []
        interval = self.total_p / batchsize
        for i in range(batchsize):
            try:
                p = np.random.uniform(i * interval, (i + 1) * interval)
            except OverflowError:
                print("OverflowError\n")
                print("Check interval:",interval)
                real_index.append(i)
            else:
                real_index.append(self.find(p))

        return real_index

class MetaBuffer(object):

    def __init__(self, shape, max_len, dtype='float32'):
        self.max_len = max_len
        self.data = np.zeros((max_len,) + shape).astype(dtype)
        self.start = 0
        self.length = 0
        self._flag = 0
        self.shape = shape
        self.max_len = max_len
        self.dtype = dtype

    # this is overridee?
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[idx]

    def sample(self, idx):

        return self.data[idx % self.length]

    def pull(self):
        return self.data[:self.length]

    def append(self, value):
        start = 0
        num = len(value)

        if self._flag + num > self.max_len:
            tail = self.max_len - self._flag
            self.data[self._flag:] = value[:tail]
            num -= tail
            start = tail
            self._flag = 0

        self.data[self._flag:self._flag + num] = value[start:]
        self._flag += num
        self.length = min(self.length + len(value), self.max_len)

    def add(self, value):
        self.data[self._flag] = value
        self._flag += 1
        self.length += 1

    def reset_new(self, start, value):
        self.data[start:self.length] = value

    def clear(self):
        self.data = np.zeros((self.max_len,) + self.shape).astype(self.dtype)
        self.start = 0
        self.length = 0
        self._flag = 0



class AgentMemory(object):
    """
    as the name indicate, it is only for a single agent which make it possible
    to single agent
    """

    def __init__(self, obs_dim, act_dim, n_step, gamma, max_buffer_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        
        self.max_buffer_size = max_buffer_size
        self.n = n_step
        self.gamma = gamma

        """ Agent  Buffer """
        self.obs_buffer = MetaBuffer(shape=self.obs_dim, max_len=self.max_buffer_size)
        self.action_buffer = MetaBuffer(shape=(), max_len=self.max_buffer_size, dtype='int16')
        self.reward_buffer = MetaBuffer(shape=(), max_len=self.max_buffer_size)
        self.old_policy = MetaBuffer(shape=(act_dim,), max_len=max_buffer_size)
        self.old_qvalue = MetaBuffer(shape=(act_dim,), max_len=max_buffer_size)

    def append(self, obs, action, reward, old_policy, old_qvalue):
        self.obs_buffer.append(np.array([obs]))
        self.action_buffer.append(np.array([action], dtype=np.int32))
        self.reward_buffer.append(np.array([reward]))
        self.old_policy.append(np.array([old_policy]))
        self.old_qvalue.append(np.array([old_qvalue]))

    def pull(self):
        res = {
            'obs': self.view_buffer.pull(),
            'action': self.action_buffer.pull(),
            'reward': self.reward_buffer.pull(),
            'old_policy': self.old_policy.pull(),
            'old_qvalue': self.old_qvalue.pull(),
        }

        return res

    def clear(self):
        self.obs_buffer.clear()
        self.action_buffer.clear()
        self.reward_buffer.clear()
        self.old_qvalue.clear()
        self.old_policy.clear()

    def reshape(self):
        if self.n != -1:
            gamma = self.gamma
            n = self.n
            N = len(self.reward_buffer)
            reward = self.reward_buffer.pull()
            assert len(reward) == N
            action = self.action_buffer.pull()
            action_indice = np.eye(self.action_space)[action]

            old_qvalue = self.old_qvalue.pull()
            value = np.sum(action_indice * old_qvalue, axis=-1)
            r = reward[0:N - 1 - n]
            v = value[0:N - 1 - n]
            for i in range(1, n):
                r = r + reward[i:N - 1 - n + i] * (gamma ** i)
            r = r + v * (gamma ** n)
            reward[0:N - 1 - n] = r
            reward[N - 1 - n:N] = reward[N - 1 - n:N] + gamma * value[N - 1 - n:N]
            self.reward_buffer.reset_new(0, reward)
        else:
            gamma = self.gamma
            N = len(self.reward_buffer)
            reward = self.reward_buffer.pull()
            action = self.action_buffer.pull()
            action_indice = np.eye(self.action_space[0])[action]
            old_qvalue = self.old_qvalue.pull()
            keep = np.sum(action_indice * old_qvalue, axis=-1)
            keep = keep[-1]
            for i in reversed(range(N)):
                keep = reward[i] + keep * gamma
                reward[i] = keep
            self.reward_buffer.reset_new(0, reward)

        return
    def tight(self):
        action = self.action_buffer.pull()
        action_indice = np.eye(self.action_space[0])[action]

        # old_policy = self.old_policy.pull()
        old_qvalue = self.old_qvalue.pull()
        value = np.sum(action_indice * old_qvalue, axis=-1)

        self.tree = SumTree(self.reward_buffer.pull(), value, 1)
        self.tree.build()
    def sample(self,batch_size):
        ids = self.tree.sample(batch_size)
        ids = np.array(ids, dtype=np.int32)

        buffer = {}

        buffer['obs'] = self.obs_buffer.sample(ids)
        buffer['action'] = self.action_buffer.sample(ids)
        buffer['reward'] = self.reward_buffer.sample(ids)
        buffer['old_policy'] = self.old_policy.sample(ids)
        buffer['old_qvalue'] = self.old_qvalue.sample(ids)
        return buffer

    def __len__(self):
        return len(self.reward_buffer)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten,self).__init__()
    def forward(self, input):
        return input.view(input.size(0), -1)


def mlp(input_size,embedded_size):
    net = nn.Sequential(
        nn.Linear(in_features=input_size,out_features=input_size),
        nn.ReLU(),
        nn.Linear(in_features=input_size, out_features=input_size),
        nn.ReLU(),
        nn.Linear(in_features=input_size,out_features=embedded_size)

    )

class Policy_Network(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(Policy_Network,self).__init__()
        dim2 = int(obs_dim/2)
        self.net = nn.Sequential(
            nn.Linear(in_features=obs_dim,out_features=dim2),
            nn.ReLU(),
            nn.Linear(in_features=dim2,out_features=dim2),
            nn.ReLU(),
            nn.Linear(in_features=dim2,out_features=act_dim)
        )
    def forward(self,obs):
        h = self.net(obs)
        log_pi = F.log_softmax(h,dim=1)
        return log_pi

class Qvalue_Network(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(Qvalue_Network,self).__init__()
        dim2 = int(obs_dim/2)
        self.net = nn.Sequential(
            nn.Linear(in_features=obs_dim,out_features=dim2),
            nn.ReLU(),
            nn.Linear(in_features=dim2,out_features=dim2),
            nn.ReLU(),
            nn.Linear(in_features=dim2,out_features=act_dim)
        )
    def forward(self,obs):
        q = self.net(obs)
        return q

class ActorCritic(nn.Module):
    def __init__(self,obs_dim,act_dim):
        super(ActorCritic,self).__init__()
        self.pi = Policy_Network(obs_dim,act_dim)
        self.q = Qvalue_Network(obs_dim,act_dim)

    def eval_policy(self,obs):
        log_pi = self.pi(obs)
        return log_pi

    def eval_qvalue(self,obs):
        return self.q(obs)

    def step(self,obs):
        with torch.no_grad():
            log_pi = self.pi(obs)
            q = self.q(obs)
            pi = torch.exp(log_pi)
            a = Categorical(logits=log_pi).sample()
        #print("Test:",np.shape(log_pi.numpy()))
        return q.numpy(),log_pi.numpy(),a.numpy()
    
    def act(self,obs):
        q,log_pi,a = self.step(view,feature)
        return a


class Agent:
    def __init__(self,obs_dim,act_dim,batch_size=1024,learning_rate=1e-5, discount_coef=0.9):

        # Basic Info
        self.learning_rate = learning_rate
        self.gamma = discount_coef
        self.batch_size = batch_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Network
        self.agent = ActorCritic(obs_dim,act_dim)
        self.pi_optimizer = torch.optim.Adam(self.agent.pi.parameters(),lr=learning_rate)
        self.q_optimizer  = torch.optim.Adam(self.agent.q.parameters(),lr=learning_rate)

        # Buffer
        self.ReplayBuffer = AgentMemory(obs_dim,act_dim,n_step = 3,gamma=discount_coef)

    def compute_loss_pi(self,data):
        obs,action,reward,pi,q = data['obs'],data['action'],data['reward'],data['old_policy'],data['old_qvalue']
        action = torch.as_tensor(action)
        reward = torch.as_tensor(reward)
        q = torch.as_tensor(q)
        obs = torch.as_tensor(obs)
        a_indice = F.one_hot(action.long(),num_classes=21)
        adv = (reward - torch.mean(q,dim=1))
        log_pi = torch.log(self.agent.pi(obs))
        log_pi = torch.sum(a_indice * log_pi,dim=1)
        loss_pi = -(torch.mean(adv * log_pi))
        return loss_pi


    def compute_loss_q(self,data):
        obs,action,reward,pi,q = data['obs'],data['action'],data['reward'],data['old_policy'],data['old_qvalue']
        reward = torch.as_tensor(reward)
        pi = torch.as_tensor(pi)
        q = self.agent.q(torch.as_tensor(obs))
        q_value = torch.sum(q*pi,dim=1)
        loss_q = torch.mean((reward - q_value)**2)
        return loss_q

    def update(self):
        self.replaybuffer.tight()
        data = self.replaybuffer.sample(self.batch_size)
        pi_loss_old = self.compute_loss_pi(data)
        q_loss_old = self.compute_loss_q(data)
        train_iteration = 1
        for i in range(train_iteration):
            self.pi_optimizer.zero_grad()
            loss_pi = self.compute_loss_pi(data)
            loss_pi.backward()
            self.pi_optimizer.step()
        for i in range(train_iteration):
            self.q_optimizer.zero_grad()
            loss_q = self.compute_loss_q(data)
            loss_q.backward()
            self.q_optimizer.step()
        return pi_loss_old,q_loss_old

    def act(self,data):
        obs = data['obs']
        res = self.agent.act(torch.as_tensor(obs)).astype(np.int32)
        
        return res
    def step(self,data):
        obs = data['obs']
        a,pi,q = self.agent.step(torch.as_tensor(obs))
        return pi,q

    def save(self,path):
        torch.save(self.agent.state_dict(),path)
        print("Saved")
        
    def load(self,path):
        self.agent.load_state_dict(torch.load(path))
        print("Loaded")

