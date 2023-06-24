# == import socket == 
import socket 

import argparse
import json
import os
import sys
import time

import cv2
import imageio
import numpy as np
import torch

import utils
from SAC import SAC
from env import HumanoidStandupEnv, HumanoidStandupVelocityEnv, HumanoidVariantStandupEnv, \
    HumanoidVariantStandupVelocityEnv
from utils import RLLogger, ReplayBuffer, organize_args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.set_printoptions(precision=5, suppress=True)


# ==tcp sending==
def UEsending(action):
    return str(action)

class UESending:
    def __init__(self):
        self.ip = "127.0.0.1"
        self.port = 1111
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client = None
        self.server.bind((self.ip, self.port))
        self.server.listen(1)
        self.client, address = self.server.accept()
        
    def test(self):
        self.ip = "127.0.0.1"
        self.port = 1111
        self.firstTime = True
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind((self.ip, self.port))
        self.server.listen(1)
        self.client, address = self.server.accept()

        if(not self.firstTime):
            print(f"Connection Established - {address[0]}:{address[1]}")
            self.firstTime = False
            senddata = "sending"
            self.client.send(senddata.encode())
            print(senddata)   

    
    def ueReceive(self):
        receiveddata = self.client.recv(1024)
        #receiveddata = receiveddata.decode("utf-8")
        #print(receiveddata.decode("utf-8"))
        receiveddata = receiveddata.decode("utf-8")
        #print("Received OK")
        #print(receiveddata)
        tempReward = None
        tempState = []
        temp1 = receiveddata.split("@")
        if(temp1[1] != None):
            tempReward = temp1[1]
            temp = temp1[0].split("#")
            for t in temp:
                if(t != ''):
                    tempState.append(float(t))
        else:
            tempReward = 0
            tempState = []
        return tempReward, tempState


    def ueSend(self, action):
        senddata = str(action)
        self.client.send(senddata.encode())
        #print(senddata)


    def ueSplitSend(self, action):
        senddata = ""
        for a in action:
            a*=100000 #[TODO] 적합한 값 찾기 100000
            if a > 0:
                senddata += " "+str(a)
            else:
                senddata += " "+str(a)
            #print(senddata)
        self.client.send(senddata.encode())
        #print("===")
            

class ArgParserTrain(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.add_argument('--env', type=str, default='HumanoidStandup',
                          choices=['HumanoidStandup', 'HumanoidVariantStandup'])
        self.add_argument('--variant', type=str, default='', choices=['Disabled', 'Noarm'])
        self.add_argument('--test_policy', default=False, action='store_true') #원래 false
        self.add_argument('--teacher_student', default=False, action='store_true')
        self.add_argument('--to_file', default=False, action='store_true')
        self.add_argument("--teacher_power", default=0.4, type=float)
        self.add_argument("--teacher_dir", default=None, type=str)
        self.add_argument("--seed", default=0, type=int)
        self.add_argument("--power", default=1.0, type=float)
        self.add_argument("--curr_power", default=1.0, type=float)
        self.add_argument("--power_end", default=0.4, type=float)
        self.add_argument("--slow_speed", default=0.2, type=float) # 0.2임
        self.add_argument("--fast_speed", default=0.8, type=float)
        self.add_argument("--target_speed", default=0.5, type=float, help="Target speed is used to test the weaker policy")
        self.add_argument("--threshold", default=60, type=float)
        self.add_argument('--max_timesteps', type=int, default=10000000, help='Number of simulation steps to run')
        self.add_argument('--test_interval', type=int, default=20000, help='Number of simulation steps between tests')
        self.add_argument('--test_iterations', type=int, default=10, help='Number of test episodes')
        self.add_argument('--replay_buffer_size', type=int, default=1e6, help='Capacity of the replay buffer')
        self.add_argument('--avg_reward', default=False, action='store_true')
        self.add_argument("--work_dir", default='./experiment/')
        self.add_argument("--load_dir", default='./experiment/standupfinal', type=str) #처음부터 학습시키려면 여길 None으로 바꾸면 된다.
        # SAC hyperparameters
        self.add_argument("--batch_size", default=1024, type=int)
        self.add_argument("--discount", default=0.97, type=float)
        self.add_argument("--init_temperature", default=0.1, type=float)
        self.add_argument("--critic_target_update_freq", default=2, type=int)
        self.add_argument("--alpha_lr", default=1e-4, type=float)
        self.add_argument("--actor_lr", default=1e-5, type=float)
        self.add_argument("--critic_lr", default=1e-4, type=float)
        self.add_argument("--tau", default=0.005)
        self.add_argument("--start_timesteps", default=10000, type=int)
        self.add_argument('--log_interval', type=int, default=100, help='log every N')
        self.add_argument("--tag", default="")


def main():
    args = ArgParserTrain().parse_args()
    trainer = Trainer(args)
    trainer.train_sac()


class Trainer():
    def __init__(self, args):
        
        # ==sending==
        self.tcp = UESending()

        args = organize_args(args)
        self.args = args
        self.setup(args)
        self.logger.log_start(sys.argv, args)
        self.env = self.create_env(args)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

        obs_dim = self.env.obs_shape
        self.act_dim = self.env.action_space

        self.buf = ReplayBuffer(obs_dim, self.act_dim, args, max_size=int(args.replay_buffer_size))
        self.env.buf = self.buf

        self.policy = SAC(obs_dim, self.act_dim,
                          init_temperature=args.init_temperature,
                          alpha_lr=args.alpha_lr,
                          actor_lr=args.actor_lr,
                          critic_lr=args.critic_lr,
                          tau=args.tau,
                          discount=args.discount,
                          critic_target_update_freq=args.critic_target_update_freq,
                          args=args)
        
        self.trashpolicy = SAC(obs_dim, self.act_dim,
                          init_temperature=args.init_temperature,
                          alpha_lr=args.alpha_lr,
                          actor_lr=args.actor_lr,
                          critic_lr=args.critic_lr,
                          tau=args.tau,
                          discount=args.discount,
                          critic_target_update_freq=args.critic_target_update_freq,
                          args=args)

        if args.test_policy or args.load_dir:
            self.policy.load(os.path.join(args.load_dir + '/model', 'best_model'), load_optimizer=False)

        if args.teacher_student:
            self.teacher_policy = SAC(self.env.teacher_env.obs_shape,
                                      self.env.teacher_env.action_space,
                                      init_temperature=args.init_temperature,
                                      alpha_lr=args.alpha_lr,
                                      actor_lr=args.actor_lr,
                                      critic_lr=args.critic_lr,
                                      tau=args.tau,
                                      discount=args.discount,
                                      critic_target_update_freq=args.critic_target_update_freq,
                                      args=args)
            self.teacher_policy.load(os.path.join(args.teacher_dir + '/model', 'best_model'), load_optimizer=False)
            for param in self.teacher_policy.parameters():
                param.requires_grad = False
            self.env.set_teacher_policy(self.teacher_policy)

    def setup(self, args):
        ts = time.gmtime()
        ts = time.strftime("%m-%d-%H-%M", ts)
        exp_name = args.env + '_' + ts + '_' + 'seed_' + str(args.seed)
        exp_name = exp_name + '_' + args.tag if args.tag != '' else exp_name
        self.experiment_dir = os.path.join(args.work_dir, exp_name)

        utils.make_dir(self.experiment_dir)
        self.video_dir = utils.make_dir(os.path.join(self.experiment_dir, 'video'))
        self.model_dir = utils.make_dir(os.path.join(self.experiment_dir, 'model'))
        self.buffer_dir = utils.make_dir(os.path.join(self.experiment_dir, 'buffer'))
        self.logger = RLLogger(self.experiment_dir)

        self.save_args(args)

    def save_args(self, args):
        with open(os.path.join(self.experiment_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

    def env_function(self):
        if self.args.env == 'HumanoidStandup':
            if self.args.teacher_student:
                return HumanoidStandupVelocityEnv
            return HumanoidStandupEnv
        elif self.args.env == "HumanoidVariantStandup":
            if self.args.teacher_student:
                return HumanoidVariantStandupVelocityEnv
            return HumanoidVariantStandupEnv

    def create_env(self, args):
        env_generator = self.env_function()
        return env_generator(args, args.seed)

    def train_sac(self):
        store_buf = False if self.args.teacher_student else True
        test_time = True if self.args.test_policy else False
        power = self.env.reset(store_buf=store_buf, test_time=test_time)
        print("power1 :", power)
        # state = 언리얼 reset하고 state 보내주기
        state = [[-7.671,-0.84,89.729,-80.895,-7.296,176.544,6.288,-90.332,10.957,90.91,5.953,7.383,-6.309,89.564,-11.028,7.838,-42.466,-37.143,-5.836,-124.583,-145.984,0.122,33.639,-52.953,10.088,30.22,56.08,9.604,130.506,-17.92,-6.444,130.198,18.608,-6.437,0,0,0.016,0,0.001,0,0,-0,0,0,0,-0,0,0,0.011,-0.061,-0,-0.015,-0.001,0,-0.001,-0.019,0.003,0,-0,0.004,0.002,-0,0,0.001,-0,-0,0]]
        state.append(power if not np.isscalar(power) else [power])
        state = np.concatenate(state)
        #state 마지막 power로 바꿔주기
        done = False 
        t = 0
        self.last_power_update = 0
        self.last_duration = np.inf
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        self.curriculum = True
        best_reward = -np.inf

        while t < int(self.args.max_timesteps):
            
            # Select action randomly or according to policy
            # action 설정
            if self.args.test_policy:
                action = self.policy.select_action(state)
                #print("1: ", action)
            elif (t < self.args.start_timesteps and not self.args.load_dir):
                action = np.clip(2 * np.random.random_sample(size=self.act_dim) - 1, -self.env.power, self.env.power)
                #print("2: ", action)
            else:
                action = self.policy.sample_action(state)
                #print("3: ", action)
            t += 1
            
            # ==step 진행 ==
            self.tcp.ueSplitSend(action)
            reward, next_state = self.tcp.ueReceive()
            
            reward = float(reward)
            next_state = [next_state]
            next_state.append(power if not np.isscalar(power) else [power])
            next_state = np.concatenate(next_state)
    
            done = self.env.step(a=action)

            episode_timesteps += 1
            self.buf.add(state, action, next_state, reward, self.env.terminal_signal)
            state = next_state
            episode_reward += reward

            # Train agent after collecting sufficient data
            if (t >= self.args.start_timesteps) and not self.args.test_policy:
                self.policy.train(self.buf, self.args.batch_size)
            else:
                self.trashpolicy.train(self.buf, self.args.batch_size)

            if done:
                self.logger.log_train_episode(t, episode_num, episode_timesteps, episode_reward, self.policy.loss_dict,
                                              self.env, self.args)
                self.policy.reset_record()
                power = self.env.reset(test_time=test_time)
                print("power2 :", power)
                # state = 언리얼 reset하고 state 보내주기
                state = [[-7.671,-0.84,89.729,-80.895,-7.296,176.544,6.288,-90.332,10.957,90.91,5.953,7.383,-6.309,89.564,-11.028,7.838,-42.466,-37.143,-5.836,-124.583,-145.984,0.122,33.639,-52.953,10.088,30.22,56.08,9.604,130.506,-17.92,-6.444,130.198,18.608,-6.437,0,0,0.016,0,0.001,0,0,-0,0,0,0,-0,0,0,0.011,-0.061,-0,-0.015,-0.001,0,-0.001,-0.019,0.003,0,-0,0.004,0.002,-0,0,0.001,-0,-0,0]]
                state.append(power if not np.isscalar(power) else [power])
                state = np.concatenate(state)
                #state 마지막 power로 바꿔주기
                done = False 
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            if t % self.args.test_interval == 0:
                test_reward, min_test_reward, video = self.run_tests(self.env.power_base, self.policy)
                
                power = self.env.reset(store_buf=store_buf, test_time=test_time)
                print("power3 :", power)
                # state = 언리얼 reset하고 state 보내주기
                state = [[-7.671,-0.84,89.729,-80.895,-7.296,176.544,6.288,-90.332,10.957,90.91,5.953,7.383,-6.309,89.564,-11.028,7.838,-42.466,-37.143,-5.836,-124.583,-145.984,0.122,33.639,-52.953,10.088,30.22,56.08,9.604,130.506,-17.92,-6.444,130.198,18.608,-6.437,0,0,0.016,0,0.001,0,0,-0,0,0,0,-0,0,0,0.011,-0.061,-0,-0.015,-0.001,0,-0.001,-0.019,0.003,0,-0,0.004,0.002,-0,0,0.001,-0,-0,0]]
                state.append(power if not np.isscalar(power) else [power])
                state = np.concatenate(state)
                #state 마지막 power로 바꿔주기
                done = False 
                
                criteria = test_reward if self.args.avg_reward else min_test_reward
                self.curriculum = self.update_power(self.env, criteria, t)
                if (test_reward > best_reward):
                    self.policy.save(os.path.join(self.model_dir, 'best_model')) # model을 .pt 파일로 저장하는 부분
                    best_reward = test_reward
                    self.logger.info("Best model saved")
                self.policy.save(os.path.join(self.model_dir, 'newest_model'))
                self.logger.log_test(test_reward, min_test_reward, self.curriculum, self.env.power_base)
            
    
    def update_power(self, env, criteria, t):
        if not self.curriculum:
            return False
        if criteria > self.args.threshold:
            env.power_base = max(env.power_end, 0.95 * env.power_base)
            self.args.curr_power = env.power_base
            self.save_args(self.args)
            if env.power_base == env.power_end:
                return False
            self.last_duration = t - self.last_power_update
            self.last_power_update = t

        else:
            current_stage_length = t - self.last_power_update
            if current_stage_length > min(1000000, max(300000, 1.5 * self.last_duration)) and env.power_base < 1.0:
                env.power_base = env.power_base / 0.95
                self.args.curr_power = env.power_base
                self.save_args(self.args)
                env.power_end = env.power_base
                return False

        return True

    def run_tests(self, power_base, test_policy):
        print("run_tests() start")
        np.random.seed(self.args.seed)
        test_env_generator = self.env_function()
        test_env = test_env_generator(self.args, self.args.seed + 10)
        test_env.power = power_base
        if self.args.teacher_student:
            test_env.set_teacher_policy(self.teacher_policy)
        test_reward = []
        speed_profile = np.linspace(self.args.slow_speed, self.args.fast_speed, num=self.args.test_iterations,
                                    endpoint=True)
        video_array = []
        for i in range(self.args.test_iterations):
            video = []

            print("run_test :", i)
            _ = self.env.reset(test_time=True)
            power = power_base
            print("power4 :", power)
            #state = 언리얼 reset하고 state 보내주기
            state = [[-7.671,-0.84,89.729,-80.895,-7.296,176.544,6.288,-90.332,10.957,90.91,5.953,7.383,-6.309,89.564,-11.028,7.838,-42.466,-37.143,-5.836,-124.583,-145.984,0.122,33.639,-52.953,10.088,30.22,56.08,9.604,130.506,-17.92,-6.444,130.198,18.608,-6.437,0,0,0.016,0,0.001,0,0,-0,0,0,0,-0,0,0,0.011,-0.061,-0,-0.015,-0.001,0,-0.001,-0.019,0.003,0,-0,0.004,0.002,-0,0,0.001,-0,-0,0]]

            state.append(power if not np.isscalar(power) else [power])
            state = np.concatenate(state)
            #state 마지막 power로 바꿔주기
            done = False
            
            episode_timesteps = 0
            episode_reward = 0

            while not done:
                # ==step 진행 ==
                action = test_policy.select_action(state)
                self.tcp.ueSplitSend(action)
                reward, next_state = self.tcp.ueReceive()
                reward = float(reward)
                next_state = [next_state]
                next_state.append(power if not np.isscalar(power) else [power])
                next_state = np.concatenate(next_state)
    
                done = self.env.step(a=action)
                
                episode_reward += reward
                state = next_state
                episode_timesteps += 1
                
                self.trashpolicy.train(self.buf, self.args.batch_size)

            test_reward.append(episode_reward)
        test_reward = np.array(test_reward)
        return test_reward.mean(), test_reward.min(), video_array


if __name__ == "__main__":
    main()
