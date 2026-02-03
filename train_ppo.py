# encoding=utf-8
import time

from spikingjelly.activation_based import functional
from torch.distributions import Categorical

from env import VADEnvironment
from model import Buffer, PPO
from real_time_plt import RTP

from tools import *
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

torch.autograd.set_detect_anomaly(True)

def train_lstm(model, envs, each_env_episodes):
    for episode in range(each_env_episodes):
        all_actor_loss = []
        all_critic_loss = []
        all_q0 = []
        q0_history = []
        all_avg_reward = []

        each_episode_reward = []

        episode_start_time = time.time()
        x = env.reset().to(training_params["device"])

        # bs = 1
        state = model.init_lstm_state()
        state_input = (x, state)
        done = False

        count = 1
        while not done:
            with torch.no_grad():
                output, state = model.old_actor(state_input)  # 单bs推理
                value = model.old_critic(state_input).item()
                q0_history.append(value)
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)

            # 与环境交互
            action_probs = torch.softmax(output, -1)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            x, reward, done = env.step(action)
            next_state = (x.to(training_params["device"]), state)

            # 存储过渡信息
            buffer.add(state_input, action.item(), reward, done, log_prob.item(), value)
            each_episode_reward.append(reward)

            if not count % training_params["update_batch"]:
                # 计算最后状态的value（用于bootstrap）
                with torch.no_grad():
                    last_value = model.old_critic(next_state).item()
                    q0_history.append(last_value)

                # 计算returns和advantages
                buffer.compute_returns_advantages(last_value)

                # 更新PPO参数
                critic_loss, actor_loss = model.update(buffer, model_type=training_params["actor"])
                buffer.clear()
                if training_params["snn"]["is_snn"]:
                    functional.reset_net(model.actor)
                print(f"[ppo][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss}  avg_q0: {sum(q0_history) / len(q0_history)}  avg_reward: {sum(each_episode_reward) / len(each_episode_reward)}")
                all_actor_loss.append(actor_loss)
                all_critic_loss.append(critic_loss)
                all_avg_reward.append(sum(each_episode_reward) / len(each_episode_reward))

                all_q0.append(sum(q0_history) / len(q0_history))
                rtp.plot_training_metrics(
                    all_actor_loss,
                    all_critic_loss,
                    all_q0,
                    all_avg_reward
                )

                q0_history = []

            state_input = next_state
            count += 1

        # 打印回合总结信息
        episode_time = time.time() - episode_start_time
        print(
            f"[Env {step} / {len(envs) - 1}] "
            f"[Episode {episode}/{each_env_episodes - 1}] Completed! "
            f"Time: {episode_time:.2f}s\n"
        )


def train_mlp(model, envs, each_env_episodes):
    for episode in range(each_env_episodes):
        all_actor_loss = []
        all_critic_loss = []
        all_q0 = []
        q0_history = []
        all_avg_reward = []

        each_episode_reward = []

        episode_start_time = time.time()
        state = env.reset().to(training_params["device"])

        # bs = 1
        done = False

        count = 1
        while not done:
            with torch.no_grad():
                output, _ = model.old_actor(state)  # 单bs推理
                value = model.old_critic(state).item()
                q0_history.append(value)
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)

            # 与环境交互
            action_probs = torch.softmax(output, -1)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            next_state, reward, done = env.step(action)
            next_state = next_state.to(training_params["device"])

            # 存储过渡信息
            buffer.add(state, action.item(), reward, done, log_prob.item(), value)
            each_episode_reward.append(reward)

            if not count % training_params["update_batch"]:
                # 计算最后状态的value（用于bootstrap）
                with torch.no_grad():
                    last_value = model.old_critic(next_state).item()
                    q0_history.append(last_value)

                # 计算returns和advantages
                buffer.compute_returns_advantages(last_value)

                # 更新PPO参数
                critic_loss, actor_loss = model.update(buffer, model_type=training_params["actor"])
                buffer.clear()
                if training_params["snn"]["is_snn"]:
                    functional.reset_net(model.actor)
                print(f"[ppo][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss}  avg_q0: {sum(q0_history) / len(q0_history)}  avg_reward: {sum(each_episode_reward) / len(each_episode_reward)}")
                all_actor_loss.append(actor_loss)
                all_critic_loss.append(critic_loss)
                all_avg_reward.append(sum(each_episode_reward) / len(each_episode_reward))

                all_q0.append(sum(q0_history) / len(q0_history))
                rtp.plot_training_metrics(
                    all_actor_loss,
                    all_critic_loss,
                    all_q0,
                    all_avg_reward
                )

                q0_history = []

            state = next_state
            count += 1

        # 打印回合总结信息
        episode_time = time.time() - episode_start_time
        print(
            f"[Env {step} / {len(envs) - 1}] "
            f"[Episode {episode}/{each_env_episodes - 1}] Completed! "
            f"Time: {episode_time:.2f}s\n"
        )


def train_ntm(model, envs, each_env_episodes):
    for episode in range(each_env_episodes):
        all_actor_loss = []
        all_critic_loss = []
        all_q0 = []
        q0_history = []
        all_avg_reward = []

        each_episode_reward = []

        episode_start_time = time.time()
        x = env.reset().to(training_params["device"])

        # bs = 1
        state = model.init_sequence()
        state_input = (x, state)
        done = False

        count = 1
        while not done:
            model.old_actor.reset_memory(1)
            with torch.no_grad():
                output, state = model.old_actor(state_input)  # 单bs推理
                value = model.old_critic(state_input).item()
                q0_history.append(value)
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)

            # 与环境交互
            action_probs = torch.softmax(output, -1)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            x, reward, done = env.step(action)
            next_state = (x.to(training_params["device"]), state)

            # 存储过渡信息
            buffer.add(state_input, action.item(), reward, done, log_prob.detach(), value)
            each_episode_reward.append(reward)

            if not count % training_params["update_batch"]:
                model.actor.reset_memory(training_params["update_batch"])
                with torch.no_grad():
                    last_value = model.old_critic(next_state).item()
                    q0_history.append(last_value)

                # 计算returns和advantages
                buffer.compute_returns_advantages(last_value)

                # 更新PPO参数
                critic_loss, actor_loss = model.update(buffer, model_type=training_params["actor"])
                buffer.clear()
                print(f"[ppo][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss}  avg_q0: {sum(q0_history) / len(q0_history)}  avg_reward: {sum(each_episode_reward) / len(each_episode_reward)}")
                all_actor_loss.append(actor_loss)
                all_critic_loss.append(critic_loss)
                all_avg_reward.append(sum(each_episode_reward) / len(each_episode_reward))

                all_q0.append(sum(q0_history) / len(q0_history))
                rtp.plot_training_metrics(
                    all_actor_loss,
                    all_critic_loss,
                    all_q0,
                    all_avg_reward
                )

                q0_history = []

            state_input = next_state
            count += 1

        # 打印回合总结信息
        episode_time = time.time() - episode_start_time
        print(
            f"[Env {step} / {len(envs) - 1}] "
            f"[Episode {episode}/{each_env_episodes - 1}] Completed! "
            f"Time: {episode_time:.2f}s\n"
        )


def train_sincnet(model, envs, each_env_episodes):
    for episode in range(each_env_episodes):
        all_actor_loss = []
        all_critic_loss = []
        all_q0 = []
        q0_history = []
        all_avg_reward = []

        each_episode_reward = []

        episode_start_time = time.time()
        x = env.reset().to(training_params["device"])

        # bs = 1
        state_input = x
        done = False

        count = 1
        while not done:
            with torch.no_grad():
                output, _ = model.old_actor(state_input)  # 单bs推理
                value = model.old_critic(state_input).item()
                q0_history.append(value)
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)

            # 与环境交互
            action_probs = torch.softmax(output, -1)
            dist = Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            x, reward, done = env.step(action)
            next_state = x.to(training_params["device"])

            # 存储过渡信息
            buffer.add(state_input, action.item(), reward, done, log_prob.item(), value)
            each_episode_reward.append(reward)

            if not count % training_params["update_batch"]:
                # 计算最后状态的value（用于bootstrap）
                with torch.no_grad():
                    last_value = model.old_critic(next_state).item()
                    q0_history.append(last_value)

                # 计算returns和advantages
                buffer.compute_returns_advantages(last_value)

                # 更新PPO参数
                critic_loss, actor_loss = model.update(buffer, model_type=training_params["actor"])
                buffer.clear()
                if training_params["snn"]["is_snn"]:
                    functional.reset_net(model.actor)
                print(f"[ppo][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss}  avg_q0: {sum(q0_history) / len(q0_history)}  avg_reward: {sum(each_episode_reward) / len(each_episode_reward)}")
                all_actor_loss.append(actor_loss)
                all_critic_loss.append(critic_loss)
                all_avg_reward.append(sum(each_episode_reward) / len(each_episode_reward))

                all_q0.append(sum(q0_history) / len(q0_history))
                rtp.plot_training_metrics(
                    all_actor_loss,
                    all_critic_loss,
                    all_q0,
                    all_avg_reward
                )

                q0_history = []

            state_input = next_state
            count += 1

        # 打印回合总结信息
        episode_time = time.time() - episode_start_time
        print(
            f"[Env {step} / {len(envs) - 1}] "
            f"[Episode {episode}/{each_env_episodes - 1}] Completed! "
            f"Time: {episode_time:.2f}s\n"
        )


if __name__ == '__main__':
    sr = 32000
    frame_duration = 0.25
    max_episodes = [30, 20, 20, 10, 10]

    # 超参数设置（新增参数保存功能）
    training_params = {
        "train": "ppo",
        "work_dir": "train-run-ntm-ppo",
        "pretrain_pth_path": r"G:\ms\snn\pretrain-ntm\model_checkpoint_epoch21.pth",
        "task": "vad",
        "snn": {
            "is_snn": False,
            "T": 8
        },
        "frame_duration": frame_duration,
        "sr": sr,
        "num_inputs": int(sr * frame_duration),
        "actor": "ntm",
        "controller_size": 1024,
        "value_hidden_size": 512,
        "controller_layers": 4,
        "dropout_rate": 0.3,
        "num_outputs": 2,
        "num_heads": 4,
        "N": 12,
        "M": 256,
        "batch_size": 1,
        "max_episodes": max_episodes,
        "update_batch": 500,
        "actor_lr": 0.000001,
        "critic_lr": 0.001,
        "gamma": 0.98,
        "ppo": {
            "K_epochs": 8,
            "eps_clip": 0.2,
            "entropy_coef": 0.01,
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "sincnet": {
            'cnn_N_filt': [80, 60, 60],
            'cnn_len_filt': [251, 5, 5],
            'cnn_max_pool_len': [4, 4, 4],
            'cnn_act': ['leaky_relu', 'leaky_relu', 'leaky_relu'],
            'cnn_drop': [0.3, 0.3, 0.3],
            'cnn_use_laynorm': [True, True, True],
            'cnn_use_batchnorm': [False, False, False],
            'cnn_use_laynorm_inp': True,
            'cnn_use_batchnorm_inp': False,
            'input_dim': int(sr * frame_duration),
            'fs': sr,
            "output_dim": 2
        }
    }

    assert training_params["actor"] in ["ntm", "lstm", "sincnet", "mlp"], "models should in [ntm, lstm, sincnet, mlp]"

    if not os.path.exists(training_params["work_dir"]):
        os.mkdir(training_params["work_dir"])
        os.mkdir(os.path.join(training_params["work_dir"], "training_plots"))

    print(training_params)
    save_params(training_params, training_params["work_dir"])

    model = PPO(
        training_params
    )
    buffer = Buffer()

    # 加载预训练模型
    if training_params["pretrain_pth_path"] != "":
        print("loading pretrain checkpoint !!!")
        checkpoint = torch.load(training_params["pretrain_pth_path"])

        # 加载模型权重
        model.actor.load_state_dict(checkpoint['model_state'])

    envs = [
        VADEnvironment(
            r"G:\ms\snn\data\train\course_study\wav0\data",
            r"G:\ms\snn\data\train\label_025",
            r"G:\ms\snn\data\train\vad_025",
            frame_duration=training_params["frame_duration"],
            sr=training_params["sr"],
            task=training_params["task"]
        ),
        VADEnvironment(
            r"G:\ms\snn\data\train\course_study\wav1\data",
            r"G:\ms\snn\data\train\vad_025",
            r"G:\ms\snn\data\train\vad_025",
            frame_duration=training_params["frame_duration"],
            sr=training_params["sr"],
            task=training_params["task"]
        ),
        VADEnvironment(
            r"G:\ms\snn\data\train\course_study\wav2\data",
            r"G:\ms\snn\data\train\vad_025",
            r"G:\ms\snn\data\train\vad_025",
            frame_duration=training_params["frame_duration"],
            sr=training_params["sr"],
            task=training_params["task"]
        ),
        VADEnvironment(
            r"G:\ms\snn\data\train\course_study\wav3\data",
            r"G:\ms\snn\data\train\vad_025",
            r"G:\ms\snn\data\train\vad_025",
            frame_duration=training_params["frame_duration"],
            sr=training_params["sr"],
            task=training_params["task"]
        ),
        VADEnvironment(
            r"G:\ms\snn\data\train\course_study\wav4\data",
            r"G:\ms\snn\data\train\vad_025",
            r"G:\ms\snn\data\train\vad_025",
            frame_duration=training_params["frame_duration"],
            sr=training_params["sr"],
            task=training_params["task"]
        ),
    ]

    # train
    for step, env in enumerate(envs):
        for each_env_episodes in max_episodes:
            print(f"--------  env [{step} / {len(envs) - 1}]  --------")

            rtp = RTP(os.path.join(training_params["work_dir"], "training_plots"), step)
            start_time = time.time()

            if training_params["actor"] == "ntm":
                train_ntm(model, envs, each_env_episodes)
            elif training_params["actor"] == "lstm":
                train_lstm(model, envs, each_env_episodes)
            elif training_params["actor"] == "mlp":
                train_mlp(model, envs, each_env_episodes)
            else:
                train_sincnet(model, envs, each_env_episodes)

            end_time = time.time()

            print(f"----------[env {step} / {len(envs) - 1}, cost time: {end_time - start_time}]----------")

            # 权重保存
            pth_path = os.path.join(training_params["work_dir"], f"actor_env{step}.pth")
            model.save_actor(pth_path)
