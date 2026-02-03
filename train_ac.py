# encoding=utf-8
import time

from spikingjelly.activation_based import functional

from env import VADEnvironment
from model import ActorCritic
from real_time_plt import RTP

from tools import *
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

torch.autograd.set_detect_anomaly(True)

def train_lstm(model, envs, each_env_episodes):
    for episode in range(each_env_episodes):
        all_actor_loss = []
        all_critic_loss = []
        all_q0 = []
        all_avg_reward = []

        episode_start_time = time.time()
        x = env.reset().to(training_params["device"])

        # bs = 1
        state = model.init_lstm_state()
        state_input = (x, state)
        done = False
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        count = 1
        while not done:
            output, state = model(state_input)  # 单bs推理
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)

            # 与环境交互
            action = torch.argmax(output)
            x, reward, done = env.step(action)
            next_state = (x.to(training_params["device"]), state)

            # 存储过渡信息
            states.append(state_input)
            actions.append(output)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(1 if done else 0)

            if not count % training_params["update_batch"]:
                states_tensor = concatenate_lstm_states(states)
                next_states_tensor = concatenate_lstm_states(next_states)
                actions_tensor = torch.cat(actions, dim=0)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(training_params["device"])
                dones_tensor = torch.tensor(dones, dtype=torch.float32).to(training_params["device"])
                critic_loss, actor_loss, avg_q0 = model.update(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)  # 批量bs反向传播
                if training_params["snn"]["is_snn"]:
                    functional.reset_net(model.actor)
                print(f"[ac][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss}  avg_q0: {avg_q0}  avg_reward: {torch.mean(rewards_tensor).cpu()}")
                all_actor_loss.append(actor_loss)
                all_critic_loss.append(critic_loss)
                all_q0.append(avg_q0)
                all_avg_reward.append(torch.mean(rewards_tensor).cpu())

                rtp.plot_training_metrics(
                    all_actor_loss,
                    all_critic_loss,
                    all_q0,
                    all_avg_reward
                )

                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

            state_input = next_state
            count += 1

        if len(states) != 0:
            states_tensor = concatenate_lstm_states(states)
            next_states_tensor = concatenate_lstm_states(next_states)
            actions_tensor = torch.cat(actions, dim=0)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(training_params["device"])
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(training_params["device"])
            critic_loss, actor_loss, avg_q0 = model.update(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)
            print(f"[ac][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss} avg_q0: {avg_q0}  avg_reward: {torch.mean(rewards_tensor).cpu()}")
            all_actor_loss.append(actor_loss)
            all_critic_loss.append(critic_loss)
            all_q0.append(avg_q0)
            all_avg_reward.append(torch.mean(rewards_tensor).cpu())

            rtp.plot_training_metrics(
                all_actor_loss,
                all_critic_loss,
                all_q0,
                all_avg_reward
            )

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
        all_avg_reward = []

        episode_start_time = time.time()
        state = env.reset().to(training_params["device"])

        # bs = 1
        done = False
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        count = 1
        while not done:
            output, _ = model(state)  # 单bs推理
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)

            # 与环境交互
            action = torch.argmax(output)
            next_state, reward, done = env.step(action)
            next_state = next_state.to(training_params["device"])

            # 存储过渡信息
            states.append(state)
            actions.append(output)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(1 if done else 0)

            if not count % training_params["update_batch"]:
                states_tensor = torch.cat(states, dim=0)
                next_states_tensor = torch.cat(next_states, dim=0)
                actions_tensor = torch.cat(actions, dim=0)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(training_params["device"])
                dones_tensor = torch.tensor(dones, dtype=torch.float32).to(training_params["device"])
                critic_loss, actor_loss, avg_q0 = model.update(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)  # 批量bs反向传播
                if training_params["snn"]["is_snn"]:
                    functional.reset_net(model.actor)
                print(f"[ac][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss}  avg_q0: {avg_q0}  avg_reward: {torch.mean(rewards_tensor).cpu()}")
                all_actor_loss.append(actor_loss)
                all_critic_loss.append(critic_loss)
                all_q0.append(avg_q0)
                all_avg_reward.append(torch.mean(rewards_tensor).cpu())

                rtp.plot_training_metrics(
                    all_actor_loss,
                    all_critic_loss,
                    all_q0,
                    all_avg_reward
                )

                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

            state = next_state
            count += 1

        if len(states) != 0:
            states_tensor = torch.cat(states, dim=0)
            next_states_tensor = torch.cat(next_states, dim=0)
            actions_tensor = torch.cat(actions, dim=0)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(training_params["device"])
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(training_params["device"])
            critic_loss, actor_loss, avg_q0 = model.update(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)
            print(f"[ac][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss} avg_q0: {avg_q0}  avg_reward: {torch.mean(rewards_tensor).cpu()}")
            all_actor_loss.append(actor_loss)
            all_critic_loss.append(critic_loss)
            all_q0.append(avg_q0)
            all_avg_reward.append(torch.mean(rewards_tensor).cpu())

            rtp.plot_training_metrics(
                all_actor_loss,
                all_critic_loss,
                all_q0,
                all_avg_reward
            )

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
        all_avg_reward = []

        episode_start_time = time.time()
        x = env.reset().to(training_params["device"])

        # bs = 1
        state = model.init_sequence()
        state_input = (x, state)
        done = False
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        count = 1
        while not done:
            model.actor.reset_memory(1)
            output, state = model(state_input)  # 单bs推理
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)

            # 与环境交互
            action = torch.argmax(output)
            x, reward, done = env.step(action)
            next_state = (x.to(training_params["device"]), state)

            # 存储过渡信息
            states.append(state_input)
            actions.append(output)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(1 if done else 0)

            if not count % training_params["update_batch"]:
                model.actor.reset_memory(len(states))

                states_tensor = concatenate_sublists(states, training_params["num_heads"])
                next_states_tensor = concatenate_sublists(next_states, training_params["num_heads"])
                actions_tensor = torch.cat(actions, dim=0)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(training_params["device"])
                dones_tensor = torch.tensor(dones, dtype=torch.float32).to(training_params["device"])
                critic_loss, actor_loss, avg_q0 = model.update(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)  # 批量bs反向传播
                if training_params["snn"]["is_snn"]:
                    functional.reset_net(model.actor)
                print(f"[ac][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss}  avg_q0: {avg_q0}  avg_reward: {torch.mean(rewards_tensor).cpu()}")
                all_actor_loss.append(actor_loss)
                all_critic_loss.append(critic_loss)
                all_q0.append(avg_q0)
                all_avg_reward.append(torch.mean(rewards_tensor).cpu())

                rtp.plot_training_metrics(
                    all_actor_loss,
                    all_critic_loss,
                    all_q0,
                    all_avg_reward
                )

                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

            state_input = next_state
            count += 1

        if len(states) != 0:
            model.actor.reset_memory(len(states))

            states_tensor = concatenate_sublists(states, training_params["num_heads"])
            next_states_tensor = concatenate_sublists(next_states, training_params["num_heads"])
            actions_tensor = torch.cat(actions, dim=0)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(training_params["device"])
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(training_params["device"])
            critic_loss, actor_loss, avg_q0 = model.update(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)
            print(f"[ac][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss} avg_q0: {avg_q0}  avg_reward: {torch.mean(rewards_tensor).cpu()}")
            all_actor_loss.append(actor_loss)
            all_critic_loss.append(critic_loss)
            all_q0.append(avg_q0)
            all_avg_reward.append(torch.mean(rewards_tensor).cpu())

            rtp.plot_training_metrics(
                all_actor_loss,
                all_critic_loss,
                all_q0,
                all_avg_reward
            )

        # 打印回合总结信息
        episode_time = time.time() - episode_start_time
        print(
            f"[Env {step} / {len(envs) - 1}] "
            f"[Episode {episode}/{each_env_episodes - 1}] Completed! "
            f"Time: {episode_time:.2f}s\n"
        )


def train_sincnet(model, envs, each_env_episodes):
    # 每个环境运行 max_episodes 个回合
    for episode in range(each_env_episodes):
        all_actor_loss = []
        all_critic_loss = []
        all_q0 = []
        all_avg_reward = []

        episode_start_time = time.time()
        x = env.reset().to(training_params["device"])

        # bs = 1
        state_input = x
        done = False
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []

        count = 1
        while not done:
            output, _ = model(state_input)  # 单bs推理
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)

            # 与环境交互
            action = torch.argmax(output)
            x, reward, done = env.step(action)
            next_state = x.to(training_params["device"])

            # 存储过渡信息
            states.append(state_input)
            actions.append(output)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(1 if done else 0)

            if not count % training_params["update_batch"]:
                states_tensor = torch.cat(states, dim=0)
                next_states_tensor = torch.cat(next_states, dim=0)
                actions_tensor = torch.cat(actions, dim=0)
                rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(training_params["device"])
                dones_tensor = torch.tensor(dones, dtype=torch.float32).to(training_params["device"])
                critic_loss, actor_loss, avg_q0 = model.update(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)  # 批量bs反向传播
                if training_params["snn"]["is_snn"]:
                    functional.reset_net(model.actor)
                print(f"[ac][env {step} / {len(envs) - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss}  avg_q0: {avg_q0}  avg_reward: {torch.mean(rewards_tensor).cpu()}")
                all_actor_loss.append(actor_loss)
                all_critic_loss.append(critic_loss)
                all_q0.append(avg_q0)
                all_avg_reward.append(torch.mean(rewards_tensor).cpu())

                rtp.plot_training_metrics(
                    all_actor_loss,
                    all_critic_loss,
                    all_q0,
                    all_avg_reward
                )

                states = []
                actions = []
                rewards = []
                next_states = []
                dones = []

            state_input = next_state
            count += 1

        if len(states) != 0:
            states_tensor = torch.cat(states, dim=0)
            next_states_tensor = torch.cat(next_states, dim=0)
            actions_tensor = torch.cat(actions, dim=0)
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(training_params["device"])
            dones_tensor = torch.tensor(dones, dtype=torch.float32).to(training_params["device"])
            critic_loss, actor_loss, avg_q0 = model.update(states_tensor, actions_tensor, rewards_tensor, next_states_tensor, dones_tensor)
            if training_params["snn"]["is_snn"]:
                functional.reset_net(model.actor)
            print(f"[ac][env {step} / {len(envs) - 1}][Episode {episode}/{max_episodes - 1}][step count({count})] critic_loss: {critic_loss}  actor_loss: {actor_loss} avg_q0: {avg_q0}  avg_reward: {torch.mean(rewards_tensor).cpu()}")
            all_actor_loss.append(actor_loss)
            all_critic_loss.append(critic_loss)
            all_q0.append(avg_q0)
            all_avg_reward.append(torch.mean(rewards_tensor).cpu())

            rtp.plot_training_metrics(
                all_actor_loss,
                all_critic_loss,
                all_q0,
                all_avg_reward
            )

        # 打印回合总结信息
        episode_time = time.time() - episode_start_time
        print(
            f"[Env {step} / {len(envs) - 1}] "
            f"[Episode {episode}/{max_episodes - 1}] Completed! "
            f"Time: {episode_time:.2f}s\n"
        )


if __name__ == '__main__':
    sr = 32000
    frame_duration = 0.25
    max_episodes = [30, 20, 20, 10, 10]

    # 超参数设置（新增参数保存功能）
    training_params = {
        "train": "ac",
        "work_dir": "train-run-ntm-ac",
        "pretrain_pth_path": r"G:\ms\snn\pretrain-ntm\model_checkpoint_epoch21.pth",
        "task": "vad",
        "shared_net": False,
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
        "update_batch": 200,
        "actor_lr": 0.000001,
        "critic_lr": 0.001,
        "gamma": 0.98,
        "ppo": {
            "K_epochs": 4,
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

    model = ActorCritic(
        training_params
    )

    # 加载预训练模型
    if training_params["pretrain_pth_path"] != "":
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
            r"G:\ms\snn\data\train\label_025",
            r"G:\ms\snn\data\train\vad_025",
            frame_duration=training_params["frame_duration"],
            sr=training_params["sr"],
            task=training_params["task"]
        ),
        VADEnvironment(
            r"G:\ms\snn\data\train\course_study\wav2\data",
            r"G:\ms\snn\data\train\label_025",
            r"G:\ms\snn\data\train\vad_025",
            frame_duration=training_params["frame_duration"],
            sr=training_params["sr"],
            task=training_params["task"]
        ),
        VADEnvironment(
            r"G:\ms\snn\data\train\course_study\wav3\data",
            r"G:\ms\snn\data\train\label_025",
            r"G:\ms\snn\data\train\vad_025",
            frame_duration=training_params["frame_duration"],
            sr=training_params["sr"],
            task=training_params["task"]
        ),
        VADEnvironment(
            r"G:\ms\snn\data\train\course_study\wav4\data",
            r"G:\ms\snn\data\train\label_025",
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