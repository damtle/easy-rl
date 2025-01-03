

def train(cfg, env, agent):
    print("start train!")
    rewards = []
    steps = []
    for i_ep in range(cfg['train_eps']):
        ep_reward = 0
        ep_step = 0
        state = env.reset()
        for _ in range(cfg['ep_max_steps']):
            ep_step += 1
            action = agent.sample_action(state)
            next_state, reward, done, _, info = env.step(action)
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
            agent.update()
            ep_reward += reward
            if done:
                break

        if (i_ep + 1) % cfg['target_update'] == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
        steps.append(ep_step)
        rewards.append(ep_reward)
        if (i_ep + 1) % 10 == 0:
            print(f"Epoch: {i_ep + 1}/{cfg['train_eps']}, reward: {ep_reward:.2f}, Epislon: {agent.epsilon:.3f}")
    print("finish train!")
    env.close()
    return {'rewards': rewards}