from solver import *


if __name__ == '__main__':
    # map = {
    #     'U': 'ggywyrwrg',
    #     'L': 'ygowryryb',
    #     'F': 'gwrggoyyg',
    #     'R': 'wbogoorbb',
    #     'B': 'broybbwob'
    # }
    # c = CuberEnv(map=map)
    c = CuberEnv(rand=5)
    solved = c.solved()
    model = './models/best_5_003.dat'
    example = lambda x: './examples/%d.jpg' % x
    state = c.render(plot=True, save=example(0))

    net_dict = torch.load(model)
    net = DQN(state.shape, c.nA)
    net.load_state_dict(net_dict)
    net = net.cpu()
    net.eval()
    try_count = 0

    done = False
    step = 0
    while not done:
        recent_a = [-1, -1, -1, -1]
        state = c.render()
        state_v = torch.tensor([state]).cpu()
        q_vals_v = net(state_v)
        _, act_v = torch.max(q_vals_v, dim=1)
        a = int(act_v.item())
        _, a2, a3, a4 = recent_a
        recent_a = [a2, a3, a4, a]
        if len(set(recent_a)) == 1 or a4 == inv(a):
            break
        _, _, done, solved = c.step(a)
        step += 1
        c.render(plot=True, save=example(step))
        if solved:
            print(c.action_his)

    # while not solved:
    #     try_count += 1
    #     print(try_count)
    #     recent_a = [-1, -1, -1, -1]
    #     while True:
    #         state = c.render()
    #         state_v = torch.tensor([state]).cpu()
    #         q_vals_v = net(state_v)
    #         _, act_v = torch.max(q_vals_v, dim=1)
    #         a = int(act_v.item())
    #         _, a2, a3, a4 = recent_a
    #         recent_a = [a2, a3, a4, a]
    #         if len(set(recent_a)) == 1:
    #             while recent_a[3] == a:
    #                 recent_a[3] = c.sample()
    #             a = recent_a[3]
    #         _, _, done, solved = c.step(a)
    #         if solved:
    #             print(c.action_his)
    #             break
    #         if done:
    #             c.reset()
    #             break
