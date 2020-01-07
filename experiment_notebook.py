from utils import Timer, AverageMeter
from interface import *
from predictors import Predictor, PredictorLSTM, PredictorLSTMTorque, PredictorOfTorque
from environment import EnvironmentState
from mappings import StateMapStatic
from torch.utils.data import DataLoader
from actors import Actor, ActorP, ActorAgregate
from rewards import *
from critics import Critic
from pymunk import Vec2d
from torch import Tensor
import torch
import time
from copy import deepcopy
import matplotlib.pyplot as plt


# -------------------- grid search on skip_step and recurrent_step, for result please refer to diary.txt -------------
def train(skip_step, recurrent_step):
    print('training start now: ')
    i = 0
    env = EnvironmentState()
    state_set = StateDataset(env, skip_step=skip_step,size=6000000)
    state_set_loader = DataLoader(state_set, batch_size=1000)

    predictor = Predictor(state_set.output_size, 2)
    predictor.stepper.recurrent_step = recurrent_step

    avm = AverageMeter()

    with Timer():
        for s in state_set_loader:
            i += 1
            d = s['s1'][:, 12:14] - s['s0'][:, 12:14] # with torques
            result, loss = predictor.optimize(s['s0'], d)
            avm.log(loss.data)
            if i % 10 ==0:
                print(f'epoch {i} (skip_step {skip_step}, recurrent_step {recurrent_step}) : train_loss {avm.value:.4f}')
                print(f'epoch {i}: trn loss {avm.value:.4f} {avm.std:.4f}, rmse {avm.value ** 0.5:.4f}, '
                      f'pred to stat error ratio: {torch.mean(torch.abs(result / d)):.4f} '
                      f'max_d {torch.max(d)} {torch.min(d)}, max_r {torch.max(result)} {torch.min(result)}')
    print('training finished.')


for skip_step, recurrent_step in [(0, 4)]:
    train(skip_step, recurrent_step)

# -------------------------------- torque movement display -----------------------------------------------
env = EnvironmentState()
env.torque_scaled_set(1, 1)
for _ in range(500):
    env.plot()
    env.step()

# -------------------------------- predictor training scripts ----------------------------------
print('training start now: ')
i = -1
env = EnvironmentState()
state_set = StateDataset(env, skip_step=8, size=7000000, random_torque=True, remove_torque=False)
state_set_loader = DataLoader(state_set, batch_size=1024)

predictor = Predictor(state_set.output_size, 2)
predictor.stepper.recurrent_step = 0

avm = AverageMeter()
with Timer():
    for s in state_set_loader:
        i += 1
        # d = s['s1'][:, 11:13] - s['s0'][:, 11:13] # without torques
        d = s['s1'][:, 12:14] - s['s0'][:, 12:14] # with torques
        r, loss = predictor.optimize(s['s0'], d)
        avm.log(loss)
        if i % 10 ==0:
            print(f'epoch {i}: trn loss {avm.value:.4f} {avm.std:.4f}, rmse {avm.value**0.5:.4f}, '
                  f'pred to stat error ratio: {torch.mean(torch.abs(r / d)):.4f} '
                  f'max_d {torch.max(d):.2f} {torch.min(d):.2f}, max_r {torch.max(r):.2f} {torch.min(r):.2f}')

print('training finished.')
# -------------------------------- torque-predictor training scripts ----------------------------------
env = EnvironmentState()
state_set = StateDatasetOfTorque(env, skip_step=24, size=8000000, random_torque=True, remove_torque=False)
state_loader = DataLoader(state_set, batch_size=1024)
predictor = PredictorOfTorque(17 - 2 + 2, 2) # states with torques removed, then added endpoint movement
criterion = torch.nn.MSELoss()
avm = AverageMeter()
i = -1
with Timer():
    for s in state_loader:
        s0, s1, t0 = s.values()
        t0_pred = predictor(s0, s1)
        loss = criterion(t0, t0_pred)
        with predictor.optimize_c():
            loss.backward()
        avm.log(loss.data)
        i += 1
        print(f'instant loss: {loss.data:.4f}', end='\r')
        if i % 20 == 0:
            print(f'epoch {i}, {state_set.skip_step} skip step, avg loss: {avm.value:.4f} {avm.std:.4f}')
# ----------------------------- predictor_lstm training scripts ------------------------------------------------
env = EnvironmentState()
state_set_lstm = StateDatasetLSTM(env, skip_step=1, seq_leng=50)
state_set_loader = DataLoader(state_set_lstm, batch_size=500)
predictor_lstm = PredictorLSTM(state_set_lstm.output_size, 2, seq_len=state_set_lstm.seq_leng, base_rnn='gru')
criterion = torch.nn.MSELoss()

with Timer() as t:
    i = -1
    avm = AverageMeter()
    for s in state_set_loader:
        i += 1
        x, y = s['s0'], s['result']
        pred = predictor_lstm(x)
        y = get_arm1_end_points(y) - get_arm1_end_points(x)[:, None]
        loss = criterion(pred, y)
        with predictor_lstm.optimize_c():
            loss.backward()
        avm.log(loss.data)
        if i % 10 == 0: print(f'epoch {i}: trn loss {avm.value:.4f} {avm.std:.4f}')
    print(f'epoch {i}: trn loss {avm.value:.4f} {avm.std:.4f}')
# ----------------------------- rnn-torque-predictor training scripts ------------------------------------------------
env = EnvironmentState()
lstm_set = StateDatasetLSTMTorque(env, skip_step=1, seq_len=50)
lstm_loader = DataLoader(lstm_set, batch_size=300)
predictor = PredictorLSTMTorque(lstm_set.output_size, 2, expand_ratio=4, base_rnn='gru')
criterion = torch.nn.MSELoss()
avm = AverageMeter()
with Timer():
    i = -1
    for s in lstm_loader:
        i += 1
        s0, torques, diffs = s.values()
        pred_diffs = predictor(s0, torques)
        loss = criterion(diffs, pred_diffs)
        with predictor.optimize_c():
            loss.backward()
        avm.log(loss.data)
        if i % 10 == 0: print(f'epoch {i}: trn loss {avm.value:.4f} {avm.std:.4f}')
    print(f'epoch {i}: trn loss {avm.value:.4f} {avm.std:.4f}')
# -------------------------- show prediction in environment plot --------------------------------
env = EnvironmentState()
state_set = StateDataset(env, skip_step=0, size=4000000, random_torque=True)
predictor = Predictor(state_set.output_size, 2).load('predictor_ratio3skip24recur0.pt')
transform_f = make_transfrom_f()
for i in range(1000):
    env.step(random_torque=True)
    if i % 50 == 0:
        env.green_dot([env.arm1.get_extend_coor() + predictor(transform_f(env.get_current_state())).data.numpy()])
    env.plot()
# -------------------------- show torque prediction in environment plot --------------------------------------
env = EnvironmentState()
torque_predictor = PredictorOfTorque(17, 2)
torque_predictor.load('pred_of_torq_r2s16r0.pt')
transform_f = make_transfrom_f()
scale = 50
target = [14., 6]
for i in range(1, 1000):
    if i % 1 == 0:
        # target_delta = torch.tensor((np.random.rand(2).astype(np.float32) - 0.5) * scale)
        s = transform_f(env.get_current_state())
        with torch.no_grad():
            # torques = torque_predictor(s, replace_arm1_endpoint(s, get_arm1_end_points(s) + target_delta))
            torques = torque_predictor(s, replace_arm1_endpoint(s, torch.tensor(target)))
        env.torque_scaled_set(torques[0], torques[1])
        # env.green_dot([env.arm1.get_extend_coor() + target_delta.numpy()])
        env.green_dot([target])
        # print(f'instant torques {torques.numpy()}, incremental target {target_delta.numpy()} scaled by {scale} from -0.5 to 0.5', end='\r')
        print(f'instant torques {torques.numpy()}', end='\r')
    if i % 100 == 0:
        env.randomize()
        target = (np.random.rand(2).astype(np.float32) - 0.5) * 80
    env.step()
    env.plot()

# -------------------------- show lstm-prediction in environment plot --------------------------------
env = EnvironmentState()
state_set = StateDataset(env, skip_step=0, size=4000000, random_torque=True)
predictor_lstm = PredictorLSTM(state_set.output_size, 2)
predictor_lstm.load('predictor_lstm_ratio2skip5recur10.pt')
transform_f = make_transfrom_f()
for i in range(2000):
    env.step()
    if i % 80 == 0:
        env.torque_random_set()
        env.step()
        env.torque_random_set()
        with torch.no_grad():
            pred = predictor_lstm(transform_f(env.get_current_state()).unsqueeze(dim=0))
        pred = torch.tensor(env.arm1.get_extend_coor()) + pred
        env.green_dot(pred.squeeze())
    env.plot()
# -------------------------- show rnn-torque-prediction in environment plot --------------------------------
env = EnvironmentState()
state_set = StateDatasetLSTMTorque(env, skip_step=0, size=4000000)
predictor_lstm = PredictorLSTMTorque(state_set.output_size, 2, expand_ratio=4)
predictor_lstm.load('predictor_torque_totaldiff_gru_r2s1r15.pt')
transform_f = make_transfrom_f()
look_ahead_step = 50
for i in range(2000):
    env.step()
    if i % look_ahead_step == 0:
        env.torque_random_set()
        env.step()
        env.torque_random_set()
        env.step()
        env.torque_random_set()
        env.step()
        env.torque_random_set()
        torques_input = torch.cat([get_torques(transform_f(env.get_current_state())).unsqueeze(0),
                                   torch.zeros(look_ahead_step - 1, 2)], 0)
        with torch.no_grad():
            pred = predictor_lstm(transform_f(env.get_current_state()).unsqueeze(0), torques_input.unsqueeze(0))
        pred = torch.tensor(env.arm1.get_extend_coor()) + pred
        env.green_dot(pred.squeeze())
    env.plot()
# -------------------------- show action movement in environment plot --------------------------------
env = EnvironmentState()
state_set = StateDataset(env, skip_step=0, size=4000000, random_torque=True, remove_torque=True)
# setup predictor to return dummy full state
actor = Actor(state_set.output_size_for_actor, 2).load('actor_1.pt')
transform_f = make_transfrom_f()
env.green_dot([[14.0, 6.0]])  # target point
for i in range(1, 3000):
    # env.step(random_torque=True)
    env.step()
    if i % 1 == 0:
        with torch.no_grad():
            actions = actor(transform_f(env.get_current_state()))
        env.torque_scaled_set(actions[0], actions[1])
        print(actions, end='\r')
    if i % 100 == 0:
        env.randomize()
    env.plot()
# -------------------------- show aggregate actor movement in environment plot ----------------------------------
env = EnvironmentState()
torque_pred_names = ['pred_of_torq_r2s4r0.pt', 'pred_of_torq_r2s16r0.pt', 'pred_of_torq_r2s24r0.pt']
torque_pred_list = [PredictorOfTorque(17 - 2 + 2, 2).load(name) for name in torque_pred_names]
actor_ag = ActorAgregate(15, 2, len(torque_pred_names), expand_ratio=3) # no torque state + target point
actor = lambda s: actor_ag(s, replace_arm1_endpoint(s, target_point), torque_pred_list)
predictor = Predictor(17, 2).load('predictor_ratio2skip24recur0.pt')
target_point = torch.tensor([14., 6])
for i in range(1, 3000):
    env.green_dot([target_point])
    env.step()
    if i % 100 == 0:
        env.randomize()
        target_point = torch.tensor(np.random.rand(2).astype(np.float32) - 0.5) * 80
    with torch.no_grad():
        torques, sm = actor(transform_f(env.get_current_state()))
    print(torques, sm, end='\r')
    env.torque_scaled_set(torques[0].data, torques[1].data)
    env.plot()

# -------------------------- show actor/predictor trajectory in environment plot --------------------------------
env = EnvironmentState()
env_c = deepcopy(env)
actor_trajectory = []
predictor_trajectory16 = []
predictor_trajectory24 = []
predictor_trajectory50 = []
for i in range(1, 3000):
    env.step()
    if i % 1 == 0:
        with torch.no_grad():
            actions = actor(transform_f(env.get_current_state()))
            env.torque_scaled_set(actions[0], actions[1])
            actor_trajectory.append(env.arm1.get_extend_coor())
            # predictor_trajectory.append(env.arm1.get_extend_coor() +
            #                             predictor(transform_f(env.get_current_state())).data.numpy())
            predictor_trajectory16.append(env.arm1.get_extend_coor() +
                                        predictor16(transform_f(env.get_current_state())).data.numpy())
            predictor_trajectory24.append(env.arm1.get_extend_coor() +
                                        predictor24(transform_f(env.get_current_state())).data.numpy())
            predictor_trajectory50.append(env.arm1.get_extend_coor() +
                                        predictor50(transform_f(env.get_current_state())).data.numpy())
        print(actions, end='\r')
        env.yellow_dot(actor_trajectory)
        env.blue_dot(predictor_trajectory16)
        env.cyan_dot(predictor_trajectory24)
        env.magenta_dot(predictor_trajectory50)

    if i % 100 == 0:
        # env.randomize()
        env = deepcopy(env_c)
        actor_trajectory = []
        predictor_trajectory16 = []
        predictor_trajectory24 = []
        predictor_trajectory50 = []
        time.sleep(3)
    env.plot()

# --------------------- parametric actor and inner train loop train script (off policy) -------------------------------
env = EnvironmentState()
state_set = StateDataset(env, skip_step=20, size=800000, random_torque=True, remove_torque=True)
state_set_loader = DataLoader(state_set, batch_size=1000)
predictor = Predictor(state_set.output_size, 2)
predictor.load('predictor_ratio2skip50recur0.pt')

# setup predictor to return dummy full state
def p(x: torch.Tensor, a: torch.Tensor, predictor):
    # no action
    a_0 = torch.zeros_like(a) # zero torques
    new_x_0 = replace_torque(x, a_0)
    pred_0 = x.narrow(-1, 12, 2) + predictor(new_x_0)
    result_0 = torch.zeros_like(new_x_0)
    result_0 = torch.cat((result_0.narrow(-1, 0, 12), pred_0, result_0.narrow(-1, 14, 3)), dim=-1)

    # action
    new_x_1 = replace_torque(x, a)
    pred_1 = x.narrow(-1, 12, 2) + predictor(new_x_1)
    result_1 = torch.zeros_like(new_x_1)
    result_1 = torch.cat((result_1.narrow(-1, 0, 12), pred_1, result_1.narrow(-1, 14, 3)), dim=-1)

    return result_0, result_1, pred_1


# critic is just a predictor armed with a reward function
config_array = [15, 30, 15, 2]
actor = ActorP(config_array)
num_loop = 20

with Timer() as t:
    i = -1
    avm = AverageMeter()
    for s in state_set_loader:
        i += 1
        for j in range(num_loop):
            actions = actor(s['s0'])
            pred_0, pred_1, pred_point = p(s['s0'], actions, predictor)
            reward_ = -reward_f1(pred_0, pred_1)
            # reward_ = -reward_dist_reduce_f1(pred_1) + reward_dist_reduce_f1(pred_0)
            with actor.optimize_c():
                reward_.backward()
            avm.log(reward_.data)
        if i % 5 == 0: print(f'epoch {i}| loop {j}: trn loss {avm.value:.4f} {avm.std:.4f}')
    print(f'epoch {i}| loop {j}: trn loss {avm.value:.4f} {avm.std:.4f}')

# ------------------- train actor step wise on policy with forward looking predictor ---------------------------------
env = EnvironmentState()
state_set = StateDataset(env, skip_step=50, size=800000, random_torque=True, remove_torque=True)
state_set_loader = DataLoader(state_set, batch_size=500)
predictor = Predictor(state_set.output_size, 2)
predictor.load('predictor_ratio2skip50recur0.pt')
transform_f = make_transfrom_f()
config_array = [15, 30, 15, 2]
actor = ActorP(config_array)

# setup predictor to return dummy full state

def p(x: Tensor, a: Tensor, predictor):
    # no action
    a_0 = torch.zeros_like(a) # zero torques
    new_x_0 = replace_torque(x, a_0)
    pred_0 = x.narrow(-1, 12, 2) + predictor(new_x_0)
    result_0 = torch.zeros_like(new_x_0)
    result_0 = torch.cat((result_0.narrow(-1, 0, 12), pred_0, result_0.narrow(-1, 14, 3)), dim=-1)

    # action
    new_x_1 = replace_torque(x, a)
    pred_1 = x.narrow(-1, 12, 2) + predictor(new_x_1)
    result_1 = torch.zeros_like(new_x_1)
    result_1 = torch.cat((result_1.narrow(-1, 0, 12), pred_1, result_1.narrow(-1, 14, 3)), dim=-1)

    return result_0, result_1, pred_1

env.blue_dot([[14.0, 6.0]])
trajectory_len = 150
n_trial = 10
num_loop = 10
avm = AverageMeter()

for k in range(100):
    # collection on policy trajectory
    trajectory_batch = []
    for i in range(1, trajectory_len * n_trial+1):
        with torch.no_grad():
            actions = actor(transform_f(env.get_current_state()))
            env.torque_scaled_set(actions[0], actions[1])
        env.step()
        trajectory_batch.append(transform_f(env.get_current_state()))
        if i % trajectory_len == 0:
            env.randomize()
    trajectory_batch = torch.stack(trajectory_batch, dim=0)

    # optimize trajectory at each step a number of times
    for j in range(num_loop):
        actions = actor(trajectory_batch)
        pred_0, pred_1, pred_point = p(trajectory_batch, actions, predictor)
        reward_ = -reward_f1(pred_0, pred_1)
        with actor.optimize_c():
            reward_.backward()
        avm.log(reward_.data)
    print(f'loop {k}: trn loss {avm.value:.4f} {avm.std:.4f}')
# ------------------------ reward/loss loop plot per optimization step ------------------------------------------------
# load predictors and whatnot
predictor_names = ['pr1', 'pr8', 'pr16', 'pr24', 'pr50', 'pr100']
predictor_list = [Predictor(state_set.output_size, 2) for i in range(len(predictor_names))]
pred_paths = ['predictor_ratio2skip1recur0.pt', 'predictor_ratio2skip8recur0.pt', 'predictor_ratio2skip16recur0.pt',
              'predictor_ratio2skip24recur0.pt', 'predictor_ratio2skip50recur0.pt', 'predictor_ratio2skip100recur0.pt']
[pr.load(pr_path) for pr, pr_path in zip(predictor_list, pred_paths)]
predictor_map = dict([(i, pr) for i, pr in zip([1, 8, 16, 24, 50, 100], predictor_list)])
transform_f = make_transfrom_f()
def p(x: Tensor, a: Tensor, predictor):
    # no action
    a_0 = torch.zeros_like(a) # zero torques
    new_x_0 = replace_torque(x, a_0)
    pred_0 = x.narrow(-1, 12, 2) + predictor(new_x_0)
    result_0 = torch.zeros_like(new_x_0)
    result_0 = torch.cat((result_0.narrow(-1, 0, 12), pred_0, result_0.narrow(-1, 14, 3)), dim=-1)

    # action
    new_x_1 = replace_torque(x, a)
    pred_1 = x.narrow(-1, 12, 2) + predictor(new_x_1)
    result_1 = torch.zeros_like(new_x_1)
    result_1 = torch.cat((result_1.narrow(-1, 0, 12), pred_1, result_1.narrow(-1, 14, 3)), dim=-1)

    return result_0, result_1, pred_1
# get an actor
config_array = [15, 30, 15, 2]
actor = ActorP(config_array)
# set env and hyper-parameters
env_cache = deepcopy(env)
trajectory_len = 120
avm = AverageMeter()


def get_trajectory_batch(env, actor, trajectory_len=120, n_trial=5, functional=True):
    if functional:
        env_ = deepcopy(env)
    else: env_ = env
    trajectory_batch = []
    for _ in range(1, trajectory_len * n_trial + 1):
        with torch.no_grad():
            actions = actor(transform_f(env_.get_current_state()))
            env_.torque_scaled_set(actions[0], actions[1])
        env_.step()
        trajectory_batch.append(transform_f(env_.get_current_state()))
        if _ % trajectory_len == 0:
            env_.randomize()
    trajectory_batch = torch.stack(trajectory_batch, dim=0)
    return trajectory_batch


# plot_test() variables needed: env_cache, plt, trajectory_len, actor, predictor_list, predictor_names
def plot_test(trajectory=None):
    plt.cla()
    # get trajectory
    if trajectory is not None:
        test_trajectory = trajectory
    else:
        test_trajectory = get_trajectory_batch(env_cache, actor, trajectory_len=120, n_trial=1, functional=True)
    # get multiple rewards
    reward_list = []
    actions = actor(test_trajectory)
    for pr in predictor_list:
        pred_0, pred_1, pred_point = p(test_trajectory, actions, pr)
        reward_list.append((-reward_dist_reduce_f1(pred_1)).sum(-1))
    # get plot
    plt.xlim(-5, 125)
    plt.ylim(0, 100)
    for r, name in zip(reward_list, predictor_names):
        y_pos = r.data.numpy()
        plt.plot(y_pos, label=name)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), shadow=True, ncol=len(predictor_names))
    plt.title('A rainbow of losses plot against time step')
    plt.text(80, 90, f'prd24 loss: {reward_list[3].sum().data:.1f}')
    plt.show()
    plt.pause(0.01)
    return reward_list


def annealing(range1, range2, step, num_loop=10):
    trajectory_batch = get_trajectory_batch(env_cache, actor, trajectory_len=120, n_trial=1, functional=True)
    # plot_test()

    for j in range(num_loop):
        actions = actor(trajectory_batch)
        _, pred_1, _ = p(trajectory_batch, actions, predictor_map[step])
        r = -reward_dist_reduce_f1(pred_1).sum(-1)
        with actor.optimize_c():
            r[range1:range2].sum().backward()
        r_scalar = r.sum().data
        avm.log(r_scalar.data)
        print(f'{r_scalar.data} {actor.fc0.weight.grad.sum()}', end='\r')
        # get plot
        plot_test()
    print(f'avg loss {avm.value:.4f} {avm.std:.4f} ')


def annealing_select(step): # picking only the upward trending tensor to optimise
    step_r_dict = {1: 0, 8: 1, 16: 2, 24: 3, 50: 4, 100: 5}
    r_list = plot_test()
    r = r_list[step_r_dict[step]]

    r_select = r[1:][(r[1:]-r[:-1]) > 0]
    r_scalar = r_select.sum()
    with actor.optimize_c():
        r_scalar.backward()
    avm.log(r_scalar.data)
    print(f'{r_scalar.data} {actor.fc0.weight.grad.sum()}', end='\r')
    # get plot
    plot_test()
    print(f'avg loss {avm.value:.4f} {avm.std:.4f} ', end='\r')


annealing(0,119,8,1) # from time step 0 to time step 119, using skip8 predictor, train 5 times
annealing_select(24) # only gd the reward tensor (by skip24 predictor) that is higher than the previous time step
# ------------------------------ train a dampened actor --------------------------------------------------------------
predictor_names = ['pr1', 'pr8', 'pr16', 'pr24', 'pr50', 'pr100']
predictor_list = [Predictor(state_set.output_size, 2) for i in range(len(predictor_names))]
pred_paths = ['predictor_ratio2skip1recur0.pt', 'predictor_ratio2skip8recur0.pt', 'predictor_ratio2skip16recur0.pt',
              'predictor_ratio2skip24recur0.pt', 'predictor_ratio2skip50recur0.pt', 'predictor_ratio2skip100recur0.pt']
[pr.load(pr_path) for pr, pr_path in zip(predictor_list, pred_paths)]

transform_f = make_transfrom_f()
config_array = [15, 30, 15, 2]
actor = ActorP(config_array)
actor.set_auto_save_name('actor_ratio2_pr24_14_6.pt').set_auto_save_delay(50).toggle_auto_save()
env_cache = None
trajectory_len = 120
n_trial = 10
num_loop = 10
avm = AverageMeter()
# setup predictor to return dummy full state
def p(x: Tensor, a: Tensor, predictor):
    # no action
    a_0 = torch.zeros_like(a) # zero torques
    new_x_0 = replace_torque(x, a_0)
    pred_0 = x.narrow(-1, 12, 2) + predictor(new_x_0)
    result_0 = torch.zeros_like(new_x_0)
    result_0 = torch.cat((result_0.narrow(-1, 0, 12), pred_0, result_0.narrow(-1, 14, 3)), dim=-1)

    # action
    new_x_1 = replace_torque(x, a)
    pred_1 = x.narrow(-1, 12, 2) + predictor(new_x_1)
    result_1 = torch.zeros_like(new_x_1)
    result_1 = torch.cat((result_1.narrow(-1, 0, 12), pred_1, result_1.narrow(-1, 14, 3)), dim=-1)

    return result_0, result_1, pred_1


def get_trajectory_batch(env, actor, trajectory_len=120, n_trial=5, functional=True):
    if functional:
        env_ = deepcopy(env)
    else: env_ = env
    trajectory_batch = []
    for _ in range(1, trajectory_len * n_trial + 1):
        with torch.no_grad():
            actions = actor(transform_f(env_.get_current_state()))
            env_.torque_scaled_set(actions[0], actions[1])
        env_.step()
        trajectory_batch.append(transform_f(env_.get_current_state()))
        if _ % trajectory_len == 0:
            env_.randomize()
    trajectory_batch = torch.stack(trajectory_batch, dim=0)
    return trajectory_batch


for j in range(250):
    env.randomize()
    trajectory_batch = get_trajectory_batch(env, actor, trajectory_len=trajectory_len, n_trial=n_trial)
    for i in range(num_loop):
        actions = actor(trajectory_batch)
        _, pred_1, _ = p(trajectory_batch, actions, predictor_list[3])  # use predictor24
        r = -reward_dist_reduce_f1(pred_1).sum()
        with actor.optimize_c():
            r.sum().backward()
        avm.log(r.data)
        print(f'{r.data} {actor.fc0.weight.grad.sum()}', end='\r')
    print(f'loop {j}, avg loss {avm.value:.4f} {avm.std:.4f} ')

    actions = actor(trajectory_batch)
    _, pred_1, _ = p(trajectory_batch, actions, predictor_list[0]) # use predictor1
    r = -reward_dist_reduce_f1(pred_1).sum()
    actor.save_by_score(r.data)
# ----------------------- target tunnelling for policy reuse --------------------------------------------------------
config_array = [15, 30, 15, 2]
actor = ActorP(config_array).load('actor_multi_14_6.pt')
transform_f = make_transfrom_f()
transformer = StateMapStatic(actor.in_features, actor.in_features)
new_actor = lambda state: actor(transformer(state))
# policy reuse
old_target = Vec2d(14,6)
new_target = old_target.rotated(3.3)*0.40
new_reward_dist_reduce_f1 = lambda pred1: reward_dist_reduce_f1(pred1, target=torch.tensor(new_target))
# setup predictor to return dummy full state
def p(x: Tensor, a: Tensor, predictor):
    # no action
    a_0 = torch.zeros_like(a) # zero torques
    new_x_0 = replace_torque(x, a_0)
    pred_0 = x.narrow(-1, 12, 2) + predictor(new_x_0)
    result_0 = torch.zeros_like(new_x_0)
    result_0 = torch.cat((result_0.narrow(-1, 0, 12), pred_0, result_0.narrow(-1, 14, 3)), dim=-1)

    # action
    new_x_1 = replace_torque(x, a)
    pred_1 = x.narrow(-1, 12, 2) + predictor(new_x_1)
    result_1 = torch.zeros_like(new_x_1)
    result_1 = torch.cat((result_1.narrow(-1, 0, 12), pred_1, result_1.narrow(-1, 14, 3)), dim=-1)

    return result_0, result_1, pred_1

def get_trajectory_batch(env, actor, trajectory_len=120, n_trial=5, functional=True):
    if functional:
        env_ = deepcopy(env)
    else: env_ = env
    trajectory_batch = []
    for _ in range(1, trajectory_len * n_trial + 1):
        with torch.no_grad():
            actions = actor(transform_f(env_.get_current_state()))
            env_.torque_scaled_set(actions[0], actions[1])
        env_.step()
        trajectory_batch.append(transform_f(env_.get_current_state()))
        if _ % trajectory_len == 0:
            env_.randomize()
    trajectory_batch = torch.stack(trajectory_batch, dim=0)
    return trajectory_batch

# let's maximise trajectory reward
trajectory_len = 120
n_trial = 10
avm = AverageMeter()

for j in range(1000):
    env.randomize()
    trajectory_batch = get_trajectory_batch(env, new_actor, trajectory_len=trajectory_len, n_trial=n_trial)
    actions = new_actor(trajectory_batch)
    _, pred_1, _ = p(trajectory_batch, actions, predictor_list[24])
    r = -new_reward_dist_reduce_f1(pred_1).sum()
    actor.zero_grad()
    with transformer.optimize_c():
        r.backward()
    avm.log(r.data)
    if j % 5 == 0:
        print(f'loop {j}, avg loss {avm.value:.4f} {avm.std:.4f}, ')
# ------------------------------------