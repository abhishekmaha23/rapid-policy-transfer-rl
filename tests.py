import torch
import matplotlib.pyplot as plt
from attempt_2.v4.utils import *


def test_target():
    generator_input_sample_length = 20
    a = torch.zeros((generator_input_sample_length), requires_grad=False)
    a[generator_input_sample_length//2:] += 1
    # print(a)
    b = a.flip(0).view(-1, 1)
    # print(b)
    a = a.view(-1, 1)
    one_hot_target = torch.cat((a,b), 1)
    print(one_hot_target)
    noise_vector_length = 7
    b = torch.randn((generator_input_sample_length, noise_vector_length), requires_grad=False)
    print(a)
    # print(b)
    c = torch.cat((b, a), 1)
    # print(c)


def test_plotting():
    plt.figure(figsize=(10, 6))
    x = [i for i in range(10)]
    y = [i*i for i in range(10)]
    # gauss_kernel = Gaussian1DKernel(3)
    print(x)
    print(y)
    # plt.plot(gauss_kernel)
    plt.plot(x, y, label='Harvest')
    # plt.grid(b=True, which='minor', linestyle='--', linewidth='0.5')
    # plt.grid(b=True, which='major', linestyle='-', linewidth='0.9')
    plt.legend(fontsize=17)
    plt.xticks(size=17)
    plt.yticks(size=17)
    # plt.grid()
    plt.xlabel("weeks", fontsize=17)
    plt.ylabel("Harvest", fontsize=17)
    plt.title("sigma = 3", fontsize=17)
    # plt.title('Gaussian sigma = 3')
    # plt.show()

    # plt.figure(figsize=(16, 9))
    # plt.plot(x_val, weight, label="Harvest")
    # plt.plot(x_val, smoothed_data_gauss, label="Gaussian Smoothed sigma = 3")
    # plt.show()
    # plt.savefig(r'Test.pdf')

    import time
    import os
    run_id = time.time()
    a = [i*i for i in range(20)]
    b = [i for i in range(20)]
    cwd = os.getcwd()
    log_path = os.path.join(os.getcwd(), str(run_id))
    os.mkdir(log_path)
    plot_fig(b, a,
             std=0.1, title="Test Plot for Dank Tots", draw_grid=True, xlabel="walks",
             ylabel="stalks", add_legend=True, label="reward", display_fig=True, save_fig=True,
             save_name=os.path.join(log_path, '-A2C_Actor_Perf.pdf'))


def test_compare_weights_of_models(model_1, model_2):
    for p1, p2 in zip(model_1.parameters(), model_2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            # print('Models are unequal in weights')
            pass
    # print('Models are equal')


def test_compare_models_on_states(model_1, model_2, states_var):
    actions_1 = model_1(states_var)
    actions_2 = model_2(states_var)
    # print('Equality of models on the states - ', torch.equal(actions_1, actions_2))