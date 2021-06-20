import numpy as np
import scipy
from scipy.optimize import minimize
from scipy.signal import convolve2d
import tqdm
from dataset import Dataset, LogChromHist, CubePlusPlus


def fun_p(img, filter):
    convolved = convolve2d(img, filter, mode='same', boundary='fill', fillvalue=0)
    return np.exp(convolved) / np.exp(convolved).sum()

def fun_l(u_range, v_range):
    l_array = np.zeros((u_range.shape[0], v_range.shape[0], 3))
    for i in range(u_range.shape[0]):
        for j in range(v_range.shape[0]):
            l_array[i, j] = [np.exp(-u_range[i]), 1, np.exp(-v_range[i])]
            l_array[i, j] /= np.linalg.norm(l_array[i, j])
    return l_array

def fun_c(gt_uv, l_array, u_range, v_range):
    l_star = np.array([np.exp(-gt_uv[0]), 1, np.exp(-gt_uv[1])]) 
    l_array = l_array.reshape((-1, 3))
    c_array = np.dot(l_array, l_star / np.linalg.norm(l_star)).reshape((len(u_range), len(v_range)))
    c_array = np.arccos(c_array)
    return c_array

def loss_fn(x, train_imgs, data_gt, l_array, lambda_param, u_range, v_range):
    kernel = x.reshape((5, 5))
    loss = 0
    for img, gt in zip(train_imgs, data_gt):
        p = fun_p(img, kernel)
        c = fun_c(gt, l_array, u_range, v_range)
        loss += (p * c).sum()
    loss += lambda_param * (kernel ** 2).sum()
    return loss

def train(lambda_param):
    lg = LogChromHist(0.0125, (-64 * 0.025, 64 * 0.025), (-64 * 0.025, 64 * 0.025))
    print("Loading dataset...")
    dataset = CubePlusPlus('/media/kvsoshin/Transcend/Work/cube++/SimpleCube++', 'train', 5, lg)
    train_imgs, data_gt = dataset.get_data()
    print(train_imgs.shape, data_gt.shape)
    # exit()
    print("Composing l_array...")
    l_array = fun_l(lg.histogram.u_coords, lg.histogram.v_coords)
    print("Minimizing...")
    res = minimize(loss_fn, np.zeros((25,)), args=(train_imgs, data_gt, l_array, lambda_param, lg.histogram.u_coords, lg.histogram.v_coords), method='L-BFGS-B', tol=1e-6)
    print("Finished!")
    print(res.x)


def rosen(x):
    # print("kek")
    return (1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2

def test():
    # res = minimize(rosen, np.array([0, 0]), method='L-BFGS-B', tol=1e-6)
    # print(res)
    train(0.1)

if __name__ == '__main__':
    test()