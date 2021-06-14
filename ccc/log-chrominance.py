import numpy as np
import matplotlib.pyplot as plt

def rgb2uvy(rgb) -> np.ndarray:
    rgb_array = np.array(rgb)
    assert rgb_array.shape[1] == 3, "Wrong input array shape"

    I_u = np.log(rgb_array[:, 1] / rgb_array[:, 0])
    I_v = np.log(rgb_array[:, 1] / rgb_array[:, 2])
    I_y = np.linalg.norm(rgb_array, axis=1)
    I_uv = np.array([I_u, I_v, I_y]).T
    return I_uv

class Histogram:
    def __init__(self, epsilon, u_range=None, v_range=None):
        assert epsilon > 0
        self.epsilon = epsilon
        self.u_range = u_range
        self.v_range = v_range
    
    def get_hist(self, uvy_array, is_normalised=True):
        uvy_array = np.array(uvy_array)
        assert uvy_array.shape[1] == 3, "Wrong input array shape"
        uvy_array = uvy_array / self.epsilon
        uvy_array[:,:2] = np.round(uvy_array[:,:2])
        num_ticks_u = int(round((self.u_range[1] - self.u_range[0]) / self.epsilon)) + 1
        num_ticks_v = int(round((self.v_range[1] - self.v_range[0]) / self.epsilon)) + 1

        u_coords = np.linspace(*self.u_range, num_ticks_u)
        v_coords = np.linspace(*self.v_range, num_ticks_v)

        u_shift = int(round(self.u_range[0] / self.epsilon))
        v_shift = int(round(self.v_range[0] / self.epsilon))

        hist = np.zeros((num_ticks_u, num_ticks_v))

        for point in uvy_array:
            # print(point[0], u_shift, point[0] - u_shift)
            hist[int(point[0]) - u_shift, int(point[1]) - v_shift] += point[2]
        
        if is_normalised:
            hist = np.sqrt(hist / hist.sum())
        return hist, u_coords, v_coords
    def get_colorhist(self, uvy_array, rgb_array, is_normalised=True):
        uvy_array = np.array(uvy_array)
        assert uvy_array.shape[1] == 3, "Wrong input array shape"
        uvy_array = uvy_array / self.epsilon
        uvy_array[:,:2] = np.round(uvy_array[:,:2])
        num_ticks_u = int(round((self.u_range[1] - self.u_range[0]) / self.epsilon)) + 1
        num_ticks_v = int(round((self.v_range[1] - self.v_range[0]) / self.epsilon)) + 1

        u_coords = np.linspace(*self.u_range, num_ticks_u)
        v_coords = np.linspace(*self.v_range, num_ticks_v)

        u_shift = int(round(self.u_range[0] / self.epsilon))
        v_shift = int(round(self.v_range[0] / self.epsilon))

        hist = np.zeros((num_ticks_u, num_ticks_v))
        colorhist = np.zeros((num_ticks_u, num_ticks_v, 3))
        numbers = np.zeros((num_ticks_u, num_ticks_v))
        
        for point, rgb in zip(uvy_array, rgb_array):
            # print(point[0], u_shift, point[0] - u_shift)
            hist[int(point[0]) - u_shift, int(point[1]) - v_shift] += point[2]
            colorhist[int(point[0]) - u_shift, int(point[1]) - v_shift] += rgb.astype(np.float64)
            numbers[int(point[0]) - u_shift, int(point[1]) - v_shift] += 1
        
        numbers[numbers == 0] = 1
        colorhist = colorhist / numbers[:, :, None]
        
        colorhist /= 255
        
        
        if is_normalised:
            hist = np.sqrt(hist / hist.sum())
            
        colorhist *= hist[:, :, None]
        colorhist /= np.amax(colorhist)
        
        return colorhist




def test2():
    x = np.random.randn(300)
    y = np.random.randn(300)
    e = np.ones((300))
    points = np.array([x, y, e]).T
    print(points.shape)
    # plt.scatter(x, y)
    # plt.show()

    hist = Histogram(0.2, (-5, 5), (-5, 5))
    h, u_range, v_range = hist.get_hist(points, is_normalised=False)
    plt.imshow(h)

    plt.show()

def test():
    print(rgb2uvy([[0, 4, 3], [1, 1, 1], [0, 1, 1]]))



if __name__ == '__main__':
    test2()




