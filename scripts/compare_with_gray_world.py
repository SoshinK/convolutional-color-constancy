import numpy as np
import tqdm
from ccc.dataset import Dataset, AugList, LogChromHist, Resize, CubePlusPlus
from ccc.log_chrominance import illum_uvy2illum_rgb
from ccc.pyramid import GaussianPyramid


def cmp_ccc_and_gray_world(gp, test_dataset, test_dataset_raw_photos):
    assert len(test_dataset) == len(test_dataset_raw_photos)
    err_gw = []
    err_barron = []
    for i in tqdm.tqdm(range(len(test_dataset))):
        hist, gt_rgb = test_dataset[i]
        img, gt_rgb_raw  = test_dataset_raw_photos[i]
        assert np.allclose(gt_rgb, gt_rgb_raw)

        filtered = gp.apply(hist)
        l_uv = np.array(np.unravel_index(np.argmax(filtered), filtered.shape)) * test_dataset.augmentations.histogram.epsilon
        l_rgb = illum_uvy2illum_rgb([l_uv])
        l_rgb_as_in_gt = l_rgb / np.sum(l_rgb)
        l_rgb_gray_world = np.mean(img, axis=(0, 1))
        l_rgb_gray_world = l_rgb_gray_world / np.sum(l_rgb_gray_world)
        print(f"Gray world: {np.round( l_rgb_gray_world, 3)}; Barron: {np.round( l_rgb_as_in_gt, 3)}, ground-truth: {np.round(gt_rgb, 3)}")
        err_gw.append(np.linalg.norm(l_rgb_gray_world - gt_rgb))
        err_barron.append(np.linalg.norm(l_rgb_as_in_gt - gt_rgb))
    return np.array(err_gw), np.array(err_barron)


def main():
    epsilon = 0.0125
    lg = LogChromHist(epsilon, (-64 * epsilon, 64 * epsilon), (-64 * epsilon, 64 * epsilon))
    train_dataset = CubePlusPlus('/media/kvsoshin/Transcend/Work/cube++/SimpleCube++', 'train', 300, lg)
    gp = GaussianPyramid(4, filter_size=5)
    gp.fit(train_dataset)
    gp.save_filters('./scripts')

    test_dataset = CubePlusPlus('/media/kvsoshin/Transcend/Work/cube++/SimpleCube++', 'train', 300, lg, return_gt_rgb=True)
    test_dataset_raw_photos = CubePlusPlus('/media/kvsoshin/Transcend/Work/cube++/SimpleCube++', 'train', 300, None, return_gt_rgb=True)

    err_gw, err_barron = cmp_ccc_and_gray_world(gp, test_dataset, test_dataset_raw_photos)

    print("Gray-world mean error: ", err_gw.mean())
    print("Barron algorithm mean error: ", err_barron.mean())


if __name__ == '__main__':
    main()