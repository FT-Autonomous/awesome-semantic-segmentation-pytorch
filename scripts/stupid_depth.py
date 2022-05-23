from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

image_size = 100

points, labels = make_blobs(centers=6, n_samples=1000, cluster_std=1.0, n_features=2, center_box=(0, 100))
points = np.int32(np.round(points))
points[points > 99] = 99
points[points < 00] = 00
#plt.scatter(points[..., 0], points[..., 1])
#plt.show()

def generate_depth_map(real_points, real_labels):
    pre = np.ix_(np.arange(image_size), np.arange(image_size))
    indices = np.empty((image_size, image_size, 2), dtype=np.int32)
    indices[..., 0] = pre[0]
    indices[..., 1] = pre[1]

    def create_gradient(shape, r=(10,0)):
        gradient = np.empty(shape)
        for i, row in enumerate(gradient):
            row[:] = r[0] + (i / gradient.shape[0]) * (r[1] - r[0])
        return gradient

    base = create_gradient((100, 100), (10, 0))
    result = base.copy()

    for i in range(labels.max()):
        blob = points[labels == i]
        low_y = blob[:, 0].max()
        whole_blob_depth = base[low_y].mean()
        result[blob[:,0], blob[:,1]] = whole_blob_depth + 0.1

    return result

def generate_segmentation_mask(real_points):
    model = DBSCAN(eps=3.0)
    pred = model.fit_predict(points)
    mask = np.zeros((100,100))
    mask[points] = 1
    return mask

seg = generate_segmentation_mask(points)
seg_non_background = np.where(seg != -1)
seg_points = np.empty((len(seg_non_background[0]), 2), dtype=np.int32)
nonzero = seg_points != 0
seg_points[:, 0] = seg_non_background[0]
seg_points[:, 1] = seg_non_background[1]

depth = generate_depth_map(points, labels)

figure, axis = plt.subplots(1, 1, subplot_kw={'projection':'3d'})
breakpoint()
axis.scatter(seg_points[:, 0], seg_points[:, 1], depth[seg_points[:, 0], seg_points[:, 1]], s=np.ones(len(seg_points))/10)
plt.show()
