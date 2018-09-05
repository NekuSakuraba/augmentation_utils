import numpy as np

from scipy import linalg
from skimage.transform import rotate
# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html

class RandomRotate:
    
    def __init__(self, angle):
        self.angles = np.arange(angle)
        self.angles = np.concatenate([self.angles, 360-self.angles])
    
    def __call__(self, img):
        return rotate(img, np.random.choice(self.angles))

		
class ZCA:
    # https://stackoverflow.com/questions/31528800/how-to-implement-zca-whitening-python
	# https://github.com/smlaine2/tempens/blob/master/zca_bn.py
	
    def __init__(self, epsilon=1e-5):
        self.epsilon = epsilon
    
    def fit(self, X):
        shape = X.shape
        
        X = X.copy().reshape((shape[0], np.prod(shape[1:])))
        mu = np.mean(X, axis=0)
        X -= mu
        
        sigma = np.dot(X.T,X)/shape[0]
        U, S, V = linalg.svd(sigma)
        
        tmp1 = np.dot(U, np.diag(1./np.sqrt(S+self.epsilon)))
        tmp2 = np.dot(U, np.diag(np.sqrt(S+self.epsilon)))
        
        self.ZCA_mat = np.dot(tmp1, U.T)
        self.inv_ZCA_mat =np.dot(tmp2, U.T)
        
        self.mean = mu
    
    def apply(self, X):
        shape = X.shape
        return np.dot(X.reshape((shape[0], np.prod(shape[1:]))) - self.mean, self.ZCA_mat).reshape(shape)

	def invert(self, X):
		shape = X.shape
		return np.dot(X.reshape((shape[0], np.prod(shape[1:]))), self.inv_ZCA_mat) + self.mean
