import numpy as np
import tensorflow as tf


class DE:
    def __init__(self, target, a, b, N=50, Fmin=0.5, Fmax=1.0, CR=0.9, max_iter=100, seed=0):
        self.F_distr = tf.random_uniform(shape=[], minval=Fmin, maxval=Fmax, dtype=tf.float32, seed=seed)
        self.F_holder = tf.get_variable('F_holder', shape=[], dtype=tf.float32)
        self.F_holder = tf.assign(self.F_holder, self.F_distr)

        self.target = target
        M = self.target.input_shape
        assert self.target.output_shape == 1, 'have to return a single real value'

        self.M = M
        self.N = N
        
        self.a = a
        self.b = b

        self.CR = tf.cast(tf.less(tf.random_uniform(shape=(N, M), minval=0, maxval=1, seed=seed), CR), tf.float32)

        self.population = tf.get_variable(name='population', shape=[N, M], initializer=tf.random_uniform_initializer(a, b, seed=seed))
        self.donors = self._get_mutation_op()
        self.candidates = self._get_reconstruction_op()


    def _get_mutation_op(self):
        b = self.__random_unequal()
        x = tf.reshape(tf.gather_nd(self.population, tf.reshape(b, [3 * self.N, 1])), [3, self.N, self.M])
        x = x[0] + self.F_holder * (x[2] - x[1])
        return x
    
    def _get_reconstruction_op(self):
        return self.population + self.CR * (self.donors - self.population)
    
    def __batch_apply(self, x):
        splits = tf.split(x, x.shape[0], 0)
        f_splits = list(map(self.target, splits))
        f_splits = tf.stack(f_splits)
        return f_splits

    def _get_selection_op(self):
        mask = tf.cast(tf.less(self.__batch_apply(self.population), self.__batch_apply(self.candidates)), tf.float32)
        return tf.assign(self.population, self.population + mask * (self.candidates - self.population))

    def __random_unequal(self, num=3):
        a = tf.random_uniform(shape=(num, self.N), minval=0, maxval=self.N - 1, dtype=tf.int64)
        b = tf.get_variable('random_holder', shape=(num, self.N), dtype=tf.int64)
        b = tf.assign(b, a)
        b = b + tf.cast(tf.less(np.int64(np.repeat(np.arange(-1, self.N - 1).reshape((1,-1)), num, axis=0)), b), tf.int64)
        return b
    
    def minimize(self):
        return self._get_selection_op()


