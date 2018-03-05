import tensorflow as tf
import DE

class Target:
    def __init__(self):
        self.input_shape = 10
        self.output_shape = 1
    def __call__(self, x):
        return tf.reshape(tf.reduce_sum(x * x), [1,])
'''
class A:
    N = 3
'''

if __name__ == '__main__':
    opt = DE.DE(Target(), 0, 1)
    #a = DE.DE._random_unequal(A)
    #print(a)
    #print(tf.Session().run(a))
    #print(tf.Session().run(a))
    #print(tf.Session().run(a))

    print(opt.minimize())
