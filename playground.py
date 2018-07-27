import tensorflow as tf

# TODO: want jupyter notebook

""" Hello, TensorFlow!

dyld: warning, LC_RPATH $ORIGIN/../../_solib_darwin_x86_64/_U_S_Stensorflow_Spython_C_Upywrap_Utensorflow_Uinternal.so___Utensorflow in /Users/bryan/Library/Python/2.7/lib/python/site-packages/tensorflow/python/_pywrap_tensorflow_internal.so being ignored in restricted program because it is a relative path
2018-06-21 14:47:19.363652: I tensorflow/core/platform/cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA
"""
hello = tf.constant("Hello, TensorFlow!")
sess = tf.Session()
print(sess.run(hello))

""" Adder
"""
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
added = tf.add(node1, node2)
print(sess.run([node1, node2, added]))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 2, 3], b: [4, 5, 6]}))

""" Tensor shapes
"""
shapes = [
    tf.constant(
        [
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ],
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9]
            ]
        ]
    ),
    tf.constant(
        [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ]
    ),
    tf.constant(
        [1, 2, 3]
    )
]
for t in shapes:
    print(t.shape)