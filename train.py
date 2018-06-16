import prog
import tensorflow as tf 

#Test different models in this file

if __name__ == "__main__":
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D())