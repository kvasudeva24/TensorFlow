import tensorflow as tf

#how to create a tensor object
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
float = tf.Variable(3.567, tf.float64)

#a tensor of no lists is a scalar, and n lists is nth degree tesnor
degree1_tensor = tf.Variable(["test"], tf.string)
degree2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

#.rank() will give you the degree/rank of the tensor
tf.rank(degree1_tensor)
tf.rank(degree2_tensor)

#.shape returns the number of elements that exist in each dimension (applied after the object so no () )
# returns [number of lists, number of items in every list]
degree1_tensor.shape
degree2_tensor.shape

#reshape resizes the tensor so long as all the dimensions multiply to the same thing
tensor1 = tf.ones([1,2,3]) #one exterior list, two interior lists, and 3 elements inside each list
tensor2 = tf.reshape(tensor1, [2,3,1]) #two exterior lists, three interior lists, and 1 element inside each list
tensor3 = tf.reshape(tensor2, [3, -1]) #-1 tells the tensor to calculate the size of the dimension in that place
                                       #resizes tensor to [3,2]



#
# Different types of tensors:
#   Varibable
#   Constant
#   Placeholder
#   SparseTensor

# besides Variable, all of these tensors are immutable

#

#to evaluate a tensor(get its value) we must run a session

#with tf.Session() as sess:
  #tensor.eval() with tensor being the name of the actual object


