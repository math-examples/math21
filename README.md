# A simple, small, math library in the 21th century.
# Everything has its mathematical form. We use this library to help us to find this form. Feel free to use and to learn.

# todo lists
* see fuse_conv_batchnorm, v4cl, is_nan_or_inf_kernel, backward node 30,
* test m21rawtensor, changed int to NumN
* bug: opencl not working on some linux system.
* bug: math21 must be static when using opencl and called by sky in windows.
* rename cuda_config.h to gpu_config.h
* get some stuff from OCL_FUNC in cv/core/src/ocl.cpp

# Change log
* add unit in clvector, not test
* Emtpy tensor can have shape, added in version 3.0.0
* tf v1 weakness
  If you need to create a variable with an initial value dependent on another
  variable, use the other variable's `initialized_value()`. This ensures that
  variables are initialized in the right order.

* may add: tf.transpose, tf.reshape, tf.split, tf.matmul
* change fminf to fmin in math21_vector_clip_cpu
* add string support to tensor, but setSize failed.
* add condition if to math21_vector_clip_wrapper in FnFullyConnected::backward.
* todo: change arrangement of vector, matrix, tensor in memory to matlab style.
* todo: serialize all states of ml net so it can restart at the backup as if it never stoped.
* todo: change size_t to NumN in math21_vector_serialize etc.
* Undefined behavior, destination object type 'struct' is not TriviallyCopy
* change c code to c++.
* set mbs to child when having child.
* change math21_c_tim(0) to math21_c_seed_get() in curandSetPseudoRandomGeneratorSeed(gen[i], math21_c_tim(0));
* remove const in math21_ml_function_dropout_parse, may need removing others.
* don't change finput in forward to constant.
* continue removing random behavior
* remove overwrite warning.
* remove random to debug.
* 2014.07.14
float representation, uniform representation of numbers.
* 2014.05.01
serialize, deserialize.
* 2013 mlp, cnn
* 2012 optimization
* 2011 matrix design using template.
* 2010 matrix design
