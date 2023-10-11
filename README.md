# cudnn-snn

nvcc snnmnist.cu readubyte.cpp nn.cu cusnn.cu -I. -lcudnn -lcublas -lcurand

./a.out

requirements:

cudnn7.6
