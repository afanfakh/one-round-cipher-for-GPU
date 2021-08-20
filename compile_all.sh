# compliation of the proposed cipher

/usr/local/cuda/bin/nvcc -ccbin g++  -I/usr/local/cuda/samples/common/inc  -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=compute_60 -w oneround2.cu -o oneround2


# compliation of the speck 2 blocks
/usr/local/cuda/bin/nvcc -ccbin g++  -I/usr/local/cuda/samples/common/inc  -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=compute_60 -w speckctr_2B.cu -o speckctr_2B


# compliation of the speck 4 blocks
/usr/local/cuda/bin/nvcc -ccbin g++  -I/usr/local/cuda/samples/common/inc  -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=compute_60 -w speckctr_4B.cu -o speckctr_4B

# compliation of the simon 2 blocks
/usr/local/cuda/bin/nvcc -ccbin g++  -I/usr/local/cuda/samples/common/inc  -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=compute_60 -w simonctr_2B.cu -o simonctr_2B


# compliation of the simon 4 blocks
/usr/local/cuda/bin/nvcc -ccbin g++  -I/usr/local/cuda/samples/common/inc  -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=compute_60 -w simonctr_4B.cu -o simonctr_4B



# compliation of the AES
/usr/local/cuda/bin/nvcc -ccbin g++  -I../../common/inc  -m64  -O3    -gencode arch=compute_70,code=sm_70 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=compute_60 -o aesTest2.o -c aesTest2.cu

/usr/local/cuda/bin/nvcc -ccbin g++ -I../../common/inc   -m64  -O3    -gencode arch=compute_70,code=sm_70 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=compute_60 -o aes aesTest2.o

# compliation of ESSENCE
/usr/local/cuda/bin/nvcc -ccbin g++  -I/usr/local/cuda/samples/common/inc  -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=compute_60 -w prng_cipher10.cu -o prng_cipher10
