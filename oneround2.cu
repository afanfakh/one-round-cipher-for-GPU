///usr/local/cuda/bin/nvcc -ccbin g++  -I/usr/local/cuda/samples/common/inc  -O3 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_60,code=compute_60 -w -Wno-deprecated-gpu-targets oneround.cu -o oneround

//NEW
///usr/local/cuda/bin/nvcc -ccbin g++  -I/usr/local/cuda/samples/common/inc  -O3  -m64  -O3    -gencode arch=compute_35,code=sm_35 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52   oneround.cu -o oneround


#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
//#include "splitmix64.h"
//#include <gmp.h>
#include <cuda.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>

//#define  size 1024
//#define nbth  (((long)402653184)/1024)       // ((long)(32768*32768/2/8))




 unsigned char DK[256];
 unsigned char Sbox1[256];
 unsigned char Sbox2[256];


typedef unsigned char   uchar;


uint64_t *v2;


static const uint64_t s_plain_text_stream[32] = {
0x20, 0x6d, 0x61, 0x64, 0x65, 0x20, 0x69, 0x74, 0x20, 0x65, 0x71, 0x75, 0x69, 0x76, 0x61, 0x6c,
0x20, 0x6d, 0x61, 0x64, 0x65, 0x20, 0x69, 0x74, 0x20, 0x65, 0x71, 0x75, 0x69, 0x76, 0x61, 0x6c,

};


__device__ __host__ static inline uint64_t splitmix64_stateless(uint64_t index) {
  uint64_t z = (index + UINT64_C(0x9E3779B97F4A7C15));
  z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
  z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
  return z ^ (z >> 31);
}


void rc4key(uchar *key, uchar *sc, int size_DK) {

  for(int i=0;i<256;i++) {
    sc[i]=i;
  }

  uchar j0 = 0;
  for(int i0=0; i0<256; i0++) {
    j0 = (j0 + sc[i0] + key[i0%size_DK])&0xFF;
    uchar tmp = sc[i0];
    sc[i0] = sc[j0];
    sc[j0] = tmp;
  }

}


__device__ uint64_t inline xorshift64( const uint64_t state)
{
  uint64_t x = state;
  x^= x << 13;
  x^= x >> 7;
  x^= x << 17;
  return x;
}


void forgpu_seed(uint64_t seed, int size, long nbth) { 

  

   v2=(uint64_t*)malloc(sizeof(uint64_t)*nbth*size);

  for(int j=0;j<nbth;j++)
     for(int i=0;i<size;i++) {
        //v1[j][i] = splitmix64_stateless(seed+i+j*nbth);
        v2[j*size+i]= splitmix64_stateless(seed+i+j*nbth);
    }
  seed+=size*nbth;

  for(int i=0;i<256;i++) {
    DK[i]=splitmix64_stateless(seed+i);
  }
  seed+=256;
  rc4key(&DK[0],  Sbox1, 48);
  rc4key(&DK[48], Sbox2, 48);
}


#define ROL64(x,r) (((x)<<(r)) | (x>>(64-(r))))
#define ROR64(x,r) (((x)>>(r)) | ((x)<<(64-(r))))

/*
static inline uint64_t rotl(const uint64_t x, int k) {
  return (x << k) | (x >> (64 - k));
}

static inline uint64_t rotr(const uint64_t x, int k) {
  return (x >> k) | (x << (64 - k));
}
*/

/*
__device__ void  apply_sub_and_prng(ulong , uchar *sbox1, uchar *sbox2) {

   uint64_t v2;
   uchar* tt;
    v2= xorshift64(v[j+i*nbth]);
     tt=(uchar*)&v2;
     tt[0]=sbox1[tt[0]];
     tt[1]=sbox1[tt[1]];
     tt[2]=sbox1[tt[2]];
     tt[3]=sbox1[tt[3]];
     tt[4]=sbox1[tt[4]];
     tt[5]=sbox1[tt[5]];
     tt[6]=sbox1[tt[6]];
     tt[7]=sbox1[tt[7]];
     out[j+i*nbth]=v2^in[j+i*nbth];

    
}
*/


__device__ inline void  cipher(const uint64_t *v,const uint64_t *in,uint64_t *out,int ind,uchar* sbox1,uchar *sbox2, uint64_t *v2) {

   uchar* tt;

  *v2= xorshift64(*v2);                                                                                                     
     tt=(uchar*)v2;                                                                                                         
     tt[0]=sbox1[tt[0]];                                                                                                     
     tt[1]=sbox2[tt[1]];                                                                                                     
     tt[2]=sbox1[tt[2]];                                                                                                     
     tt[3]=sbox2[tt[3]];                                                                                                     
     tt[4]=sbox1[tt[4]];                                                                                                     
     tt[5]=sbox2[tt[5]];                                                                                                     
     tt[6]=sbox1[tt[6]];                                                                                                     
     tt[7]=sbox2[tt[7]];                                                                                                     
     out[ind]=*v2^in[ind];
  
  
}




__global__ void encrypt(uint64_t *  __restrict__ v, const uint64_t* __restrict__ in, uint64_t *out, const uchar*__restrict__ box1, const uchar* __restrict__ box2 ,int size, long nbth) {
  __shared__ uchar sbox1[256];
  __shared__ uchar sbox2[256];

  int  j= blockIdx.x*blockDim.x+threadIdx.x;

  if(threadIdx.x<256){
    sbox1[threadIdx.x]=box1[threadIdx.x];
    sbox2[threadIdx.x]=box2[threadIdx.x];	
  }
  __syncthreads();
   
  if (j<nbth)
  {
  
   uchar* tt;
   uint64_t v2=v[j];

  #pragma unroll 64
   for(int i=0;i<size;i++) 
    { 

      //      cipher(&v2,in,out,j+i*nbth,sbox1,sbox2,&v2);
      v2=splitmix64_stateless(v2);
      //     v2= xorshift64(v2);
     tt=(uchar*)&v2;
     tt[0]=sbox1[tt[0]];
     tt[1]=sbox2[tt[1]];
     tt[2]=sbox1[tt[2]];
     tt[3]=sbox2[tt[3]];
     tt[4]=sbox1[tt[4]];
     tt[5]=sbox2[tt[5]];
     tt[6]=sbox1[tt[6]];
     tt[7]=sbox2[tt[7]];
     out[j+i*nbth]=v2^in[j+i*nbth];       
     }
   v[j]=v2;
  }
}












int main(int argc, char **argv) 

{
      int  blockSize, size;
      long nbth; 

      if  ( (strncmp(argv[1],"blsize",6)==0)  && (strncmp(argv[3],"size",4)==0)  &&       (strncmp(argv[5],"nbth",4)==0) )   
     {  
         blockSize = atoi((argv[2]));  
         size = atoi((argv[4]));
         nbth = atoi((argv[6]));   
         nbth=(nbth/size) ; 

     // int  blockSize =256*2;   //never less than 256
      int  gridSize = nbth/blockSize;
      if (gridSize<1)
         gridSize=1;
      // gridSize=512;
       
       
     // printf("nb blocks %d nb thread per block %d\n",gridSize,blockSize);
    //CPU variables 
      uint8_t *plain_text_stream= NULL;
       uint8_t *crypted_text_stream= NULL;
       uint8_t *decrypted_text_stream= NULL;
       uint64_t *v_stream;
      
      
   //allocate CPU variables   
      plain_text_stream = ( uint8_t*)malloc(8*nbth*size);
      crypted_text_stream = ( uint8_t*)malloc(8*nbth*size);
      decrypted_text_stream = ( uint8_t*)malloc(8*nbth*size);

      
      for (int i=0; i<nbth*size; i++)
       plain_text_stream[i]=1;
        //memcpy(plain_text_stream + (i * size), s_plain_text_stream, sizeof(s_plain_text_stream));
     
      
   //GPU variable
       uint8_t *dev_plain_text_stream;
       uint8_t *dev_plain_text_stream2;
       uint8_t *dev_crypted_text_stream;
       uint8_t *dev_decrypted_text_stream;
       uint64_t *dev_v_stream;
       uchar *dev_Sbox1;
       uchar *dev_Sbox2;

      
    //allocate GPU variables
      cudaMalloc((void **)&dev_plain_text_stream, 8*nbth*size*sizeof( uint8_t));   
      cudaMalloc((void **)&dev_crypted_text_stream, 8*nbth*size*sizeof( uint8_t));     
      cudaMalloc((void **)&dev_plain_text_stream2, 8*nbth*size*sizeof( uint8_t));
                                                                  
      cudaMalloc((void **)&dev_decrypted_text_stream, 8*nbth*size*sizeof(uint8_t));             
      cudaMalloc((void **)&dev_v_stream, nbth*size *sizeof(uint64_t)); 
      cudaMalloc((void **)&dev_Sbox1, 256 *sizeof(uchar)); 
      cudaMalloc((void **)&dev_Sbox2, 256 *sizeof(uchar)); 
      
    //call seed function to generate random vector V2   
      forgpu_seed(10,size,nbth);    
      

   // for (int i=0; i<nbth*size; i++) 
   //  printf("v2[%d]= %ld \n", i,v2[i]);
    
   //copy variables to device memory
     cudaMemcpy(dev_v_stream, v2, nbth*sizeof(uint64_t), cudaMemcpyHostToDevice);
     cudaMemcpy(dev_plain_text_stream, plain_text_stream, 8*nbth*size *sizeof(uint8_t), cudaMemcpyHostToDevice);
     cudaMemcpy(dev_Sbox1,Sbox1 ,256 *sizeof(uchar), cudaMemcpyHostToDevice);
     cudaMemcpy(dev_Sbox2,Sbox2 ,256 *sizeof(uchar), cudaMemcpyHostToDevice);
      encrypt<<<gridSize, blockSize,512>>>((uint64_t*)dev_v_stream, (uint64_t*)dev_plain_text_stream, (uint64_t*)dev_crypted_text_stream, dev_Sbox1, dev_Sbox2,size,nbth);
   //printf("okkkkkkkkkkk\n") ;
      
      /*     encrypt<<<gridSize, blockSize,512>>>((uint64_t*)dev_v_stream, (uint64_t*)dev_plain_text_stream, (uint64_t*)dev_crypted_text_stream, dev_Sbox1, dev_Sbox2);
        cudaMemcpy(dev_v_stream, v2, nbth*sizeof(uint64_t), cudaMemcpyHostToDevice);
      cudaEvent_t start, stop;
                float time;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
                cudaEventRecord(start, 0);

                
     encrypt<<<gridSize, blockSize,512>>>((uint64_t*)dev_v_stream, (uint64_t*)dev_plain_text_stream, (uint64_t*)dev_crypted_text_stream, dev_Sbox1, dev_Sbox2);
		//              cudaThreadSynchronize();                                                                              
                                cudaDeviceSynchronize();
                cudaEventRecord(stop, 0);
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&time, start, stop);
		printf("GPU processing time : %f (ms) \n", time);
      */

		
          StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    for(int i=0;i<100;i++) {

   // call keranl function
     
     Sbox1[i]++;
     rc4key(Sbox2, Sbox1, 128);
     rc4key(Sbox1, Sbox2,256);
     
     cudaMemcpy(dev_Sbox1,Sbox1 ,256 *sizeof(uchar), cudaMemcpyHostToDevice);
     cudaMemcpy(dev_Sbox2,Sbox2 ,256 *sizeof(uchar), cudaMemcpyHostToDevice);
     
     
     sdkStartTimer(&timer);
     encrypt<<<gridSize, blockSize,512>>>((uint64_t*)dev_v_stream, (uint64_t*)dev_plain_text_stream, (uint64_t*)dev_crypted_text_stream, dev_Sbox1, dev_Sbox2,size,nbth);
     //     cudaThreadSynchronize();
   
   
      cudaDeviceSynchronize();
      
   
     sdkStopTimer(&timer);
             cudaMemcpy(dev_v_stream, v2, nbth*sizeof(uint64_t), cudaMemcpyHostToDevice);
    }


float reduceTime_GPU = sdkGetAverageTimerValue(&timer)/1000;
      

//float reduceTime_GPU=		time;		
 printf("Size message %ld bytes Throughput on GPU %f Gbps  Time on GPU %f s gridsize %d  thread per block %d size %d \n",(long)nbth*(long)size*8,((double)64*nbth*size)/reduceTime_GPU/1e9,reduceTime_GPU,gridSize, blockSize, size);

 // printf("Throughput on GPU %f Gbps\n ", ((double)64*nbth*size)/reduceTime_GPU/1e9);     
 //copy cypher text from device to host
   
  //cudaMemcpy(v_stream, dev_v_stream, nbth*size*sizeof(uint64_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(crypted_text_stream, dev_crypted_text_stream, 8*nbth*size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
  
  
  //copy cypher text from host to device
 
 
 
 
  cudaMemcpy(dev_crypted_text_stream, crypted_text_stream, 8*nbth*size*sizeof(uint8_t),cudaMemcpyHostToDevice);
  cudaMemcpy(dev_v_stream, v2, nbth*sizeof(uint64_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_Sbox1,Sbox1 ,256 *sizeof(uchar), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_Sbox2,Sbox2 ,256 *sizeof(uchar), cudaMemcpyHostToDevice);
  
    encrypt<<<gridSize, blockSize,512>>>((uint64_t*)dev_v_stream, (uint64_t*)dev_crypted_text_stream, (uint64_t*)dev_plain_text_stream2, dev_Sbox1, dev_Sbox2,size,nbth);
  
  

     
 cudaMemcpy(decrypted_text_stream, dev_plain_text_stream2, 8*nbth*size*sizeof(uint8_t), cudaMemcpyDeviceToHost);
   int flag=0;
    for (int i = 0; i < nbth*size; i++)  {

      if (plain_text_stream[i] != decrypted_text_stream[i]) 
          flag=1;
          

    }
//for (int i = nbth*size-10; i <nbth*size; i++) 
     // printf("%u  %u \n",plain_text_stream[i],crypted_text_stream[i]);
   
    if (flag==0)
          printf("\n");     // printf("success encryption-decryption test\n");
    else if (flag==1)
     printf("Not success encryption-decryption test\n");
         
  
   
        cudaFree(dev_plain_text_stream);
        cudaFree(dev_crypted_text_stream);
        cudaFree(dev_plain_text_stream2);
        cudaFree(dev_decrypted_text_stream);
        cudaFree(dev_v_stream); 
        cudaFree(dev_Sbox1);
        cudaFree(dev_Sbox2);
        
        
    free(plain_text_stream);
    free(crypted_text_stream);
    free(decrypted_text_stream);

  }
  
  else 
  printf("Wrong arguments\n");
  
    
     //printf(" finish \n"); 
        
   return 0;
}
