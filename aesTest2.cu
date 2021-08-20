//utilisation ./aes key myfile out 32 32
//head -c 1024 </dev/urandom >myfile

//./aes key 512x512x3 out 500 512
//./aes key 16384x16384x3 out 16384 512



//Bitsliced High-Performance AES-ECB on GPUs


#include "aes_kernel2.cu"

#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <sys/time.h>

__host__ static inline void cudaCheck(const cudaError_t ret, const char *const message)
{
   if(ret != cudaSuccess)
   {
      fprintf(stderr,"%s\n",message);
      fprintf(stderr,"%s\n",cudaGetErrorString(cudaGetLastError() ) );
      exit(1);
   }
}

__host__ static inline void checkError(const void *const retcode, const char *const message)
{
   if(retcode == NULL)
   {
      fprintf(stderr,"%s\n",message);
      perror("");
      exit(1);
   }
}

__host__ static inline void checkError(const int retcode, const char *const message)
{
   if(retcode != 0)
   {
      fprintf(stderr,"%s\n",message);
      perror("");
      exit(1);
   }
}

__host__ static inline void checkError2(const long retcode, const char *const message)
{
   if(retcode == -1)
   {
      fprintf(stderr,"%s\n",message);
      perror("");
      exit(1);
   }
}

int main(int argc, char *argv[])
{
   if(argc < 6)
   {
      fprintf(stderr, "Need more arguments\n");
      fprintf(stderr, "key_file input_file output_file blockCount threadsPerBlock\n");
      exit(1);
   }
   const int STREAM_COUNT = 2;
   cudaStream_t cudaStreams[STREAM_COUNT];
   unsigned char *key, *input[STREAM_COUNT], *output[STREAM_COUNT], *temp_input;
   vec2 * bs_subkey;
   vec2 *d_bs_subkey, *d_input[STREAM_COUNT], *d_output[STREAM_COUNT];
   timespec start1, end1, start2, end2, start3, end3, start4, end4, start5, end5, start6, end6, start7, end7;
   timeval start2_1, end2_1;
   double time1, time2, time2_1, time3, time4, time5, time6, time7;
   int device;
   cudaDeviceProp deviceProp;
   FILE *fp;
   long pos;
   unsigned int fileSize;
   int i, j;
   const int blockCount = atoi(argv[4]), threadsPerBlock = atoi(argv[5]);
   key = new unsigned char[16];
   bs_subkey = new vec2[88];

   printf("blockCount: %d, threadsPerBlock: %d\n", blockCount, threadsPerBlock);
   printf("Stream count: %d\n", STREAM_COUNT);
   fp = fopen(argv[1], "rb");
   checkError(fp, "fopen error");
   if(fread(key, 1, 16, fp) != 16)
   {
      fprintf(stderr, "fread error\n");
      perror("");
      exit(1);
   }
   checkError(fclose(fp), "fclose error");

   fp = fopen(argv[2], "rb");
   checkError(fp, "fopen error");
   checkError(fseek(fp, 0, SEEK_END), "fseek error");
   pos = ftell(fp);
   checkError2(pos, "ftell error");
   checkError(fseek(fp, 0, SEEK_SET), "fseek error");
   if(pos % 16 != 0 || pos < 0)
   {
      fprintf(stderr, "Size must be divisible by 16\n");
      exit(1);
   }
   fileSize = static_cast<unsigned int>(pos);
   for(i = 0; i < STREAM_COUNT; i++)
   {
      cudaCheck(cudaMallocHost(&input[i], fileSize * sizeof(unsigned char), cudaHostAllocPortable | cudaHostAllocWriteCombined), "cudaMallocHost Error");
      cudaCheck(cudaMallocHost(&output[i], fileSize * sizeof(unsigned char), cudaHostAllocPortable), "cudaMallocHost Error");
   }
   temp_input = new unsigned char[fileSize];
   if(fread(temp_input, 1, fileSize, fp) != fileSize)
   {
      fprintf(stderr, "fread error\n");
      perror("");
      exit(1);
   }
   checkError(fclose(fp), "fclose error");
   checkError(clock_gettime(CLOCK_MONOTONIC, &start1), "clock_gettime error"); //init
   checkError(clock_gettime(CLOCK_MONOTONIC, &start1), "clock_gettime error");
   for(i = 0; i < STREAM_COUNT; i++)
   {
      memcpy(input[i], temp_input, fileSize);
   }
   checkError(clock_gettime(CLOCK_MONOTONIC, &end1), "clock_gettime error");
   delete [] temp_input;
   temp_input = NULL;

   AES_keygen_bs(key, bs_subkey);

   cudaCheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1), "cudaThreadSetCacheConfig Error");

   cudaCheck(cudaGetDevice(&device), "cudaGetDevice Error");
   cudaCheck(cudaGetDeviceProperties(&deviceProp, device), "cudaGetDeviceProperties Error");
   printf("Using device %d - %s, async count: %d\n", device, deviceProp.name, deviceProp.asyncEngineCount);

   cudaCheck(cudaMalloc(&d_bs_subkey, 88 * sizeof(vec2) ), "cudaMalloc Error");
   for(i = 0; i < STREAM_COUNT; i++)
   {
      cudaCheck(cudaMalloc(&d_input[i], fileSize), "cudaMalloc Error");
      cudaCheck(cudaMalloc(&d_output[i], fileSize), "cudaMalloc Error");
   }

   cudaCheck(cudaMemcpy(d_bs_subkey, bs_subkey, 88 * sizeof(vec2), cudaMemcpyHostToDevice), "cudaMemcpy Error");
   for(i = 0; i < STREAM_COUNT; i++)
   {
      cudaCheck(cudaMemcpy(d_input[i], input[0], fileSize, cudaMemcpyHostToDevice), "cudaMemcpy Error1");
      cudaCheck(cudaMemcpy(d_output[i], input[0], fileSize, cudaMemcpyHostToDevice), "cudaMemcpy Error2"); //overwrite
   }

	 cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize 1 Error");
   AES_kernel<<<blockCount, threadsPerBlock>>> (d_output[0], d_input[0], fileSize / 8, d_bs_subkey); //init
	 cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize 2 Error");
   cudaCheck(cudaMemcpy(d_output[0], input[0], fileSize, cudaMemcpyHostToDevice), "cudaMemcpy Error3"); //overwrite
	 cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize 3 Error");
   gettimeofday(&start2_1, NULL); //rep
   checkError(clock_gettime(CLOCK_MONOTONIC, &start2), "clock_gettime error"); //rep

   gettimeofday(&start2_1, NULL);
   checkError(clock_gettime(CLOCK_MONOTONIC, &start2), "clock_gettime error");
   for(i = 0; i < 1; i++)
   {
		 AES_kernel<<<blockCount, threadsPerBlock>>> (d_output[0], d_input[0], fileSize / 8, d_bs_subkey);
   }
   cudaCheck(cudaDeviceSynchronize(), "cudaDeviceSynchronize 2 Error");
   checkError(clock_gettime(CLOCK_MONOTONIC, &end2), "clock_gettime error");
   gettimeofday(&end2_1, NULL);
	 
   cudaCheck(cudaMemcpy(d_output[0], input[0], fileSize, cudaMemcpyHostToDevice), "cudaMemcpy Error"); //overwrite

   for(i = 0; i < STREAM_COUNT; i++)
   {
      cudaCheck(cudaStreamCreate(&cudaStreams[i]), "cudaStreamCreate Error");
   }

   //init
   for(i = 0; i < STREAM_COUNT; i++)
   {
      cudaCheck(cudaMemcpyAsync(d_input[i], input[i], fileSize, cudaMemcpyHostToDevice, cudaStreams[i]), "cudaMemcpyAsync Error");
      AES_kernel<<<blockCount, threadsPerBlock, 0, cudaStreams[i]>>> (d_output[i], d_input[i], fileSize / 8, d_bs_subkey);
      cudaCheck(cudaMemcpyAsync(output[i], d_output[i], fileSize, cudaMemcpyDeviceToHost, cudaStreams[i]), "cudaMemcpyAsync Error");
   }
   for(i = 0; i < STREAM_COUNT; i++)
   {
      cudaCheck(cudaStreamSynchronize(cudaStreams[i]), "cudaStreamSynchronize Error");
   }

   checkError(clock_gettime(CLOCK_MONOTONIC, &start3), "clock_gettime error");
   for(j = 0; j < 1; j++)
   {
      for(i = 0; i < STREAM_COUNT; i++)
      {
         cudaCheck(cudaMemcpyAsync(d_input[i], input[i], fileSize, cudaMemcpyHostToDevice, cudaStreams[i]), "cudaMemcpyAsync Error");
         AES_kernel<<<blockCount, threadsPerBlock, 0, cudaStreams[i]>>> (d_output[i], d_input[i], fileSize / 8, d_bs_subkey);
         cudaCheck(cudaMemcpyAsync(output[i], d_output[i], fileSize, cudaMemcpyDeviceToHost, cudaStreams[i]), "cudaMemcpyAsync Error");
      }
      for(i = 0; i < STREAM_COUNT; i++)
      {
         cudaCheck(cudaStreamSynchronize(cudaStreams[i]), "cudaStreamSynchronize Error");
      }
   }
   checkError(clock_gettime(CLOCK_MONOTONIC, &end3), "clock_gettime error");
	 
   checkError(clock_gettime(CLOCK_MONOTONIC, &start4), "clock_gettime error");
   for(j = 0; j < 1; j++)
   {
      for(i = 0; i < STREAM_COUNT; i++)
      {
         cudaCheck(cudaMemcpyAsync(d_input[i], input[i], fileSize, cudaMemcpyHostToDevice, cudaStreams[i]), "cudaMemcpyAsync Error");
      }
      for(i = 0; i < STREAM_COUNT; i++)
      {
         AES_kernel<<<blockCount, threadsPerBlock, 0, cudaStreams[i]>>> (d_output[i], d_input[i], fileSize / 8, d_bs_subkey);
      }
      for(i = 0; i < STREAM_COUNT; i++)
      {
         cudaCheck(cudaMemcpyAsync(output[i], d_output[i], fileSize, cudaMemcpyDeviceToHost, cudaStreams[i]), "cudaMemcpyAsync Error");
      }
      for(i = 0; i < STREAM_COUNT; i++)
      {
         cudaCheck(cudaStreamSynchronize(cudaStreams[i]), "cudaStreamSynchronize Error");
      }
   }
   checkError(clock_gettime(CLOCK_MONOTONIC, &end4), "clock_gettime error");
	
   checkError(clock_gettime(CLOCK_MONOTONIC, &start5), "clock_gettime error");
   for(j = 0; j < 1; j++)
   {
      for(i = 0; i < STREAM_COUNT; i++)
      {
         cudaCheck(cudaMemcpyAsync(d_input[i], input[i], fileSize, cudaMemcpyHostToDevice, cudaStreams[i]), "cudaMemcpyAsync Error");
         cudaCheck(cudaMemcpyAsync(output[i], d_output[i], fileSize, cudaMemcpyDeviceToHost, cudaStreams[i]), "cudaMemcpyAsync Error");
      }
      for(i = 0; i < STREAM_COUNT; i++)
      {
         cudaCheck(cudaStreamSynchronize(cudaStreams[i]), "cudaStreamSynchronize Error");
      }
   }
   checkError(clock_gettime(CLOCK_MONOTONIC, &end5), "clock_gettime error");
	 
   checkError(clock_gettime(CLOCK_MONOTONIC, &start6), "clock_gettime error");
   for(j = 0; j < 1; j++)
   {
      for(i = 0; i < STREAM_COUNT; i++)
      {
         cudaCheck(cudaMemcpyAsync(d_input[i], input[i], fileSize, cudaMemcpyHostToDevice, cudaStreams[i]), "cudaMemcpyAsync Error");
      }
      for(i = 0; i < STREAM_COUNT; i++)
      {
         cudaCheck(cudaMemcpyAsync(output[i], d_output[i], fileSize, cudaMemcpyDeviceToHost, cudaStreams[i]), "cudaMemcpyAsync Error");
      }
      for(i = 0; i < STREAM_COUNT; i++)
      {
         cudaCheck(cudaStreamSynchronize(cudaStreams[i]), "cudaStreamSynchronize Error");
      }
   }
   checkError(clock_gettime(CLOCK_MONOTONIC, &end6), "clock_gettime error");

   for(i = 0; i < STREAM_COUNT; i++)
   {
      cudaCheck(cudaStreamDestroy(cudaStreams[i]), "cudaStreamDestroy Error");
   }

   //cudaCheck(cudaMemcpy(output[0], d_output[0], fileSize, cudaMemcpyDeviceToHost), "cudaMemcpy Error");

   checkError(clock_gettime(CLOCK_MONOTONIC, &start7), "clock_gettime error");
   for(i = 1; i < STREAM_COUNT; i++)
   {
      if(memcmp(output[0], output[i], fileSize) != 0)
      {
         fprintf(stderr, "block mismatch error\n");
         exit(1);
      }
   }
   checkError(clock_gettime(CLOCK_MONOTONIC, &end7), "clock_gettime error");
 

   fp = fopen(argv[3], "wb");
   checkError(fp, "fopen error");
   if(fwrite(output[0], 1, fileSize, fp) != fileSize)
   {
      fprintf(stderr, "fwrite error\n");
      perror("");
      exit(1);
   }
   checkError(fflush(fp), "fflush error");
   checkError(fclose(fp), "fclose error");

   time1 = static_cast<double>(end1.tv_sec - start1.tv_sec) + static_cast<double>(end1.tv_nsec - start1.tv_nsec) * 0.000000001;
   time2 = static_cast<double>(end2.tv_sec - start2.tv_sec) + static_cast<double>(end2.tv_nsec - start2.tv_nsec) * 0.000000001;
   time2_1 = static_cast<double>(end2_1.tv_sec - start2_1.tv_sec) + static_cast<double>(end2_1.tv_usec - start2_1.tv_usec) * 0.000001;
   time3 = static_cast<double>(end3.tv_sec - start3.tv_sec) + static_cast<double>(end3.tv_nsec - start3.tv_nsec) * 0.000000001;
   time4 = static_cast<double>(end4.tv_sec - start4.tv_sec) + static_cast<double>(end4.tv_nsec - start4.tv_nsec) * 0.000000001;
   time5 = static_cast<double>(end5.tv_sec - start5.tv_sec) + static_cast<double>(end5.tv_nsec - start5.tv_nsec) * 0.000000001;
   time6 = static_cast<double>(end6.tv_sec - start6.tv_sec) + static_cast<double>(end6.tv_nsec - start6.tv_nsec) * 0.000000001;
   time7 = static_cast<double>(end7.tv_sec - start7.tv_sec) + static_cast<double>(end7.tv_nsec - start7.tv_nsec) * 0.000000001;

   printf("Size: %u\n", fileSize);
   printf("end1 - start1:     %.9f s\n", time1);
   printf("end2 - start2:     %.9f s\n", time2);
   printf("end2_1 - start2_1: %.6f s\n", time2_1);
   printf("end3 - start3:     %.9f s\n", time3);
   printf("end4 - start4:     %.9f s\n", time4);
   printf("end5 - start5:     %.9f s\n", time5);
   printf("end6 - start6:     %.9f s\n", time6);
   printf("end7 - start7:     %.9f s\n", time7);
   printf("speed: %f\n", (static_cast<double>(fileSize) / time2) );
   printf("speed (transfer 1): %f\n", (static_cast<double>(fileSize * STREAM_COUNT) / time3) );
   printf("speed (transfer 2): %f\n\n", (static_cast<double>(fileSize * STREAM_COUNT) / time4) );   

   cudaCheck(cudaFree(d_bs_subkey), "cudaFree Error");
   for(i = 0; i < STREAM_COUNT; i++)
   {
      cudaCheck(cudaFree(d_input[i]), "cudaFree Error");
      cudaCheck(cudaFree(d_output[i]), "cudaFree Error");
   }

   delete [] key;
   delete [] bs_subkey;
   for(i = 0; i < STREAM_COUNT; i++)
   {
      cudaCheck(cudaFreeHost(input[i]), "cudaFreeHost Error");
      cudaCheck(cudaFreeHost(output[i]), "cudaFreeHost Error");
   }
   key = NULL;
   bs_subkey = NULL;
   for(i = 0; i < STREAM_COUNT; i++)
   {
      input[i] = NULL;
      output[i] = NULL;
   }

   d_bs_subkey = NULL;
   for(i = 0; i < STREAM_COUNT; i++)
   {
      d_input[i] = NULL;
      d_output[i] = NULL;
   }

   cudaCheck(cudaDeviceReset(), "cudaDeviceReset Error");

   return 0;
}
