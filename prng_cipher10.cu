//nvcc  prng21.cu  -I/home/couturie/TestU01-inst/include -I/home/couturie/NVIDIA_GPU_Computing_SDK/CUDALibraries/common/inc/ -o prng21 -ltestu01 -lprobdist -lmylib -lm   -L/usr/local/cuda/lib64   -lcuda -lcudart

 
//nvcc  perf_opti_bbs2.cu  -I ~/TestU01-inst2/include -I ~/NVIDIA_GPU_Computing_SDK/C/common/inc/ -o perf_opti_bbs2 -ltestu01 -lprobdist -lmylib -lm   -L/usr/local/cuda/lib64   -lcuda -lcudart  -L/$HOME/NVIDIA_GPU_Computing_SDK/C/lib -lcutil -arch=sm_13 -O3


#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <math.h>

#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>


/*
extern "C" {

#include "unif01.h"
#include "bbattery.h"
}
*/
//extern "C" {
//	int load_RGB_pixmap(char *filename, int *width, int *height, unsigned char**R_data, unsigned char**G_data, unsigned char**B_data);
	//void store_RGB_pixmap(char *filename, unsigned char *R_data, unsigned char *G_data, unsigned char *B_data, int width, int height);
//}


typedef unsigned char   uchar;


int nb=64;//512*2;
uint size=512;


//const uint nb_ele=8192*8192/4*3;//1024*1024*2;
//const uint nb_ele=9000*9000/4*3;//1024*1024*2;
uint nb_ele;




const    uint ssize=512;
uint blocks;

ulong nb_numbers=0;




typedef unsigned int uint;
typedef unsigned long ulong;
typedef unsigned short ushort;



__device__ inline ulong rotl(const ulong x, int k) {
	return (x << k) | (x >> (64 - k));
}

__device__ inline
ulong xoroshiro128plus(ulong2* rng) {
	const ulong s0 = rng->x;
	ulong s1 = rng->y;
	const ulong result = rng->x + rng->y;

	s1 ^= s0;
	rng->x = rotl(s0, 24) ^ s1 ^ (s1 << 16); // a, b
	rng->y = rotl(s1, 37); // c

	return result;
}


__device__ inline
ulong xorshift64(ulong t)
{
	/* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
	ulong x = t;
	x ^= x >> 12; // a
	x ^= x << 25; // b
	x ^= x >> 27; // c


	return x;
}



ulong xor128(ulong t) {
  static uint x = 123456789, 
		y = 362436069, 
		z = 521288629, 
		w = 88675123;
 
  t = x ^ (x << 11);
  x = y; y = z; z = w;
  w = w ^ (w >> 19) ^ (t ^ (t >> 8));
	//printf("%u %u %u %u %u\n",x,y,z,w,t);
	return w;
}



ulong *d_random;
uchar *d_Pbox;
uchar *d_Sbox;
ulong2 *d_val;


uint *h_v;
uint *h_x;

uint *h_random;
ulong2 *h_val;


//typedef struct { ulong state;  ulong inc; } pcg32_random_t;
__device__ inline
uint pcg32_random_r(ulong2* rng)
{
	//	pcg32_random_t *rng=(pcg32_random_t*)rng2;
	ulong oldstate = rng->x;
	// Advance internal state
	rng->x = oldstate * 6364136223846793005ULL + (rng->y|1);
	// Calculate output function (XSH RR), uses old state for max ILP
	uint xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
	uint rot = oldstate >> 59u;
	return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}







__device__ __host__ inline
uint xorshift32(const uint t)
{
	/* Algorithm "xor" from p. 4 of Marsaglia, "Xorshift RNGs" */
	uint x = t;
	x ^= x << 13;
	x ^= x >> 17;
	x ^= x << 5;
	return x;
}





__device__
ulong xor128_device(ulong4 *d) {
	
	ulong t = d->x ^ (d->x << 11);
	d->x = d->y; d->y = d->z; d->z = d->w;
	d->w = d->w ^ (d->w >> 19) ^ (t ^ (t >> 8));
	return d->w;
}


__device__
ulong xorshift_device(ulong *v, ulong4 *d) {
	ulong t = d->x^(d->x>>7);
	d->x=d->y; d->y=d->z; d->z=d->w; d->w=*v;
	*v=(*v^(*v<<6))^(t^(t<<13));
	return (d->y+d->y+1)*(*v);
}

__device__
unsigned long xorwow_device(ulong2 *v, ulong4 *d){

	ulong t=(d->x^(d->x>>2)); 
	d->x=d->y; 	d->y=d->z; d->z=d->w; d->w=v->x; 
	v->x=(v->x^(v->x<<4))^(t^(t<<1));
	return (v->y+=362437)+v->x;
}



__device__ __host__
ulong
xorshift64star(ulong *s)
{
    ulong x = s[0];
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    s[0] = x;
    return x * (ulong)(0x2545f4914f6cdd1d);
}



__device__
ulong xor64_device(uint4 *d, uint *t) {
        
	*t = d->x ^ (d->x << 11);
	d->x = d->y; d->y = d->z; d->z= d->w;
	d->w = d->w ^ (d->w >> 19) ^ ((*t) ^ ((*t) >> 8));
	return d->w;
}



typedef ulong uint64_t ;

#define INT64_C(c) (c ## LL)
#define UINT64_C(c) (c ## ULL)

__device__
static inline uint64_t splitmix64(ulong *nb) {
	uint64_t z = ((*nb) += UINT64_C(0x9E3779B97F4A7C15));
	z = (z ^ (z >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
	z = (z ^ (z >> 27)) * UINT64_C(0x94D049BB133111EB);
	return z ^ (z >> 31);
}





extern __shared__ unsigned long shmem[];



const int size_pbox=32;
const int nb_sbox=16;

static int width;
static	int height;




__global__
void prng_kernel( ulong2  *dpcg, uchar * __restrict__ Pbox, uchar *  __restrict__ Sbox2, ulong *d_random, ulong * __restrict__ d_seq,int nb_ele,int nb) {
	uint i = blockIdx.x*blockDim.x + threadIdx.x;


	

	if(i<nb_ele) {
		ulong2 pcg=dpcg[i];

		ulong w,w2,res;
		uchar* res2;
		unsigned offset=threadIdx.x & (size_pbox-1);
		int n;

		unsigned base=threadIdx.x-offset;
		uchar *Sbox;
		//Sbox=&Sbox2[256*(pcg.x&(nb_sbox-1))];
		//Sbox=&Sbox2[256*(threadIdx.x&(nb_sbox-1))];
		Sbox=Sbox2;


		for(int j=0;j<nb;j++)  {
			//			w=splitmix64(&pcg.x);
			//w=xorshift64star(&pcg.x);
			int o0=base+ Pbox[size_pbox*(pcg.x&15)+offset];
			int o1=base+ Pbox[size_pbox*(16+pcg.y&15)+offset];			
			w=xoroshiro128plus(&pcg);

			//too slow
			/*w=pcg32_random_r(&pcg);
			w=w<<32;
			w=w|pcg32_random_r(&pcg);
			*/


			shmem[threadIdx.x]=w;     
      w2=xorshift64(w);
			//			__syncthreads();
			w2=w2^shmem[o0]^shmem[o1];  

			res= w^w2;


			//			if(i==0)
			//		printf("%u\n",(w&(nb_sbox-1))<<7);
			
			res2=(uchar*)&res;
			res2[0]=Sbox[res2[0]];
			res2[1]=Sbox[res2[1]];
			res2[2]=Sbox[res2[2]];
			res2[3]=Sbox[res2[3]];
			res2[4]=Sbox[res2[4]];
			res2[5]=Sbox[res2[5]];
			res2[6]=Sbox[res2[6]];
			res2[7]=Sbox[res2[7]];
			
			/*if(i==0)
				printf("%u\n",res);
			*/
			d_random[i+j*nb_ele]= res^d_seq[i+j*nb_ele];
		}


		dpcg[i]=pcg;
		
	}


}






void rc4keyperm(uchar *key,int len, int rp,uchar *sc, int size_DK) {

	//sc=1:len;



	for (int i=0;i<len;i++) {
		sc[i]=i;
	}
	for (int it = 0; it < rp; it++) {
		int j0 = 1;
		for(int i0 = 0; i0<len; i0++) {
			j0 = (j0 + sc[i0] + sc[j0] + key[i0%size_DK] )% len;
			int tmp = sc[i0];
			sc[i0] = sc[j0];
			sc[j0] = tmp;
		}

	}
}


void rc4key(uchar *key, uchar *sc, int size_DK) {

	for(int i=0;i<256;i++) {
		sc[i]=i;
	}


	uchar j0 = 0;
	for(int i0=0; i0<256; i0++) {
		j0 = (j0 + sc[i0] + key[i0%size_DK] )&0xFF;
		uchar tmp = sc[i0];
		sc[i0] = sc[j0 ];
		sc[j0] = tmp;
	}
}




uint test(int argc, char** argv)
{
	
	/*	
			static ulong t=122190821;
			t=xor128(t);
			return (uint)t;
	*/
	
	
	
	static int str=0, old_str=0;
	static ulong need_generation=1;
	static ushort init=1;
	
	ulong dum,j;
	

	static 	uchar *data_R, *data_G, *data_B;
	static 	long imsize;
	static uchar* seq;
	static uchar* seq2;
	static ulong* d_seq;
	static int oneD;
	static uchar  *Pbox;
	static uchar  *Sbox;
	
	if(init==1) {
		
		// h_val=(ulong2*)malloc(nb_ele*sizeof(ulong2));
		h_val=(ulong2*)malloc(nb_ele*sizeof(ulong2));

		Pbox=new uchar[32*size_pbox];
		Sbox=new uchar[256*nb_sbox];

		
		cudaMallocHost((void**)&h_random,nb_ele*nb*sizeof(ulong));
		
		//ulong myseed=121;
		ulong s1,s2;
		sscanf(argv[3], "%lu", &s1);
		sscanf(argv[4], "%lu", &s2);


		for(int i=0;i<32;i++) {
			ulong val[2];
			val[0]=xorshift64star(&s1);
			val[1]=xorshift64star(&s2);
			uchar *DK=(uchar*)val;
			rc4keyperm(DK, size_pbox, 1, &Pbox[size_pbox*i], 16);
			
		}

		for(int i=0;i<nb_sbox;i++)
			rc4key(&Pbox[i*8], &Sbox[256*i], 8);

		
		
		//for(int i=0;i<32;i++) {
			//for(int j=0;j<size_pbox;j++)
				//printf("%u ",Pbox[size_pbox*i+j]);
			//printf("\n\n");
		//}

		


		printf("\n %lu %lu \n",s1,s2);
		for(int i=0;i<nb_ele;i++) {

			h_val[i].x=xorshift64star(&s1);
			h_val[i].y=xorshift64star(&s2);
			if(i==0) {
				//printf("%lu %lu\n",h_val[i].x,h_val[i].y);
			}
			
		}




		
		cudaMalloc((void**) &d_random, nb_ele*nb*sizeof(ulong)) ;
		cudaMalloc((void**) &d_Pbox,size_pbox*32*sizeof(uchar)) ;
		cudaMalloc((void**) &d_Sbox,nb_sbox*256*sizeof(uchar)) ;
		cudaMalloc((void**) &d_val, nb_ele*sizeof(ulong2)) ;


		cudaMemcpy(d_val, h_val, nb_ele*sizeof(ulong2), cudaMemcpyHostToDevice) ;
		cudaMemcpy(d_Pbox, Pbox, 32*size_pbox*sizeof(uchar), cudaMemcpyHostToDevice) ;
		cudaMemcpy(d_Sbox, Sbox, nb_sbox*256*sizeof(uchar), cudaMemcpyHostToDevice) ;

		/*if(size==32768) {
			load_RGB_pixmap("32768.ppm", &width, &height, &data_R, &data_G, &data_B);
			//			width=height=32768;
		}
		if(size==16384)
			load_RGB_pixmap("16384.ppm", &width, &height, &data_R, &data_G, &data_B);
		if(size==8192)
			load_RGB_pixmap("8192.ppm", &width, &height, &data_R, &data_G, &data_B);
		if(size==4096)
			load_RGB_pixmap("4096.ppm", &width, &height, &data_R, &data_G, &data_B);
		if(size==2048)
			load_RGB_pixmap("2048.ppm", &width, &height, &data_R, &data_G, &data_B);
		if(size==1024)
			load_RGB_pixmap("1024.ppm", &width, &height, &data_R, &data_G, &data_B);
		if(size==512)
			load_RGB_pixmap("lena.ppm", &width, &height, &data_R, &data_G, &data_B);
			
			
           */

                
                  

		
		//		store_RGB_pixmap("test.ppm", data_R, data_G, data_B, width, height);
		imsize=(long)width*(long)height*3;
		printf("size image %ld\n",imsize);
		printf("eee1\n");		
		seq= new uchar[imsize];
		printf("eee2\n");		
		seq2= new uchar[imsize];
		printf("eee3\n");		
		oneD=width*height;
		printf("size %d %d\n",width,height);

		//if(size!=32768) {
			for(int i=0;i<oneD;i++) {
				seq[i]= 1;//data_R[i];
				seq[oneD+i]=128; //data_G[i];
				seq[2*oneD+i]= 255; //data_B[i];
			}
			/*	}
		else {

		}*/
		int val=cudaMalloc((void**)&d_seq,imsize*sizeof(uchar));
		//		printf("malloc %d\n",val);
		if(val==cudaSuccess)
			printf("OK \n",val);
						
		val=cudaMemcpy(d_seq,seq, imsize*sizeof(uchar), cudaMemcpyHostToDevice);
		//printf("memcpy %d\n",val);
		if(val==cudaSuccess)
			printf("OK \n",val);
		
		init=0;
	}
	


	/*	cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	*/

	
	if(need_generation==1) {
		
		cudaEvent_t start, stop;
		float time;
		prng_kernel<<<blocks,ssize,ssize*8>>>(d_val, d_Pbox, d_Sbox,d_random, d_seq,nb_ele,nb);
		
		printf("nb blocks %d nb thd blocks %d\n",blocks,ssize);

		//cudaEventCreate(&start);
		//cudaEventCreate(&stop);
		//cudaEventRecord(start, 0);
		StopWatchInterface *timer = 0;
                 sdkCreateTimer(&timer);
                sdkResetTimer(&timer);
		for(int i=0;i<100;i++) {
		 sdkStartTimer(&timer);
		
		prng_kernel<<<blocks,ssize,ssize*8>>>(d_val, d_Pbox, d_Sbox,d_random, d_seq,nb_ele,nb);

		//		cudaThreadSynchronize();
				cudaDeviceSynchronize();
		sdkStopTimer(&timer);	
		}
		time = sdkGetAverageTimerValue(&timer);	
		//cudaEventRecord(stop, 0);
		//cudaEventSynchronize(stop);
		//cudaEventElapsedTime(&time, start, stop);
		printf("GPU processing time : %f s \n", time/1000); 
		
             printf("Throughput  %f Gbits/s\n", ((double)width*height*3*8)/time/1e6);
             printf("image size  : %ld Bytes \n", (long) width*height*3); 

		
		
		//		cudaMemcpy(h_random, d_random, nb_ele*nb*sizeof(uint), cudaMemcpyDeviceToHost) ;
		cudaMemcpy(seq2, d_random, nb_ele*nb*sizeof(ulong), cudaMemcpyDeviceToHost) ;

		//if(size!=32768) {
			//for(int i=0;i<oneD;i++) {
				//data_R[i]=seq2[i];
				//data_G[i]=seq2[oneD+i];
				//data_B[i]=seq2[2*oneD+i];
			//}
		//	store_RGB_pixmap("lena2.ppm", data_R, data_G, data_B, width, height);
			//}

		cudaMemcpy(d_val, h_val, nb_ele*sizeof(ulong2), cudaMemcpyHostToDevice) ;
		//		cudaMemcpy(d_val2, h_val2, nb_ele*sizeof(uint4), cudaMemcpyHostToDevice) ;


		cudaMemcpy(d_seq,seq2, imsize*sizeof(uchar), cudaMemcpyHostToDevice);


		prng_kernel<<<blocks,ssize,ssize*8>>>(d_val, d_Pbox, d_Sbox,d_random, d_seq,nb_ele,nb);
		
		prng_kernel<<<blocks,ssize,ssize*8>>>( d_val, d_Pbox, d_Sbox, d_random, d_seq,nb_ele,nb);
		cudaMemcpy(seq2, d_random, nb_ele*nb*sizeof(ulong), cudaMemcpyDeviceToHost) ;

		//if(size!=32768) {
			//for(int i=0;i<oneD;i++) {
				//data_R[i]=seq2[i];
				//data_G[i]=seq2[oneD+i];
				//data_B[i]=seq2[2*oneD+i];
			//}
			//store_RGB_pixmap("lena3.ppm", data_R, data_G, data_B, width, height);
			//}
		
		/*for(int i=0;i<100;i++) {
			printf("%d ",h_random[i]);
		}
		printf("\n");
		*/
		
		//nb_numbers+=nb*nb_ele;

		need_generation=1+nb*nb_ele;
	}
	//		printf("ici\n");
	/*	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("CPU processing time : %f (ms) \n", time);
	*/
	need_generation--;

	//printf("%d\n ", h_random[nb*nb_ele-(need_generation)]);
	return  h_random[nb*nb_ele-(need_generation)];

}


int main (int argc, char** argv) 
{


	
	/*
	const int size=1024;
	const   int sizeMat=size*size;
  float *h_arrayA=(float*)malloc(sizeMat*sizeof(float));
	float *h_arrayB=(float*)malloc(sizeMat*sizeof(float));
	float *h_arrayC=(float*)malloc(sizeMat*sizeof(float));
	float *h_arrayCgpu=(float*)malloc(sizeMat*sizeof(float));
	

	srand48(32);
	for(int i=0;i<sizeMat;i++) {
		h_arrayA[i]=drand48();
		h_arrayB[i]=drand48();
		h_arrayC[i]=0;
		h_arrayCgpu[i]=0;

	}
	

	
  cudaEvent_t start, stop;
	float time;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	for(int i=0;i<size;i++) {
		for(int j=0;j<size;j++) {
			for(int k=0;k<size;k++) {
				                                h_arrayC[size*i+j]+=h_arrayA[size*i+k]*h_arrayB[\
																																												size*k+j];
			}
		}
	}
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);
	printf("CPU processing time : %f (ms) \n", time);

	*/	


	if(argc!=5) {
		printf("%s size nb random1 random2\n",argv[0]);
		exit(0);
	}

	size=atoi(argv[1]);
	  width =size;
          height=size;
                  
	nb=atoi(argv[2]);
	printf("size image %d\n",size);
	
	
	if(size!=512 && size!=1024 && size!=2048 && size!=4096 && size!=8192 && size!=16384 && size!=32768) {
		printf("wrong size\n");
		exit(0);
	}
	if(nb<1 || nb>32768) {
		printf("nb not good\n");
		exit(0);

	}
	printf("nb %d\n",nb);
	nb_ele=size*size/8*3/nb;
	printf("nb_ele %d\n",nb_ele);
	blocks=(nb_ele+ssize-1)/ssize;
	
	
	test(argc, argv);
	printf("nb numbers %lu\n",nb_numbers);
	/*
   unif01_Gen *gen;


	 gen = unif01_CreateExternGenBits ("raph", test);
   bbattery_BigCrush (gen);
   unif01_DeleteExternGenBits (gen);
	*/
   return 0;
	
}
