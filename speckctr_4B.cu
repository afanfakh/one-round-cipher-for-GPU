
//g++ speck128128ctr.cpp -o speck128128ctr



/*
 * Copyright (c) 2016-2017 Naruto TAKAHASHI <tnaruto@gmail.com>
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
 #include <stdbool.h>
//#include "speck.h"
#include <cuda.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <helper_functions.h>
#include <helper_cuda.h>

#include <cuda_runtime.h>




#define ROUNDS 32
#define WORDS 8
#define BLOCK_SIZE (WORDS * 2)
#define MAX_KEY_WORDS 4
#define LANE_NUM 1

#define  bl_size  16

enum speck_encrypt_type {
    SPECK_ENCRYPT_TYPE_32_64 = 0,
    SPECK_ENCRYPT_TYPE_48_72,
    SPECK_ENCRYPT_TYPE_48_96,
    SPECK_ENCRYPT_TYPE_64_96,
    SPECK_ENCRYPT_TYPE_64_128,
    SPECK_ENCRYPT_TYPE_96_96,
    SPECK_ENCRYPT_TYPE_96_144,
    SPECK_ENCRYPT_TYPE_128_128,
    SPECK_ENCRYPT_TYPE_128_192,
    SPECK_ENCRYPT_TYPE_128_256,
};

struct speck_ctx_t_ {
    int round;
    uint64_t *key_schedule;
    enum speck_encrypt_type type;
};



typedef struct  speck_ctx_t_ speck_ctx_t;



static inline uint64_t rdtsc() {
  uint32_t lo, hi;
  asm volatile (
                "cpuid \n" /* serializing */
                "rdtsc"
                : "=a"(lo), "=d"(hi) /* outputs */
                : "a"(0) /* inputs */
                : "%ebx", "%ecx");
  /* clobbers*/
  return ((uint64_t) lo) | (((uint64_t) hi) << 32);
}

static inline void cast_uint8_array_to_uint64(uint64_t *dst, const uint8_t *array) {
    // TODO: byte order
    *dst = (uint64_t)array[7] << 56 | (uint64_t)array[6] << 48 | (uint64_t)array[5] << 40 | (uint64_t)array[4] << 32 | (uint64_t)array[3] << 24 | (uint64_t)array[2] << 16 | (uint64_t)array[1] << 8 | (uint64_t)array[0];
}

static inline void cast_uint64_to_uint8_array(uint8_t *dst, uint64_t src) {
    // TODO: byte order
    dst[0] = (uint8_t)(src & 0x00000000000000ffULL);
    dst[1] = (uint8_t)((src & 0x000000000000ff00ULL) >> 8);
    dst[2] = (uint8_t)((src & 0x0000000000ff0000ULL) >> 16);
    dst[3] = (uint8_t)((src & 0x00000000ff000000ULL) >> 24);
    dst[4] = (uint8_t)((src & 0x000000ff00000000ULL) >> 32);
    dst[5] = (uint8_t)((src & 0x0000ff0000000000ULL) >> 40);
    dst[6] = (uint8_t)((src & 0x00ff000000000000ULL) >> 48);
    dst[7] = (uint8_t)((src & 0xff00000000000000ULL) >> 56);
}



int is_validate_key_len(enum speck_encrypt_type type, int key_len) {
    int ret;

    switch (type) {
        case SPECK_ENCRYPT_TYPE_128_128:
            ret = (key_len == (128 / 8));
            break;
        case SPECK_ENCRYPT_TYPE_128_192:
            ret = (key_len == (192 / 8));
            break;
        case SPECK_ENCRYPT_TYPE_128_256:
            ret = (key_len == (256 / 8));
            break;
        default:
            ret = -1;
            break;
    }

    return ret;
}

int get_round_num(enum speck_encrypt_type type) {
    int ret;

    switch (type) {
        case SPECK_ENCRYPT_TYPE_128_128:
            ret = 32;
            break;
        case SPECK_ENCRYPT_TYPE_128_192:
            ret = 33;
            break;
        case SPECK_ENCRYPT_TYPE_128_256:
            ret = 34;
            break;
        default:
            ret = -1;
            break;
    }

    return ret;
}

int get_key_words_num(enum speck_encrypt_type type) {
    int ret;

    switch (type) {
        case SPECK_ENCRYPT_TYPE_128_128:
            ret = 2;
            break;
        case SPECK_ENCRYPT_TYPE_128_192:
            ret = 3;
            break;
        case SPECK_ENCRYPT_TYPE_128_256:
            ret = 4;
            break;
        default:
            ret = -1;
            break;
    }

    return ret;
}

//typedef struct  speck_ctx_t_ speck_ctx_t;
// https://eprint.iacr.org/2013/404.pdf
//
// Speck128/128
//  Key:        0f0e0d0c0b0a0908 0706050403020100
//  Plaintext:  6c61766975716520 7469206564616d20
//  Ciphertext: a65d985179783265 7860fedf5c570d18
static const uint64_t s_key[2] = {0x0706050403020100, 0x0f0e0d0c0b0a0908};
static const uint64_t s_plain_text[2] = {0x7469206564616d20, 0x6c61766975716520};
static const uint64_t s_cipher_text[2] = {0x7860fedf5c570d18, 0xa65d985179783265};

//static const uint8_t s_key_stream[16] = {
//    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
//};
static const uint8_t s_plain_text_stream[16] = {
    0x20, 0x6d, 0x61, 0x64, 0x65, 0x20, 0x69, 0x74, 0x20, 0x65, 0x71, 0x75, 0x69, 0x76, 0x61, 0x6c,
};
static const uint8_t s_cipher_text_stream[16] = {
    0x18, 0x0d, 0x57, 0x5c, 0xdf, 0xfe, 0x60, 0x78, 0x65, 0x32, 0x78, 0x79, 0x51, 0x98, 0x5d, 0xa6,
};

//#define BLOCK_SIZE 16


__device__  void speck_round(uint64_t *x, uint64_t *y, const uint64_t *k) {
 //uint64_t h;
//h=k;
//printf(" k from round =%llu          x=%llu \n", k,x);
   *x = (*x >> 8) | (*x << (8 * sizeof(*x) - 8));  // x = ROTR(x, 8)
    *x += *y;   
    *x^= *k; 
    *y = (*y << 3) | (*y >> (8 * sizeof(*y) - 3));  // y = ROTL(y, 3)
    *y ^= *x;
    
}

static inline void speck_round_host(uint64_t *x, uint64_t *y, uint64_t *k) {
    *x = (*x >> 8) | (*x << (8 * sizeof(*x) - 8));  // x = ROTR(x, 8)
    *x += *y; 
    *x ^=*k;
    *y = (*y << 3) | (*y >> (8 * sizeof(*y) - 3));  // y = ROTL(y, 3)
    *y ^= *x;
}




void speck_finish(speck_ctx_t **ctx) {
    if (!ctx) return;
    free((*ctx)->key_schedule);
    free(*ctx);
}

speck_ctx_t *speck_init(enum speck_encrypt_type type, const uint8_t *key, int key_len) {
    if (key == NULL) return NULL;
    if (!is_validate_key_len(type, key_len)) return NULL;

    speck_ctx_t *ctx = (speck_ctx_t *)calloc(1, sizeof(speck_ctx_t));
    if (!ctx) return NULL;
    ctx->type = type;
    ctx->round = get_round_num(type);
    
    ctx->key_schedule = (uint64_t*)calloc(1, ctx->round * sizeof(uint64_t));    /// this line has been changed by Ahmed
    if (!ctx->key_schedule) return NULL;

    // calc key schedule
    uint64_t b;
    uint64_t a;
    uint64_t k;
    int key_words_num = get_key_words_num(ctx->type);
    uint64_t keys[MAX_KEY_WORDS];
    for (int i = 0; i < key_words_num; i++) {
        cast_uint8_array_to_uint64(&keys[i], key + (WORDS * i));
    }
    ctx->key_schedule[0] = keys[0];
    for (int i = 0; i < ctx->round - 1; i++) {
        b = keys[0];
        a = keys[1];
        k = (uint64_t)i;
        speck_round_host(&a, &b, &k);
        keys[0] = b;

        if (key_words_num != 2) {
            for (int j = 1; j < (key_words_num - 1); j++) {
                keys[j] = keys[j + 1];
            }
        }
        keys[key_words_num - 1] = a;

        ctx->key_schedule[i + 1] = keys[0];
    }

    return ctx;
}

void generate_random_array(uint8_t *iv, size_t iv_len) {
      std::random_device rd;  //Will be used to obtain a seed for the random number engine
      std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
      std::uniform_int_distribution<> dis;

    for(int i=0; i<iv_len; i++) {
        iv[i] = static_cast<uint8_t>(dis(gen));
     
   //  for(int i=0; i<iv_len; i++) {
     // iv[i] = rand();
    }
}

void show_array(const char *explain, const uint8_t *array, size_t len) {
    printf("%20s ", explain);
    for(int i=len-1; i >= 0; i--) {
        printf("%02x ", array[i]);
    }
    printf("\n");
}

 __device__ void show_array2(const char *explain, uint64_t *array, size_t len) {
    printf("%20s ", explain);
    for(int i=len-1; i >= 0; i--) {
        printf("%02x ", array[i]);
    }
    printf("\n");
}


/*
static inline uint8_t *ctr128_inc(uint8_t counter[16]) {
    uint32_t n = 16;
    uint8_t c;
    do {
        --n;
        c = counter[n];
        ++c;
        counter[n] = c;
        if (c) return counter;
    } while (n);

    return counter;
}

static void cast_uint8_array_to_uint64_len(uint64_t *dst, const uint8_t *array, int len) {
    uint8_t tmp[8] = {0};
    int i;
    for(i=0; i<len; i++) {
        tmp[i] = array[i];
    }
    *dst =  (uint64_t)tmp[7] << 56 | (uint64_t)tmp[6] << 48 | (uint64_t)tmp[5] << 40 | (uint64_t)tmp[4] << 32 | (uint64_t)tmp[3] << 24 | (uint64_t)tmp[2] << 16 | (uint64_t)tmp[1] <<  8 | (uint64_t)tmp[0];
}

static inline void cast_uint64_to_uint8_array_len(uint8_t *dst, uint64_t src, int len) {
    for(int i=0; i<len; i++) {
        dst[i] = (uint8_t)((src & (0x00000000000000ffULL << (8 * i)) ) >> (8 *  i));
    }
}
*/


__global__  void speck_ctr_encrypt(uint64_t *key_schedule,  uint64_t *in, uint64_t *out, int len, uint64_t *iv, int iv_len, enum speck_encrypt_type type) {
  //      printf("erarar %d %d\n");
  /*    if (len < 0) {
        return -1;
    }
  */

    int count = len/2;//(len / (BLOCK_SIZE * LANE_NUM));
    //    printf("count %d\n",count);    
    //int remain_bytes = len % BLOCK_SIZE;
    int i;
    uint64_t crypted_iv_block[2];
    //    uint64_t plain_block[2];
   uint64_t  iv_block[2];
 // __shared__ uint64_t * tmp;
  // tmp =  (uint64_t*)malloc(len*8);
    iv_block[0]=iv[0];
    iv_block[1]=iv[1];
  //printf("len= %d     len_iv=%d \n", len , iv_len);

   //printf("okkkkkkkkkkk\n");
    //printf("key %u  \n", ctx->key_schedule[1]);
    
    //printf("count = %d  \n", count);
    int   id= blockIdx.x*blockDim.x+threadIdx.x;
/*
  if  (id==0) ///printing the recieved variables
  {  
 
    //show_array2("--- iv", iv, iv_len*8);
  //  show_array2("--- plain", in, len*8);
     printf("plain -- ");
       for (int i = 0; i < len*8; i++) 
       printf("%llu  ", &in[i]);
    printf("\n key -- ");
    for (int i = 0; i < ctx->round; i++) 
       printf("%d = %llu -- ", i,&ctx->key_schedule[i] );
     printf("\n");
}
 */
         
   if (id <count)
    {
     	//printf("id =%d\n",id);
       iv_block[0]+=id;

	  crypted_iv_block[0] = iv_block[0];
	  crypted_iv_block[1] = iv_block[1]; 

	  //	  for (int i = 0; i < ctx->round; i++) {
	  
	  
	 //speck_encrypt(ctx, iv_block, crypted_iv_block);
	  
	  

	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[0]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[1]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[2]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[3]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[4]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[5]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[6]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[7]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[8]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[9]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[10]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[11]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[12]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[13]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[14]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[15]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[16]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[17]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[18]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[19]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[20]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[21]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[22]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[23]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[24]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[25]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[26]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[27]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[28]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[29]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[30]);
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[31]);


            switch (type) {

        case SPECK_ENCRYPT_TYPE_128_192:

            speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[32]);

            break;

        case SPECK_ENCRYPT_TYPE_128_256:

            speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[32]);

	        speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[33]);

            break; 

        default:

            break;
            
            
            
            
            
            
            
            

    }
	
//printf("okkkkkkkkkkk\n");
	
	//printf("iv bloc %d %u %u\n",i,iv_block[0],iv_block[1]);
	//printf("cr block %d %u %u\n",i,crypted_iv_block[0],crypted_iv_block[1]);
//__syncthreads(); 
	out[2*id]=crypted_iv_block[0] ^ in[2*id];
	out[2*id+1]=crypted_iv_block[1] ^ in[2*id+1];



    }
 //if (id==0)
 //for(; id<count; id+=gridDim.x*blockDim.x)
//memcpy(&out[id], &tmp[id], sizeof(uint64_t)*len);
//if (id==0)
	     //for (int i = 0; i < len ; i++) 
	    // {
	             //    out[i]=tmp[i];
	              //   printf("%02x   ", out[i]);

	   // }

}








__global__  void speck_ctr_encrypt4(uint64_t *key_schedule,  uint64_t *in, uint64_t *out, int len, uint64_t *iv, int iv_len, enum speck_encrypt_type type) {
  //      printf("erarar %d %d\n");
  /*    if (len < 0) {
        return -1;
    }
  */

    int count = len/4;   //(len / (BLOCK_SIZE * LANE_NUM));
   
     int i;
     uint64_t crypted_iv_block[4];
  
     uint64_t  iv_block[4];
 
    iv_block[0]=iv[0];
    iv_block[1]=iv[1];
    iv_block[2]=iv[2];
    iv_block[3]=iv[3];
    
    int   id= blockIdx.x*blockDim.x+threadIdx.x;

         
   if (id <count)
    {
     	  for (i = 0; i < 32; i++){
            iv_block[0]+=id;

	    crypted_iv_block[0] = iv_block[0];
	    crypted_iv_block[1] = iv_block[1]; 
          
	    speck_round(&crypted_iv_block[1], &crypted_iv_block[0], &key_schedule[i]);
	    
            iv_block[2]+=id;
            crypted_iv_block[2] = iv_block[2];
	    crypted_iv_block[3] = iv_block[3]; 
            speck_round(&crypted_iv_block[3], &crypted_iv_block[2], &key_schedule[i]);
	  }


	out[4*id]=crypted_iv_block[0] ^ in[4*id];
	out[4*id+1]=crypted_iv_block[1] ^ in[4*id+1];
        out[4*id+2]=crypted_iv_block[2] ^ in[4*id+2];
	out[4*id+3]=crypted_iv_block[3] ^ in[4*id+3];
	
   }
}




void speck_ctr_encrypt_cpu(speck_ctx_t *ctx, const uint64_t *in, uint64_t *out, int len, uint64_t *iv, int iv_len,enum speck_encrypt_type type) {

    int count = len/2;//(len / (BLOCK_SIZE * LANE_NUM));
    //    printf("count %d\n",count);    
    //int remain_bytes = len % BLOCK_SIZE;
    int i;
    uint64_t crypted_iv_block[2];
    //    uint64_t plain_block[2];
    uint64_t iv_block[2];
    iv_block[0]=iv[0];
    iv_block[1]=iv[1];
    

      //printf("key0= %u  key1= %u \n", &ctx->key_schedule[0], &ctx->key_schedule[1] );
      //  printf("len= %d     len_iv=%d \n", len , iv_len);
    for (i = 0; i < count; i++) {   // replaced with thread id in gpu
    
    /*
     if  (i==0) ///printing the recieved variables
  {  
 
    show_array2("--- iv", iv, len*8);
    show_array2("--- plain", in, len*8);
    printf("plain2 -- ");
       for (int i = 0; i < len*8; i++) 
     printf("%llu ", in[i]);
    printf("\n key -- ");
    for (int i = 0; i < ctx->round; i++) 
       printf("%d = %u -- ", i,&ctx->key_schedule[i] );
     printf("\n");
}
*/    
      //cast_uint8_array_to_uint64(&iv_block[0], iv + (WORDS * 0));
      //cast_uint8_array_to_uint64(&iv_block[1], iv + (WORDS * 1));
      //  ctr128_inc(iv);

      
      iv_block[0]+=i;

      
      //speck_encrypt(ctx, iv_block, crypted_iv_block);
	
	  crypted_iv_block[0] = iv_block[0];
	  crypted_iv_block[1] = iv_block[1];
	  //	  for (int i = 0; i < ctx->round; i++) {
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[0]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[1]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[2]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[3]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[4]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[5]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[6]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[7]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[8]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[9]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[10]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[11]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[12]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[13]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[14]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[15]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[16]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[17]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[18]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[19]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[20]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[21]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[22]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[23]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[24]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[25]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[26]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[27]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[28]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[29]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[30]);
	    speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[31]);
        
            switch (type) {

        case SPECK_ENCRYPT_TYPE_128_192:

            speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[32]);

            break;

        case SPECK_ENCRYPT_TYPE_128_256:

            speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[32]);

	        speck_round_host(&crypted_iv_block[1], &crypted_iv_block[0], &ctx->key_schedule[33]);

            break; 

        default:

            break;

       }


	 out[2*i]=crypted_iv_block[0] ^ in[2*i];
	 out[2*i+1]=crypted_iv_block[1] ^ in[2*i+1];
	
    }


}




int encrypt_decrypt_stream_test(long block_num,enum speck_encrypt_type type,uint8_t  s_key_stream[], uint8_t key_s) {
    int r = 0;
    
    /// define CPU variables 
    speck_ctx_t *ctx = NULL;
    uint8_t *plain_text_stream = NULL;
    uint8_t *crypted_text_stream = NULL;
    uint8_t *decrypted_text_stream = NULL;
    uint8_t *iv_text_stream = NULL;
    uint8_t *origin_iv_text_stream = NULL;
    int *bl1,*bl2;
    
    
    
    // Define GPU variable
      speck_ctx_t  *dev_ctx= NULL;
      
      uint64_t *dev_key_schedule=NULL;
      uint8_t *dev_plain_text_stream= NULL;
      uint8_t *dev_crypted_text_stream=NULL;
      uint8_t *dev_decrypted_text_stream= NULL;
      uint8_t *dev_iv_text_stream= NULL;
     //uint64_t *key;
    
    cudaStream_t  streams[2];
    
     // specfy gridSize and blockSize of threads
    int  blockSize =1024;
    int  gridSize = block_num/blockSize; /// ????
    
    //allocate CPU variables   
   plain_text_stream = (uint8_t*)malloc(BLOCK_SIZE * block_num);
    if (!plain_text_stream) {
        r = 1;
      // goto finish;
    }
    crypted_text_stream = (uint8_t*)malloc(BLOCK_SIZE * block_num);
    if (!crypted_text_stream) {
        r = 1;
      //  goto finish;
    }
    decrypted_text_stream = (uint8_t*)malloc(BLOCK_SIZE * block_num);
    if (!decrypted_text_stream) {
        r = 1;
      // goto finish;
    }
    iv_text_stream = (uint8_t*)malloc(BLOCK_SIZE);
    if (!iv_text_stream) {
        r = 1;
     //   goto finish;
    }
    origin_iv_text_stream = (uint8_t*)malloc(BLOCK_SIZE);
    if (!origin_iv_text_stream) {
        r = 1;
    //    goto finish;
    }



    printf("Size message %ld bytes\n",(long)BLOCK_SIZE * block_num);
    
    //allocate GPU variables
     cudaMalloc(&dev_key_schedule, sizeof(uint64_t)*ROUNDS);
     cudaMalloc((void **)&dev_plain_text_stream, BLOCK_SIZE * block_num*sizeof(uint8_t));        
     cudaMalloc((void **)&dev_crypted_text_stream, BLOCK_SIZE * block_num*sizeof(uint8_t));                                                                   
     cudaMalloc((void **)&dev_decrypted_text_stream, BLOCK_SIZE * block_num*sizeof(uint8_t));             
     cudaMalloc((void **)&dev_iv_text_stream, BLOCK_SIZE *sizeof(uint8_t));     
     //cudaMalloc(&key, BLOCK_SIZE *sizeof(uint64_t));     
      
    for (int i = 0; i < block_num; i++) 
        {
            memcpy(plain_text_stream + (i * BLOCK_SIZE), s_plain_text_stream, sizeof(s_plain_text_stream));
        }
       generate_random_array(origin_iv_text_stream, BLOCK_SIZE);

     ctx = speck_init(type, s_key_stream, key_s);
    if (!ctx) {
        r = 1;
      //  goto finish;
    }
    memcpy(iv_text_stream, origin_iv_text_stream, BLOCK_SIZE);
    


    ////////////////////////////////////           Speck on GPU     ////////////////////////////////////////////

    printf(" Run Speck on GPU \n");
 
    float elapsed=0;
    cudaEvent_t start, stop;



    //copy all needed variables to device memory
    cudaMemcpy(dev_key_schedule, ctx->key_schedule,sizeof(uint64_t)*ROUNDS, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_plain_text_stream, plain_text_stream, BLOCK_SIZE * block_num*sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_iv_text_stream, iv_text_stream,BLOCK_SIZE *sizeof(uint8_t), cudaMemcpyHostToDevice);
    
    /*cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    */
    StopWatchInterface *timer = 0;
    sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    for(int i=0;i<100;i++) {
      sdkStartTimer(&timer);

    
    
      speck_ctr_encrypt4<<<gridSize, blockSize>>>( dev_key_schedule,  (uint64_t*)dev_plain_text_stream, (uint64_t*)dev_crypted_text_stream , BLOCK_SIZE * block_num/8 , (uint64_t*)dev_iv_text_stream,BLOCK_SIZE/8,type);   //encryption 
    
    
      cudaDeviceSynchronize();
      
    /*cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);
    */

      sdkStopTimer(&timer);

    }
      
    //copy cypher text from device to host
    cudaMemcpy( crypted_text_stream, dev_crypted_text_stream , BLOCK_SIZE * block_num*sizeof(uint8_t), cudaMemcpyDeviceToHost );

    float reduceTime_GPU = sdkGetAverageTimerValue(&timer);
    printf("Average time: %f s\n", reduceTime_GPU/1000);
    printf("Throughput on GPU %f Gbps\n ", 8*(double)BLOCK_SIZE * (double)block_num/reduceTime_GPU/1e6);
    
    
 /*
  printf("\n encr---\n");
 for (int i = 0; i < BLOCK_SIZE * block_num; i++) 
 printf("%02x   ",crypted_text_stream[i]);
  printf("\n");
  
 */
  
  
memcpy(iv_text_stream, origin_iv_text_stream, BLOCK_SIZE);

 //recopy the iv to device memory
 cudaMemcpy(dev_iv_text_stream, iv_text_stream,BLOCK_SIZE *sizeof(uint8_t), cudaMemcpyHostToDevice);
 
cudaMemcpy(dev_crypted_text_stream, crypted_text_stream , BLOCK_SIZE * block_num*sizeof(uint8_t), cudaMemcpyHostToDevice);

speck_ctr_encrypt4<<<gridSize, blockSize>>>( dev_key_schedule  ,  (uint64_t*)dev_crypted_text_stream, (uint64_t*)dev_decrypted_text_stream , BLOCK_SIZE * block_num/8 , (uint64_t*)dev_iv_text_stream, BLOCK_SIZE/8,type );   //decryption 
 
 //copy decryphed text from device to host
cudaMemcpy(decrypted_text_stream, dev_decrypted_text_stream , BLOCK_SIZE * block_num*sizeof(uint8_t), cudaMemcpyDeviceToHost );
   

/*
cudaEventElapsedTime(&elapsed, start, stop);
float time_GPU =elapsed/1000;
cudaEventDestroy(start);
cudaEventDestroy(stop);

printf("Elapsed time on GPU %f seconds \n ", time_GPU);
 printf("Throughput on GPU %f Gbps\n ", 8*BLOCK_SIZE * block_num/time_GPU/1e9);
*/

 
/*
printf("\n decr---\n");
 for (int i = 0; i < BLOCK_SIZE * block_num; i++) 
 printf("%02x  ",decrypted_text_stream[i]);
 printf("\n");
*/
    //show_array("plain", plain_text_stream, block_num * BLOCK_SIZE);
   // show_array("decrypted", decrypted_text_stream, block_num * BLOCK_SIZE);
   
    for (int i = 0; i < BLOCK_SIZE * block_num; i++)  {
        //printf("%d %u %u\n",i,plain_text_stream[i],decrypted_text_stream[i]);
      if (plain_text_stream[i] != decrypted_text_stream[i]) {
            printf("block_num:%d idx:%d  0x%02x != 0x%02x\n", block_num, i, plain_text_stream[i], decrypted_text_stream[i]);
            show_array("iv", origin_iv_text_stream, BLOCK_SIZE);
            show_array("plain", plain_text_stream, block_num * BLOCK_SIZE);
            show_array("decrypted", decrypted_text_stream, block_num * BLOCK_SIZE);
            show_array("counted iv", iv_text_stream, BLOCK_SIZE);
            printf("\n");

            r = 1;
          // goto finish;
        }
        
}

    ////////////////////////////////////           Speck on CPU     ////////////////////////////////////////////  
 
    printf(" Run Speck on CPU \n");
    
    double time_CPU = 0.0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    
    speck_ctr_encrypt_cpu(ctx, (uint64_t*)plain_text_stream,  (uint64_t*)crypted_text_stream, BLOCK_SIZE * block_num/8,  (uint64_t*)iv_text_stream, BLOCK_SIZE/8,type);   //encryption 

    cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);

    
    memcpy(iv_text_stream, origin_iv_text_stream, BLOCK_SIZE);
    
    speck_ctr_encrypt_cpu(ctx, (uint64_t*)crypted_text_stream,  (uint64_t*)decrypted_text_stream, BLOCK_SIZE * block_num/8,  (uint64_t*)iv_text_stream, BLOCK_SIZE/8,type);   // decryption
    

    cudaEventElapsedTime(&elapsed, start, stop);
    time_CPU =elapsed/1000;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    printf("Elapsed time on CPU %f seconds \n ", time_CPU);
/*
    clock_t end = clock();
    time_CPU += (double)(end - begin) / CLOCKS_PER_SEC;
    
    printf("Elapsed time on CPU %f seconds \n", time_CPU);
    */
    
    //show_array("plain", plain_text_stream, block_num * BLOCK_SIZE);
   // show_array("decrypted", decrypted_text_stream, block_num * BLOCK_SIZE);
    
    
    /////////////////////     speedup   ////////////////////////////
    
    printf("The speedup = %f \n",time_CPU / (reduceTime_GPU/1000.));
    
    for (int i = 0; i < BLOCK_SIZE * block_num; i++) {
        //printf("%d %u %u\n",i,plain_text_stream[i],decrypted_text_stream[i]);
      if (plain_text_stream[i] != decrypted_text_stream[i]) {
            printf("block_num:%d idx:%d  0x%02x != 0x%02x\n", block_num, i, plain_text_stream[i], decrypted_text_stream[i]);
            show_array("iv", origin_iv_text_stream, BLOCK_SIZE);
            show_array("plain", plain_text_stream, block_num * BLOCK_SIZE);
            show_array("decrypted", decrypted_text_stream, block_num * BLOCK_SIZE);
            show_array("counted iv", iv_text_stream, BLOCK_SIZE);
            printf("\n");

            r = 1;
            goto finish;
        }
    }



    finish:
    
      // Release device memory
        cudaFree(dev_plain_text_stream);
        cudaFree(dev_crypted_text_stream);
        cudaFree(dev_decrypted_text_stream);
        cudaFree(dev_iv_text_stream); 
        cudaFree(&dev_ctx);
       
    free(plain_text_stream);
    free(crypted_text_stream);
    free(decrypted_text_stream);
    free(iv_text_stream);
    free(origin_iv_text_stream);
    speck_finish(&ctx);
    return r;
}


 int main(int argc, char **argv) {

    cudaDeviceProp deviceProp = { 3 };
    int dev;
    dev = findCudaDevice(argc, (const char **)argv);
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));



int t, flag=1;

 

 enum speck_encrypt_type type;

uint8_t key_stream[32] = {

    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,

};



uint8_t *s_key_stream;

long psize;
 psize = atoi((argv[1]));  

 // s_plain_text_stream=(uint8_t*)malloc(16);

//  s_cipher_text_stream=(uint8_t*)malloc(16);

  

 

 printf("Encryption type:  \n");

  printf("1 --> for SPECK_ENCRYPT_TYPE_128_128 \n2 --> for SPECK_ENCRYPT_TYPE_128_192 \n3 --> for SPECK_ENCRYPT_TYPE_128_256\n");

  uint8_t key_s;

//scanf("%d ", &t);

 t=1;



   switch (t) {

        case 1:

                 // s_key_stream=(uint8_t*)malloc(16);

                  type=SPECK_ENCRYPT_TYPE_128_128;

                  key_s=16;

                  s_key_stream=(uint8_t*)malloc(key_s);

                  for(int i=1; i<key_s; i++)

                      s_key_stream[i] = key_stream[i];

                  break;  

        case 2:

                 //s_key_stream=(uint8_t*)malloc(24);

                 type=SPECK_ENCRYPT_TYPE_128_192;

                  key_s=24;

                   s_key_stream=(uint8_t*)malloc(key_s);

                  for(int i=1; i<key_s; i++)

                      s_key_stream[i] = key_stream[i];

                 break; 

        case 3:

                 // s_key_stream=(uint8_t*)malloc(32);

                  type=SPECK_ENCRYPT_TYPE_128_256;

                   key_s=32;

                  s_key_stream=(uint8_t*)malloc(key_s);

                  for(int i=1; i<key_s; i++)

                      s_key_stream[i]=key_stream[i];

                  break; 

        default:

                flag=0;

                break; 

    }

    

 
   printf("***************************\n");
   printf("selected type --> %d \n", t);
   printf("***************************\n");

 

    printf("test encrypt_decrypt_stream_test\n");

   if (flag) 

        {

      

	  int r = encrypt_decrypt_stream_test(psize,type, key_stream, key_s);

            if(r != 0) 

             {

               return r;

              }

      

    printf("success encrypt_decrypt_stream_test\n");

}

else

printf("Wrong Speck Encrypted Type \n");



    
    return 0;
}
