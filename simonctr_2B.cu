

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
#include <cuda.h>
#include <random>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>     
//#define ROUNDS 67
#define WORDS  8
#define BLOCK_SIZE (WORDS * 2)
#define MAX_KEY_WORDS 4
#define LANE_NUM 1

#define  bl_size  16


//#include <helper_functions.h>
//#include <helper_cuda.h>

//#include <cuda_runtime.h>

int ROUNDS;
/*
#define ROTL32(x,r) (((x)<<(r)) | (x>>(32-(r))))
#define ROTR32(x,r) (((x)>>(r)) | ((x)<<(32-(r))))
#define ROTL64(x,r) (((x)<<(r)) | (x>>(64-(r))))
#define ROTR64(x,r) (((x)>>(r)) | ((x)<<(64-(r))))
#define f64(x) ((ROTL64(x,1) & ROTL64(x,8)) ^ ROTL64(x,2))
#define R64x2(x,y,k1,k2) (y^=f64(x), y^=k1, x^=f64(y), x^=k2)
*/

enum simon_encrypt_type {
    simon_ENCRYPT_TYPE_32_64 = 0,
    simon_ENCRYPT_TYPE_48_72,
    simon_ENCRYPT_TYPE_48_96,
    simon_ENCRYPT_TYPE_64_96,
    simon_ENCRYPT_TYPE_64_128,
    simon_ENCRYPT_TYPE_96_96,
    simon_ENCRYPT_TYPE_96_144,
    simon_ENCRYPT_TYPE_128_128,
    simon_ENCRYPT_TYPE_128_192,
    simon_ENCRYPT_TYPE_128_256,
};

struct simon_ctx_t_ {
    int round;
    uint64_t *key_schedule;
    enum simon_encrypt_type type;
};



typedef struct  simon_ctx_t_ simon_ctx_t;



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



int is_validate_key_len(enum simon_encrypt_type type, int key_len) {
    int ret;

    switch (type) {
        case simon_ENCRYPT_TYPE_128_128:
            ret = (key_len == (128 / 8));
            break;
        case simon_ENCRYPT_TYPE_128_192:
            ret = (key_len == (192 / 8));
            break;
        case simon_ENCRYPT_TYPE_128_256:
            ret = (key_len == (256 / 8));
            break;
        default:
            ret = -1;
            break;
    }

    return ret;
}

int get_round_num(enum simon_encrypt_type type) {
    int ret;

    switch (type) {
        case simon_ENCRYPT_TYPE_128_128:
            ret = 32;
            break;
        case simon_ENCRYPT_TYPE_128_192:
            ret = 33;
            break;
        case simon_ENCRYPT_TYPE_128_256:
            ret = 34;
            break;
        default:
            ret = -1;
            break;
    }

    return ret;
}

int get_key_words_num(enum simon_encrypt_type type) {
    int ret;

    switch (type) {
        case simon_ENCRYPT_TYPE_128_128:
            ret = 2;
            break;
        case simon_ENCRYPT_TYPE_128_192:
            ret = 3;
            break;
        case simon_ENCRYPT_TYPE_128_256:
            ret = 4;
            break;
        default:
            ret = -1;
            break;
    }

    return ret;
}

//typedef struct  simon_ctx_t_ simon_ctx_t;
// https://eprint.iacr.org/2013/404.pdf
//
// simon128/128
//  Key:        0f0e0d0c0b0a0908 0706050403020100
//  Plaintext:  6c61766975716520 7469206564616d20
//  Ciphertext: a65d985179783265 7860fedf5c570d18
static const uint64_t s_key[2] = {0x0706050403020100, 0x0f0e0d0c0b0a0908};
static const uint64_t s_plain_text[2] = {0x7469206564616d20, 0x6c61766975716520};
static const uint64_t s_cipher_text[2] = {0x7860fedf5c570d18, 0xa65d985179783265};

/// for type 128_128

/*
static const uint8_t s_key_stream1[16] = {
   0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
};



////  for type  128_192
static const uint8_t s_key_stream2[24] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17,
};


////       for type 128_256

static const uint8_t s_key_stream3[32] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
};


*/
static const uint8_t s_plain_text_stream[16] = {
  0x20, 0x6d, 0x61, 0x64, 0x65, 0x20, 0x69, 0x74, 0x20, 0x65, 0x71, 0x75, 0x69, 0x76, 0x61, 0x6c,
};

//static const uint8_t  s_plain_text_stream[16] = {[0 ... 15] = 0x20};
//static const uint8_t  s_cipher_text_stream[16] = {[0 ... 15] = 0x20};

static const uint8_t s_cipher_text_stream[16] = {
   0x18, 0x0d, 0x57, 0x5c, 0xdf, 0xfe, 0x60, 0x78, 0x65, 0x32, 0x78, 0x79, 0x51, 0x98, 0x5d, 0xa6,
};

//#define BLOCK_SIZE 16




__host__ __device__ uint64_t  ROTR64(uint64_t *x, int r)
{
return (((*x)>>(r)) | ((*x)<<(64-(r))));
}


__host__ __device__ uint64_t  ROTL64(uint64_t *x,  int r)
{
return (((*x)<<(r)) | ((*x)>>(64-(r))));
}



__host__ __device__  uint64_t  f64(uint64_t *x) 
{
 return  ((ROTL64(x,1) & ROTL64(x,8)) ^ ROTL64(x,2));
}

__host__ __device__  void R64x2(uint64_t *x, uint64_t *y, uint64_t *k1, uint64_t *k2)
{

*y^=f64(x);
*y^=*k1;
*x^=f64(y);
*x^=*k2;
}


void Simon128128KeySchedule( uint8_t  *K , uint64_t  *rk)  // for init key
{
  int i;
  uint64_t  B=K[1],  A=K[0]; 
  uint64_t  c=0xfffffffffffffffcLL, z=0x7369f885192c0ef5LL;
  for(i=0;i<64;)
  {
     rk[i++]=A; 
     A^=c^(z&1)^ROTR64(&B,3)^ROTR64(&B,4);   
     z>>=1;
     rk[i++]=B; 
     B^=c^(z&1)^ROTR64(&A,3)^ROTR64(&A,4); 
     z>>=1;
  }
     rk[64]=A;   A^=c^1^ROTR64(&B,3)^ROTR64(&B,4);
     rk[65]=B;    B^=c^0^ROTR64(&A,3)^ROTR64(&A,4); 
     rk[66]=A; 
     rk[67]=B;
}

void Simon128192KeySchedule(uint8_t  *K, uint64_t *rk)
 {
   uint64_t i,C=K[2],B=K[1],A=K[0];
   uint64_t c=0xfffffffffffffffcLL, z=0xfc2ce51207a635dbLL;
   for(i=0;i<63;)
      {
   		  rk[i++]=A; 
   		 A^=c^(z&1)^ROTR64(&C,3)^ROTR64(&C,4); z>>=1;
   		 rk[i++]=B; B^=c^(z&1)^ROTR64(&A,3)^ROTR64(&A,4); z>>=1;
  	 	 rk[i++]=C; C^=c^(z&1)^ROTR64(&B,3)^ROTR64(&B,4); z>>=1;
    	}
  			rk[63]=A; A^=c^1^ROTR64(&C,3)^ROTR64(&C,4);
   			rk[64]=B; B^=c^0^ROTR64(&A,3)^ROTR64(&A,4);
  			rk[65]=C; C^=c^1^ROTR64(&B,3)^ROTR64(&B,4);
  			rk[66]=A; 
  			rk[67]=B; 
  			rk[68]=C;
 }




void Simon128256KeySchedule(uint8_t *K,uint64_t *rk)
{
    uint64_t i,D=K[3],C=K[2],B=K[1],A=K[0];
    uint64_t c=0xfffffffffffffffcLL, z=0xfdc94c3a046d678bLL;
    for(i=0;i<64;)
     {
        rk[i++]=A; A^=c^(z&1)^ROTR64(&D,3)^ROTR64(&D,4)^B^ROTR64(&B,1); z>>=1;
        rk[i++]=B; 
        B^=c^(z&1)^ROTR64(&A,3)^ROTR64(&A,4)^C^ROTR64(&C,1); z>>=1;
        rk[i++]=C; C^=c^(z&1)^ROTR64(&B,3)^ROTR64(&B,4)^D^ROTR64(&D,1); z>>=1;
        rk[i++]=D; D^=c^(z&1)^ROTR64(&C,3)^ROTR64(&C,4)^A^ROTR64(&A,1); z>>=1;
     }
        rk[64]=A; A^=c^0^ROTR64(&D,3)^ROTR64(&D,4)^B^ROTR64(&B,1);
        rk[65]=B; B^=c^1^ROTR64(&A,3)^ROTR64(&A,4)^C^ROTR64(&C,1);
        rk[66]=C; C^=c^0^ROTR64(&B,3)^ROTR64(&B,4)^D^ROTR64(&D,1);
        rk[67]=D; D^=c^0^ROTR64(&C,3)^ROTR64(&C,4)^A^ROTR64(&A,1);
        rk[68]=A; 
        rk[69]=B; 
        rk[70]=C; 
        rk[71]=D;
}


/*
void Simon128128Encrypt(uint64_t  Pt[], uint64_t  Ct[], uint64_t  rk[])
{
int i;
Ct[0]=Pt[0]; Ct[1]=Pt[1];
for(i=0; i<68; i+=2) 
     R64x2(Ct[1], Ct[0], rk[i], rk[i+1]);
}


void Simon128128Decrypt(uint64_t  Pt[], uint64_t  Ct[],  uint64_t  rk[])
{
int i;
Pt[0]=Ct[0]; Pt[1]=Ct[1];
for(i=67;i>=0;i-=2) 
     R64x2(Pt[0], Pt[1], rk[i],  rk[i-1]);
}


*/




void simon_finish(simon_ctx_t **ctx) {
    if (!ctx) return;
    free((*ctx)->key_schedule);
    free(*ctx);
}

simon_ctx_t *simon_init(enum simon_encrypt_type type, uint8_t *key, int key_len) {
    if (key == NULL) return NULL;
    if (!is_validate_key_len(type, key_len)) return NULL;

    simon_ctx_t *ctx = (simon_ctx_t *)calloc(1, sizeof(simon_ctx_t));
    if (!ctx) return NULL;
    ctx->type = type;
    ctx->round = ROUNDS; // get_round_num(type);

    ctx->key_schedule = (uint64_t*)calloc(1, ROUNDS * sizeof(uint64_t));    /// this line has been changed by Ahmed
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
    
      switch (type) 
          {
           case simon_ENCRYPT_TYPE_128_128:
                Simon128128KeySchedule(key, ctx->key_schedule);
              break;
          case simon_ENCRYPT_TYPE_128_192:
                Simon128192KeySchedule(key, ctx->key_schedule);
              break;
          case simon_ENCRYPT_TYPE_128_256:
                Simon128256KeySchedule(key, ctx->key_schedule);
             break; 
          default:
            break;
       }
	  
    

   
    return ctx;
}

void generate_random_array(uint8_t *iv, size_t iv_len) {
std::random_device rd;  //Will be used to obtain a seed for the random number engine
std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
 std::uniform_int_distribution<> dis;

    for(int i=0; i<iv_len; i++) 
        iv[i] = static_cast<uint8_t>(dis(gen));
     
   //for(int i=0; i<iv_len; i++) 
     // iv[i] = rand();
    
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


__global__  void simon_ctr_encrypt(uint64_t *key_schedule,  uint64_t *in, uint64_t *out, int len, uint64_t *iv, int iv_len, enum simon_encrypt_type type) {
  //      printf("erarar %d %d\n");
  /*    if (len < 0) {
        return -1;
    }
  */

    int count = len/2;  //(len / (BLOCK_SIZE * LANE_NUM));
    //    printf("count %d\n",count);    
    //int remain_bytes = len % BLOCK_SIZE;
    int i;
    uint64_t crypted_iv_block[2];
    uint64_t plain_block[2];
    uint64_t  iv_block[2], t;
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
         
	R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[0], &key_schedule[1] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[2], &key_schedule[3] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[4], &key_schedule[5] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[6], &key_schedule[7] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[8], &key_schedule[9] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[10], &key_schedule[11] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[12], &key_schedule[13] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[14], &key_schedule[15] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[16], &key_schedule[17] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[18], &key_schedule[19] );
        
  
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[20], &key_schedule[21] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[22], &key_schedule[23] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[24], &key_schedule[25] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[26], &key_schedule[27] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[28], &key_schedule[29] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[30], &key_schedule[31] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[32], &key_schedule[33] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[34], &key_schedule[35] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[36], &key_schedule[37] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[38], &key_schedule[39] );
        
        
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[40], &key_schedule[41] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[42], &key_schedule[43] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[44], &key_schedule[45] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[46], &key_schedule[47] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[48], &key_schedule[49] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[50], &key_schedule[51] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[52], &key_schedule[53] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[54], &key_schedule[55] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[56], &key_schedule[57] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[58], &key_schedule[59] );
        
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[60], &key_schedule[61] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[62], &key_schedule[63] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[64], &key_schedule[65] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[66], &key_schedule[67] );
       
       /*
       switch (type) 
        {
         case simon_ENCRYPT_TYPE_128_192:
            t=crypted_iv_block[1];   
            crypted_iv_block[1]=crypted_iv_block[0]^f64(&crypted_iv_block[1])^key_schedule[68]; 
            crypted_iv_block[0]=t;
            break;
          case simon_ENCRYPT_TYPE_128_256:
            R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[68], &key_schedule[69] );
            R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[70], &key_schedule[71] );
          break; 
          default:
          break;
       }
	  
	*/
	
	//printf("iv bloc %d %u %u\n",i,iv_block[0],iv_block[1]);
	//printf("cr block %d %u %u\n",i,crypted_iv_block[0],crypted_iv_block[1]);
    __syncthreads(); 
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




__global__  void simon_ctr_encrypt4(uint64_t *key_schedule,  uint64_t *in, uint64_t *out, int len, uint64_t *iv, int iv_len, enum simon_encrypt_type type) {
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
     	  for (i = 0; i < 34; i++){
            iv_block[0]+=id;

	    crypted_iv_block[0] = iv_block[0];
	    crypted_iv_block[1] = iv_block[1]; 
            R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[i], &key_schedule[i+1] );
	    
	    
            iv_block[2]+=id;
            crypted_iv_block[2] = iv_block[2];
	    crypted_iv_block[3] = iv_block[3]; 
             R64x2(&crypted_iv_block[2],  &crypted_iv_block[3],  &key_schedule[i+2], &key_schedule[i+3]);
            
	  }

	out[4*id]=crypted_iv_block[0] ^ in[4*id];
	out[4*id+1]=crypted_iv_block[1] ^ in[4*id+1];
        out[4*id+2]=crypted_iv_block[2] ^ in[4*id+2];
	out[4*id+3]=crypted_iv_block[3] ^ in[4*id+3];
	
   }
}






void simon_ctr_encrypt_cpu(uint64_t *key_schedule, const uint64_t *in, uint64_t *out, int len, uint64_t *iv, int iv_len, enum simon_encrypt_type type)
 {

    int count = len/2;//(len / (BLOCK_SIZE * LANE_NUM));
    //    printf("count %d\n",count);    
    //int remain_bytes = len % BLOCK_SIZE;
    int i;
    uint64_t crypted_iv_block[2];
    uint64_t plain_block[2];
    uint64_t iv_block[2],t;
    iv_block[0]=iv[0];
    iv_block[1]=iv[1];
    

      //printf("key0= %u  key1= %u \n", &key_schedule[0], &key_schedule[1] );
      //  printf("len= %d     len_iv=%d \n", len , iv_len);
 for (i = 0; i < count; i++) 
   {   // replaced with thread id in gpu
   
      
      iv_block[0]+=i;

      
       //simon_encrypt(ctx, iv_block, crypted_iv_block);
	
	     crypted_iv_block[0] = iv_block[0];
	     crypted_iv_block[1] = iv_block[1];
     

   //    R64x2(Ct[1], Ct[0], rk[i], rk[i+1]);
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[0], &key_schedule[1] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[2], &key_schedule[3] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[4], &key_schedule[5] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[6], &key_schedule[7] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[8], &key_schedule[9] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[10], &key_schedule[11] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[12], &key_schedule[13] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[14], &key_schedule[15] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[16], &key_schedule[17] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[18], &key_schedule[19] );
        
  
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[20], &key_schedule[21] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[22], &key_schedule[23] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[24], &key_schedule[25] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[26], &key_schedule[27] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[28], &key_schedule[29] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[30], &key_schedule[31] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[32], &key_schedule[33] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[34], &key_schedule[35] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[36], &key_schedule[37] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[38], &key_schedule[39] );
        
        
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[40], &key_schedule[41] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[42], &key_schedule[43] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[44], &key_schedule[45] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[46], &key_schedule[47] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[48], &key_schedule[49] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[50], &key_schedule[51] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[52], &key_schedule[53] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[54], &key_schedule[55] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[56], &key_schedule[57] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[58], &key_schedule[59] );
        
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[60], &key_schedule[61] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[62], &key_schedule[63] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[64], &key_schedule[65] );
        R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[66], &key_schedule[67] );
       
       /* switch (type) 
         {
          case simon_ENCRYPT_TYPE_128_192:
              t=crypted_iv_block[1];   
              crypted_iv_block[1]=crypted_iv_block[0]^f64(&crypted_iv_block[1])^key_schedule[68]; 
              crypted_iv_block[0]=t;
           break;
           case simon_ENCRYPT_TYPE_128_256:
              R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[68], &key_schedule[69] );
              R64x2(&crypted_iv_block[1],  &crypted_iv_block[0],  &key_schedule[70], &key_schedule[71] );
           break; 
           default:
           break;
        }

*/
	 out[2*i]=crypted_iv_block[0] ^ in[2*i];
	 out[2*i+1]=crypted_iv_block[1] ^ in[2*i+1];
	
    }
}



int encrypt_decrypt_stream_test(int block_num,  enum simon_encrypt_type type,uint8_t  *s_key_stream, uint8_t key_s ) {
    int r = 0;
    
    /// define CPU variables 
    simon_ctx_t *ctx = NULL;
    uint8_t *plain_text_stream = NULL;
    uint8_t *crypted_text_stream = NULL;
    uint8_t *decrypted_text_stream = NULL;
    uint8_t *iv_text_stream = NULL;
    uint8_t *origin_iv_text_stream = NULL;
    int *bl1,*bl2;
    
    
    
    // Define GPU variable
   //   simon_ctx_t  *dev_ctx= NULL;
      
      uint64_t *dev_key_schedule=NULL;
      uint8_t *dev_plain_text_stream= NULL;
      uint8_t *dev_crypted_text_stream=NULL;
      uint8_t *dev_decrypted_text_stream= NULL;
      uint8_t *dev_iv_text_stream= NULL;
     //uint64_t *key;
    
    //cudaStream_t  streams[2];
    
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


    
    //allocate GPU variables
    cudaMalloc((void **)&dev_key_schedule, sizeof(uint64_t)*ROUNDS);
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

     ctx = simon_init(type, s_key_stream, key_s);
   //ctx->key_schedule = (uint64_t*)malloc(ROUNDS);
     // Simon128128KeySchedule(s_key_stream  , ctx->key_schedule); 
    if (!ctx) {
         r = 1;
         //  goto finish;
    }
    memcpy(iv_text_stream, origin_iv_text_stream, BLOCK_SIZE);
    

//cast_uint8_array_to_uint64(c_plain_text_stream,plain_text_stream);
/*
printf("\n Plain---\n");
 for (int i = 0; i < BLOCK_SIZE * block_num; i++) 
 printf("%02x   ",plain_text_stream[i]);
  printf("\n");
*/




 ////////////////////////////////////           simon on GPU     ////////////////////////////////////////////

 printf(" Run simon on GPU \n");
 float elapsed=0;
cudaEvent_t start, stop;

//copy all needed variables to device memory
cudaMemcpy(dev_key_schedule, ctx->key_schedule,sizeof(uint64_t)*ROUNDS, cudaMemcpyHostToDevice);
cudaMemcpy(dev_plain_text_stream, plain_text_stream, BLOCK_SIZE * block_num*sizeof(uint8_t), cudaMemcpyHostToDevice);
cudaMemcpy(dev_iv_text_stream, iv_text_stream,BLOCK_SIZE *sizeof(uint8_t), cudaMemcpyHostToDevice);





//StopWatchInterface *timer = 0;
   //sdkCreateTimer(&timer);
    //sdkResetTimer(&timer);
 cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

for(int i=0;i<100;i++)
{
      
 //sdkStartTimer(&timer);


simon_ctr_encrypt<<<gridSize, blockSize>>>( dev_key_schedule,  (uint64_t*)dev_plain_text_stream, (uint64_t*)dev_crypted_text_stream , BLOCK_SIZE * block_num/8 , (uint64_t*)dev_iv_text_stream,BLOCK_SIZE/8, type);   //encryption 


  cudaDeviceSynchronize();

  //sdkStopTimer(&timer);

}


cudaEventRecord(stop, 0);
    cudaEventSynchronize (stop);
    
cudaEventElapsedTime(&elapsed, start, stop);
float time_GPU =elapsed/100000;
cudaEventDestroy(start);
cudaEventDestroy(stop); 
printf("Size message %ld bytes\n",(long)BLOCK_SIZE * block_num);
printf("Elapsed time on GPU %f seconds \n ", time_GPU);
printf("Throughput on GPU %f Gbps\n ",(long) 8*BLOCK_SIZE * block_num/time_GPU/1e9);  


/*
cudaEventRecord(stop, 0);
cudaEventSynchronize (stop);

cudaEventElapsedTime(&elapsed, start, stop);
float time_GPU =elapsed/1000;
cudaEventDestroy(start);
cudaEventDestroy(stop);
*/

 //float reduceTime_GPU = sdkGetAverageTimerValue(&timer);
 //printf("Average time on GPU: %f seconds\n", reduceTime_GPU/1000);


 //copy cypher text from device to host
cudaMemcpy(crypted_text_stream, dev_crypted_text_stream , BLOCK_SIZE * block_num*sizeof(uint8_t), cudaMemcpyDeviceToHost );
 
 
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

simon_ctr_encrypt<<<gridSize, blockSize>>>( dev_key_schedule  ,  (uint64_t*)dev_crypted_text_stream, (uint64_t*)dev_decrypted_text_stream , BLOCK_SIZE * block_num/8 , (uint64_t*)dev_iv_text_stream, BLOCK_SIZE/8, type);   //decryption 
 
 

 //copy decryphed text from device to host
cudaMemcpy(decrypted_text_stream, dev_decrypted_text_stream , BLOCK_SIZE * block_num*sizeof(uint8_t), cudaMemcpyDeviceToHost );
   

//printf("Elapsea time on GPU %f seconds \n ", time_GPU);

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



 ////////////////////////////////////           simon on CPU     ////////////////////////////////////////////  
 
 /*
 printf(" Run simon on CPU \n");
 
// double time_CPU = 0.0;
//clock_t begin = clock();
 
 

//cudaEvent_t start, stop;

//cudaEventCreate(&start);

//cudaEventCreate(&stop);
//cudaEventRecord(start, 0);

sdkCreateTimer(&timer);
    sdkResetTimer(&timer);

    for(int i=0;i<100;i++) {
      sdkStartTimer(&timer);


 simon_ctr_encrypt_cpu(ctx->key_schedule, (uint64_t*)plain_text_stream,  (uint64_t*)crypted_text_stream, BLOCK_SIZE * block_num/8,  (uint64_t*)iv_text_stream, BLOCK_SIZE/8,type);   //encryption 



sdkStopTimer(&timer);


 }
    float time_CPU = sdkGetAverageTimerValue(&timer);
    printf("Average time on CPU: %f seconds\n", time_CPU/1000);



//clock_t end = clock();
//time_CPU += (double)(end - begin) / CLOCKS_PER_SEC;

//cudaEventRecord(stop, 0);
//cudaEventSynchronize (stop);

//cudaEventElapsedTime(&elapsed, start, stop);
//float time_CPU =elapsed;
//cudaEventDestroy(start);
//cudaEventDestroy(stop);


memcpy(iv_text_stream, origin_iv_text_stream, BLOCK_SIZE);

simon_ctr_encrypt_cpu(ctx->key_schedule, (uint64_t*)crypted_text_stream,  (uint64_t*)decrypted_text_stream, BLOCK_SIZE * block_num/8,  (uint64_t*)iv_text_stream, BLOCK_SIZE/8,type);   // decryption


printf("Elapsea time on CPU %f seconds \n", time_CPU);
    
  
    //show_array("plain", plain_text_stream, block_num * BLOCK_SIZE);
   // show_array("decrypted", decrypted_text_stream, block_num * BLOCK_SIZE);
    
    */
 /////////////////////     speedup   ////////////////////////////
    
//  printf("The speedup = %f \n",time_CPU / time_GPU);
 
 //   printf("The speedup = %f \n",(time_CPU/1000) / (reduceTime_GPU/1000));


   
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
   
    //   Release device memory
    //  cudaFree(dev_key_schedule);
      //cudaFree(dev_plain_text_stream);
     //cudaFree(dev_crypted_text_stream);
     // cudaFree(dev_decrypted_text_stream);
     //  cudaFree(dev_iv_text_stream); 
      //cudaFree(&dev_ctx);
       
    free(plain_text_stream);
    free(crypted_text_stream);
    free(decrypted_text_stream);
    free(iv_text_stream);
    free(origin_iv_text_stream);
    simon_finish(&ctx);
   // printf("okkkkkkkkkkkkk\n"); 
   return r;

}


 int main(int argc, char **argv ) 
 {


// cudaDeviceProp deviceProp = { 3 };
    //int dev;
    //dev = findCudaDevice(argc, (const char **)argv);
   // checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));


 int t, flag=1;
 
 enum simon_encrypt_type type;
uint8_t key_stream[32] = {
    0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
};

uint8_t *s_key_stream;

 // s_plain_text_stream=(uint8_t*)malloc(16);
//  s_cipher_text_stream=(uint8_t*)malloc(16);
  
 
 printf("Enter Encryption type:  \n");
  //printf("Enter 1 --> for simon_ENCRYPT_TYPE_128_128 \nEnter 2 --> for simon_ENCRYPT_TYPE_128_192 \nEnter 3 --> for simon_ENCRYPT_TYPE_128_256\n");
  uint8_t key_s;
//scanf("%d ", &t);
// t=3;


int keysize;

 if  ( (strncmp(argv[1],"keysize",7)==0)  && (strncmp(argv[3],"nblocks",7)==0)  )
 {
   keysize = atoi((argv[2]));
   printf("  %s   = %s\n", argv[1], argv[2]);
   
   long nblocks=atoi((argv[4]));
   printf("***************************\n");
   printf("SIMON_ENCRYPT_TYPE_128_%d\n",keysize);
   printf("***************************\n");


   switch (keysize) {
        case 128:
                 // s_key_stream=(uint8_t*)malloc(16);
                  type=simon_ENCRYPT_TYPE_128_128;
                  key_s=24;
                  ROUNDS=67;
                  s_key_stream=(uint8_t*)malloc(key_s);
                  for(int i=1; i<key_s; i++)
                      s_key_stream[i] = key_stream[i];
                  break;  
        case 192:
                 //s_key_stream=(uint8_t*)malloc(24);
                 type=simon_ENCRYPT_TYPE_128_192;
                  key_s=24;
                  ROUNDS=68;
                   s_key_stream=(uint8_t*)malloc(key_s);
                  for(int i=1; i<key_s; i++)
                      s_key_stream[i] = key_stream[i];
                 break; 
        case 256:
                 // s_key_stream=(uint8_t*)malloc(32);
                  type=simon_ENCRYPT_TYPE_128_256;
                   key_s=32;
                   ROUNDS=71;
                  s_key_stream=(uint8_t*)malloc(key_s);
                  for(int i=1; i<key_s; i++)
                      s_key_stream[i]=key_stream[i];
                  break; 
        default:
                flag=0;
                break; 
    }
    
 
   //printf(" type = %d \n", type);
 
    printf("test encrypt_decrypt_stream_test\n");
   if (flag) 
        {
      
           int r = encrypt_decrypt_stream_test(nblocks,type, key_stream, key_s);
            if(r != 0) 
             {
               return r;
              }
      
          printf("success encrypt_decrypt_stream_test\n");
    
    }
 else
       printf("Wrong simon Encrypted Type \n");
    
 } // end   if(strncmp(argv[1],"keysize",7)==0)
    
    free(s_key_stream);
    return 0;
}
