#ifndef AES_KERNEL_CU
#define AES_KERNEL_CU

class vec2
{
private:
   uint2 value;

public:

   __host__ __device__ explicit inline vec2()
   {
   }

   __host__ __device__ explicit inline vec2(const unsigned int val0, const unsigned int val1) 
   {
      value.x = val0;
      value.y = val1;
   }

   __host__ __device__ void inline assign(const unsigned int val0, const unsigned int val1) 
   {
      value.x = val0;
      value.y = val1;
   }

   __host__ __device__ void inline assignTo(unsigned int & __restrict__ val0, unsigned int & __restrict__ val1) const
   {
      val0 = value.x;
      val1 = value.y;
   }

   __host__ void inline assign(const unsigned int values[2])
   {
      this->value.x = values[0];
      this->value.y = values[1];
   }

   __host__ __device__ inline vec2 & operator=(const vec2 & rhs)
   {
      this->value.x = rhs.value.x;
      this->value.y = rhs.value.y;
      return *this;
   }

   __host__ __device__ inline volatile vec2 & operator=(const vec2 & rhs) volatile
   {
      this->value.x = rhs.value.x;
      this->value.y = rhs.value.y;
      return *this;
   }

   __host__ __device__ inline vec2 & operator^=(const vec2 & rhs)
   {
      this->value.x ^= rhs.value.x;
      this->value.y ^= rhs.value.y;
      return *this;
   }

   __host__ __device__ inline vec2 & operator^=(const volatile vec2 & __restrict__ rhs)
   {
      this->value.x ^= rhs.value.x;
      this->value.y ^= rhs.value.y;
      return *this;
   }

   __host__ __device__ inline vec2 & operator&=(const vec2 & rhs)
   {
      this->value.x &= rhs.value.x;
      this->value.y &= rhs.value.y;
      return *this;
   }

   __host__ __device__ inline vec2 & operator|=(const vec2 & rhs)
   {
      this->value.x |= rhs.value.x;
      this->value.y |= rhs.value.y;
      return *this;
   }

   __host__ __device__ friend void xor_equ(vec2 & dst, const vec2 & op1, const vec2 &op2);

   __host__ __device__ friend void or_equ(vec2 & dst, const vec2 & op1, const vec2 &op2);

   __host__ __device__ friend void and_equ(vec2 & dst, const vec2 & op1, const vec2 &op2);

   __host__ __device__ friend void swapmove(vec2 & __restrict__ a, vec2 & __restrict__ b, const unsigned int n, const vec2 & __restrict__ m, vec2 & __restrict__ t);

   __device__ friend void shufdw0x93(vec2 & dst, const vec2 & src, vec2 & __restrict__ tmp);

   __device__ friend void shufdw0x4E(vec2 & dst, const vec2 & src, vec2 & __restrict__ tmp);

   __device__ friend void shufbSR(vec2 & dst, const vec2 & src, vec2 & __restrict__ tmp);

   __device__ friend void shufbSRM0(vec2 & dst, const vec2 & src, vec2 & __restrict__ tmp);

   __host__ __device__ friend void shufbM0sl(vec2 & dst1, vec2 & dst2, const vec2 & src1, const vec2 & src2, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2);

   __host__ friend void shufbsl(vec2 & dst1, vec2 & dst2, const vec2 & src1, const vec2 & src2, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2);

   __device__ friend void shufbslrv(vec2 & dst1, vec2 & dst2, const vec2 & src1, const vec2 & src2, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2);
};

__host__ __device__ inline void xor_equ(vec2 & dst, const vec2 & op1, const vec2 &op2)
{
   dst.value.x = op1.value.x ^ op2.value.x;
   dst.value.y = op1.value.y ^ op2.value.y;
}

__host__ __device__ inline void or_equ(vec2 & dst, const vec2 & op1, const vec2 &op2)
{
   dst.value.x = op1.value.x | op2.value.x;
   dst.value.y = op1.value.y | op2.value.y;
}

__host__ __device__ inline void and_equ(vec2 & dst, const vec2 & op1, const vec2 &op2)
{
   dst.value.x = op1.value.x & op2.value.x;
   dst.value.y = op1.value.y & op2.value.y;
}

class vec4
{
private:
   uint4 value;

public:

   __host__ __device__ explicit inline vec4()
   {
   }

   __host__ __device__ inline vec4 & operator=(const vec4 & rhs)
   {
      this->value.x = rhs.value.x;
      this->value.y = rhs.value.y;
      this->value.z = rhs.value.z;
      this->value.w = rhs.value.w;
      return *this;
   }

   __host__ __device__ inline void assignTo(vec2 & __restrict__ op1, vec2 & __restrict__ op2) const
   {
      op1.assign(this->value.x, this->value.y);
      op2.assign(this->value.z, this->value.w);
   }

   __host__ __device__ inline void assignFrom(const vec2 & __restrict__ op1, const vec2 & __restrict__ op2)
   {
      op1.assignTo(this->value.x, this->value.y);
      op2.assignTo(this->value.z, this->value.w);
   }
};


__host__ __device__ inline void swapmove(vec2 & __restrict__ a, vec2 & __restrict__ b, const unsigned int n, const vec2 & __restrict__ m, vec2 & __restrict__ t)
{
   t = b;
   t.value.x >>= n;
   t.value.y >>= n;
   t ^= a;
   t &= m;
   a ^= t;
   t.value.x <<= n;
   t.value.y <<= n;
   b ^= t;
}

__host__ __device__ static inline void bitslice(vec2 & __restrict__ x0, vec2 & __restrict__ x1, vec2 & __restrict__ x2, vec2 & __restrict__ x3, vec2 & __restrict__ x4, vec2 & __restrict__ x5, vec2 & __restrict__ x6, vec2 & __restrict__ x7, vec2 & __restrict__ t)
{
   const vec2 BS0(0x55555555, 0x55555555);
   const vec2 BS1(0x33333333, 0x33333333);
   const vec2 BS2(0x0f0f0f0f, 0x0f0f0f0f);

   swapmove(x0, x1, 1, BS0, t);
   swapmove(x2, x3, 1, BS0, t);
   swapmove(x4, x5, 1, BS0, t);
   swapmove(x6, x7, 1, BS0, t);

   swapmove(x0, x2, 2, BS1, t);
   swapmove(x1, x3, 2, BS1, t);
   swapmove(x4, x6, 2, BS1, t);
   swapmove(x5, x7, 2, BS1, t);

   swapmove(x0, x4, 4, BS2, t);
   swapmove(x1, x5, 4, BS2, t);
   swapmove(x2, x6, 4, BS2, t);
   swapmove(x3, x7, 4, BS2, t);
}

__device__ inline void InBasisChange(vec2 & __restrict__ b0, vec2 & __restrict__ b1, vec2 & __restrict__ b2, vec2 & __restrict__ b3, vec2 & __restrict__ b4, vec2 & __restrict__ b5, vec2 & __restrict__ b6, vec2 & __restrict__ b7)
{
   b5 ^= b6;
   b2 ^= b1;
   b5 ^= b0;
   b6 ^= b2;
   b3 ^= b0;

   b6 ^= b3;
   b3 ^= b7;
   b3 ^= b4;
   b7 ^= b5;
   b3 ^= b1;

   b4 ^= b5;
   b2 ^= b7;
   b1 ^= b5;
}

__device__ inline void OutBasisChange(vec2 & __restrict__ b0, vec2 & __restrict__ b1, vec2 & __restrict__ b2, vec2 & __restrict__ b3, vec2 & __restrict__ b4, vec2 & __restrict__ b5, vec2 & __restrict__ b6, vec2 & __restrict__ b7)
{
   b0 ^= b6;
   b1 ^= b4;
   b2 ^= b0;
   b4 ^= b6;
   b6 ^= b1;

   b1 ^= b5;
   b5 ^= b3;
   b2 ^= b5;
   b3 ^= b7;
   b7 ^= b5;

   b4 ^= b7;
}

__device__ inline void Mul_GF4(vec2 & __restrict__ x0, vec2 & __restrict__ x1, const vec2 & __restrict__ y0, const vec2 & __restrict__ y1, vec2 & __restrict__ t0)
{
   xor_equ(t0, y0, y1);
   t0 &= x0;
   x0 ^= x1;
   x0 &= y1;
   x1 &= y0;
   x0 ^= x1;
   x1 ^= t0;
}

__device__ inline void Mul_GF4_com(vec2 & __restrict__ x0, vec2 & __restrict__ x1, const vec2 & __restrict__ y0, const vec2 & __restrict__ y1, const vec2 & __restrict__ y0y1, vec2 & __restrict__ t0)
{
   and_equ(t0, y0y1, x0);
   x0 ^= x1;
   x0 &= y1;
   x1 &= y0;
   x0 ^= x1;
   x1 ^= t0;
}

__device__ inline void Mul_GF4_N(vec2 & __restrict__ x0, vec2 & __restrict__ x1, const vec2 & __restrict__ y0, const vec2 & __restrict__ y1, vec2 & __restrict__ t0)
{
   xor_equ(t0, y0, y1);
   t0 &= x0;
   x0 ^= x1;
   x0 &= y1;
   x1 &= y0;
   x1 ^= x0;
   x0 ^= t0;
}

__device__ inline void Mul_GF16_2(vec2 & __restrict__ x0, vec2 & __restrict__ x1, vec2 & __restrict__ x2, vec2 & __restrict__ x3, vec2 & __restrict__ x4, vec2 & __restrict__ x5, vec2 & __restrict__ x6, vec2 & __restrict__ x7, vec2 & __restrict__ y0, vec2 & __restrict__ y1, vec2 & __restrict__ y2, vec2 & __restrict__ y3, vec2 & __restrict__ t0, vec2 & __restrict__ t1, vec2 & __restrict__ t2, vec2 & __restrict__ t3, vec2 & __restrict__ tmp0, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2)
{
   xor_equ(t0, x0, x2);
   xor_equ(t1, x1, x3);
   xor_equ(tmp2, y0, y1);
   Mul_GF4_com(x0, x1, y0, y1, tmp2, t2);
   xor_equ(tmp0, y0, y2);
   xor_equ(tmp1, y1, y3);
   Mul_GF4_N(t0, t1, tmp0, tmp1, t3);
   Mul_GF4(x2, x3, y2, y3, t2);

   x0 ^= t0;
   x2 ^= t0;
   x1 ^= t1;
   x3 ^= t1;

   xor_equ(t0, x4, x6);
   xor_equ(t1, x5, x7);
   Mul_GF4_N(t0, t1, tmp0, tmp1, t3);
   Mul_GF4(x6, x7, y2, y3, t2);
   Mul_GF4_com(x4, x5, y0, y1, tmp2, t3);

   x4 ^= t0;
   x6 ^= t0;
   x5 ^= t1;
   x7 ^= t1;
}

__device__ inline void Inv_GF256(vec2 & __restrict__ x0, vec2 & __restrict__ x1, vec2 & __restrict__ x2, vec2 & __restrict__ x3, vec2 & __restrict__ x4, vec2 & __restrict__ x5, vec2 & __restrict__ x6, vec2 & __restrict__ x7, vec2 & __restrict__ t0, vec2 & __restrict__ t1, vec2 & __restrict__ t2, vec2 & __restrict__ t3, vec2 & __restrict__ s0, vec2 & __restrict__ s1, vec2 & __restrict__ s2, vec2 & __restrict__ s3, vec2 & __restrict__ tmp0, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2)
{
   xor_equ(t3, x4, x6);
   xor_equ(t2, x5, x7);
   xor_equ(t1, x1, x3);
   xor_equ(s1, x7, x6);
   xor_equ(s0, x0, x2);

   xor_equ(s3, t3, t2);
   and_equ(s2, t3, s0);
   and_equ(t0, t2, t1);

   t2 |= t1;
   t3 |= s0;
   s0 ^= t1;
   s3 &= s0;
   xor_equ(s0, x3, x2);
   s1 &= s0;
   t3 ^= s1;
   t2 ^= s1;
   xor_equ(s1, x4, x5);
   xor_equ(s0, x1, x0);
   or_equ(t1, s1, s0);
   s1 &= s0;
   t0 ^= s1;
   t3 ^= s3;
   t2 ^= s2;
   t1 ^= s3;
   t0 ^= s2;
   t1 ^= s2;
   and_equ(s0, x7, x3);
   and_equ(s1, x6, x2);
   and_equ(s2, x5, x1);
   or_equ(s3, x4, x0);
   t3 ^= s0;
   t2 ^= s1;
   t1 ^= s2;
   t0 ^= s3;

   xor_equ(s0, t3, t2);
   t3 &= t1;
   xor_equ(s2, t0, t3);
   and_equ(s3, s0, s2);
   s3 ^= t2;
   xor_equ(s1, t1, t0);
   t3 ^= t2;
   s1 &= t3;
   s1 ^= t0;
   t1 ^= s1;
   xor_equ(t2, s2, s1);
   t2 &= t0;
   t1 ^= t2;
   s2 ^= t2;
   s2 &= s3;
   s2 ^= s0;

   Mul_GF16_2(x0, x1, x2, x3, x4, x5, x6, x7, s3, s2, s1, t1, s0, t0, t2, t3, tmp0, tmp1, tmp2);
}

__device__ inline void sbox(vec2 & __restrict__ b0, vec2 & __restrict__ b1, vec2 & __restrict__ b2, vec2 & __restrict__ b3, vec2 & __restrict__ b4, vec2 & __restrict__ b5, vec2 & __restrict__ b6, vec2 & __restrict__ b7, vec2 & __restrict__ t0, vec2 & __restrict__ t1, vec2 & __restrict__ t2, vec2 & __restrict__ t3, vec2 & __restrict__ s0, vec2 & __restrict__ s1, vec2 & __restrict__ s2, vec2 & __restrict__ s3, vec2 & __restrict__ tmp0, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2)
{
   InBasisChange(b0, b1, b2, b3, b4, b5, b6, b7);
   Inv_GF256(b6, b5, b0, b3, b7, b1, b4, b2, t0, t1, t2, t3, s0, s1, s2, s3, tmp0, tmp1, tmp2);
   OutBasisChange(b7, b1, b4, b2, b6, b5, b0, b3);
}

//0x93 - 10 01 00 11

__device__ inline void shufdw0x93(vec2 & dst, const vec2 & src, vec2 & __restrict__ tmp)
{
   tmp = src;

   dst.value.x = __byte_perm(tmp.value.x, tmp.value.y, 0x1076);
   dst.value.y = __byte_perm(tmp.value.x, tmp.value.y, 0x5432);
}

//0x4E - 01 00 11 10
__device__ inline void shufdw0x4E(vec2 & dst, const vec2 & src, vec2 & __restrict__ tmp)
{
   tmp = src;

   dst.value.x = tmp.value.y;
   dst.value.y = tmp.value.x;
}

__device__ inline void mixcolumns(vec2 & __restrict__ x0, vec2 & __restrict__ x1, vec2 & __restrict__ x2, vec2 & __restrict__ x3, vec2 & __restrict__ x4, vec2 & __restrict__ x5, vec2 & __restrict__ x6, vec2 & __restrict__ x7, vec2 & __restrict__ t0, vec2 & __restrict__ t1, vec2 & __restrict__ t2, vec2 & __restrict__ t3, vec2 & __restrict__ t4, vec2 & __restrict__ t5, vec2 & __restrict__ t6, vec2 & __restrict__ t7, vec2 & __restrict__ tmp)
{
   shufdw0x93(t0, x0, tmp);
   shufdw0x93(t1, x1, tmp);
   shufdw0x93(t2, x2, tmp);
   shufdw0x93(t3, x3, tmp);
   shufdw0x93(t4, x4, tmp);
   shufdw0x93(t5, x5, tmp);
   shufdw0x93(t6, x6, tmp);
   shufdw0x93(t7, x7, tmp);

   x0 ^= t0;
   x1 ^= t1;
   x2 ^= t2;
   x3 ^= t3;
   x4 ^= t4;
   x5 ^= t5;
   x6 ^= t6;
   x7 ^= t7;

   t0 ^= x7;
   t1 ^= x0;
   t2 ^= x1;
   t1 ^= x7;
   t3 ^= x2;
   t4 ^= x3;
   t5 ^= x4;
   t3 ^= x7;
   t6 ^= x5;
   t7 ^= x6;
   t4 ^= x7;

   shufdw0x4E(x0, x0, tmp);
   shufdw0x4E(x1, x1, tmp);
   shufdw0x4E(x2, x2, tmp);
   shufdw0x4E(x3, x3, tmp);
   shufdw0x4E(x4, x4, tmp);
   shufdw0x4E(x5, x5, tmp);
   shufdw0x4E(x6, x6, tmp);
   shufdw0x4E(x7, x7, tmp);

   t0 ^= x0;
   t1 ^= x1;
   t2 ^= x2;
   t3 ^= x3;
   t4 ^= x4;
   t5 ^= x5;
   t6 ^= x6;
   t7 ^= x7;
}

__device__ __forceinline__ unsigned int bfe_func(const unsigned int x, const unsigned int startBit, const unsigned int numBits)
{
   unsigned int ret;
   asm("bfe.u32 %0, %1, %2, %3;" : "=r"(ret) : "r"(x), "r"(startBit), "r"(numBits) );
   return ret;
}

__device__ __forceinline__ void bfi_func(unsigned int & dst, const unsigned int op, const unsigned int startBit, const unsigned int numBits)
{
   asm("bfi.b32 %0, %1, %0, %2, %3;" : "+r"(dst) : "r"(op), "r"(startBit), "r"(numBits) );
}


__device__ inline unsigned short covtToShort(const unsigned int val)
{
   return static_cast<unsigned short>(val);
}

__device__ inline unsigned int rotateL(const unsigned int val, const unsigned int numBits)
{
   return ( (val << numBits) + (val >> (16 - numBits)) );
}

__device__ inline unsigned int rotateR(const unsigned int val, const unsigned int numBits)
{
   return ( (val >> numBits) + (val << (16 - numBits)) );
}

//SR: .quad 0x05 04 07 06 00 03 02 01, 0x0f 0e 0d 0c 0a 09 08 0b

__device__ inline void shufbSR(vec2 & dst, const vec2 & src, vec2 & __restrict__ tmp)
{
   tmp = src;

   dst.value.x = ((__byte_perm(tmp.value.x, tmp.value.y, 0x5401) & 0xF0F0F0F) << 4);
   dst.value.x += ((__byte_perm(tmp.value.x, tmp.value.y, 0x4510) & 0xF0F0F0F0) >> 4);

   dst.value.y = dst.value.x;

   dst.value.x = __byte_perm(dst.value.x, tmp.value.x, 0x6710);
   dst.value.y = __byte_perm(dst.value.y, tmp.value.y, 0x7632);
}

//SRM0:	.quad 0x03 04 09 0e 00 05 0a 0f, 0x01 06 0b 0c 02 07 08 0d

__device__ inline void shufbSRM0(vec2 & dst, const vec2 & src, vec2 & __restrict__ tmp)
{
   tmp = src;

   dst.value.x = ((__byte_perm(tmp.value.x, tmp.value.y, 0x1405) & 0xF0F0F0F) << 4);
   dst.value.y = (__byte_perm(tmp.value.x, tmp.value.y, 0x0514) & 0xF0F0F0F0);
   
   dst.value.x += ((__byte_perm(tmp.value.x, tmp.value.y, 0x3627) & 0xF0F0F0F0) >> 4);
   dst.value.y |= (__byte_perm(tmp.value.x, tmp.value.y, 0x3627) & 0xF0F0F0F);

   tmp = dst;

   dst.value.x = __byte_perm(tmp.value.x, tmp.value.y, 0x5410);
   dst.value.y = __byte_perm(tmp.value.x, tmp.value.y, 0x7632);
}

__host__ static inline unsigned int selectByte(const unsigned int input, const unsigned int byteNum, const unsigned int outByteLoc)
{
   return ( ((input >> (byteNum * 8) ) & 0xFF) << (outByteLoc * 8) );
}

//M0:  .quad 0x02 06 0a 0e 03 07 0b 0f, 0x00 04 08 0c 01 05 09 0d

__host__ __device__ inline void shufbM0sl(vec2 & dst1, vec2 & dst2, const vec2 & src1, const vec2 & src2, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2)
{
   tmp1 = src1;
   tmp2 = src2;

#ifdef __CUDA_ARCH__
   dst1.value.x = __byte_perm(tmp1.value.y, tmp2.value.y, 0x2637);
   dst1.value.y = __byte_perm(tmp1.value.y, tmp2.value.y, 0x0415);

   dst2.value.x = __byte_perm(tmp1.value.x, tmp2.value.x, 0x2637);
   dst2.value.y = __byte_perm(tmp1.value.x, tmp2.value.x, 0x0415);
#else

   dst1.value.x = selectByte(tmp2.value.y, 3, 0) | selectByte(tmp1.value.y, 3, 1) | selectByte(tmp2.value.y, 2, 2) | selectByte(tmp1.value.y, 2, 3);   
   dst1.value.y = selectByte(tmp2.value.y, 1, 0) | selectByte(tmp1.value.y, 1, 1) | selectByte(tmp2.value.y, 0, 2) | selectByte(tmp1.value.y, 0, 3);

   dst2.value.x = selectByte(tmp2.value.x, 3, 0) | selectByte(tmp1.value.x, 3, 1) | selectByte(tmp2.value.x, 2, 2) | selectByte(tmp1.value.x, 2, 3);   
   dst2.value.y = selectByte(tmp2.value.x, 1, 0) | selectByte(tmp1.value.x, 1, 1) | selectByte(tmp2.value.x, 0, 2) | selectByte(tmp1.value.x, 0, 3);
#endif
}

__host__ inline void shufbsl(vec2 & dst1, vec2 & dst2, const vec2 & src1, const vec2 & src2, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2)
{
   tmp1 = src1;
   tmp2 = src2;

   dst1.value.x = selectByte(tmp1.value.x, 0, 0) | selectByte(tmp1.value.x, 2, 1) | selectByte(tmp1.value.y, 0, 2) | selectByte(tmp1.value.y, 2, 3);   
   dst1.value.y = selectByte(tmp2.value.x, 0, 0) | selectByte(tmp2.value.x, 2, 1) | selectByte(tmp2.value.y, 0, 2) | selectByte(tmp2.value.y, 2, 3);   

   dst2.value.x = selectByte(tmp1.value.x, 1, 0) | selectByte(tmp1.value.x, 3, 1) | selectByte(tmp1.value.y, 1, 2) | selectByte(tmp1.value.y, 3, 3);   
   dst2.value.y = selectByte(tmp2.value.x, 1, 0) | selectByte(tmp2.value.x, 3, 1) | selectByte(tmp2.value.y, 1, 2) | selectByte(tmp2.value.y, 3, 3); 
}

__device__ inline void shufbslrv(vec2 & dst1, vec2 & dst2, const vec2 & src1, const vec2 & src2, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2)
{
   tmp1 = src1;
   tmp2 = src2;

   dst1.value.x = __byte_perm(tmp1.value.x, tmp2.value.x, 0x5140);
   dst1.value.y = __byte_perm(tmp1.value.x, tmp2.value.x, 0x7362);

   dst2.value.x = __byte_perm(tmp1.value.y, tmp2.value.y, 0x5140);
   dst2.value.y = __byte_perm(tmp1.value.y, tmp2.value.y, 0x7362);
}


__device__ inline void roundkey(vec2 & __restrict__ x0, vec2 & __restrict__ x1, vec2 & __restrict__ x2, vec2 & __restrict__ x3, vec2 & __restrict__ x4, vec2 & __restrict__ x5, vec2 & __restrict__ x6, vec2 & __restrict__ x7, const unsigned int i, const volatile vec2 * __restrict__ bskey)
{
   const volatile vec2 & __restrict__ bskey0 = bskey[8 * (i - 1)];
   const volatile vec2 & __restrict__ bskey1 = bskey[8 * (i - 1) + 1];
   const volatile vec2 & __restrict__ bskey2 = bskey[8 * (i - 1) + 2];
   const volatile vec2 & __restrict__ bskey3 = bskey[8 * (i - 1) + 3];
   const volatile vec2 & __restrict__ bskey4 = bskey[8 * (i - 1) + 4];
   const volatile vec2 & __restrict__ bskey5 = bskey[8 * (i - 1) + 5];
   const volatile vec2 & __restrict__ bskey6 = bskey[8 * (i - 1) + 6];
   const volatile vec2 & __restrict__ bskey7 = bskey[8 * (i - 1) + 7];
   x0 ^= bskey0;
   x1 ^= bskey1;
   x2 ^= bskey2;
   x3 ^= bskey3;
   x4 ^= bskey4;
   x5 ^= bskey5;
   x6 ^= bskey6;
   x7 ^= bskey7;
}

__device__ inline void shiftrows1(vec2 & __restrict__ x0, vec2 & __restrict__ x1, vec2 & __restrict__ x2, vec2 & __restrict__ x3, vec2 & __restrict__ x4, vec2 & __restrict__ x5, vec2 & __restrict__ x6, vec2 & __restrict__ x7, vec2 & __restrict__ tmp)
{
   shufbSR(x0, x0, tmp);
   shufbSR(x1, x1, tmp);
   shufbSR(x2, x2, tmp);
   shufbSR(x3, x3, tmp);
   shufbSR(x4, x4, tmp);
   shufbSR(x5, x5, tmp);
   shufbSR(x6, x6, tmp);
   shufbSR(x7, x7, tmp);
}

__device__ inline void shiftrows2(vec2 & __restrict__ x0, vec2 & __restrict__ x1, vec2 & __restrict__ x2, vec2 & __restrict__ x3, vec2 & __restrict__ x4, vec2 & __restrict__ x5, vec2 & __restrict__ x6, vec2 & __restrict__ x7, vec2 & __restrict__ tmp)
{
   shufbSRM0(x0, x0, tmp);
   shufbSRM0(x1, x1, tmp);
   shufbSRM0(x2, x2, tmp);
   shufbSRM0(x3, x3, tmp);
   shufbSRM0(x4, x4, tmp);
   shufbSRM0(x5, x5, tmp);
   shufbSRM0(x6, x6, tmp);
   shufbSRM0(x7, x7, tmp);
}

__device__ inline void preround(vec2 & __restrict__ b0, vec2 & __restrict__ b1, vec2 & __restrict__ b2, vec2 & __restrict__ b3, vec2 & __restrict__ b4, vec2 & __restrict__ b5, vec2 & __restrict__ b6, vec2 & __restrict__ b7, const volatile vec2 * __restrict__ bskey)
{
   roundkey(b0, b1, b2, b3, b4, b5, b6, b7, 1, bskey);
}

__device__ inline void aesround(const unsigned int i, vec2 & __restrict__ b0, vec2 & __restrict__ b1, vec2 & __restrict__ b2, vec2 & __restrict__ b3, vec2 & __restrict__ b4, vec2 & __restrict__ b5, vec2 & __restrict__ b6, vec2 & __restrict__ b7, vec2 & __restrict__ t0, vec2 & __restrict__ t1, vec2 & __restrict__ t2, vec2 & __restrict__ t3, vec2 & __restrict__ t4, vec2 & __restrict__ t5, vec2 & __restrict__ t6, vec2 & __restrict__ t7, const volatile vec2 * __restrict__ bskey, vec2 & __restrict__ tmp0, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2)
{
   sbox(b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, tmp0, tmp1, tmp2);
   shiftrows1(b0, b1, b2, b3, b4, b5, b6, b7, tmp0);
   mixcolumns(b0, b1, b4, b6, b3, b7, b2, b5, t0, t1, t2, t3, t4, t5, t6, t7, tmp0);
   roundkey(t0, t1, t2, t3, t4, t5, t6, t7, i, bskey);
}

__device__ inline void lastround(vec2 & __restrict__ b0, vec2 & __restrict__ b1, vec2 & __restrict__ b2, vec2 & __restrict__ b3, vec2 & __restrict__ b4, vec2 & __restrict__ b5, vec2 & __restrict__ b6, vec2 & __restrict__ b7, vec2 & __restrict__ t0, vec2 & __restrict__ t1, vec2 & __restrict__ t2, vec2 & __restrict__ t3, vec2 & __restrict__ t4, vec2 & __restrict__ t5, vec2 & __restrict__ t6, vec2 & __restrict__ t7, const volatile vec2 * __restrict__ bskey, vec2 & __restrict__ tmp0, vec2 & __restrict__ tmp1, vec2 & __restrict__ tmp2)
{
   sbox(b0, b1, b2, b3, b4, b5, b6, b7, t0, t1, t2, t3, t4, t5, t6, t7, tmp0, tmp1, tmp2);
   shiftrows2(b0, b1, b2, b3, b4, b5, b6, b7, tmp0);
   roundkey(b0, b1, b4, b6, b3, b7, b2, b5, 11, bskey);
}

__global__ void AES_kernel(vec2 * __restrict__ dst, const vec2 * __restrict__ src, const unsigned int count, const vec2 * __restrict__ bskey)
{
   vec2 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7;
   vec2 xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15;
   vec2 tmp1, tmp2, tmp3;
   __shared__ volatile vec2 shared_bskey[88];
   vec4 * dst4 = reinterpret_cast<vec4 *>(dst);
   const vec4 * src4 = reinterpret_cast<const vec4 *>(src);
	 //RAPH
	 //	 	 #define SHARE_LOAD

#ifdef SHARE_LOAD
   __shared__ vec2 shared_temp[4360];
   vec2  values[8];
   const int      bid = blockDim.x * blockIdx.x;
   const int warp_idx = threadIdx.x >> 5;
   const int warp_tid = threadIdx.x & 31;
#endif
   vec4  values4[4];
   const int      tid = blockDim.x * blockIdx.x + threadIdx.x;
   const int THREAD_N = blockDim.x * gridDim.x;
   int i, j, k;
#ifdef SHARE_LOAD
   unsigned int count2;
#endif

   if( (count & 0x7) != 0)
   {
      asm("trap;");
   }
   for(i = threadIdx.x; i < 88; i+= blockDim.x)
   {
      shared_bskey[i] = bskey[i];
   }
   __syncthreads();

#ifdef SHARE_LOAD
   count2 = (count / (blockDim.x * 8) ) * (blockDim.x * 8);
   for(i = bid * 8; i < count2; i += THREAD_N * 8)
#else
   for(i = 4 * tid; i < count / 2; i += THREAD_N * 4)
#endif
   {

#ifdef SHARE_LOAD
      for(j = 0; j < 8; j++)
      {
         values[j] = src[i + (warp_idx << 8) + (j << 5) + warp_tid];
      }
      for(j = 0; j < 8; j++)
      {
         shared_temp[(warp_idx << 8) + (j << 5) + warp_tid + (warp_tid >> 4) + (j << 1) + (warp_idx << 4)] = values[j];
      }
      
      xmm0 = shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1)];
      xmm4 = shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 1];
      xmm1 = shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 2];
      xmm5 = shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 3];
      xmm2 = shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 4];
      xmm6 = shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 5];
      xmm3 = shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 6];
      xmm7 = shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 7];
#else
      for(j = 0; j < 4; j++)
      {
         values4[j] = src4[i + j];
      }
      values4[0].assignTo(xmm0, xmm4);
      values4[1].assignTo(xmm1, xmm5);
      values4[2].assignTo(xmm2, xmm6);
      values4[3].assignTo(xmm3, xmm7);
#endif
      shufbM0sl(xmm0, xmm4, xmm0, xmm4, tmp1, tmp2);
      shufbM0sl(xmm1, xmm5, xmm1, xmm5, tmp1, tmp2);
      shufbM0sl(xmm2, xmm6, xmm2, xmm6, tmp1, tmp2);
      shufbM0sl(xmm3, xmm7, xmm3, xmm7, tmp1, tmp2);

     
      bitslice(xmm7, xmm6, xmm5, xmm4, xmm3, xmm2, xmm1, xmm0, xmm8);

      preround(xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, shared_bskey);

      #pragma unroll 1
      for(k = 2; k <= 8; k+=2)
      {
         aesround(k, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15, shared_bskey, tmp1, tmp2, tmp3);
         aesround(k + 1, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, shared_bskey, tmp1, tmp2, tmp3);
      }
      aesround(10, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15, shared_bskey, tmp1, tmp2, tmp3);

 

      lastround(xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, shared_bskey, tmp1, tmp2, tmp3);

      bitslice(xmm13, xmm10, xmm15, xmm11, xmm14, xmm12, xmm9, xmm8, xmm0);

      shufbslrv(xmm8, xmm11, xmm8, xmm11, tmp1, tmp2);
      shufbslrv(xmm9, xmm15, xmm9, xmm15, tmp1, tmp2);
      shufbslrv(xmm12, xmm10, xmm12, xmm10, tmp1, tmp2);
      shufbslrv(xmm14, xmm13, xmm14, xmm13, tmp1, tmp2);

#ifdef SHARE_LOAD
      shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1)] = xmm8;
      shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 1] = xmm11;
      shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 2] = xmm9;
      shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 3] = xmm15;
      shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 4] = xmm12;
      shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 5] = xmm10;
      shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 6] = xmm14;
      shared_temp[(warp_idx << 8) + (warp_idx << 4) + (warp_tid << 3) + (warp_tid >> 1) + 7] = xmm13;

      for(j = 0; j < 8; j++)
      {
         values[j] = shared_temp[(warp_idx << 8) + (j << 5) + warp_tid + (warp_tid >> 4) + (j << 1) + (warp_idx << 4)];
      }
      for(j = 0; j < 8; j++)
      {
         dst[i + (warp_idx << 8) + (j << 5) + warp_tid] = values[j];
      }
#else
      values4[0].assignFrom(xmm8, xmm11);
      values4[1].assignFrom(xmm9, xmm15);
      values4[2].assignFrom(xmm12, xmm10);
      values4[3].assignFrom(xmm14, xmm13);
      for(j = 0; j < 4; j++)
      {
         dst4[i + j] = values4[j];
      }
#endif
   }

#ifdef SHARE_LOAD
   for(i = count2 / 2 + 4 * tid; i < count / 2; i += THREAD_N * 4)
   {
      for(j = 0; j < 4; j++)
      {
         values4[j] = src4[i + j];
      }
      values4[0].assignTo(xmm0, xmm4);
      values4[1].assignTo(xmm1, xmm5);
      values4[2].assignTo(xmm2, xmm6);
      values4[3].assignTo(xmm3, xmm7);

      shufbM0sl(xmm0, xmm4, xmm0, xmm4, tmp1, tmp2);
      shufbM0sl(xmm1, xmm5, xmm1, xmm5, tmp1, tmp2);
      shufbM0sl(xmm2, xmm6, xmm2, xmm6, tmp1, tmp2);
      shufbM0sl(xmm3, xmm7, xmm3, xmm7, tmp1, tmp2);

      bitslice(xmm7, xmm6, xmm5, xmm4, xmm3, xmm2, xmm1, xmm0, xmm8);

      preround(xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, shared_bskey);

      for(k = 2; k <= 8; k+=2)
      {
         aesround(k, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15, shared_bskey, tmp1, tmp2, tmp3);
         aesround(k + 1, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, shared_bskey, tmp1, tmp2, tmp3);
      }
      aesround(10, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15, shared_bskey, tmp1, tmp2, tmp3);

      lastround(xmm8, xmm9, xmm10, xmm11, xmm12, xmm13, xmm14, xmm15, xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, shared_bskey, tmp1, tmp2, tmp3);


      bitslice(xmm13, xmm10, xmm15, xmm11, xmm14, xmm12, xmm9, xmm8, xmm0);

      shufbslrv(xmm8, xmm11, xmm8, xmm11, tmp1, tmp2);
      shufbslrv(xmm9, xmm15, xmm9, xmm15, tmp1, tmp2);
      shufbslrv(xmm12, xmm10, xmm12, xmm10, tmp1, tmp2);
      shufbslrv(xmm14, xmm13, xmm14, xmm13, tmp1, tmp2);

      values4[0].assignFrom(xmm8, xmm11);
      values4[1].assignFrom(xmm9, xmm15);
      values4[2].assignFrom(xmm12, xmm10);
      values4[3].assignFrom(xmm14, xmm13);
      for(j = 0; j < 4; j++)
      {
         dst4[i + j] = values4[j];
      }
   }
#endif
}

__host__ static inline void bitslicekey0(const vec2 & __restrict__ key1, const vec2 & __restrict__ key2, vec2 * __restrict__ bskey)
{
   vec2 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
   vec2 tmp1, tmp2;
   xmm0 = key1;
   xmm4 = key2;
   shufbM0sl(xmm0, xmm4, xmm0, xmm4, tmp1, tmp2);
   xmm1 = xmm0;
   xmm2 = xmm0;
   xmm3 = xmm0;
   xmm5 = xmm4;
   xmm6 = xmm4;
   xmm7 = xmm4;

   bitslice(xmm7, xmm6, xmm5, xmm4, xmm3, xmm2, xmm1, xmm0, xmm8);

   bskey[0] = xmm0;
   bskey[1] = xmm1;
   bskey[2] = xmm2;
   bskey[3] = xmm3;
   bskey[4] = xmm4;
   bskey[5] = xmm5;
   bskey[6] = xmm6;
   bskey[7] = xmm7;
}

__host__ static inline void bitslicekey10(const vec2 & __restrict__ key1, const vec2 & __restrict__ key2, vec2 * __restrict__ bskey)
{
   const vec2 ONE(0xffffffff, 0xffffffff);
   vec2 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
   vec2 tmp1, tmp2;
   xmm0 = key1;
   xmm4 = key2;
   shufbsl(xmm0, xmm4, xmm0, xmm4, tmp1, tmp2);
   xmm1 = xmm0;
   xmm2 = xmm0;
   xmm3 = xmm0;
   xmm5 = xmm4;
   xmm6 = xmm4;
   xmm7 = xmm4;

   bitslice(xmm7, xmm6, xmm5, xmm4, xmm3, xmm2, xmm1, xmm0, xmm8);

   xmm6 ^= ONE;
   xmm5 ^= ONE;
   xmm1 ^= ONE;
   xmm0 ^= ONE;

   bskey[0] = xmm0;
   bskey[1] = xmm1;
   bskey[2] = xmm2;
   bskey[3] = xmm3;
   bskey[4] = xmm4;
   bskey[5] = xmm5;
   bskey[6] = xmm6;
   bskey[7] = xmm7;
}

__host__ static inline void bitslicekey(const unsigned int i, const vec2 & __restrict__ key1, const vec2 & __restrict__ key2, vec2 * __restrict__ bskey)
{
   const vec2 ONE(0xffffffff, 0xffffffff);
   vec2 xmm0, xmm1, xmm2, xmm3, xmm4, xmm5, xmm6, xmm7, xmm8;
   vec2 tmp1, tmp2;
   xmm0 = key1;
   xmm4 = key2;
   shufbM0sl(xmm0, xmm4, xmm0, xmm4, tmp1, tmp2);
   xmm1 = xmm0;
   xmm2 = xmm0;
   xmm3 = xmm0;
   xmm5 = xmm4;
   xmm6 = xmm4;
   xmm7 = xmm4;

   bitslice(xmm7, xmm6, xmm5, xmm4, xmm3, xmm2, xmm1, xmm0, xmm8);

   xmm6 ^= ONE;
   xmm5 ^= ONE;
   xmm1 ^= ONE;
   xmm0 ^= ONE;

   bskey[0 + 8 * i] = xmm0;
   bskey[1 + 8 * i] = xmm1;
   bskey[2 + 8 * i] = xmm2;
   bskey[3 + 8 * i] = xmm3;
   bskey[4 + 8 * i] = xmm4;
   bskey[5 + 8 * i] = xmm5;
   bskey[6 + 8 * i] = xmm6;
   bskey[7 + 8 * i] = xmm7;
}

__host__ static inline void byteRotateLeft8(unsigned char value[4])
{
   unsigned char tmp;
   
   tmp = value[0];
   value[0] = value[1];
   value[1] = value[2];
   value[2] = value[3];
   value[3] = tmp;
}

__host__ static inline void AES_keyGen(const unsigned char key[16], unsigned char subkey[176])
{
   static const unsigned char SBox[256] = 
   {0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76, 
    0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0, 
    0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15, 
    0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75, 
    0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84, 
    0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf, 
    0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8, 
    0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2, 
    0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73, 
    0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb, 
    0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79, 
    0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08, 
    0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a, 
    0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e, 
    0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf, 
    0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16};

   unsigned int i, j, k, counter;
   unsigned int rcon = 1;
   static const unsigned char RCon[15] = {0x8d,0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x80,0x1b,0x36,0x6c,0xd8,0xab,0x4d};
   unsigned char *t;
   t = new unsigned char[4];

   for(i = 0; i < 16; i++)
   {
      subkey[i] = key[i];
   }
   counter = 16;
   for(i = 0; i < 10; i++)
   {
      for(j = 0; j < 4; j++)
      {
         t[j] = subkey[counter - 4 + j];
      }
      byteRotateLeft8(t);
      for(j = 0; j < 4; j++)
      {
         t[j] = SBox[t[j]];
      }
      t[0] ^= RCon[rcon]; 

      rcon++;
      for(j = 0; j < 4; j++)
      {
         subkey[counter + j] = subkey[counter - 16 + j] ^ t[j];
      }
      counter+=4;
      for(j = 0; j < 3; j++)
      {
         for(k = 0; k < 4; k++)
         {
            subkey[counter + k] = subkey[counter - 16 + k] ^ subkey[counter - 4 + k];
         }
         counter+=4;
      }	
   }
   delete [] t;
}

__host__ static inline void AES_keygen_bs(const unsigned char key[16], vec2 bs_subkey[88])
{
   vec2 tmp, tmp2;
   unsigned char *subkey;
   subkey = new unsigned char[176];

   AES_keyGen(key, subkey);

   tmp.assign(reinterpret_cast<const unsigned int *>(&subkey[0]) );
   tmp2.assign(reinterpret_cast<const unsigned int *>(&subkey[8]) );
   bitslicekey0(tmp, tmp2, &bs_subkey[0]);
   for(int i = 1; i <= 9; i++)
   {
      tmp.assign(reinterpret_cast<const unsigned int *>(&subkey[16 * i]) );
      tmp2.assign(reinterpret_cast<const unsigned int *>(&subkey[16 * i + 8]) );
      bitslicekey(i, tmp, tmp2, &bs_subkey[0]);
   }
   tmp.assign(reinterpret_cast<const unsigned int *>(&subkey[160]) );
   tmp2.assign(reinterpret_cast<const unsigned int *>(&subkey[168]) );
   bitslicekey10(tmp, tmp2, &bs_subkey[80]);

   delete [] subkey;
}

#endif
