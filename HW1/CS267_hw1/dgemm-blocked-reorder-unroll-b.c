const char* dgemm_desc = "Simple blocked dgemm, reorder and unroll B.";

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif

#include <immintrin.h>
#define min(a, b) (((a) < (b)) ? (a) : (b))



static void pad(double* padA, double*  A, const int lda, const int lda_)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      padA[i + j*lda_] = A[i + j*lda];
    }   
  }   
}

static void unpad(double* padA, double*  A, const int lda, const int lda_)
{
  for (int j = 0; j < lda; j++) {
    for (int i = 0; i < lda; i++) {
      A[i + j*lda] = padA[i + j*lda_];
    }   
  }   
}



static void do_block_unroll(const int lda, const int M, const int N, const int K, double*  A, double*  B, double* restrict C)
{
  double b1, b2, b3, b4, b5, b6, b7, b8;
  for (int j= 0; j < N; j++) 
  {
    for (int k = 0; k < (K-7); k+=8) 
    {
      b1 = B[j*lda + k];
      b2 = B[j*lda + k + 1];
      b3 = B[j*lda + k + 2];
      b4 = B[j*lda + k + 3];
      b5 = B[j*lda + k + 4];
      b6 = B[j*lda + k + 5];
      b7 = B[j*lda + k + 6];
      b8 = B[j*lda + k + 7];

      for (int i = 0; i < M; i++) 
      {
        C[j*lda + i] += A[k*lda + i] * b1;
        C[j*lda + i] += A[(k+1)*lda + i] * b2;
        C[j*lda + i] += A[(k+2)*lda + i] * b3;
        C[j*lda + i] += A[(k+3)*lda + i] * b4;
        C[j*lda + i] += A[(k+4)*lda + i] * b5;
        C[j*lda + i] += A[(k+5)*lda + i] * b6;
        C[j*lda + i] += A[(k+6)*lda + i] * b7;
        C[j*lda + i] += A[(k+7)*lda + i] * b8;

      }


    }
  }
  
  if(K % 8) 
  {
    int j,k;
    do 
    {
      b1 = B[j*lda + k];
      for (int i = 0; i < M; ++i) 
      {
        C[j*lda + i] += A[k*lda + i] * b1;
      }
    }
    while(++k < K);

  }

}


/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (const int lda, double* A, double*  B, double* restrict C)
{

  int lda_ = lda;
  int mult = 8;
  if (lda % mult)
  {
    int t = lda % mult;
    lda_ = lda + (mult-t) + mult;
  }

  double* padA = (double*) _mm_malloc(lda_ * lda_ * sizeof(double), 32);
  pad(padA, A, lda, lda_);

  double* padB = (double*) _mm_malloc(lda_ * lda_ * sizeof(double), 32);
  pad(padB, B, lda, lda_);

  double* padC = (double*) _mm_malloc(lda_ * lda_ * sizeof(double), 32);
  pad(padC, C, lda, lda_);

  /* For each block-row of A */ 
  for (int i = 0; i < lda_; i += BLOCK_SIZE)
  {
    /* For each block-column of B */
    for (int j = 0; j < lda_; j += BLOCK_SIZE)
    {
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < lda_; k += BLOCK_SIZE)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int M = min (BLOCK_SIZE, lda_-i);
        int N = min (BLOCK_SIZE, lda_-j);
        int K = min (BLOCK_SIZE, lda_-k);
        /* Perform individual block dgemm */
        do_block_unroll(lda_, M, N, K, padA + i + k*lda_, padB + k + j*lda_, padC + i + j*lda_);

      }
    }
  }

  unpad(padC, C, lda, lda_);
  _mm_free(padA);
  _mm_free(padB);

}
