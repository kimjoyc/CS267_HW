const char* dgemm_desc = "Simple blocked dgemm and vectorization.";

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


static void avx_unroll(const int lda, const int M, const int N, const int K, double*  A, double*  B, double* restrict C)
{
  for (int i = 0; i < M; i += 8) 
  {
    for (int j = 0; j < N; j += 4) 
    {
      __m256d c1 = _mm256_load_pd(C+i+j*lda);
      __m256d c2 = _mm256_load_pd(C+i+(j+1)*lda);
      __m256d c3 = _mm256_load_pd(C+i+(j+2)*lda);
      __m256d c4 = _mm256_load_pd(C+i+(j+3)*lda);
        

      __m256d c5 = _mm256_load_pd(C+i+4+j*lda);
      __m256d c6 = _mm256_load_pd(C+i+4+(j+1)*lda);
      __m256d c7 = _mm256_load_pd(C+i+4+(j+2)*lda);
      __m256d c8 = _mm256_load_pd(C+i+4+(j+3)*lda);

      for (int k = 0; k < K; ++k) 
      {
        __m256d a1 = _mm256_load_pd(A+i+k*lda);
        __m256d a2 = _mm256_load_pd(A+i+4+(k)*lda);

        __m256d b1 = _mm256_broadcast_sd(B+k+j*lda);
        __m256d b2 = _mm256_broadcast_sd(B+k+(j+1)*lda);
        __m256d b3 = _mm256_broadcast_sd(B+k+(j+2)*lda);
        __m256d b4 = _mm256_broadcast_sd(B+k+(j+3)*lda);

        c1 = _mm256_fmadd_pd(a1,b1,c1);
        c2 = _mm256_fmadd_pd(a1,b2,c2);
        c3 = _mm256_fmadd_pd(a1,b3,c3);
        c4 = _mm256_fmadd_pd(a1,b4,c4);

        c5 = _mm256_fmadd_pd(a2,b1,c5);
        c6 = _mm256_fmadd_pd(a2,b2,c6);
        c7 = _mm256_fmadd_pd(a2,b3,c7);
        c8 = _mm256_fmadd_pd(a2,b4,c8);

      }
      _mm256_store_pd(C+i+j*lda, c1);
      _mm256_store_pd(C+i+(j+1)*lda, c2);
      _mm256_store_pd(C+i+(j+2)*lda, c3);
      _mm256_store_pd(C+i+(j+3)*lda, c4);

      _mm256_store_pd(C+i+4+(j)*lda, c5);
      _mm256_store_pd(C+i+4+(j+1)*lda, c6);
      _mm256_store_pd(C+i+4+(j+2)*lda, c7);
      _mm256_store_pd(C+i+4+(j+3)*lda, c8);

    }
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
  if (lda % mult){
    int t = lda % mult;
    lda_ = lda + (mult-t);
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
        avx_unroll(lda_, M, N, K, padA + i + k*lda_, padB + k + j*lda_, padC + i + j*lda_);

      }
    }
  }

  unpad(padC, C, lda, lda_);
  _mm_free(padA);
  _mm_free(padB);

}
