#include "../include/MatrixFP32.cuh"
#include <assert.h>

__global__ void naive_mat_mul_kernel(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols)
{
    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int col = blockDim.y*blockIdx.y + threadIdx.y;

    if (row < C_n_rows && col < C_n_cols)
    {
        float value = 0;
        for (int k = 0; k < A_n_cols; k++)
        {
            value += d_A_ptr[row*A_n_cols + k] * d_B_ptr[k*C_n_cols + col];
        }

        d_C_ptr[row*C_n_cols + col] = value;
    }
}

void naive_xgemm(float *d_A_ptr, float *d_B_ptr, float *d_C_ptr, int C_n_rows, int C_n_cols, int A_n_cols)
{
    dim3 dim_block(32, 32, 1);
    dim3 dim_grid(ceil(C_n_rows/(float)(32)), ceil(C_n_cols/(float)(32)), 1);
    naive_mat_mul_kernel<<<dim_grid, dim_block>>>(d_A_ptr, d_B_ptr, d_C_ptr, C_n_rows, C_n_cols, A_n_cols);
}