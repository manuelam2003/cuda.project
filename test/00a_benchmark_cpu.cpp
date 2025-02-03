#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <random>
#include <chrono>

#include "../include/MatrixFP32.cuh"
#include "../include/utils.cuh"

void cpu_xgemm(MatrixFP32 A_mat, MatrixFP32 B_mat, MatrixFP32 C_mat)
{
    // Getting A Matrix Dimension
    int A_n_rows = A_mat.n_rows; 
    int A_n_cols = A_mat.n_cols;

    // Getting B Matrix Dimension
    int B_n_rows = B_mat.n_rows; 
    int B_n_cols = B_mat.n_cols;

    // Getting C Matrix Dimension
    int C_n_rows = C_mat.n_rows; 
    int C_n_cols = C_mat.n_cols;

    // Asserting dimensions
    assert (A_n_cols == B_n_rows && "Matrices A & B must have one common dimension");
    assert (A_n_rows == C_n_rows && "A rows must be equal to C rows");
    assert (B_n_cols == C_n_cols && "B cols must be equal to C cols");

    // Matrix Multiplication
    for (int row = 0; row < A_n_rows; row++)
    {
        for (int col = 0; col < B_n_cols; col++)
        {
            float val = 0.0f;
            for (int k = 0; k < A_n_cols; k++)
            {
                val += A_mat.ptr[row*A_n_cols + k] * B_mat.ptr[k*B_n_cols + col];
            }
            C_mat.ptr[row*C_n_cols + col] = val;
        }
    }
}


int main(int argc, char const *argv[])
{
    int mat_sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);

    double cblas_time[n_sizes];
    double cblas_gflops[n_sizes];

    auto start = std::chrono::high_resolution_clock::now();
    auto stop = std::chrono::high_resolution_clock::now();
    auto elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
    {
        // Matrix Size
        int n = mat_sizes[mat_size];

        // Define MatrixFP32
        MatrixFP32 A_FP32 = MatrixFP32(n, n, false);
        MatrixFP32 B_FP32 = MatrixFP32(n, n, false);
        MatrixFP32 C_FP32 = MatrixFP32(n, n, false);

        // Initialize Matrices
        random_init_mat(A_FP32, -10, 10); // Random Initialization between -10 and 10
        random_init_mat(B_FP32, -10, 10); // Random Initialization between -10 and 10

        // Perform matrix multiplication: C = A * B
        cpu_xgemm(A_FP32, B_FP32,  C_FP32); 

        //----------------------------------------------------//
        //---------------------- cBLAS -----------------------//
        //----------------------------------------------------//
        start = std::chrono::high_resolution_clock::now();
        for (int n_runs = 0; n_runs < 5; n_runs++)
            cpu_xgemm(A_FP32, B_FP32,  C_FP32);
        stop = std::chrono::high_resolution_clock::now();

        elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

        cblas_time[mat_size] = (elapsed_time.count() / 1e+6) / 5;
        cblas_gflops[mat_size] = 2. * 1e-9 * 10 * n * n * n / (elapsed_time.count() / 1e+6);

        A_FP32.free_mat();
        B_FP32.free_mat();
    }

    std::cout << "Matrix Size: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << mat_sizes[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cBLAS Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cblas_time[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cBLAS GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cblas_gflops[mat_size] << " ";
    std::cout << "\n \n";

    // Saving to benchmark file
    update_benckmark_txt("txt_benchmarks/00a_blas.txt", cblas_time, cblas_gflops, mat_sizes, n_sizes);
    return 0;
}
