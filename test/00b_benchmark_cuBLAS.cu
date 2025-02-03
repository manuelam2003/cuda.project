#include <iostream>
#include <iomanip>
#include <cublas_v2.h>
#include <device_launch_parameters.h>

#include "../include/MatrixFP32.cuh"
#include "../include/utils.cuh"

int main(int argc, char const *argv[])
{
    int mat_sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    double cublas_time[n_sizes];
    double cublas_gflops[n_sizes];

    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
    {
        int n = mat_sizes[mat_size];

        MatrixFP32 A_FP32 = MatrixFP32(n, n, false);
        MatrixFP32 B_FP32 = MatrixFP32(n, n, false);
        MatrixFP32 C_FP32_cublas = MatrixFP32(n, n, false);

        random_init_mat(A_FP32, -10, 10);    
        random_init_mat(B_FP32, -10, 10);    
        init_mat(C_FP32_cublas, 1.0f);    

        MatrixFP32 d_A_FP32 = MatrixFP32(n, n, true); 
        A_FP32.copy_to_device(d_A_FP32);
        MatrixFP32 d_B_FP32 = MatrixFP32(n, n, true); 
        B_FP32.copy_to_device(d_B_FP32);
        MatrixFP32 d_C_FP32_cublas = MatrixFP32(n, n, true); 
        C_FP32_cublas.copy_to_device(d_C_FP32_cublas);
        cudaDeviceSynchronize();

        //----------------------------------------------------//
        //-------------------- Warmup Run --------------------//
        //----------------------------------------------------//
        // Create and initialize cuBLAS handle
        cublasHandle_t handle;
        cublas_check(cublasCreate(&handle));
        
        float alpha = 1;
        float beta = 0;
        cublas_check(cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                d_C_FP32_cublas.n_cols, d_C_FP32_cublas.n_rows, d_A_FP32.n_cols, 
                                &alpha,
                                d_B_FP32.ptr, d_B_FP32.n_cols, 
                                d_A_FP32.ptr, d_A_FP32.n_cols, 
                                &beta,
                                d_C_FP32_cublas.ptr, d_C_FP32_cublas.n_cols) 
                    );
        cudaDeviceSynchronize();

        //----------------------------------------------------//
        //--------------------- cuBLAS -----------------------//
        //----------------------------------------------------//
        cudaEventRecord(beg);
        for (int n_runs = 0; n_runs < 10; n_runs++)
        {
            float alpha = 1;
            float beta = 0;
            cublas_check(cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                d_C_FP32_cublas.n_cols, d_C_FP32_cublas.n_rows, d_A_FP32.n_cols,
                                &alpha,
                                d_B_FP32.ptr, d_B_FP32.n_cols, 
                                d_A_FP32.ptr, d_A_FP32.n_cols, 
                                &beta,
                                d_C_FP32_cublas.ptr, d_C_FP32_cublas.n_cols) 
                    );
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.;

        cublas_time[mat_size] = (elapsed_time) / 10;
        cublas_gflops[mat_size] = 2. * 1e-9 * 10 * n * n * n / (elapsed_time);

        A_FP32.free_mat();
        B_FP32.free_mat();
        C_FP32_cublas.free_mat();

        d_A_FP32.free_mat();
        d_B_FP32.free_mat();
        d_C_FP32_cublas.free_mat();
    }

    std::cout << "Matrix Size: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << mat_sizes[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cuBLAS Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cublas_time[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "cuBLAS GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << cublas_gflops[mat_size] << " ";
    std::cout << "\n \n";

    update_benckmark_txt("txt_benchmarks/00b_cublas.txt", cublas_time, cublas_gflops, mat_sizes, n_sizes);

    return 0;
}
