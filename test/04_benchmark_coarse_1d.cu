#include <iostream>
#include <iomanip>
#include <cublas_v2.h>

#include "../include/MatrixFP32.cuh"
#include "../include/utils.cuh"

#include "../include/04_coarse_1d_xgemm.cuh"

int main(int argc, char const *argv[])
{
    int mat_sizes[] = {128, 256, 512, 1024, 2048, 4096};
    int n_sizes = sizeof(mat_sizes) / sizeof(mat_sizes[0]);

    float elapsed_time;
    cudaEvent_t beg, end;
    cudaEventCreate(&beg);
    cudaEventCreate(&end);

    double xgemm_time[n_sizes];
    double xgemm_gflops[n_sizes];

    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
    {
        int n = mat_sizes[mat_size];

        MatrixFP32 A_FP32 = MatrixFP32(n, n, false);
        MatrixFP32 B_FP32 = MatrixFP32(n, n, false);
        MatrixFP32 C_FP32_xgemm = MatrixFP32(n, n, false);

        random_init_mat(A_FP32, -10, 10);          
        random_init_mat(B_FP32, -10, 10);          
        init_mat(C_FP32_xgemm, -1.0f);   

        MatrixFP32 d_A_FP32 = MatrixFP32(n, n, true); 
        A_FP32.copy_to_device(d_A_FP32);
        MatrixFP32 d_B_FP32 = MatrixFP32(n, n, true); 
        B_FP32.copy_to_device(d_B_FP32);
        MatrixFP32 d_C_FP32_xgemm = MatrixFP32(n, n, true); 
        C_FP32_xgemm.copy_to_device(d_C_FP32_xgemm);
        cudaDeviceSynchronize();

        //----------------------------------------------------//
        //-------------------- Warmup Run --------------------//
        //----------------------------------------------------//
      
        coarse_1d_xgemm(d_A_FP32.ptr, d_B_FP32.ptr, d_C_FP32_xgemm.ptr, d_C_FP32_xgemm.n_rows, d_C_FP32_xgemm.n_cols, d_A_FP32.n_cols);
        cudaDeviceSynchronize();
        
        //----------------------------------------------------//
        //---------------------- xGeMM -----------------------//
        //----------------------------------------------------//
        cudaEventRecord(beg);
        for (int n_runs = 0; n_runs < 10; n_runs++)
        {
            coarse_1d_xgemm(d_A_FP32.ptr, d_B_FP32.ptr, d_C_FP32_xgemm.ptr, d_C_FP32_xgemm.n_rows, d_C_FP32_xgemm.n_cols, d_A_FP32.n_cols);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);
        cudaEventElapsedTime(&elapsed_time, beg, end);
        elapsed_time /= 1000.;

        xgemm_time[mat_size] = (elapsed_time) / 10;
        xgemm_gflops[mat_size] = 2. * 1e-9 * 10 * n * n * n / (elapsed_time);

        A_FP32.free_mat();
        B_FP32.free_mat();
        C_FP32_xgemm.free_mat();

        d_A_FP32.free_mat();
        d_B_FP32.free_mat();
        d_C_FP32_xgemm.free_mat();
    }

    std::cout << "Matrix Size: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << mat_sizes[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "xGeMM Time (seconds): ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << xgemm_time[mat_size] << " ";
    std::cout << "\n \n";

    std::cout << "xGeMM GFLOPS: ";
    for (int mat_size = 0; mat_size < n_sizes; mat_size++)
        std::cout << xgemm_gflops[mat_size] << " ";
    std::cout << "\n \n";

    update_benckmark_txt("txt_benchmarks/04_coarse_1d_xgemm.txt", xgemm_time, xgemm_gflops, mat_sizes, n_sizes);

    return 0;
}
