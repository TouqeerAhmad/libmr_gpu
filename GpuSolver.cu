#include <stdio.h>

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}


__global__
void normalization_and_sum(int size, double maxx, double range, double *inputArr, double *x0, double *mean)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < size)
  {
    x0[i] = (inputArr[i] - maxx) / range;
    atomicAdd(mean,x0[i]);
  }
}


__global__
void compute_std(int size, double *x0, double *mean, double *myStd)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  double tempVal, tempValSq;
  if (i < size)
  {
    tempVal = x0[i] - *(mean);
    tempValSq = tempVal * tempVal;
    atomicAdd(myStd,tempValSq);
  }
}



#ifdef __cplusplus
extern "C" {
#endif

void runKernels()
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));
  
  cudaMalloc(&d_x, N*sizeof(float)); 
  cudaMalloc(&d_y, N*sizeof(float));
  
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }
  
  cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);
  
  // Perform SAXPY on 1M elements
  saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);
  
  cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
  
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = max(maxError, abs(y[i]-4.0f));
  printf("Max error: %f\n", maxError);
  
  cudaFree(d_x);
  cudaFree(d_y);
  free(x);
  free(y);
}


void runKernels_ComputeMeanAndStd(double * inputData, double * x0, double *mean, double *myStd, double maxx, double range, int size)
{
  
  double *device_inputData, *device_x0, *device_mean, *device_myStd;
  
  cudaMalloc(&device_inputData, size*sizeof(double)); 
  cudaMalloc(&device_x0, size*sizeof(double));
  cudaMalloc(&device_mean, sizeof(double));
  cudaMalloc(&device_myStd, sizeof(double));
  
  
  cudaMemcpy(device_inputData, inputData, size*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_x0, x0, size*sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_mean, mean, sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(device_myStd, myStd, sizeof(double), cudaMemcpyHostToDevice);
  
  
  normalization_and_sum<<<(size+255)/256, 256>>>(size, maxx, range, device_inputData, device_x0, device_mean);
  
  // copy the sum to host
  cudaMemcpy(mean, device_mean, sizeof(double), cudaMemcpyDeviceToHost);
  
  // compute the mean 
  *(mean) = *(mean) / size; 
  
  // copy mean to the device
  cudaMemcpy(device_mean, mean, sizeof(double), cudaMemcpyHostToDevice);
  
  
  compute_std<<<(size+255)/256, 256>>>(size, device_x0, device_mean, device_myStd);
  
  // copy the sum to host
  cudaMemcpy(myStd, device_myStd, sizeof(double), cudaMemcpyDeviceToHost);
  
  *(myStd) = *(myStd) / (size-1);
  *(myStd) = sqrt(*(myStd));
      
  // copy x0 to host
  cudaMemcpy(x0, device_x0, size*sizeof(double), cudaMemcpyDeviceToHost);
  
  
  cudaFree(device_myStd);
  cudaFree(device_mean);
  cudaFree(device_x0);
  cudaFree(device_inputData);
  
}


#ifdef __cplusplus
}
#endif

