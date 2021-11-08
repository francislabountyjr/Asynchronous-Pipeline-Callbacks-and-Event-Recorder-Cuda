#include "operator.cuh"

__global__ void vecAdd_kernel(float* c, const float* a, const float* b)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i = 0; i < 500; i++)
	{
		c[idx] = a[idx] + b[idx];
	}
}

void init_buffer(float* data, const int size)
{
	for (int i = 0; i < size; i++)
	{
		data[i] = rand() / (float)RAND_MAX;
	}
}

void CUDART_CB Operator::Callback(cudaStream_t stream, cudaError_t status, void* userData)
{
	Operator* this_ = (Operator*)userData;
	this_->print_time();
}

void Operator::print_time()
{
	sdkStopTimer(&p_timer);
	float elapsed_time_msed = sdkGetTimerValue(&p_timer);
	printf("Stream %2d - elapsed %.3f ms\n", _index, elapsed_time_msed);
}

void Operator::async_operation(float* h_c, const float* h_a, const float* h_b, float* d_c, float* d_a, float* d_b, const int size, const int bufsize)
{
	sdkStartTimer(&p_timer);

	// copy host memory to device memory
	cudaMemcpyAsync(d_a, h_a, bufsize, cudaMemcpyHostToDevice, stream);
	cudaMemcpyAsync(d_b, h_b, bufsize, cudaMemcpyHostToDevice, stream);

	// record the event before the kernel execution
	cudaEventRecord(start, stream);

	// launch cuda kernel;
	dim3 block(256);
	dim3 grid(size / block.x);

	vecAdd_kernel<<<grid, block, 0, stream>>>(d_c, d_a, d_b);

	// record the event right after kernel execution is complete
	cudaEventRecord(stop, stream);

	// copy device memory to host memory
	cudaMemcpyAsync(h_c, d_c, bufsize, cudaMemcpyDeviceToHost, stream);

	// register callback function
	cudaStreamAddCallback(stream, Operator::Callback, this, 0);
}

void Operator::print_kernel_time()
{
	float elapsed_time_msed = 0.f;
	cudaEventElapsedTime(&elapsed_time_msed, start, stop);
	printf("Kernel in stream %2d - elapsed %.3f ms\n", _index, elapsed_time_msed);
}0