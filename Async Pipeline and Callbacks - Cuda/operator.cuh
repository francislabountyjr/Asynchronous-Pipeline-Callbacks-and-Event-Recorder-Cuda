#pragma once

#include <cstdio>
#include <helper_timer.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

class Operator
{
private:
	int _index;
	StopWatchInterface* p_timer;
	cudaEvent_t start, stop;

	static void CUDART_CB Callback(cudaStream_t stream, cudaError_t status, void* userData);
	void print_time();

protected:
	cudaStream_t stream = nullptr;

public:
	Operator(bool create_stream = true)
		:_index()
	{
		if (create_stream)
		{
			cudaStreamCreate(&stream);
		}		
		
		sdkCreateTimer(&p_timer);

		// create CUDA events
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~Operator()
	{
		if (stream != nullptr)
		{
			cudaStreamDestroy(stream);
		}

		sdkDeleteTimer(&p_timer);

		// terminate CUDA events
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void set_index(int index) { _index = index; }
	void async_operation(float* h_c, const float* h_a, const float* h_b, float* d_c, float* d_a, float* d_b, const int size, const int bufsize);
	void print_kernel_time();
};

class Operator_with_priority : public Operator
{
public:
	Operator_with_priority() : Operator(false) {}

	void set_priority(int priority)
	{
		cudaStreamCreateWithPriority(&stream, cudaStreamNonBlocking, priority);
	}
};

__global__ void vecAdd_kernel(float* c, const float* a, const float* b);

void init_buffer(float* data, const int size);