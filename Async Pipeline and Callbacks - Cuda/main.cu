#include "operator.cuh"

int main(int argc, char* argv[])
{
	float* h_a, * h_b, * h_c;
	float* d_a, * d_b, * d_c;

	int size = 1 << 24;
	int bufsize = size * sizeof(float);
	int num_operator = 4;

	if (argc != 1)
	{
		num_operator = atoi(argv[1]);
	}

	// initialize timer
	StopWatchInterface* timer;
	sdkCreateTimer(&timer);

	// allocate pinned host memory
	cudaMallocHost((void**)&h_a, bufsize);
	cudaMallocHost((void**)&h_b, bufsize);
	cudaMallocHost((void**)&h_c, bufsize);

	// initialize host values
	srand(2019);
	init_buffer(h_a, size);
	init_buffer(h_b, size);
	init_buffer(h_c, size);

	// allocate device memory
	cudaMalloc((void**)&d_a, bufsize);
	cudaMalloc((void**)&d_b, bufsize);
	cudaMalloc((void**)&d_c, bufsize);

	// create list of operation elements
	Operator_with_priority* ls_operator = new Operator_with_priority[num_operator];

	// get priority range
	int priority_low, priority_high;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
	printf("Priority Range: low(%d), high(%d)\n", priority_low, priority_high);

	// start measuring execution time
	sdkStartTimer(&timer);

	// execute each operator with corresponding data
	// priority settings for each CUDA stream
	for (int i = 0; i < num_operator; i++)
	{
		ls_operator[i].set_index(i);

		if (i + 1 == num_operator)
		{
			ls_operator[i].set_priority(priority_high);
		}
		else
		{
			ls_operator[i].set_priority(priority_low);
		}
	}

	// operation (copy(H2D), kernel execution, copy(D2H))
	for (int i = 0; i < num_operator; i++)
	{
		int offset = i * size / num_operator;
		ls_operator[i].async_operation(&h_c[offset], &h_a[offset], &h_b[offset], &d_c[offset], &d_a[offset], &d_b[offset], size / num_operator, bufsize / num_operator);
	}

	// synchronize all steam operations
	cudaDeviceSynchronize();

	// stop timer to measure execution time
	sdkStopTimer(&timer);

	// print each CUDA stream execution time
	for (int i = 0; i < num_operator; i++)
	{
		ls_operator[i].print_kernel_time();
	}

	// print out the result
	int print_idx = 256;
	printf("Compared a sample result...\nHost: %.6f, device: %.6f\n", h_a[print_idx] + h_b[print_idx], h_c[print_idx]);

	// compute and print the performance
	float elapsed_time_msed = sdkGetTimerValue(&timer);
	float bandwidth = 3 * bufsize * sizeof(float) / elapsed_time_msed / 1e6;
	printf("Time= %.3f msec, bandwidth= %f GB/s\n", elapsed_time_msed, bandwidth);

	// cleanup

	// delete timer
	sdkDeleteTimer(&timer);

	// terminate operatos
	delete[] ls_operator;

	// terminate device memory
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// terminate host memory
	cudaFreeHost(h_a);
	cudaFreeHost(h_b);
	cudaFreeHost(h_c);

	return 0;
}