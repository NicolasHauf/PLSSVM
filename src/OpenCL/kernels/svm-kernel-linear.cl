//#include "../include/typedef.hpp"

#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable
void __attribute__((overloadable)) AtomicAdd(__global double *source, double delta) {
    union {
	    double f;
    	ulong i;
    } oldVal;
    union {
    	double f;
		ulong i;
    } newVal;
    do {
    	oldVal.f = *source;
		newVal.f = oldVal.f + delta;
    } while (atom_cmpxchg ( (volatile __global ulong *)source, oldVal.i, newVal.i) != oldVal.i);
}



// void __attribute__((overloadable)) AtomicAdd(__global float *source, float delta) {
//     union {
// 	    float f;
//     	unsigned i;
//     } oldVal;
//     union {
//     	float f;
// 		unsigned i;
//     } newVal;
//     do {
//     	oldVal.f = *source;
// 		newVal.f = oldVal.f + delta;
//     } while (atom_cmpxchg ( (volatile __global unsigned *)source, oldVal.i, newVal.i) != oldVal.i);
// }
__kernel void kernel_linear(__global const real_t *q, __global real_t *ret, __global const real_t *d, __global const real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows, const int add, const int start_block_x, const int start_block_y){  
//  const unsigned CUDABLOCK_SIZE = 16;
//  const int BLOCKING_SIZE_THREAD = 6;


	// int i =  blockIdx.x * blockDim.x * BLOCKING_SIZE_THREAD;
	int i =  (get_group_id(0) + start_block_x) * get_local_size(0) * BLOCKING_SIZE_THREAD ;
	// int j = blockIdx.y * blockDim.y * BLOCKING_SIZE_THREAD;
	int j =  (get_group_id(1) + start_block_y) * get_local_size(1) * BLOCKING_SIZE_THREAD ;

	// __shared__ real_t data_intern_i [CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
	__local real_t data_intern_i [BLOCK][BLOCKING_SIZE_THREAD];
	// __shared__ real_t data_intern_j [CUDABLOCK_SIZE][BLOCKING_SIZE_THREAD];
	__local real_t data_intern_j [BLOCK][BLOCKING_SIZE_THREAD];
	real_t matr[BLOCKING_SIZE_THREAD][BLOCKING_SIZE_THREAD] = {};
	real_t data_j[BLOCKING_SIZE_THREAD];
	
	if(i >= j){
		// i += threadIdx.x * BLOCKING_SIZE_THREAD;
		i += 	get_local_id(0) * BLOCKING_SIZE_THREAD;
		const int ji = j + get_local_id(0) * BLOCKING_SIZE_THREAD;
		// j += threadIdx.y * BLOCKING_SIZE_THREAD;
		j += 	get_local_id(1) * BLOCKING_SIZE_THREAD;
		// printf("%d,%d\n", i,j);
		for(int vec_index = 0; vec_index < Ncols * Nrows ; vec_index += Nrows){
			{
				//#pragma unroll(BLOCKING_SIZE_THREAD)
				for(int block_id = 0; block_id < BLOCKING_SIZE_THREAD; ++block_id){
					const int data_index = vec_index + block_id;
					// if(threadIdx.y == block_id ) data_intern_i[threadIdx.x][block_id] = data_d[data_index + i ];  
					if(	get_local_id(1) == block_id ) data_intern_i[ get_local_id(0) ][block_id] = data_d[data_index + i ];  
					// if(threadIdx.y == block_id * 2 ) data_intern_j[threadIdx.x][block_id] = data_d[data_index + ji];
					if(	get_local_id(1) == block_id * 2 ) data_intern_j[ get_local_id(0) ][block_id] = data_d[data_index + ji];
				}

			}
			// __syncthreads();
			barrier(CLK_GLOBAL_MEM_FENCE);

			//#pragma unroll(BLOCKING_SIZE_THREAD)
			for(int data_index = 0; data_index < BLOCKING_SIZE_THREAD; ++data_index){
				data_j[data_index] = data_intern_j[get_local_id(1)][data_index];
			}
			// __syncthreads();
			barrier(CLK_GLOBAL_MEM_FENCE);
			//#pragma unroll(BLOCKING_SIZE_THREAD)
			for(int x = 0; x < BLOCKING_SIZE_THREAD; ++x){
				const real_t data_i = data_intern_i[get_local_id(0)][x];				
			//	#pragma unroll(BLOCKING_SIZE_THREAD)
				for(int y = 0; y < BLOCKING_SIZE_THREAD; ++y){
					matr[x][y] += data_i * data_j[y];
				}
			}
		}
		//#pragma unroll(BLOCKING_SIZE_THREAD)
		for(int x = 0; x < BLOCKING_SIZE_THREAD; ++x){
			//#pragma unroll(BLOCKING_SIZE_THREAD)
			for(int y = 0; y < BLOCKING_SIZE_THREAD; ++y){
				const real_t temp = (matr[x][y]  + QA_cost - q[i + x] - q[j + y]) * add;
				if(i + x > j + y){
						AtomicAdd(&ret[i + x], temp * d[j + y]);
					// ret[i+x] = temp * d[j + y];
						AtomicAdd(&ret[j + y], temp * d[i + x]);
					// ret[i+x] = temp * d[i + x];
				}else if(i + x == j + y){
			
						AtomicAdd(&ret[j + y], (temp + cost * add) * d[i + x]);
					// ret[j+y] = temp * d[i + x];
				}
			}
		}
	}
}