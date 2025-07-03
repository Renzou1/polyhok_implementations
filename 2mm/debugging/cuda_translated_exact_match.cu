
extern "C" __global__ void mm2_kernel(int ni, int nj, int nk, int nl, float alpha, float beta, float *tmp, float *a, float *b)
{
	int j = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int i = ((blockIdx.y * blockDim.y) + threadIdx.y);
if(((i < 1024) && (j < 1024)))
{
	tmp[((i * 1024) + j)] = 0.0;
for( int k = 0; k<nk; k+=1){
	tmp[((i * 1024) + j)] = (tmp[((i * 1024) + j)] + ((alpha * a[((i * 1024) + k)]) * b[((k * 1024) + j)]));
}

}

}



extern "C" __global__ void mm2_kernel2(int ni, int nj, int nk, int nl, float alpha, float beta, float *tmp, float *c, float *d)
{
	int j = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int i = ((blockIdx.y * blockDim.y) + threadIdx.y);
if(((i < 1024) && (j < 1024)))
{
	d[((i * 1024) + j)] = (d[((i * 1024) + j)] * beta);
for( int k = 0; k<1024; k+=1){
	d[((i * 1024) + j)] = (d[((i * 1024) + j)] + (tmp[((i * 1024) + k)] * c[((k * 1024) + j)]));
}

}

}

