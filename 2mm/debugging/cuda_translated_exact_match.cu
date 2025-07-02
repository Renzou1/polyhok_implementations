
extern "C" __global__ void mm2_kernel(int ni, int nj, int nk, int nl, int alpha, int beta, float *tmp, float *a, float *b)
{
	int j = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int i = ((blockIdx.y * blockDim.y) + threadIdx.y);
if(((i < ni) && (j < nj)))
{
	tmp[((i * nj) + j)] = 0.0;
for( int k = 0; k<nk; k+=1){
	tmp[((i * nj) + j)] = (tmp[((i * nj) + j)] + ((alpha * a[((i * nk) + k)]) * b[((k * nj) + j)]));
}

}

}



extern "C" __global__ void mm2_kernel2(int ni, int nj, int nk, int nl, int alpha, int beta, float *tmp, float *c, float *d)
{
	int j = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int i = ((blockIdx.y * blockDim.y) + threadIdx.y);
if(((i < ni) && (j < nl)))
{
	d[((i * nl) + j)] = (d[((i * nl) + j)] * beta);
for( int k = 0; k<nj; k+=1){
	d[((i * nl) + j)] = (d[((i * nl) + j)] + (tmp[((i * nj) + k)] * c[((k * nl) + j)]));
}

}

}
