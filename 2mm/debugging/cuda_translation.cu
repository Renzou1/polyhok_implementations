
__device__
float mm2_kernel_helper(int nj, int nk, int alpha, float *a, float *b, int j, int i, float *tmp)
{
for( int k = 0; k<nk; k+=1){
	tmp[((i * nj) + j)] = (tmp[((i * nj) + j)] + ((alpha * a[((i * nk) + k)]) * b[((k * nj) + j)]));
}

return (tmp[((i * nj) + j)]);
}


extern "C" __global__ void mm2_kernel(int ni, int nj, int nk, int nl, int alpha, int beta, float *tmp, float *a, float *b)
{
	int j = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int i = ((blockIdx.y * blockDim.y) + threadIdx.y);
if(((i < ni) && (j < nj)))
{
	tmp[((i * nj) + j)] = 0.0;
	tmp[((i * nj) + j)] = mm2_kernel_helper(nj, nk, alpha, a, b, j, i, tmp);
}

}



__device__
float mm2_kernel2_helper(int nj, int nl, int beta, float *tmp, float *c, int j, int i, float *d)
{
for( int k = 0; k<nj; k+=1){
	d[((i * nl) + j)] = (d[((i * nl) + j)] + (tmp[((i * nj) + k)] * c[((k * nl) + j)]));
}

return (d[((i * nl) + j)]);
}


extern "C" __global__ void mm2_kernel2(int ni, int nj, int nk, int nl, int alpha, int beta, float *tmp, float *c, float *d)
{
	int j = ((blockIdx.x * blockDim.x) + threadIdx.x);
	int i = ((blockIdx.y * blockDim.y) + threadIdx.y);
if(((i < ni) && (j < nl)))
{
	d[((i * nl) + j)] = (d[((i * nl) + j)] * beta);
	d[((i * nl) + j)] = mm2_kernel2_helper(nj, nl, beta, tmp, c, j, i, d);
}

}


