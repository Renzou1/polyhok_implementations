require PolyHok

PolyHok.defmodule MM2 do

def init_array(ni, nj, nk, nl, a, b, c, d) do
    alpha = 32412
    beta = 2123

    #a = Nx.tensor()
    a =
        for i <- 0..ni-1 do
            for j <- 0..nk-1 do
                (i * j) / ni
            end
        end

    b = 
        for i <- 0..nk-1 do
            for j <- 0..nj-1 do
                (i * (j + 1))  / nj
            end
        end

    c = 
        for i <- 0..nl-1 do
            for j <- 0..nj-1 do
                (i * (j + 3)) / nl
            end
        end

    d = 
        for i <- 0..ni-1 do
            for j <- 0..nl-1 do
                (i * (j + 2)) / nk
            end
        end


    {alpha, beta}
end

def compare_results(ni, nl, d, d_outputFromGpu) do

end

def gpu_argv_init() do

end

defk mm2_kernel1(ni, nj, nk, nl, alpha, beta, tmp, a, b) do
	j = blockIdx.x * blockDim.x + threadIdx.x
	i = blockIdx.y * blockDim.y + threadIdx.y

    if (i < ni && j < nj) do
        tmp[i * nj + j] = map_kernel1_helper(nj, nk, alpha, a, b, j, i)
    end

end

defd mm2_kernel1_helper(nj, nk, alpha, a, b, j, i) do 
    acc = 0
    for k in range(0, nk, 1) do
        acc = acc + alpha * a[i * nk + k] * b[k * nj + j]
    end

    return acc
end

defk mm2_kernel2(ni, nj, nk, nl, alpha, beta, tmp, c, d) do
    j = blockIdx.x * blockDim.x + threadIdx.x
	i = blockIdx.y * blockDim.y + threadIdx.y

	if ((i < ni) && (j < nl)) do 
        D[i * nl + j] = mm2_kernel2_helper(nj, nl, D[i * nl + j], beta, tmp, c,  j, i)
	end
end

defd mm2_kernel2_helper(nj, nl, acc, beta, tmp, c,  j, i) do
    acc = acc * beta

    for k in range(0, nj, 1) do
        acc =  acc + tmp[i * nj + k] * c[k * nl + j]
    end

    return acc
end

def print_array(ni, nl, d) do

end

def mm2_polyhok(ni, nj, nk, nl, alpha, beta, temp, a, b, c, d, d_outputFromGpu) do

end

end