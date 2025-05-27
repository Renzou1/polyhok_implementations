require PolyHok

PolyHok.defmodule MM2 do
    @ni 1024 # probably needs to be changable later
    @nj 1024
    @nl 1024
    @nk 1024

def init_array(ni, nj, nk, nl, a, b, c, d) do
    alpha = 32412
    beta = 2123

    a =
        for i <- 0..ni-1 do
            for j <- 0..nk-1 do
                (i * j) / @ni
            end
        end

    b = 
        for i <- 0..nk-1 do
            for j <- 0..nj-1 do
                (i * (j + 1))  / @nj
            end
        end

    c = 
        for i <- 0..nl-1 do
            for j <- 0..nj-1 do
                (i * (j + 3)) / @nl
            end
        end

    d = 
        for i <- 0..ni-1 do
            for j <- 0..nl-1 do
                (i * (j + 2)) / @nk
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

    if (i < @ni && j < @nj) do
        tmp[i * @nj + j] = 0
        for 0..@nk - 1 do
            #tmp[i * @nj + j] = alpha * a[i * @nk + k] * b[k * @nj + j] + tmp[i * @nj + j] 
        end
    end

end

defk mm2_kernel2(ni, nj, nk, nl, alpha, beta, tmp, c, d) do

end

def print_array(ni, nl, d) do

end

def mm2_polyhok(ni, nj, nk, nl, alpha, beta, temp, a, b, c, d, d_outputFromGpu) do

end

end