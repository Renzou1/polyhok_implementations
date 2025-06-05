require PolyHok

PolyHok.defmodule MM2 do

def init_array(ni, nj, nk, nl) do
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


    {alpha, beta, a, b, c, d}
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

def mm2_polyhok(ni, nj, nk, nl, alpha, beta, tmp, a, b, c, d) do    

tmp_gpu = PolyHok.new_gnx(tmp, type: f32) #hardcoded types for now
a_gpu = PolyHok.new_gnx(a, type: f32)
b_gpu = PolyHok.new_gnx(b, type: f32)
c_gpu = PolyHok.new_gnx(c, type: f32)
d_gpu = PolyHok.new_gnx(d, type: f32)

block = {32, 8, 1}
x = 0
y = 1
z = 2
grid1 = {Float.ceil(nj / elem(block, x)), Float.ceil(ni / elem(block, y)), 1}
grid2 = {Float.ceil(nl / elem(block, x)), Float.ceil(ni / elem(block,y)), 1}

PolyHok.spawn(&MM2.mm2_kernel1/9, grid1, block, [ni, nj, nk, nl, alpha, beta, tmp_gpu, a_gpu, b_gpu])
PolyHok.spawn(&MM2.mm2_kernel2/9, grid2, block, [ni, nj, nk, nl, alpha, beta, tmp_gpu, c_gpu, d_gpu])

d_outputFromGpu = PolyHok.get_gnx(d_gpu)
d_outputFromGpu
end
end

ni = 1024
nj = 1024
nk = 1024
nl = 1024

{alpha, beta, a, b, c, d} = init_array(ni, nj, nk, nl)

mm2_polyhok(ni, nj, nk, nl, alpha, beta, tmp, a, b, c, d, d_outputFromGpu)