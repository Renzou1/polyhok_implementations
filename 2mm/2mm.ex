require PolyHok

PolyHok.defmodule MM2 do

def init_array(ni, nj, nk, nl, type) do
    alpha = 32412.0
    beta = 2123.0

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

    a = Nx.tensor(a, type: type)
    b = Nx.tensor(b, type: type)
    c = Nx.tensor(c, type: type)
    d = Nx.tensor(d, type: type)
    tmp = Nx.broadcast(Nx.tensor(0, type: type), {ni, nj})

    {alpha, beta, a, b, c, d, tmp}
end

defk mm2_kernel(ni, nj, nk, nl, alpha, beta, tmp, a, b) do
	j = blockIdx.x * blockDim.x + threadIdx.x
	i = blockIdx.y * blockDim.y + threadIdx.y

    if (i < ni && j < nj) do
        tmp[i * nj + j] = 0.0
        tmp[i * nj + j] = mm2_kernel_helper(nj, nk, alpha, a, b, j, i, tmp)
    end

end

defd mm2_kernel_helper(nj, nk, alpha, a, b, j, i, tmp) do
    for k in range(0, nk, 1) do
        tmp[i * nj + j] = tmp[i * nj + j] + alpha * a[i * nk + k] * b[k * nj + j]
    end

    return tmp[i * nj + j]
end

defk mm2_kernel2(ni, nj, nk, nl, alpha, beta, tmp, c, d) do
    j = blockIdx.x * blockDim.x + threadIdx.x
	i = blockIdx.y * blockDim.y + threadIdx.y

	if ((i < ni) && (j < nl)) do
        d[i * nl + j] = d[i * nl + j] * beta
        d[i * nl + j] = mm2_kernel2_helper(nj, nl, beta, tmp, c,  j, i, d)
	end
end

defd mm2_kernel2_helper(nj, nl, beta, tmp, c,  j, i, d) do
    for k in range(0, nj, 1) do
        d[i * nl + j] = d[i * nl + j] + tmp[i * nj + k] * c[k * nl + j]
    end

    return d[i * nl + j]
end

def write_tensor_to_file(list, file_name) do
list
|> Nx.to_flat_list()
|> Enum.map(&Float.to_string/1)
|> Enum.join(" ")
|> then(&File.write!(file_name, &1))

end

def mm2_polyhok(ni, nj, nk, nl, alpha, beta, tmp, a, b, c, d) do

tmp_gpu = PolyHok.new_gnx(tmp)
a_gpu = PolyHok.new_gnx(a)
b_gpu = PolyHok.new_gnx(b)
c_gpu = PolyHok.new_gnx(c)
d_gpu = PolyHok.new_gnx(d)

block = {32, 8, 1}
x = 0
y = 1
_z = 2
grid1 = {ceil(nj / elem(block, x)), ceil(ni / elem(block, y)), 1}
grid2 = {ceil(nl / elem(block, x)), ceil(ni / elem(block,y)), 1}

prev = System.monotonic_time()

PolyHok.spawn(&MM2.mm2_kernel/9, grid1, block, [ni, nj, nk, nl, alpha, beta, tmp_gpu, a_gpu, b_gpu])
PolyHok.spawn(&MM2.mm2_kernel2/9, grid2, block, [ni, nj, nk, nl, alpha, beta, tmp_gpu, c_gpu, d_gpu])

next = System.monotonic_time()

#MM2.write_tensor_to_file(PolyHok.get_gnx(d_gpu), "polyhok_d_gpu.txt")
IO.puts "PolyHok\t#{inspect(PolyHok.get_gnx(d_gpu))}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "


end
end

[size_string] = System.argv
size = String.to_integer(size_string)
ni = size
nj = size
nk = size
nl = size
type = {:f, 32}

{alpha, beta, a, b, c, d, tmp} = MM2.init_array(ni, nj, nk, nl, type)

MM2.mm2_polyhok(ni, nj, nk, nl, alpha, beta, tmp, a, b, c, d)
