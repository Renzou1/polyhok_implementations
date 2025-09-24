require PolyHok

PolyHok.defmodule JACOBI1D do

def init_arrays(n) do

	i = Nx.iota({n})
  a = (4 * i + 10) / n
  b = (7 * i + 11) / n

  {a, b}
end

defk runJacobiCUDA_kernel1(n, a, b, f) do
	i = blockIdx.x * blockDim.x + threadIdx.x

  if (i > 0 && i < n-1) do
		B[i] = f(i, a)
  end

end

defd runJacobiCuda_kernel1_helper(i, a) do
  return 0.33333 * (A[i-1] + A[i] + A[i + 1])
end


defk runJacobiCUDA_kernel2(n, a, b) do
	j = blockIdx.x * blockDim.x + threadIdx.x

	if ((j > 0) && (j < (n-1))) do
		A[j] = B[j]
  end
end


def jacobi_polyhok(m, n, data, mean, stddev, symmat, float_n, eps) do
  data_gpu = PolyHok.new_gnx(data)
  symmat_gpu = PolyHok.new_gnx(symmat)
  stddev_gpu = PolyHok.new_gnx(stddev)
  mean_gpu = PolyHok.new_gnx(mean)

  block1 = {256, 1, 1}
  grid1 = { ceil(m / 256), 1, 1}

  block2 = {256, 1, 1}
  grid2 = { ceil(m / 256), 1, 1}

  block3 = {32, 8, 1}
  grid3 = { ceil(m / 32), ceil(n / 8), 1}

  block4 = {256, 1, 1}
  grid4 = { ceil(m / 256), 1, 1}

  prev = System.monotonic_time()

  PolyHok.spawn(&CORR.mean_kernel/6, grid1, block1, [m, n, mean_gpu, data_gpu, float_n, &CORR.mean_kernel_helper/5])
  PolyHok.spawn(&CORR.std_kernel/8, grid2, block2, [m, n, mean_gpu, stddev_gpu, data_gpu, float_n, eps, &CORR.std_kernel_helper/7])
  PolyHok.spawn(&CORR.reduce_kernel/7, grid3, block3, [m, n, mean_gpu, stddev_gpu, data_gpu, float_n, &CORR.reduce_kernel_helper/4])
  PolyHok.spawn(&CORR.corr_kernel/4, grid4, block4, [m, n, symmat_gpu, data_gpu]) #most computation happens here (around 2/3)

  next = System.monotonic_time()

  symmat_gpu_output = PolyHok.get_gnx(symmat_gpu)
  symmat_gpu_output = Nx.indexed_put(symmat_gpu_output, Nx.tensor([[m-1, m-1]]), Nx.tensor([1.0])) # i dont know why this is done

  #CORR.write_tensor_to_file(symmat_gpu_output, "polyhok_output.txt")
  IO.puts "PolyHok\t#{inspect(symmat_gpu_output)}\t#{System.convert_time_unit(next-prev,:native,:millisecond)} "
end

def write_tensor_to_file(list, file_name) do
list
|> Nx.to_flat_list()
|> Enum.map(&Float.to_string/1)
|> Enum.join(" ")
|> then(&File.write!(file_name, &1))

end

end

float_n = 3214212.01
eps = 0.005
type = {:f, 32}
[size_string] = System.argv
size = String.to_integer(size_string)
m = size
n = size

data = CORR.init_arrays(m, n)
mean = Nx.broadcast(Nx.tensor(0, type: type), {m})
stddev = Nx.broadcast(Nx.tensor(0, type: type), {m})
symmat = Nx.broadcast(Nx.tensor(0, type: type), {m, n})

CORR.correlation_polyhok(m, n, data, mean, stddev, symmat, float_n, eps)
