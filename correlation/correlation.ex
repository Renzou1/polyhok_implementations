require PolyHok

PolyHok.defmodule CORR do

def init_arrays(m, n) do

  i = Nx.tensor(Enum.to_list(0..(m - 1)))
  j = Nx.tensor(Enum.to_list(0..(n - 1)))

  # Reshape for broadcasting: i as column vector, j as row vector
  i_mat = Nx.reshape(i, {m, 1})
  j_mat = Nx.reshape(j, {1, n})

  # Broadcasted multiply then divide
  data = Nx.divide(Nx.multiply(i_mat , j_mat), Nx.tensor(m, type: {:f, 32}))

  #write_tensor_to_file(data, "data.txt")

  data
end

defk mean_kernel(m, n, mean, data, float_n, f) do
	j = blockIdx.x * blockDim.x + threadIdx.x

    if (j < m) do
      mean[j] = f(m, n, data, float_n, j)
    end

end

defd mean_kernel_helper(m, n, data, float_n, j) do
  mean_j = 0.0

  for i in range(0, n, 1) do
    mean_j = mean_j + data[i*m + j]
  end

  return (mean_j / float_n)

end


defk std_kernel(m, n, mean, std, data, float_n, eps, f) do
  j = blockIdx.x * blockDim.x + threadIdx.x

  if (j < m) do
    std[j] = f(m, n, data, mean, float_n, eps, j)
  end
end

defd std_kernel_helper(m, n, data, mean, float_n, eps, j) do
		std_j = 0.0

    for i in range(0, n, 1) do
			std_j = std_j + (data[i*m + j] - mean[j]) * (data[i*m + j] - mean[j])
    end

		std_j = std_j / float_n
		std_j = sqrt(std_j)
		if(std_j <= eps) do
			std_j = 1.0
    end

    return std_j

end

defk reduce_kernel(m, n, mean, std, data, float_n, f) do
	j = blockIdx.x * blockDim.x + threadIdx.x
	i = blockIdx.y * blockDim.y + threadIdx.y

  if(i < n && j < m) do
    data[i * m + j] = f(data[i * m + j], mean[j], std[j], float_n)
  end
end

defd reduce_kernel_helper(data_im_j, mean_j, std_j, float_n) do
  return (data_im_j - mean_j) / (sqrt(float_n) * std_j)
end


defk corr_kernel(m, n, symmat, data) do
  j1 = blockIdx.x * blockDim.x + threadIdx.x

  if (j1 < m - 1) do

		symmat[j1*m + j1] = 1.0

    for j2 in range(j1 + 1, m, 1) do

      symmat[j1*m + j2] = 0.0

      for i in range(0, n, 1) do

				symmat[j1*m + j2] = symmat[j1*m + j2] + data[i*m + j1] * data[i*m + j2]

      end

			symmat[j2*m + j1] = symmat[j1*m + j2]

    end

  end
end

def correlation_polyhok(m, n, data, mean, stddev, symmat, float_n, eps) do
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

  CORR.write_tensor_to_file(symmat_gpu_output, "polyhok_output.txt")
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
