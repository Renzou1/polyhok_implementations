require PolyHok

PolyHok.defmodule CORR do

def init_arrays(m, n) do

  i = Nx.tensor(Enum.to_list(0..(m - 1)))
  j = Nx.tensor(Enum.to_list(0..(n - 1)))

  # Reshape for broadcasting: i as column vector, j as row vector
  i_mat = Nx.reshape(i, {m, 1})
  j_mat = Nx.reshape(j, {1, n})

  # Broadcasted multiply then divide
  data = Nx.divide(Nx.multiply(i_mat , j_mat), m)

  #write_tensor_to_file(data, "data.txt")

  data
end

defk mean_kernel(m, n, mean, data, float_n, f) do
	j = blockIdx.x * blockDim.x _ threadIdx.x

    if (j < m) do
      mean[j] = f(mean[j], data, float_n, j)
    end

end

defd mean_kernel_helper(mean_j, data, float_n, j) do
  mean_j = 0.0

  for i in range(0, n, 1) do
    mean_j = mean_j + data[i*m + j]
  end

  return (mean_j / float_n)

end


defk std_kernel(m, n, mean, std, data, float_n, eps, f) do
  j = blockIdx.x * blockDim.x + threadIdx.x

  if (j < m) do
    std[j] = f(std[j], data, mean, float_n, eps)
  end
end

defd std_kernel_helper(std_j, data, mean, float_n, eps) do
		std_j = 0.0

    for i in range(0, n, 1) do
			std_j = std_j + (data[i*M + j] - mean[j]) * (data[i*M + j] - mean[j])
    end

		std_j = std_j / float_n
		std_j = math.sqrt(std_j)
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
  data_im_j = data_im_j - mean_j
  data_im_j = data_im_j / (math.sqrt(float_n) * std_j)
  return data_im_j
end


defk corr_kernel(m, n, symmat, data) do
  j1 = blockIdx.x * blockDim.x + threadIdx.x

  if (j1 < m - 1) do

		symmat[j1*M + j1] = 1.0

    for j2 in range(j1 + 1, m, 1) do

      symmat[j1*M + j2] = 0.0

      for i in range(0, n, 1) do

				symmat[j1*M + j2] = symmat[j1*M + j2] + data[i*M + j1] * data[i*M + j2]

      end

			symmat[j2*M + j1] = symmat[j1*M + j2]

    end

  end
end

def correlation_polyhok(m, n, data, mean, stddev, symmat, symmat_gpu_output, float_n, eps) do
  data_gpu = PolyHok.new_gnx(data)
  symmat_gpu = PolyHok.new_gnx(symmat)
  stddev_gpu = PolyHok.new_gnx(stddev)
  mean_gpu = PolyHok.new_gnx(mean)

  block1 = {256, 1, 1}
  grid1 = { ceil(m / 256), 1, 1}

  block2 = {256, 1, 1}
  grid2 = { ceil(m / 256), 1, 1}

  block3 = {32, 8, 1}
  grid3 = { ceil(m / 32), n / 8, 1}

  block4 = {256, 1, 1}
  grid4 = { ceil(m / 256), 1, 1}

  #defk mean_kernel(m, n, mean, data, float_n, f) do, defd mean_kernel_helper(mean_j, data, float_n, j) do
  #defk std_kernel(m, n, mean, std, data, float_n, eps, f) do, defd std_kernel_helper(std_j, data, mean, float_n, eps) do
  #defk reduce_kernel(m, n, mean, std, data, float_n, f) do, defd reduce_kernel_helper(data_im_j, mean_j, std_j, float_n) do
  #defk corr_kernel(m, n, symmat, data) do

  prev = System.monotonic_time()

  PolyHok.spawn(&CORR.mean_kernel/5, grid1, block1, [m, n, mean_gpu, data_gpu, float_n, &CORR.mean_kernel_helper/4])
  PolyHok.spawn(&CORR.std_kernel/8, grid2, block2, [m, n, mean_gpu, stddev_gpu, data_gpu, float_n, eps, &CORR.std_kernel_helper/5])
  PolyHok.spawn(&CORR.reduce_kernel/7, grid3, block3, [m, n, mean_gpu, stddev_gpu, data_gpu, float_n, &CORR.reduce_kernel_helper/4])
  PolyHok.spawn(&CORR.corr_kernel/4, grid3, block3, [m, n, symmat_gpu, data_gpu])

  next = System.monotonic_time()

  symmat_gpu_output = PolyHok.get_gnx(symmat_gpu)
  symmat_gpu_output[(m-1)*m + (m-1)] = 1.0 # i dont know why this is done

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
data = CORR.init_arrays(10, 10)

data = Nx.broadcast(Nx.tensor(0, type: type), {m, n})
mean = Nx.broadcast(Nx.tensor(0, type: type), {m, n})
stddev = Nx.broadcast(Nx.tensor(0, type: type), {m, n})
symmat = Nx.broadcast(Nx.tensor(0, type: type), {m, n})
symmat_gpu_output = Nx.broadcast(Nx.tensor(0, type: type), {m, n})
