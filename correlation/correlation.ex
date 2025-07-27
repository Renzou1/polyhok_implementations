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


end

def write_tensor_to_file(list, file_name) do
list
|> Nx.to_flat_list()
|> Enum.map(&Float.to_string/1)
|> Enum.join(" ")
|> then(&File.write!(file_name, &1))

end

end

CORR.init_arrays(100, 100)
