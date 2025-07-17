require PolyHok

PolyHok.defmodule SR do

defd get_row(i, c) do
  return i / c
end

defk reduce_by_key() do


end

end


r = 5
c = 8

key = Nx.Random.key(1701)
{x, _new_key} = Nx.Random.randint(key, 10, 100, shape: {r, c}, type: :u32)
row_sums = Nx.broadcast(0, {r})
row_indices = Nx.broadcast(0, {r})
