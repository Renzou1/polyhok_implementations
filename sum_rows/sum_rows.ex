require PolyHok

PolyHok.defmodule SR do
include CAS

defd get_row(i, c) do
  return i / c
end

def reduce_by_key(arr, columns, row_indices, row_sums) do
# if i/c + 1 != i/c then last_instance_of_this_key[i/c] = i
# just use reduce every c indexes
# for every c-sized slice of arr, reduce(start, c)

#reduce(arr[0..c], 0.0, PolyHok.phok fn ...)
#reduce(arr[c..c*2])...

end

defk reduce_by_key_kernel() do

end

def reduce(ref, initial, f) do

    {l,c} = PolyHok.get_shape_gnx(ref)
    type = PolyHok.get_type_gnx(ref)
    size = l*c
    result_gpu  = PolyHok.new_gnx(Nx.tensor([[initial]] , type: type))

    threadsPerBlock = 256
    blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
    numberOfBlocks = blocksPerGrid
    PolyHok.spawn(&DP.reduce_kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial,f, size])
    result_gpu
end

defk reduce_kernel(a, ref4, initial,f,n) do

  __shared__ cache[256]

  tid = threadIdx.x + blockIdx.x * blockDim.x;
  cacheIndex = threadIdx.x

  temp = initial

  while (tid < n) do
    temp = f(a[tid], temp)
    tid = blockDim.x * gridDim.x + tid
  end

  cache[cacheIndex] = temp
    __syncthreads()

  i = blockDim.x/2

  while (i != 0 ) do  ###&& tid < n) do
    #tid = blockDim.x * gridDim.x + tid
    if (cacheIndex < i) do
      cache[cacheIndex] = f(cache[cacheIndex + i] , cache[cacheIndex])
    end

  __syncthreads()
  i = i/2
  end

  if (cacheIndex == 0) do
    current_value = ref4[0]
    while(!(current_value == atomic_cas(ref4,current_value,f(cache[0],current_value)))) do
      current_value = ref4[0]
    end
  end

end

end

r = 5
c = 8

{array, _new_key} = Nx.Random.randint(Nx.Random.key(1701), 10, 100, shape: {r * c}, type: :u32)
row_sums = Nx.broadcast(0, {r})
row_indices = Nx.broadcast(0, {r})

row_sums_gpu = PolyHok.new_gnx(row_sums)
row_indices_gpu = PolyHok.new_gnx(row_indices_gpu)
