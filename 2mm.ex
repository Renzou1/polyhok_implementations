require PolyHok

PolyHok.defmodule MM2 do
    @ni 1024 # probably needs to be changable later
    @nj 1024
    @nl 1024
    @nk 1024

def init_array(ni, nj, nk, nl, A, B, C, D) do
    alpha = 32412
    beta = 2123

    A =
    |> Enum.with_index()
    |> Enum.map(fn {row, i} ->
        row
        |> Enum.with_index()
        |> Enum.map(fn {val, j} ->
        val * (i + j) #change
        end)
    end)    

    {alpha, beta}
end

def compare_results() do

end

def gpu_argv_init() do

end

defk mm2_kernel1() do

end

defk mm2_kernel2() do

end

def print_array() do

end

def mm2_polyhok() do

end

end