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
        for i <- 0..ni-1 do
            for j <- 0..nk-1 do
                (i * j) / @ni
            end
        end

    B = 
        for i <- 0..nk-1 do
            for j <- 0..nj-1 do
                (i * (j + 1))  / @nj
            end
        end

    C = 
        for i <- 0..nl-1 do
            for j <- 0..nj-1 do
                (i * (j + 3)) / @nl
            end
        end

    D = 
        for i <- 0..ni-1 do
            for j <- 0..nl-1 do
                (i * (j + 2)) / @nk
            end
        end


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