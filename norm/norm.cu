#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>

//   This example computes the norm [1] of a vector.  The norm is 
// computed by squaring all numbers in the vector, summing the 
// squares, and taking the square root of the sum of squares.  In
// Thrust this operation is efficiently implemented with the 
// transform_reduce() algorith.  Specifically, we first transform
// x -> x^2 and the compute a standard plus reduction.  Since there
// is no built-in functor for squaring numbers, we define our own
// square functor.
//
// [1] http://en.wikipedia.org/wiki/Norm_(mathematics)#Euclidean_norm


// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__
        T operator()(const T& x) const { 
            return x * x;
        }
};

int main(void)
{
    // initialize host array
    float x[4] = {1.0, 2.0, 3.0, 4.0};

    // start timer
    float time;
    cudaEvent_t start, stop;   
    cudaEventCreate(&start) ;
    cudaEventCreate(&stop) ;
    cudaEventRecord(start, 0) ;    
    
    // transfer to device
    thrust::device_vector<float> d_x(x, x + 4);

    // setup arguments
    square<float>        unary_op;
    thrust::plus<float> binary_op;
    float init = 0;

    // compute norm
    float norm = std::sqrt( thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op) );

    // end timer
    cudaEventRecord(stop, 0) ;
    cudaEventSynchronize(stop) ;
    cudaEventElapsedTime(&time, start, stop) ;

    printf("CUDA\t%3.1f\n", time);

    std::cout << "norm is " << norm << std::endl;

    return 0;
}

/*
relevant mess:

    //constructor used is probably this one
    template<typename InputIterator>
    device_vector(InputIterator first, InputIterator last)
      :Parent(first,last) {}

    //parent is vector_base

    // so it goes to this:
    template<typename InputIterator>
    vector_base(InputIterator first, InputIterator last);

    // which then goes to this
    template<typename T, typename Alloc>
    template<typename InputIterator>
        void vector_base<T,Alloc>
        ::range_init(InputIterator first,
                    InputIterator last)
    {
    range_init(first, last,
        typename thrust::iterator_traversal<InputIterator>::type());
    } // end vector_base::range_init()


    // then one of these two:

    template<typename T, typename Alloc>
    template<typename InputIterator>
        void vector_base<T,Alloc>
        ::range_init(InputIterator first,
                    InputIterator last,
                    thrust::incrementable_traversal_tag)
    {
    for(; first != last; ++first)
        push_back(*first);
    } // end vector_base::range_init()

    template<typename T, typename Alloc>
    template<typename ForwardIterator>
        void vector_base<T,Alloc>
        ::range_init(ForwardIterator first,
                    ForwardIterator last,
                    thrust::random_access_traversal_tag)
    {
    size_type new_size = thrust::distance(first, last);

    allocate_and_copy(new_size, first, last, m_storage);
    m_size    = new_size;
    } // end vector_base::range_init()
*/
