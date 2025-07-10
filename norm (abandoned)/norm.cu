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


    // finding out the type (random_access_traversal_tagE)
    /*
    float* ptr = nullptr;  // just need a pointer of the right type

    using InputIterator = decltype(ptr);
    using Traversal = typename thrust::iterator_traversal<InputIterator>::type;

    std::cout << typeid(Traversal).name() << "\n";  // will be mangled
    */

/*
flow for device_vector:

    //constructor used is probably this one
    template<typename InputIterator>
    device_vector(InputIterator first, InputIterator last)
      :Parent(first,last) {}

    //parent is vector_base

    // so it goes to this:
    template<typename InputIterator>
    vector_base(InputIterator first, InputIterator last);

    // then this

    template<typename T, typename Alloc>
    template<typename InputIterator>
        vector_base<T,Alloc>
        ::vector_base(InputIterator first,
                        InputIterator last)
            :m_storage(),
            m_size(0)
    {
    // check the type of InputIterator: if it's an integral type,
    // we need to interpret this call as (size_type, value_type)
    typedef thrust::detail::is_integral<InputIterator> Integer;

    init_dispatch(first, last, Integer());
    } // end vector_base::vector_base()

    // then this:

    template<typename T, typename Alloc>
    template<typename InputIterator>
        void vector_base<T,Alloc>
        ::init_dispatch(InputIterator first,
                        InputIterator last,
                        false_type)
    {
    range_init(first, last);
    } // end vector_base::init_dispatch()

    // then this:

    template<typename T, typename Alloc>
    template<typename InputIterator>
        void vector_base<T,Alloc>
        ::range_init(InputIterator first,
                    InputIterator last)
    {
    range_init(first, last,
        typename thrust::iterator_traversal<InputIterator>::type());
    } // end vector_base::range_init()

    // then this:

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


/*
flow for transform reduce:
template<typename InputIterator, 
         typename UnaryFunction, 
         typename OutputType,
         typename BinaryFunction>
  OutputType transform_reduce(InputIterator first,
                              InputIterator last,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  using thrust::system::detail::generic::select_system;

  typedef typename thrust::iterator_system<InputIterator>::type System;

  System system;

  return thrust::transform_reduce(select_system(system), first, last, unary_op, init, binary_op);
} // end transform_reduce()

// then this

__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator, 
         typename UnaryFunction, 
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
  OutputType transform_reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  using thrust::system::detail::generic::transform_reduce;
  return transform_reduce(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, unary_op, init, binary_op);
} // end transform_reduce()


// then this

template<typename DerivedPolicy,
         typename InputIterator, 
         typename UnaryFunction, 
         typename OutputType,
         typename BinaryFunction>
__host__ __device__
  OutputType transform_reduce(thrust::execution_policy<DerivedPolicy> &exec,
                              InputIterator first,
                              InputIterator last,
                              UnaryFunction unary_op,
                              OutputType init,
                              BinaryFunction binary_op)
{
  thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> xfrm_first(first, unary_op);
  thrust::transform_iterator<UnaryFunction, InputIterator, OutputType> xfrm_last(last, unary_op);

  return thrust::reduce(exec, xfrm_first, xfrm_last, init, binary_op);
} // end transform_reduce()


// then this
__thrust_exec_check_disable__
template<typename DerivedPolicy,
         typename InputIterator,
         typename T,
         typename BinaryFunction>
__host__ __device__
  T reduce(const thrust::detail::execution_policy_base<DerivedPolicy> &exec,
           InputIterator first,
           InputIterator last,
           T init,
           BinaryFunction binary_op)
{
  using thrust::system::detail::generic::reduce;
  return reduce(thrust::detail::derived_cast(thrust::detail::strip_const(exec)), first, last, init, binary_op);
} // end reduce()
 
*/

