#ifndef ARRAYS_INTERNAL_H
#define ARRAYS_INTERNAL_H

/* --- 
   Internal routines and boilerplate expression templates
   for the Arrays class
   
   Coded by J. de la Cruz Rodriguez (ISP-SU 2019)
   --- */

#include <array>
#include <iostream>
#include <string>
#include <cassert>
#include <csignal>

//

namespace mem{
#if defined(DEBUG_ARRAYS)
  constexpr static const int NBUFFER = 130;
#endif
  
  namespace internal{

    M_INLINE void array_error(std::string const &error_msg, std::string const &error_funct){
      fprintf(stderr, "[error] %s: %s\n", error_funct.c_str(), error_msg.c_str());
      std::raise(SIGINT);
    }

    // --------------------------------------------------------------- //

    template<size_t idx, typename T, size_t N>
    struct linearize_dimensions{
      static  M_INLINE T run(std::array<T,N> const& Dim,
				      std::array<T,N> const& Off,
				      std::array<T,N> const& xx)
      {
#if defined(DEBUG_ARRAYS)
	T const elem = (std::get<idx>(xx)-std::get<idx>(Off));
	if((elem >= std::get<idx>(Dim)) || (elem < 0)){
	  char buffer[NBUFFER]; snprintf(buffer,NBUFFER, "array offset in dimension %ld is out of range [%ld] !C [%ld - %ld], exiting!\n",
				   idx, std::get<idx>(xx),std::get<idx>(Off),std::get<idx>(Off)+std::get<idx>(Dim)-1);
	  
	  array_error(std::string(buffer), "linearize_dimensions::run"); 
	}
	return elem + std::get<idx>(Dim) * linearize_dimensions<idx-1,T,N>::run(Dim,Off,xx);
#else
	return (std::get<idx>(xx)-std::get<idx>(Off)) +
	  std::get<idx>(Dim) * linearize_dimensions<idx-1,T,N>::run(Dim,Off,xx);
#endif
	
      }
    };

    template<typename T, size_t N>
    struct linearize_dimensions<0,T,N>{
      static  M_INLINE T run(std::array<T,N> const& Dim,
				      std::array<T,N> const& Off,
				      std::array<T,N> const& xx)
      {
#if defined(DEBUG_ARRAYS)
	T const elem = (std::get<0>(xx)-std::get<0>(Off));
	if((elem >= std::get<0>(Dim)) || (elem < 0)){
	  char buffer[NBUFFER]; snprintf(buffer,NBUFFER,"array offset in dimension %ld is out of range [%ld] !C [%ld - %ld], exiting!\n",
				   long(0), std::get<0>(xx),std::get<0>(Off),std::get<0>(Off)+std::get<0>(Dim)-1);
	  array_error(std::string(buffer), "linearize_dimensions::run");
	}
	return elem;
#else
	return (std::get<0>(xx)-std::get<0>(Off));
#endif
      }
    };
    
    // --------------------------------------------------------------- //
    
    template<size_t ii, typename T, size_t N>
      M_INLINE void init2Arrays(std::array<T,N> &arr1, std::array<T,N> &arr2, T const v1, T const v2){
      std::get<ii>(arr1) = v2-v1+1; // Dimension
      std::get<ii>(arr2) = v1;      // Offset
    }
    
    template<size_t ii, typename T, size_t N, typename ...S>
      M_INLINE void init2Arrays(std::array<T,N> &arr1, std::array<T,N> &arr2, T const v1, T const v2, S... others){
      std::get<ii>(arr1) = v2-v1+1; // Dimension
      std::get<ii>(arr2) = v1;      // Offset
      init2Arrays<ii+1, T, N>(arr1, arr2, others...);
    }
    
    // --------------------------------------------------------------- //

    template<size_t idx, typename T, size_t N>
    struct array_element_product{
      static constexpr M_INLINE T run(std::array<T,N> const& arr){
	return std::get<idx>(arr) * array_element_product<idx-1,T,N>::run(arr);
      }
    };

    template<typename T, size_t N>
    struct array_element_product<0,T,N>{
      static constexpr M_INLINE T run(std::array<T,N> const& arr){
	return std::get<0>(arr);
      }
    };
    
    // --------------------------------------------------------------- //

    template<typename T, size_t N>
    struct DDim{
      using idx_t = typename Traits<T>::index;
      std::array<idx_t, N> Dim, Off;
      
      M_INLINE DDim(){};

      template<size_t N1>
      explicit DDim(std::array<idx_t,N1> const &in){
	static_assert((N1 == N || N1 == 2*N),
		      "[error] array::DDim: wrong number of dimensions");

	if constexpr (N1 == N){
	    Dim = in;
	    Off = {};
	  }else if(N1 == 2*N)
	  init2Arrays<0,T,N>(Dim,Off,in);
      }
      
      template<typename ...S>
      explicit DDim(const S... in)
      {
	static_assert((sizeof...(S) == N || sizeof...(S) == 2*N),
		      "[error] array::DDim: wrong number of dimensions");
	
	if constexpr (sizeof...(S) == N){
	    Dim = {in...};//std::array<idx_t,N>(in...);
	    Off = {};
	  }else if(sizeof...(S) == 2*N)
	  init2Arrays<0,T,N>(Dim,Off,in...);
	
      }
      
      M_INLINE DDim(DDim<T,N> const& in)
      {
	Dim = in.Dim, Off = in.Off;
      }
      
      
      DDim<T,N> &operator=( DDim<T,N> const& in)
      {
	Dim = in.Dim, Off = in.Off;
	return *this;
      }

      M_INLINE constexpr T size()const{return array_element_product<N-1, T, N>::run(Dim);}

      std::array<T,N> const& getConstRefDim()const{return *static_cast<std::array<T,N> const*>(&Dim);}
      std::array<T,N> const& getConstRefOff()const{return *static_cast<std::array<T,N> const*>(&Off);}
      
    
      };

    // --------------------------------------------------------------- //
    
    template<class T>
    struct Storage{
      size_t n_elements;
      T* data;
      bool allocated;
      
      M_INLINE void allocate(size_t const siz)
      {

	if(!allocated && !data){

	  n_elements = siz;
	  data = amem::aligned_malloc<T>(n_elements*sizeof(T));
	  allocated = true;

	}else if(allocated && data){

	  if(siz != n_elements){
	    //data = amem::aligned_realloc<T>(data, siz, n_elements);
	    amem::aligned_free<T>(data);
	    data = amem::aligned_malloc<T>(n_elements*sizeof(T));
	    
	    n_elements = siz;
	  }
	}else if(data && !allocated){

	  if(n_elements != siz){
	    n_elements = siz; data = NULL;
	    data = amem::aligned_malloc<T>(n_elements*sizeof(T));
	    allocated = true;
	  }
	}
      }
      
      M_INLINE explicit Storage(): n_elements(0), data(NULL), allocated(false){};
      
      M_INLINE explicit Storage(size_t const siz): n_elements(0), data(NULL), allocated(false)
      {
	allocate(siz);
      }
      
      M_INLINE explicit Storage(size_t const siz, T* idata):
	n_elements(siz), data(idata), allocated(false){};
      
      M_INLINE explicit Storage(Storage<T> const& istor):
        Storage<T>()
      {
	allocate(istor.n_elements);
	memcpy(data, istor.data, n_elements*sizeof(T));
      }

      
      M_INLINE explicit Storage(Storage<T> &&istor):
	n_elements(istor.n_elements), data(istor.data), allocated(true)
      {
	istor.allocated = false;
	istor.data = NULL;
	istor.n_elements = 0;
      }

      
      M_INLINE Storage<T>& operator=(Storage<T> const& istor)
      {
	allocate(istor.size());
	
	T* __restrict__ mine = this->data;
	T* const  __restrict__ other = istor.getData();
	for(size_t ii=0; ii<n_elements; ++ii) mine[ii] = other[ii];
	
	return *this;
      }
      
      
      constexpr M_INLINE size_t size()const{return n_elements;}
      M_INLINE        T* getData()      const{return data;}
      M_INLINE  const T*  getDataConst()const{return static_cast<const T*>(data);}

      void resize(size_t const siz)
      {
	allocate(siz);
      }

      void Zero()const{std::memset(data,0,sizeof(T)*n_elements);}
      
      ~Storage()
      {
	if(data && allocated) amem::aligned_free<T>(data);
	data = NULL;
      }

      //M_INLINE       T& operator[](size_t const ii)     {return getData()[ii];}
      M_INLINE  T& operator[](size_t const ii)const{return getData()[ii];}

      
      M_INLINE Storage<T>& operator=(Storage<T> && istor)
      {
	if(allocated && data)
	  amem::aligned_free<T>(data);
	
	data = NULL;
	data = istor.data;
	n_elements = istor.n_elements;


	istor.n_elements = 0;
	istor.data = NULL;
	istor.allocated = 0;
	  	
	return *this;
      }
      
    };
    // --------------------------------------------------------------- //
    
    
  }
}

#endif

