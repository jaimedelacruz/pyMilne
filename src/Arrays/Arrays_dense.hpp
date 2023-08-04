#ifndef ARRAYS_DENSE_H
#define ARRAYS_DENSE_H

/* ---
   Multidimensional Array class based on templates

   It also allows starting the indexing of a given
   dimension with an offset, very useful when dealing
   with domain decomposition.

   Coded by J. de la Cruz Rodriguez (ISP-SU, 2019)
   --- */

namespace mem{
  template<typename T, size_t N, typename Rep = internal::Storage<T>>
  class Array{
    using idx_t = typename Traits<T>::index;
  private:
    internal::DDim<idx_t,N> Dimensions;
    Rep Data;
  public:

    // ------------------------------------------------------------------ //

    M_INLINE idx_t  shape(size_t const ii)const{return Dimensions.Dim[ii];}
    M_INLINE idx_t offset(size_t const ii)const{return Dimensions.Off[ii];}
    
    // ------------------------------------------------------------------ //
    
    M_INLINE Rep const &getStorage()const
    {return *static_cast<Rep const*>(&Data);}
    
    // ------------------------------------------------------------------ //
    M_INLINE Rep &getStorage()
    {return Data;}
    
    // ------------------------------------------------------------------ //

    M_INLINE T* getDataBlock(){return Data.getData();}
      // ------------------------------------------------------------------ //

    M_INLINE internal::Storage<T> &getData(){return Data;}
    
    // ------------------------------------------------------------------ //

    M_INLINE Array():Dimensions(), Data(){}
    
    // ------------------------------------------------------------------ //

    M_INLINE Array(mem::Array<T,N> &&in):
      Dimensions(in.getDimensions()), Data(std::move(in.getStorage())){}
    


    // ------------------------------------------------------------------ //

    M_INLINE internal::DDim<idx_t,N> const& getDimensions()const
      {return *static_cast<internal::DDim<idx_t,N> const*>(&Dimensions);}

    // ------------------------------------------------------------------ //

    template<typename ...S> M_INLINE explicit Array(const S... xx):
    Dimensions(xx...), Data(Dimensions.size()){}

    // ------------------------------------------------------------------ //

    M_INLINE Array(Array<T,N> const &in):
    Dimensions(in.getDimensions()), Data(in.getStorage()){};
    
    // ------------------------------------------------------------------ //

    M_INLINE explicit Array(Rep const& in): Data(in){}

    // ------------------------------------------------------------------ //

    template<typename ...S> M_INLINE explicit Array(T* datain, const S... xx):
      Dimensions(xx...), Data(Dimensions.size(),datain){}
    
    // ------------------------------------------------------------------ //
    
    template<typename ...S>
    M_INLINE T &operator()(const S... indexes)
    {
      static_assert(sizeof...(S) == N,
		    "mem::Arrays<T,N>::operator(): You are indexing this array with the wrong number of indexes!");
      return Data.getData()
	  [internal::linearize_dimensions<N-1,idx_t,N>::run(Dimensions.getConstRefDim(),
							    Dimensions.getConstRefOff(),
							    {indexes...})];
      }
    
    // ------------------------------------------------------------------ //

    template<typename ...S>
    M_INLINE  const T &operator()(const S... indexes)const
    {
      static_assert(sizeof...(S) == N,
		    "mem::Arrays<T,N>::operator(): You are indexing this array with the wrong number of indexes!");
      return Data.getDataConst()
	[internal::linearize_dimensions<N-1,idx_t,N>::run(Dimensions.getConstRefDim(),
							  Dimensions.getConstRefOff(),
							  {indexes...})];
    }
    
    // ------------------------------------------------------------------ //

    M_INLINE  T& operator[](size_t const ii){return Data[ii];}
    M_INLINE decltype(auto) operator[](size_t const ii)const{return Data[ii];}

    // ------------------------------------------------------------------ //

    M_INLINE void Zero()const
    {
      if constexpr(mem::Traits<T>::zero_with_memset)
	Data.Zero();
      else
	this->operator=(static_cast<T>(0));
    }
    
    // ------------------------------------------------------------------ //

    M_INLINE void ZeroMem()const
    {
      Data.Zero();
    }
    
    // ------------------------------------------------------------------ //

    M_INLINE size_t size()const{return static_cast<size_t>(Dimensions.size());}

    // ------------------------------------------------------------------ //

    template<typename ...S>
      M_INLINE void reshape(const S... indexes)
      {
	static_assert((sizeof...(S) == N || sizeof...(S) == 2*N));
	
	internal::DDim<idx_t, N> const nDim(indexes...);
	
	if(nDim.size() != Dimensions.size()){
	  char tmp[100];
	  sprintf(tmp,"attempted array reshape with wrong element number [%ld] != [%ld]",
		  nDim.size(), Dimensions.size());
	  internal::array_error(std::string(tmp), __FUNCTION__);
	}
	
	Dimensions = nDim;
      }
    
    // ------------------------------------------------------------------ //
    
    Array<T,N,Rep>& operator=(Array<T, N, Rep> && b)
    {
      Dimensions = b.getDimensions();
      Data = std::move(b.getData());
      
      return *this;
    }

    // ------------------------------------------------------------------ //

    Array<T,N,Rep> &operator=(Array<T,N,Rep> const& in)
      {
       Dimensions = in.getDimensions();
       Data = in.getStorage();
       return *this;
      }

    // ------------------------------------------------------------------ //
    
    template<typename T2, typename Rep2>
    Array<T,N,Rep>& operator=(Array<T2, N, Rep2> const& b)
    {
      size_t const nel = Data.size();
      T* __restrict__ iData = Data.getData();
      for(size_t ii=0; ii<nel; ++ii)
	iData[ii] = b[ii];

      return *this;
    }
    
    // ------------------------------------------------------------------ //

    M_INLINE void operator=(T const val)const
      {
       const size_t imax = Data.size();
       T* __restrict__ iData = Data.getData();
       for(size_t ii=0; ii<imax; ++ii) iData[ii] = val;
      }
   
    // ------------------------------------------------------------------ //

    template<typename ...S>
    void resize(const S... indexes)
    {
      static_assert(sizeof...(S) == N || sizeof...(S) == 2*N,
		    "wrong number of indexes");
      
      internal::DDim<idx_t, N> nDim(indexes...);
      
      if(nDim.size() != Dimensions.size())
	Data.allocate(nDim.size());
      
      Dimensions = nDim;
    }

    // ------------------------------------------------------------------ //

    void resize(std::array<idx_t,N> const& din)
    {
      internal::DDim<idx_t, N>nDim(din);
      if(nDim.size() != Dimensions.size())
	Data.allocate(nDim.size());
      
      Dimensions = nDim;
    }
    
    // ------------------------------------------------------------------ //  

    T norm()const{
      const size_t imax = Data.size();
      const T* __restrict__ iData = Data.getData();
      T iNorm = static_cast<T>(0);
      for(size_t ii=0; ii<imax; ++ii) iNorm += iData[ii]*iData[ii];
      return sqrt(iNorm);
    }
    
    // ------------------------------------------------------------------ //  

    Rep const& rep()const{return Data;}
    Rep&       rep()     {return Data;}

    // ------------------------------------------------------------------ //  

    T max()const{
      const size_t nel = Data.size();
      T max_element = Data[0];

      for(size_t ii=1;ii<nel;++ii)
	max_element = (Data[ii] > max_element)? Data[ii] : max_element;

      return max_element;
    }
    
    // ------------------------------------------------------------------ //  
    
    T min()const{
      const size_t nel = Data.size();
      T min_element = Data[0];

      for(size_t ii=1;ii<nel;++ii)
	min_element = (Data[ii] < min_element)? Data[ii] : min_element;

      return min_element;
    }

    // ------------------------------------------------------------------ //  

    idx_t argmin()const
    {
      const size_t nel = Data.size();
      T min_element = Data[0];
      idx_t imin =0;
      
      for(size_t ii=1;ii<nel;++ii){
	if(Data[ii] < min_element){
	  min_element = Data[ii];
	  imin = ii;
	}
      }
      return imin;
    }
    
    // ------------------------------------------------------------------ //  

    idx_t argmax()const
    {
      const size_t nel = Data.size();
      T max_element = Data[0];
      idx_t imax =0;
      
      for(size_t ii=1;ii<nel;++ii){
	if(Data[ii] > max_element){
	  max_element = Data[ii];
	  imax = ii;
	}
      }
      return imax;
    }
    
    // ------------------------------------------------------------------ //  
    
    M_INLINE T sum()const
    {
      const size_t nel = Data.size();
      const T* const __restrict__ iData = Data.getData();

      T sum = static_cast<T>(0);
      for(size_t ii=0; ii<nel;++ii) sum += iData[ii];

      return sum;
    }

    // ------------------------------------------------------------------ //  
    
  }; // class Array
}


#endif

