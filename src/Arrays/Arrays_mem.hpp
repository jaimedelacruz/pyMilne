/* ---- 
   Routines and macros extracted/ported from the Eigen3 library:
   http://eigen.tuxfamily.org/index.php?title=Main_Page

   Basically they make sure that allocated mem is correctly aligned
   for better performance in loops
   
   Modified and adapted by J. de la Cruz Rodriguez (ISP-SU 2019)
   
   ---- */
#ifndef MEMALIG_H
#define MEMALIG_H

#include <cassert>
#include <cstring>
#include <iostream>
  // ************************************************************************************** //

#define ALIGN_TO_BOUNDARY(n) __attribute__((aligned(n)))

#define MIN_ALIGN_BYTES 16
#if defined(VECTORIZE_AVX512)
// 64 bytes static alignmeent is preferred only if really required
#define IDEAL_MAX_ALIGN_BYTES 64
#elif defined(__AVX__)
// 32 bytes static alignmeent is preferred only if really required
#define IDEAL_MAX_ALIGN_BYTES 32
#else
#define IDEAL_MAX_ALIGN_BYTES 16
#endif

// Shortcuts to ALIGN_TO_BOUNDARY
#define ALIGN8  ALIGN_TO_BOUNDARY(8)
#define ALIGN16 ALIGN_TO_BOUNDARY(16)
#define ALIGN32 ALIGN_TO_BOUNDARY(32)
#define ALIGN64 ALIGN_TO_BOUNDARY(64)

#if IDEAL_MAX_ALIGN_BYTES > MAX_ALIGN_BYTES
#define DEFAULT_ALIGN_BYTES IDEAL_MAX_ALIGN_BYTES
#else
#define DEFAULT_ALIGN_BYTES MAX_ALIGN_BYTES
#endif

#ifndef MALLOC_ALREADY_ALIGNED
#if defined(__GLIBC__) && ((__GLIBC__>=2 && __GLIBC_MINOR__ >= 8) || __GLIBC__>2) \
 && defined(__LP64__) && ! defined( __SANITIZE_ADDRESS__ ) && (DEFAULT_ALIGN_BYTES == 16)
  #define GLIBC_MALLOC_ALREADY_ALIGNED 1
#else
  #define GLIBC_MALLOC_ALREADY_ALIGNED 0
#endif

#if (OS_MAC && (DEFAULT_ALIGN_BYTES == 16))     \
 || (OS_WIN64 && (DEFAULT_ALIGN_BYTES == 16))   \
 || GLIBC_MALLOC_ALREADY_ALIGNED              \
 || FREEBSD_MALLOC_ALREADY_ALIGNED
  #define MALLOC_ALREADY_ALIGNED 1
#else
  #define MALLOC_ALREADY_ALIGNED 0
#endif

#endif

#define ALIGN_TO_BOUNDARY(n) __attribute__((aligned(n)))


// ************************************************************************************** //
namespace amem{

  
  inline void* handmade_aligned_malloc(std::size_t size)
  {
    void *original = std::malloc(size+DEFAULT_ALIGN_BYTES);
    if (original == 0) return 0;
    void *aligned = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(original) & ~(std::size_t(DEFAULT_ALIGN_BYTES-1))) + DEFAULT_ALIGN_BYTES);
    *(reinterpret_cast<void**>(aligned) - 1) = original;
    return aligned;
  }
  
  // ************************************************************************************** //

  /** \internal Frees memory allocated with handmade_aligned_malloc */
  inline void handmade_aligned_free(void *ptr)
  {
    if (ptr) std::free(*(reinterpret_cast<void**>(ptr) - 1));
  }
  
  // ************************************************************************************** //

    inline void* handmade_aligned_realloc(void* ptr, std::size_t size, std::size_t = 0)
    {
      if (ptr == 0) return handmade_aligned_malloc(size);
      void *original = *(reinterpret_cast<void**>(ptr) - 1);
      std::ptrdiff_t previous_offset = static_cast<char *>(ptr)-static_cast<char *>(original);
      
      original = std::realloc(original,size+DEFAULT_ALIGN_BYTES);
      
      if (original == 0) return 0;
      void *aligned = reinterpret_cast<void*>((reinterpret_cast<std::size_t>(original) &
					       ~(std::size_t(DEFAULT_ALIGN_BYTES-1))) + DEFAULT_ALIGN_BYTES);
      void *previous_aligned = static_cast<char *>(original)+previous_offset;
      if(aligned!=previous_aligned)
	std::memmove(aligned, previous_aligned, size);
      
      *(reinterpret_cast<void**>(aligned) - 1) = original;
      
      return aligned;
    }
  
  // ************************************************************************************** //

  template<typename T>
  inline T* aligned_malloc(std::size_t size)
  {
    //check_that_malloc_is_allowed();
    
    void *result;
#if (DEFAULT_ALIGN_BYTES==0) || MALLOC_ALREADY_ALIGNED
    result = std::malloc(size);
#if DEFAULT_ALIGN_BYTES==16
    assert((size<16 || (std::size_t(result)%16)==0) && "System's malloc returned an unaligned pointer. Compile with MALLOC_ALREADY_ALIGNED=0 to fallback to handmade alignd memory allocator.");
#endif
#else
    
    result = handmade_aligned_malloc(size);
#endif
    
    return static_cast<T*>(result);
  }
  
  // ************************************************************************************** //

  template<typename T>
    inline void aligned_free(T *ptr)
    {
      
#if (DEFAULT_ALIGN_BYTES==0) || MALLOC_ALREADY_ALIGNED
      std::free((void*)ptr);
#else
      handmade_aligned_free((void*)ptr);
#endif
    }

  // ************************************************************************************** //
  
  template<typename T> void ignore_unused_variable(const T&) {}
  
  // ************************************************************************************** //
  
  template<typename T>
    inline T* aligned_realloc(T *ptr, std::size_t new_size, std::size_t old_size)
    {
      ignore_unused_variable(old_size);
      
      void *result;
#if (DEFAULT_ALIGN_BYTES==0) || MALLOC_ALREADY_ALIGNED
      result = std::realloc((void*)ptr,new_size);
#else
      result = handmade_aligned_realloc(static_cast<void*>(ptr),new_size,old_size);
#endif
      
      return static_cast<T*>(result);
    }
  
  // ************************************************************************************** //

}


#endif

