#ifndef ARRAYS_TRAITS_H
#define ARRAYS_TRAITS_H

#include <limits>

namespace mem{
  
  template<class T>
  struct Traits{
    using ctype = T;
    using index = long;
    static constexpr const T max = std::numeric_limits<T>::max();
    static constexpr const T min = std::numeric_limits<T>::min();
    static constexpr const bool zero_with_memset = false;
  };

  template<> struct Traits<float>{
    using ctype = double;
    using index = long;
    static constexpr const float max = std::numeric_limits<float>::max();
    static constexpr const float min = std::numeric_limits<float>::min();
    static constexpr const bool zero_with_memset = true;

  };

  template<> struct Traits<int>{
    using ctype = long;
    using index = long;
    static constexpr const int max = std::numeric_limits<int>::max();
    static constexpr const int min = std::numeric_limits<int>::min();
    static constexpr const bool zero_with_memset = true;
  };
  
  template<> struct Traits<size_t>{
    using ctype = size_t;
    using index = long;
    static constexpr const size_t max = std::numeric_limits<size_t>::max();
    static constexpr const size_t min = std::numeric_limits<size_t>::min();
    static constexpr const bool zero_with_memset = true;
  };

  template<> struct Traits<unsigned>{
    using ctype = size_t;
    using index = long;
    static constexpr const size_t max = std::numeric_limits<unsigned>::max();
    static constexpr const size_t min = std::numeric_limits<unsigned>::min();
    static constexpr const bool zero_with_memset = true;
  };

  template<> struct Traits<char>{
    using ctype = long;
    using index = long;
    static constexpr const size_t max = std::numeric_limits<char>::max();
    static constexpr const size_t min = std::numeric_limits<char>::min();
    static constexpr const bool zero_with_memset = true;
  };

  template<> struct Traits<short>{
    using ctype = long;
    using index = long;
    static constexpr const size_t max = std::numeric_limits<short>::max();
    static constexpr const size_t min = std::numeric_limits<short>::min();
    static constexpr const bool zero_with_memset = true;
  };
  
  template<> struct Traits<unsigned short>{
    using ctype = long;
    using index = long;
    static constexpr const size_t max = std::numeric_limits<short>::max();
    static constexpr const size_t min = std::numeric_limits<short>::min();
    static constexpr const bool zero_with_memset = true;
  };
  
}


#endif
