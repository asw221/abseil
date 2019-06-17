
#include <type_traits>


#ifndef _ABSEIL_TYPE_DEFINITIONS_
#define _ABSEIL_TYPE_DEFINITIONS_


// --- has_value_type ----------------------------------------------------------
template< typename T, typename = void >
struct has_value_type : std::false_type
{};


template< typename T >
struct has_value_type<T,
  std::void_t< typename T::value_type > >
  : std::true_type
{};


template< typename T >
inline constexpr bool has_value_type_v = has_value_type<T>::value;


// --- is_iterable -------------------------------------------------------------
template< typename T, typename = void, typename = void >
struct is_iterable : std::false_type
{};


template< typename T >
struct is_iterable<T, std::void_t< decltype(std::declval<T>().begin()) >,
		   std::void_t< decltype(std::declval<T>().end()) > >
  : std::true_type
{};


template< typename T >
inline constexpr bool is_iterable_v = is_iterable<T>::value;


// --- has_size ----------------------------------------------------------------
template< typename T, typename = void >
struct has_size : std::false_type
{};


template< typename T >
struct has_size<T, std::void_t< decltype(std::declval<T>().size()) > >
  : std::true_type
{};


template< typename T >
inline constexpr bool has_size_v = has_size<T>::value;

// template< typename T >
// using has_size_t = has_size<T>::type;




// --- is_indexable ------------------------------------------------------------
template< typename T, typename = void, typename = void >
struct is_indexable : std::false_type
{};


template< typename T >
struct is_indexable<T, std::void_t< decltype(std::declval<T>().size()) >,
		    std::void_t< decltype(std::declval<T>().operator[](0)) > >
  : std::true_type
{};


template< typename T >
inline constexpr bool is_indexable_v = is_indexable<T>::value;



// --- is_vector_like_v --------------------------------------------------------
template< typename T >
inline constexpr bool is_vector_like_v = is_iterable<T>::value ||
  (is_indexable<T>::value && has_size<T>::value);


#endif  // _ABSEIL_TYPE_DEFINITIONS_
