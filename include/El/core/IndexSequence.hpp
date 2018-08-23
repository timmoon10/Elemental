#ifndef EL_CORE_INDEXSEQUENCE_HPP_
#define EL_CORE_INDEXSEQUENCE_HPP_

// This is a wrapper file providing a streamlined interface to the
// STL's index_sequence class (technically, typedef). Since I only
// need the index_sequence capability and not integer_sequence, I
// haven't bothered to expose the latter. If the need arises, we can.

#ifdef USE_STL_INDEXSEQUENCE
#include <utility>

namespace El
{
template <size_t... Is>
using IndexSequence = std::index_sequence<Is...>;

template <size_t N>
using MakeIndexSequence = std::make_index_sequence<N>;
}// namespace El
#else

namespace El
{
template <typename IntT, IntT... Is>
struct IntegerSequence {};

template <size_t... Is>
using IndexSequence = IntegerSequence<size_t, Is...>;

template <typename Seq1, typename Seq2>
struct MergeIndexSequences;

template <size_t... Is1, size_t... Is2>
struct MergeIndexSequences<IndexSequence<Is1...>, IndexSequence<Is2...>>
{
    using type = IndexSequence<Is1..., (Is2 + sizeof...(Is1))...>;
};

template <size_t N>
struct GenerateIndexSequence
    : MergeIndexSequences<typename GenerateIndexSequence<N/2>::type,
                          typename GenerateIndexSequence<N-N/2>::type>
{};

template <>
struct GenerateIndexSequence<1>
{
    using type = IndexSequence<0>;
};

template <size_t N>
using MakeIndexSequence = typename GenerateIndexSequence<N>::type;
}// namespace El
#endif // USE_STL_INDEXSEQUENCE

#endif // EL_CORE_INDEXSEQUENCE_HPP_
