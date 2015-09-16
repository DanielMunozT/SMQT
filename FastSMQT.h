/* Created by daniel.munoz.trejo at “the e-mail of google that com” for MiddleMatter Inc. */
#ifndef FASTSMQT_H
#define FASTSMQT_H

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <vector>
#include <iostream>
#include <limits>
#include <type_traits>
#include <omp.h>

template<
        /* Data type of the input vector. */
        class InputType,

        /* Data type of the output vector. This doesn't necessarily has to be the
           same type as the input. This type must be able to hold the
           output range, which is: 1 << L
           So it must at least have L bits. */
        class OutputType,

        /* The accumulator created with this type must be able to hold the sum of
           all the input elements. */
        class AccumType,

        /* Output quantization levels. */
        uint8_t L,

        /* Parallelize the process. Since many instances of this class could be run
           in parallel to process different input vectors, the user may not want to
           parallelize each of those processes. */
        bool Parallel,

        /* The maximum value of the input. By default it'll be the maximum value that
           the destination type can hold. But if the user knows that the maximum value
           will never be greater than certain value (which is less than the maximum
           value for its type), setting the correct InputMax value will save memory
           and time. */
        size_t InputMax = static_cast<size_t>(std::numeric_limits<InputType>::max()) + 1>
class FastSMQT
{
public:
    FastSMQT();
    ~FastSMQT();

    /* Definitions exposed to the user and to methods. */
    typedef std::vector<InputType> InputVectorType;
    typedef std::vector<OutputType> OutputVectorType;
    static const uint8_t valueL = L;
    static const bool valueParallel = Parallel;
    static const size_t inputMax = InputMax;
    static const size_t outputRange = 1 << L;

    /* Method that will do the SMQT transformation.
       "input" and "output" can be the same vector if they are of the same type
       and the transformation will be done in place. */
    void transform(const InputVectorType& input, OutputVectorType& output);

private:
    void countInputValuesSingle(const InputVectorType& input);
    void countInputValuesParallel(const InputVectorType& input);

    void accumulateAndSum();
    void calculateVals(size_t iBegin, size_t iEnd, OutputType weight);
    void sumVals();

    void transformOutputsSingle(const InputVectorType& input, OutputVectorType& output);
    void transformOutputsParallel(const InputVectorType& input, OutputVectorType& output);

private:
    typedef std::vector<AccumType> CountVectorType;
    CountVectorType accumSum;
    CountVectorType accumCount;
    CountVectorType count;
    InputVectorType next;
    InputVectorType prev;
    OutputVectorType val;
    size_t iMin;
    size_t iMax;
};

template<class InputType, class OutputType, class AccumType, uint8_t L, bool Parallel, size_t InputMax>
FastSMQT<InputType, OutputType, AccumType, L, Parallel, InputMax>::FastSMQT()
    : accumSum(InputMax)
    , accumCount(InputMax)
    , next(InputMax)
    , prev(InputMax)
{
    static_assert(std::numeric_limits<size_t>::max() >= InputMax,
                  "size_t must be able to hold InputMax. See InputMax declaration.");
}

template<class InputType, class OutputType, class AccumType, uint8_t L, bool Parallel, size_t InputMax>
FastSMQT<InputType, OutputType, AccumType, L, Parallel, InputMax>::~FastSMQT()
{
}

template<class InputType, class OutputType, class AccumType, uint8_t L, bool Parallel, size_t InputMax>
void FastSMQT<InputType, OutputType, AccumType, L, Parallel, InputMax>::transform(const InputVectorType& input, OutputVectorType& output)
{
    assert((InputMax * input.size()) <= std::numeric_limits<AccumType>::max());

    // Reset count and val values.
    count.assign(InputMax, InputType());
    val.assign(InputMax, InputType());

    // Resize output to match input size.
    output.resize(input.size());

    // Compile time if, will be optimized.
    if (Parallel) {
        countInputValuesParallel(input);
    } else {
        countInputValuesSingle(input);
    }

    accumulateAndSum();
    calculateVals(iMin, iMax, static_cast<OutputType>(outputRange / 2));
    sumVals();

    if (Parallel) {
        transformOutputsParallel(input, output);
    } else {
        transformOutputsSingle(input, output);
    }
}

// Complexity: N
template<class InputType, class OutputType, class AccumType, uint8_t L, bool Parallel, size_t InputMax>
void FastSMQT<InputType, OutputType, AccumType, L, Parallel, InputMax>::countInputValuesSingle(const InputVectorType& input)
{
    const size_t lastChunk = ((input.size() / 8) * 8);
    for (size_t i = 0; i < lastChunk; i += 8) { // Loop unrolling
            ++count[input[i]];
            ++count[input[i + 1]];
            ++count[input[i + 2]];
            ++count[input[i + 3]];
            ++count[input[i + 4]];
            ++count[input[i + 5]];
            ++count[input[i + 6]];
            ++count[input[i + 7]];
    }
    for (size_t i = lastChunk, size = input.size(); i < size; ++i) {
        ++count[input[i]];
    }
}

// Complexity: N
template<class InputType, class OutputType, class AccumType, uint8_t L, bool Parallel, size_t InputMax>
void FastSMQT<InputType, OutputType, AccumType, L, Parallel, InputMax>::countInputValuesParallel(const InputVectorType& input)
{
    std::vector<CountVectorType> counts;

#pragma omp parallel
    {
#pragma omp single
        counts.assign(omp_get_num_threads(), count);
#pragma omp barrier
        const int tNum = omp_get_thread_num();
        CountVectorType& myCount = counts[tNum];
        const int totalThreads = omp_get_num_threads();
        size_t iPoints = input.size() / totalThreads;
        const size_t iStart = tNum * iPoints;
        if (tNum == (totalThreads - 1)) {
            iPoints = input.size() - iStart;
        }
        const size_t iEnd = iPoints + iStart;
        const size_t lastChunk = ((iPoints / 8) * 8) + iStart;
        for (size_t i = iStart; i < lastChunk; i += 8) { // Loop unrolling
            ++myCount[input[i]];
            ++myCount[input[i + 1]];
            ++myCount[input[i + 2]];
            ++myCount[input[i + 3]];
            ++myCount[input[i + 4]];
            ++myCount[input[i + 5]];
            ++myCount[input[i + 6]];
            ++myCount[input[i + 7]];
        }
        for (size_t i = lastChunk; i < iEnd; ++i) {
            ++myCount[input[i]];
        }
    }

    const size_t totalThreads = counts.size();

    for (size_t thread = 0; thread < totalThreads; ++thread) {
#pragma omp simd
        for (size_t i = 0; i < InputMax; ++i) {
            count[i] += counts[thread][i];
        }
    }
}

// Complexity <= 256 (2^L)
template<class InputType, class OutputType, class AccumType, uint8_t L, bool Parallel, size_t InputMax>
void FastSMQT<InputType, OutputType, AccumType, L, Parallel, InputMax>::accumulateAndSum()
{
    iMin = 0;
    while (!count[iMin]) ++iMin;
    AccumType ac = AccumType();
    AccumType acCnt = AccumType();
    size_t lastElem = iMin;
    for (size_t i = iMin; i < InputMax; ++i) {
        if (count[i]) {
            next[lastElem] = static_cast<InputType>(i);
            prev[i] = static_cast<InputType>(lastElem);
            next[i] = static_cast<InputType>(i);
            lastElem = static_cast<InputType>(i);
            accumCount[i] = acCnt = count[i] + acCnt;
            accumSum[i] = ac = count[i] * i + ac;
            iMax = i;
        } else {
            prev[i] = lastElem;
        }
    }
}

// Complexity <= 256 (2^L)
template<class InputType, class OutputType, class AccumType, uint8_t L, bool Parallel, size_t InputMax>
void FastSMQT<InputType, OutputType, AccumType, L, Parallel, InputMax>::calculateVals(size_t iBegin, size_t iEnd, OutputType weight)
{
    if (accumCount[iBegin] < accumCount[iEnd]) { // If there is only one element, it will never be higher than itself (the mean)
        const bool first = prev[iBegin] == iBegin;
        const AccumType lastAcSum = first ? 0 : accumSum[prev[iBegin]];
        const AccumType lastAcCount = first ? 0 : accumCount[prev[iBegin]];
        const size_t mean = (accumSum[iEnd] - lastAcSum) / (accumCount[iEnd] - lastAcCount);
        const size_t iUpper = count[mean] ? next[mean] : next[prev[mean]];
        val[iUpper] += weight;
        if (next[iEnd] != iEnd) {
            val[next[iEnd]] -= weight;
        }
        const OutputType nextWeight = weight >> 1;
        if (nextWeight) {
            calculateVals(iBegin, prev[iUpper], nextWeight);
            calculateVals(iUpper, iEnd, nextWeight);
        }
    }
}

// Complexity <= 256 (2^L)
template<class InputType, class OutputType, class AccumType, uint8_t L, bool Parallel, size_t InputMax>
void FastSMQT<InputType, OutputType, AccumType, L, Parallel, InputMax>::sumVals()
{
    if (iMin < iMax) {
        size_t i = next[iMin];
        for (; i < iMax; i = next[i]) {
            val[i] += val[prev[i]];
        }
        val[i] += val[prev[i]];
    }
}

// Complexity = N
template<class InputType, class OutputType, class AccumType, uint8_t L, bool Parallel, size_t InputMax>
void FastSMQT<InputType, OutputType, AccumType, L, Parallel, InputMax>::transformOutputsSingle(const InputVectorType& input, OutputVectorType &output)
{
    const size_t lastChunk = ((input.size() / 8) * 8);

    for (size_t i = 0; i < lastChunk; i += 8) { // Loop unrolling
        output[i] = val[input[i]];
        output[i + 1] = val[input[i + 1]];
        output[i + 2] = val[input[i + 2]];
        output[i + 3] = val[input[i + 3]];
        output[i + 4] = val[input[i + 4]];
        output[i + 5] = val[input[i + 5]];
        output[i + 6] = val[input[i + 6]];
        output[i + 7] = val[input[i + 7]];
    }

    for (size_t i = lastChunk, size = output.size(); i < size; ++i) {
        output[i] = val[input[i]];
    }
}

// Complexity = N
template<class InputType, class OutputType, class AccumType, uint8_t L, bool Parallel, size_t InputMax>
void FastSMQT<InputType, OutputType, AccumType, L, Parallel, InputMax>::transformOutputsParallel(const InputVectorType& input, OutputVectorType &output)
{
    const size_t lastChunk = ((input.size() / 8) * 8);

#pragma omp parallel for firstprivate(lastChunk)
    for (size_t i = 0; i < lastChunk; i += 8) { // Loop unrolling
        output[i] = val[input[i]];
        output[i + 1] = val[input[i + 1]];
        output[i + 2] = val[input[i + 2]];
        output[i + 3] = val[input[i + 3]];
        output[i + 4] = val[input[i + 4]];
        output[i + 5] = val[input[i + 5]];
        output[i + 6] = val[input[i + 6]];
        output[i + 7] = val[input[i + 7]];
    }

    for (size_t i = lastChunk, size = output.size(); i < size; ++i) {
        output[i] = val[input[i]];
    }
}

#endif // FASTSMQT_H
