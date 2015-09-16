/* Created by daniel.munoz.trejo at “the e-mail of google dot com” for MiddleMatter Inc. */
#ifndef SMQT_H
#define SMQT_H

#include <vector>
#include <algorithm>
#include <cstdint>
#include <omp.h>

/* I tried enabling nested threads for best performance: omp_set_nested(1), but for some reason
 * It made the unittests crash from time to time arguing that a thread couldn't be created */
template<typename SrcType, typename DstType, typename IndexType, typename AccumulatorType, int L, bool Parallel>
class SMQT
{
public:
    SMQT();
    ~SMQT();

    typedef std::vector<SrcType> InputVectorType;
    typedef std::vector<DstType> OutputVectorType;
    static const uint8_t valueL = L;
    static const bool valueParallel = Parallel;

    void transform(const InputVectorType& input, OutputVectorType& output);

private:
    void smqt(const InputVectorType& input, OutputVectorType& output, IndexType iBegin, IndexType iEnd, int l, size_t currIdx);
    static size_t getNextIdx(size_t currIdx) { return currIdx ? 0 : 1; }

private:
    class Indexes
    {
        IndexType count;

    public:
        Indexes() : count(0) {}
        IndexType operator()() {return count++;}
    };

private:
    typedef std::vector<IndexType> IndexVectorType;
    static const IndexType minSamplesMultiThread = 127;
    IndexVectorType idx[2];
};

template<typename SrcType, typename DstType, typename IndexType, typename AccumulatorType, int L, bool Parallel>
SMQT<SrcType, DstType, IndexType, AccumulatorType, L, Parallel>::SMQT()
{
}

template<typename SrcType, typename DstType, typename IndexType, typename AccumulatorType, int L, bool Parallel>
SMQT<SrcType, DstType, IndexType, AccumulatorType, L, Parallel>::~SMQT()
{
}

template<typename SrcType, typename DstType, typename IndexType, typename AccumulatorType, int L, bool Parallel>
void SMQT<SrcType, DstType, IndexType, AccumulatorType, L, Parallel>::transform(
        const InputVectorType &input,
        OutputVectorType &output)
{
    output.assign(input.size(), DstType()); // Zero fill
    idx[0].resize(input.size());
    std::generate(idx[0].begin(), idx[0].end(), Indexes()); // Fill with [0 to (n - 1)]
    idx[1] = idx[0];
    smqt(input, output, 0, input.size(), 1, 0);
}

template<typename SrcType, typename DstType, typename IndexType, typename AccumulatorType, int L, bool Parallel>
void SMQT<SrcType, DstType, IndexType, AccumulatorType, L, Parallel>::smqt(const InputVectorType& input, OutputVectorType& output, IndexType iBegin, IndexType iEnd, int l, size_t currIdx)
{
    if (l > L || iBegin == iEnd) return;

    const size_t nextIdxPos = getNextIdx(currIdx);
    IndexVectorType& myIdx = idx[currIdx];
    IndexVectorType& nextIdx = idx[nextIdxPos];

    AccumulatorType accum = AccumulatorType();
    for (IndexType i = iBegin; i < iEnd; ++i) {
        accum += input[myIdx[i]];
    }
    const SrcType mean = SrcType(accum / (iEnd - iBegin));
    const DstType weight = DstType(1 << (L - l));
    if (l < L) {
        IndexType iUpper = iBegin;
        for (IndexType i = iBegin; i < iEnd; ++i) {
            if (input[myIdx[i]] <= mean) {
                ++iUpper;
            }
        }
        for (IndexType i = iBegin, i0 = iBegin, i1 = iUpper; i < iEnd; ++i) {
            if (input[myIdx[i]] > mean) {
                output[myIdx[i]] += weight;
                nextIdx[i1++] = myIdx[i];
            } else {
                nextIdx[i0++] = myIdx[i];
            }
        }
        if (Parallel && (iUpper - iBegin) >= minSamplesMultiThread && (iEnd - iUpper) >= minSamplesMultiThread) {
            IndexType positions[] = {iBegin, iUpper, iEnd};
            // Must enable nested parallelism for best performance: omp_set_nested(1);
#pragma omp parallel for
            for (int i = 0; i < 2; ++i) {
                smqt(input, output, positions[i], positions[i + 1], l + 1, nextIdxPos);
            }
        } else {
            smqt(input, output, iBegin, iUpper, l + 1, nextIdxPos);
            smqt(input, output, iUpper, iEnd, l + 1, nextIdxPos);
        }
    } else {
        for (IndexType i = iBegin; i < iEnd; ++i) {
            if (input[myIdx[i]] > mean) {
                output[myIdx[i]] += weight;
            }
        }
    }
}

#endif // SMQT_H
