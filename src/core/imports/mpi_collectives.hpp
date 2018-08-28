/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   Copyright (c) 2013, Jeff Hammond
   All rights reserved.

   Copyright (c) 2013, Jed Brown
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El-lite.hpp>
#include "mpi_utils.hpp"

#include <El/core/imports/aluminum.hpp>

namespace El
{

#ifdef HYDROGEN_HAVE_ALUMINUM
Al::ReductionOperator MPI_Op2ReductionOperator(MPI_Op op)
{
    if (op == MPI_SUM)
        return Al::ReductionOperator::sum;
    else if (op == MPI_PROD)
        return Al::ReductionOperator::prod;
    else if (op == MPI_MIN)
        return Al::ReductionOperator::min;
    else if (op == MPI_MAX)
        return Al::ReductionOperator::max;
    else
        LogicError("Given reduction operator not supported.");

    // Silence compiler warning
    return Al::ReductionOperator::sum;
}
#endif // HYDROGEN_HAVE_ALUMINUM

} // namespace El

#include "mpi/AllGather.hpp"
#include "mpi/AllReduce.hpp"
#include "mpi/Broadcast.hpp"
//#include "mpi/Reduce.hpp"
#include "mpi/ReduceScatter.hpp"

// These don't exist in NCCL, so they're left here as a reminder of
// all your unfulfilled hopes and dreams.

//#include "mpi/Gather.hpp"
//#include "mpi/Scatter.hpp"
