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


#ifndef EL_IMPORTS_MPIUTILS_HPP
#define EL_IMPORTS_MPIUTILS_HPP

namespace {

inline void
SafeMpi( int mpiError ) EL_NO_RELEASE_EXCEPT
{
    EL_DEBUG_ONLY(
      if( mpiError != MPI_SUCCESS )
      {
          char errorString[MPI_MAX_ERROR_STRING];
          int lengthOfErrorString;
          MPI_Error_string( mpiError, errorString, &lengthOfErrorString );
          El::RuntimeError( std::string(errorString) );
      }
    )
}

template<typename T>
MPI_Op NativeOp( const El::mpi::Op& op )
{
    MPI_Op opC;
    if( op == El::mpi::SUM )
        opC = El::mpi::SumOp<T>().op;
    else if( op == El::mpi::PROD )
        opC = El::mpi::ProdOp<T>().op;
    else if( op == El::mpi::MAX )
        opC = El::mpi::MaxOp<T>().op;
    else if( op == El::mpi::MIN )
        opC = El::mpi::MinOp<T>().op;
    else
        opC = op.op;
    return opC;
}
}
#endif // ifndef EL_IMPORTS_MPIUTILS_HPP

