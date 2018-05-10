/*
   Copyright (c) 2009-2016, Jack Poulson
   All rights reserved.

   This file is part of Elemental and is under the BSD 2-Clause License,
   which can be found in the LICENSE file in the root directory, or at
   http://opensource.org/licenses/BSD-2-Clause
*/
#include <El.hpp>
using namespace El;

template<typename T,Device D=Device::CPU>
void TestGemv
(Orientation orientA,
 Int m,
 T alpha,
 T beta,
 bool print,
 const Grid& g)
{
    OutputFromRoot(g.Comm(),"Testing with ",TypeName<T>());
    PushIndent();

    DistMatrix<T,MC,MR,ELEMENT,D> A(g), x(g), y(g);

    Uniform(A, m, m);
    Uniform(x, m, 1);
    Uniform(y, m, 1);
    if (print)
    {
        Print(A, "A");
        Print(x, "x");
        Print(y, "y");
    }

    // Test Gemv
    OutputFromRoot(g.Comm(),"Starting Gemv");
    mpi::Barrier(g.Comm());
    Timer timer;
    timer.Start();
    Gemv(orientA, alpha, A, x, beta, y);
    mpi::Barrier(g.Comm());
    const double runTime = timer.Stop();
    const double realGFlops = 2.*double(m)*double(m)/(1.e9*runTime);
    const double gFlops = (IsComplex<T>::value ? 4*realGFlops : realGFlops);
    OutputFromRoot
    (g.Comm(),"Finished in ",runTime," seconds (",gFlops," GFlop/s");
    if (print)
        Print(y, BuildString("y := ",alpha," Symm(A) x + ",beta," y"));

    PopIndent();
}

int
main(int argc, char* argv[])
{
    Environment env(argc, argv);
    mpi::Comm comm = mpi::COMM_WORLD;

    try
    {
        int gridHeight = Input("--gridHeight","height of process grid",0);
        const bool colMajor = Input("--colMajor","column-major ordering?",true);
        const char transA = Input("--transA","orientation of A: N/T/C",'N');
        const Int m = Input("--m","height of matrix",100);
        const Int nb = Input("--nb","algorithmic blocksize",96);
        const bool print = Input("--print","print matrices?",false);
        ProcessInput();
        PrintInputReport();


        const Orientation orientA = CharToOrientation(transA);
        if (gridHeight == 0)
            gridHeight = Grid::DefaultHeight(mpi::Size(comm));
        const GridOrder order = (colMajor ? COLUMN_MAJOR : ROW_MAJOR);
        const Grid g(comm, gridHeight, order);
        SetBlocksize(nb);

        ComplainIfDebug();
        OutputFromRoot(comm,"Will test Gemv ",transA);

#ifdef HYDROGEN_HAVE_CUDA
        TestGemv<float,Device::GPU>
            (orientA, m,
             float(3), float(4),
             print, g);
        TestGemv<double,Device::GPU>
            (orientA, m,
             double(3), double(4),
             print, g);
#endif // HYDROGEN_HAVE_CUDA

        TestGemv<float>
        (orientA, m,
          float(3), float(4),
          print, g);
        TestGemv<Complex<float>>
        (orientA, m,
          Complex<float>(3), Complex<float>(4),
          print, g);

        TestGemv<double>
        (orientA, m,
          double(3), double(4),
          print, g);
        TestGemv<Complex<double>>
        (orientA, m,
          Complex<double>(3), Complex<double>(4),
          print, g);

#ifdef EL_HAVE_QD
        TestGemv<DoubleDouble>
        (orientA, m,
          DoubleDouble(3), DoubleDouble(4),
          print, g);
        TestGemv<QuadDouble>
        (orientA, m,
          QuadDouble(3), QuadDouble(4),
          print, g);

        TestGemv<Complex<DoubleDouble>>
        (orientA, m,
          Complex<DoubleDouble>(3), Complex<DoubleDouble>(4),
          print, g);
        TestGemv<Complex<QuadDouble>>
        (orientA, m,
          Complex<QuadDouble>(3), Complex<QuadDouble>(4),
          print, g);
#endif

#ifdef EL_HAVE_QUAD
        TestGemv<Quad>
        (orientA, m,
          Quad(3), Quad(4),
          print, g);
        TestGemv<Complex<Quad>>
        (orientA, m,
          Complex<Quad>(3), Complex<Quad>(4),
          print, g);
#endif

#ifdef EL_HAVE_MPC
        TestGemv<BigFloat>
        (orientA, m,
          BigFloat(3), BigFloat(4),
          print, g);
        TestGemv<Complex<BigFloat>>
        (orientA, m,
          Complex<BigFloat>(3), Complex<BigFloat>(4),
          print, g);
#endif
    }
    catch(exception& e)
    { ReportException(e); }

    return 0;
}
