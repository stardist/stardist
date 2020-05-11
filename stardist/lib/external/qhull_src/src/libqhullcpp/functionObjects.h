/****************************************************************************
**
** Copyright (c) 2008-2018 C.B. Barber. All rights reserved.
** $Id: //main/2015/qhull/src/libqhullcpp/functionObjects.h#3 $$Change: 2549 $
** $DateTime: 2018/12/28 22:24:20 $$Author: bbarber $
**
****************************************************************************/

#ifndef QHFUNCTIONOBJECTS_H
#define QHFUNCTIONOBJECTS_H

#include <stdlib.h>
#include <math.h>

namespace orgQhull {

#//!\name Defined here

    //! Sum of absolute values of the elements in a container
    class AbsoluteSumOf;
    //! Sum of the elements in a container
    class SumOf;
    //! Sum of squares of the elements in a container
    class SumSquaresOf;

#//!\name Class

//! Absolute sum of the elements in a container
class AbsoluteSumOf
{
private:
    double sum;
public:
    inline AbsoluteSumOf() : sum(0.0) {}
    inline void operator()(double v) { sum += fabs(v); }
    inline operator double() { return sum; }
};//AbsoluteSumOf

//! Sum of the elements in a container
class SumOf
{
private:
    double sum;
public:
    inline SumOf() : sum(0.0) {}
    inline void operator()(double v) { sum += v; }
    inline operator double() { return sum; }
};//SumOf


//! Sum of squares of the elements in a container
class SumSquaresOf
{
private:
    double sum;
public:
    inline SumSquaresOf() : sum(0.0) {}
    inline void operator()(double v) { sum += v*v; }
    inline operator double() { return sum; }
};//SumSquaresOf


}//orgQhull


#endif //QHFUNCTIONOBJECTS_H

