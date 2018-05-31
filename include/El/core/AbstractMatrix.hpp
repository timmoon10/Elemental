#ifndef HYDROGEN_ABSTRACTMATRIX_HPP_
#define HYDROGEN_ABSTRACTMATRIX_HPP_

namespace El
{

template <typename T>
class AbstractMatrix
{
public:
    AbstractMatrix() = default;
    AbstractMatrix(AbstractMatrix<T> const&) = default;
    AbstractMatrix<T>& operator=(AbstractMatrix<T> const& A);
    AbstractMatrix(AbstractMatrix<T>&&) = default;
    AbstractMatrix& operator=(AbstractMatrix<T>&&) = default;

    virtual ~AbstractMatrix() = default;

    Device GetDevice() const EL_NO_EXCEPT;

    Int Height() const EL_NO_EXCEPT;
    Int Width() const EL_NO_EXCEPT;
    Int LDim() const EL_NO_EXCEPT;
    Int MemorySize() const EL_NO_EXCEPT;
    Int DiagonalLength(Int offset = Int{0}) const EL_NO_EXCEPT;

    bool Viewing() const EL_NO_EXCEPT;
    bool FixedSize() const EL_NO_EXCEPT;
    bool Locked() const EL_NO_EXCEPT;

    void FixSize() EL_NO_EXCEPT;

    void Empty(bool freeMemory=true);
    void Resize(Int height, Int width);
    void Resize(Int height, Int width, Int leadingDimension);

    void ShallowSwap(AbstractMatrix<T>& A);

    // Advanced functions
    void SetViewType(El::ViewType viewType) EL_NO_EXCEPT;
    El::ViewType ViewType() const EL_NO_EXCEPT;
    virtual void SetMemoryMode(unsigned int mode) = 0;
    virtual unsigned int MemoryMode() const EL_NO_EXCEPT = 0;

    virtual T* Buffer() EL_NO_EXCEPT = 0;
    virtual T* Buffer(Int i, Int j) EL_NO_EXCEPT = 0;
    virtual T const* LockedBuffer() const EL_NO_EXCEPT = 0;
    virtual T const* LockedBuffer(Int i, Int j) const EL_NO_EXCEPT = 0;

    virtual void Attach(
        Int height, Int width, T* buffer, Int leadingDimension) = 0;
    virtual void LockedAttach(
        Int height, Int width, const T* buffer, Int leadingDimension) = 0;

    // Assertions
    void AssertValidDimensions(
        Int height, Int width, Int leadingDimension) const;
    void AssertValidEntry(Int i, Int j) const;

    //
    // Operator overloading
    //

    // Type conversion
    operator Matrix<T, Device::CPU>& () {
      if(this->GetDevice() != Device::CPU) {
        LogicError("Illegal conversion from AbstractMatrix to incompatible CPU Matrix ref");
      }
      return static_cast<Matrix<T, Device::CPU>&>(*this);
    }
    operator Matrix<T, Device::CPU>const& () const {
      if(this->GetDevice() != Device::CPU) {
        LogicError("Illegal conversion from AbstractMatrix to incompatible const CPU Matrix ref");
      }
      return static_cast<const Matrix<T, Device::CPU>&>(*this);
    }

#ifdef HYDROGEN_HAVE_CUDA
    operator Matrix<T, Device::GPU>& () {
      if(this->GetDevice() != Device::GPU) {
        LogicError("Illegal conversion from AbstractMatrix to incompatible GPU Matrix ref");
      }
      return static_cast<Matrix<T, Device::GPU>&>(*this);
    }
    operator Matrix<T, Device::GPU>const& () const {
      if(this->GetDevice() != Device::GPU) {
        LogicError("Illegal conversion from AbstractMatrix to incompatible const GPU Matrix ref");
      }
      return static_cast<const Matrix<T, Device::GPU>&>(*this);
    }
#endif // HYDROGEN_HAVE_CUDA
    // Rescaling
    AbstractMatrix<T> const& operator*=(T const& alpha);

    // Addition/substraction
    AbstractMatrix<T> const&
    operator+=(AbstractMatrix<T> const& A);

    AbstractMatrix<T> const&
    operator-=(AbstractMatrix<T> const& A);

    //
    // Basic queries
    //

    // virtual T* Buffer() EL_NO_RELEASE_EXCEPT;
    // virtual T* Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT;
    // virtual const T* LockedBuffer() const EL_NO_EXCEPT;
    // virtual const T* LockedBuffer(Int i, Int j) const EL_NO_EXCEPT;

    // Single-entry manipulation
    // =========================
    T Get(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT;

    void Set(Int i, Int j, T const& alpha) EL_NO_RELEASE_EXCEPT;
    void Set(Entry<T> const& entry) EL_NO_RELEASE_EXCEPT;

    // Return a reference to a single entry without error-checking
    // -----------------------------------------------------------
    virtual T const& CRef(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT = 0;
    virtual T const& operator()(Int i, Int j=0) const EL_NO_RELEASE_EXCEPT = 0;

    virtual T& Ref(Int i, Int j=0) EL_NO_RELEASE_EXCEPT = 0;
    virtual T& operator()(Int i, Int j=0) EL_NO_RELEASE_EXCEPT = 0;

protected:

    AbstractMatrix(Int height, Int width, Int ldim);
    AbstractMatrix(El::ViewType view, Int height, Int width, Int ldim);

    void SetSize_(Int height, Int width, Int ldim);

private:
    template<typename S> friend class AbstractDistMatrix;
    template<typename S> friend class ElementalMatrix;
    template<typename S> friend class BlockMatrix;

    // These don't have debugging checks
    virtual void Attach_(
        Int height, Int width, T* buffer, Int leadingDimension) = 0;
    virtual void LockedAttach_(
        Int height, Int width, const T* buffer, Int leadingDimension) = 0;

    void Empty_(bool freeMemory=true);
    void Resize_(Int height, Int width);
    void Resize_(Int height, Int width, Int leadingDimension);

    virtual Int do_get_memory_size_() const EL_NO_EXCEPT = 0;
    virtual Device do_get_device_() const EL_NO_EXCEPT = 0;
    virtual void do_empty_(bool freeMemory) = 0;
    virtual void do_resize_() = 0;

private:

    El::ViewType viewType_=OWNER;

    Int height_=0, width_=0, leadingDimension_=1;

};// class AbstractMatrix

template <typename T>
inline Device AbstractMatrix<T>::GetDevice() const EL_NO_EXCEPT { return do_get_device_(); }

template <typename T>
inline Int AbstractMatrix<T>::Height() const EL_NO_EXCEPT { return height_; }

template <typename T>
inline Int AbstractMatrix<T>::Width() const EL_NO_EXCEPT { return width_; }

template <typename T>
inline Int AbstractMatrix<T>::LDim() const EL_NO_EXCEPT
{ return leadingDimension_; }

template <typename T>
inline Int AbstractMatrix<T>::DiagonalLength(Int offset) const EL_NO_EXCEPT
{ return El::DiagonalLength(height_,width_,offset); }

template <typename T>
inline Int AbstractMatrix<T>::MemorySize() const EL_NO_EXCEPT
{ return this->do_get_memory_size_(); }

template <typename T>
inline bool AbstractMatrix<T>::Viewing() const EL_NO_EXCEPT
{ return IsViewing(viewType_); }

template <typename T>
inline void AbstractMatrix<T>::FixSize() EL_NO_EXCEPT
{
    // A view is marked as fixed if its second bit is nonzero
    // (and OWNER_FIXED is zero except in its second bit).
    viewType_ = static_cast<El::ViewType>(viewType_ | OWNER_FIXED);
}

template <typename T>
inline bool AbstractMatrix<T>::FixedSize() const EL_NO_EXCEPT
{ return IsFixedSize(viewType_); }

template <typename T>
inline bool AbstractMatrix<T>::Locked() const EL_NO_EXCEPT
{ return IsLocked(viewType_); }

template <typename T>
inline void AbstractMatrix<T>::SetViewType(El::ViewType viewType) EL_NO_EXCEPT
{ viewType_ = viewType; }

template <typename T>
inline El::ViewType AbstractMatrix<T>::ViewType() const EL_NO_EXCEPT
{ return viewType_; }

template <typename T>
inline void AbstractMatrix<T>::Empty(bool freeMemory)
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(
            if (this->FixedSize())
                LogicError("Cannot empty a fixed-size matrix");
            )

        this->Empty_(freeMemory);
}

template <typename T>
inline void AbstractMatrix<T>::Empty_(bool freeMemory)
{
    viewType_ = static_cast<El::ViewType>(viewType_ & ~LOCKED_VIEW);
    this->SetSize_(0,0,1);
    do_empty_(freeMemory);
}

template <typename T>
inline void AbstractMatrix<T>::Resize(Int height, Int width)
{
    Resize(height, width, Max(leadingDimension_,height));
}

template <typename T>
inline void AbstractMatrix<T>::Resize(
    Int height, Int width, Int leadingDimension)
{
    EL_DEBUG_CSE
#ifndef EL_RELEASE
        AssertValidDimensions(height, width, leadingDimension);
    if (this->FixedSize() &&
        (height != height_ || width != this->Width() ||
         leadingDimension != leadingDimension_))
    {
        LogicError
            ("Cannot resize this matrix from ",
             height_," x ",this->Width()," (",leadingDimension_,") to ",
             height," x ",width," (",leadingDimension,")");
    }
    if (this->Viewing() && (height > height_ || width > this->Width() ||
                            leadingDimension != leadingDimension_))
        LogicError("Cannot increase the size of this matrix");
#endif // !EL_RELEASE

    this->Resize_(height,width,leadingDimension);
}

template <typename T>
inline void AbstractMatrix<T>::Resize_(Int height, Int width)
{
    Resize_(height, width, Max(leadingDimension_,height));
}

template <typename T>
inline void AbstractMatrix<T>::Resize_(
    Int height, Int width, Int leadingDimension)
{
    if (height != this->Height()
        || width != this->Width()
        || leadingDimension != this->LDim())
    {
        this->SetSize_(height, width, leadingDimension);
        do_resize_();
    }
}

template <typename T>
void AbstractMatrix<T>::ShallowSwap(AbstractMatrix<T>& A)
{
    std::swap(viewType_, A.viewType_);
    std::swap(height_, A.height_);
    std::swap(width_, A.width_);
    std::swap(leadingDimension_, A.leadingDimension_);
}

template <typename T>
inline void AbstractMatrix<T>::AssertValidDimensions(
    Int height, Int width, Int leadingDimension) const
{
    EL_DEBUG_CSE
        if (height < 0 || width < 0)
            LogicError("Height and width must be non-negative");
    if (leadingDimension < height)
        LogicError("Leading dimension must be no less than height");
    if (leadingDimension == 0)
        LogicError("Leading dimension cannot be zero (for BLAS compatibility)");
}

template <typename T>
inline void AbstractMatrix<T>::AssertValidEntry(Int i, Int j) const
{
    EL_DEBUG_CSE
        if (i == END) i = height_ - 1;
    if (j == END) j = width_ - 1;
    if (i < 0 || j < 0)
        LogicError("Indices must be non-negative");
    if (i >= height_ || j >= width_)
        LogicError
            ("Out of bounds: (",i,",",j,") of ",height_," x ",
             width_," Matrix");
}

template <typename T>
AbstractMatrix<T>::AbstractMatrix(Int height, Int width, Int ldim)
    : AbstractMatrix{OWNER, height, width, ldim}
{
}

template <typename T>
AbstractMatrix<T>::AbstractMatrix(
    El::ViewType view, Int height, Int width, Int ldim)
    : viewType_{view}, height_{height}, width_{width}, leadingDimension_{ldim}
{
    EL_DEBUG_CSE
        EL_DEBUG_ONLY(AssertValidDimensions(height, width, ldim))
        }

template <typename T>
inline void AbstractMatrix<T>::SetSize_(
    Int height, Int width, Int leadingDimension)
{
    height_ = height;
    width_ = width;
    leadingDimension_ = leadingDimension;
}

// Single-entry manipulation
// =========================

template<typename T>
T AbstractMatrix<T>::Get(Int i, Int j) const
    EL_NO_RELEASE_EXCEPT
{
    switch(this->GetDevice())
    {
    case Device::CPU:
        return static_cast<const Matrix<T,Device::CPU>*>(this)->Get(i,j);
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        return static_cast<const Matrix<T,Device::GPU>*>(this)->Get(i,j);
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Unsupported device type.");
        return T{};
    }
}

template<typename T>
void AbstractMatrix<T>::Set(Int i, Int j, T const& alpha)
    EL_NO_RELEASE_EXCEPT
{
    switch(this->GetDevice())
    {
    case Device::CPU:
        static_cast<Matrix<T,Device::CPU>*>(this)->Set(i,j, alpha);
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        static_cast<Matrix<T,Device::GPU>*>(this)->Set(i,j, alpha);
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Unsupported device type.");
    }
}

template<typename T>
void AbstractMatrix<T>::Set(Entry<T> const& entry)
    EL_NO_RELEASE_EXCEPT
{ Set(entry.i, entry.j, entry.value); }

// Operator overloading
// ====================

// Assignment
// ----------
template<typename T>
AbstractMatrix<T>&
AbstractMatrix<T>::operator=(AbstractMatrix<T> const& A)
{
    Copy( A, *this );
    return *this;
}

// Rescaling
// ---------
template<typename T>
AbstractMatrix<T> const&
AbstractMatrix<T>::operator*=(T const& alpha)
{
    switch(this->GetDevice()) {
    case Device::CPU:
      return static_cast<Matrix<T,Device::CPU>*>(this)->operator*=(alpha);
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
      return static_cast<Matrix<T,Device::GPU>*>(this)->operator*=(alpha);
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Unsupported device type.");
    }
    return *this;// silence compiler warning
}

// Addition/subtraction
// --------------------
template<typename T>
AbstractMatrix<T> const&
AbstractMatrix<T>::operator+=(AbstractMatrix<T> const& A)
{
    if (this->GetDevice() != A.GetDevice())
        LogicError("operator= requires matching device types.");

    switch(this->GetDevice()) {
    case Device::CPU:
      return static_cast<Matrix<T,Device::CPU>*>(this)->operator+=(A);
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
      return static_cast<Matrix<T,Device::GPU>*>(this)->operator+=(A);
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Unsupported device type.");
    }
    return *this;// silence compiler warning
}

template<typename T>
AbstractMatrix<T> const&
AbstractMatrix<T>::operator-=(AbstractMatrix<T> const& A)
{
    if (this->GetDevice() != A.GetDevice())
        LogicError("operator= requires matching device types.");

    switch(this->GetDevice()) {
    case Device::CPU:
      return static_cast<Matrix<T,Device::CPU>*>(this)->operator-=(A);
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
      return static_cast<Matrix<T,Device::GPU>*>(this)->operator-=(A);
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Unsupported device type.");
    }
    return *this;// silence compiler warning
}

// Basic queries
// =============

// template<typename T>
// Int AbstractMatrix<T>::do_get_memory_size_() const EL_NO_EXCEPT
// {
//     if ((this->GetDevice() == Device::CPU)) {
//       return static_cast<const Matrix<T,Device::CPU>*>(this)->do_get_memory_size_();
//     }else if ((this->GetDevice() == Device::GPU)) {
//       return static_cast<const Matrix<T,Device::GPU>*>(this)->do_get_memory_size_());
//     }else {
//       LogicError("Unsupported device type.");
//     }
// }

// template <typename T>
// Device AbstractMatrix<T>::do_get_device_() const EL_NO_EXCEPT
// {
//     if ((this->GetDevice() == Device::CPU)) {
//       return static_cast<const Matrix<T,Device::CPU>*>(this)->do_get_memory_size_();
//     }else if ((this->GetDevice() == Device::GPU)) {
//       return static_cast<const Matrix<T,Device::GPU>*>(this)->do_get_memory_size_());
//     }else {
//       LogicError("Unsupported device type.");
//     }
// }

// template<typename T>
// T* AbstractMatrix<T>::Buffer() EL_NO_RELEASE_EXCEPT
// {
//     if ((this->GetDevice() == Device::CPU)) {
//       return static_cast<Matrix<T,Device::CPU>*>(this)->Buffer();
//     }else if ((this->GetDevice() == Device::GPU)) {
//       return static_cast<Matrix<T,Device::GPU>*>(this)->Buffer();
//     }else {
//       LogicError("Unsupported device type.");
//     }
// }

// template<typename T>
// T* AbstractMatrix<T>::Buffer(Int i, Int j) EL_NO_RELEASE_EXCEPT
// {
//     if ((this->GetDevice() == Device::CPU)) {
//       return static_cast<Matrix<T,Device::CPU>*>(this)->Buffer(i,j);
//     }else if ((this->GetDevice() == Device::GPU)) {
//       return static_cast<Matrix<T,Device::GPU>*>(this)->Buffer(i,j);
//     }else {
//       LogicError("Unsupported device type.");
//     }
// }

// template<typename T>
// const T* AbstractMatrix<T>::LockedBuffer() const EL_NO_EXCEPT
// {
//     if ((this->GetDevice() == Device::CPU)) {
//       return static_cast<const Matrix<T,Device::CPU>*>(this)->Buffer();
//     }else if ((this->GetDevice() == Device::GPU)) {
//       return static_cast<const Matrix<T,Device::GPU>*>(this)->Buffer();
//     }else {
//       LogicError("Unsupported device type.");
//     }
// }

// template<typename T>
// const T*
// AbstractMatrix<T>::LockedBuffer(Int i, Int j) const EL_NO_EXCEPT
// {
//     if ((this->GetDevice() == Device::CPU)) {
//       return static_cast<const Matrix<T,Device::CPU>*>(this)->Buffer(i,j);
//     }else if ((this->GetDevice() == Device::GPU)) {
//       return static_cast<const Matrix<T,Device::GPU>*>(this)->Buffer(i,j);
//     }else {
//       LogicError("Unsupported device type.");
//     }
// }

// Return a reference to a single entry without error-checking
// ===========================================================
#if 0
template<typename T>
T const& AbstractMatrix<T>::CRef(Int i, Int j) const
    EL_NO_RELEASE_EXCEPT
{
    switch(this->GetDevice())
    {
    case Device::CPU:
        return static_cast<Matrix<T,Device::CPU> const*>(this)->CRef(i,j);
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        return static_cast<Matrix<T,Device::GPU> const*>(this)->CRef(i,j);
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Unsupported device type.");
    }
}

template<typename T>
T const& AbstractMatrix<T>::operator()(Int i, Int j) const
    EL_NO_RELEASE_EXCEPT
{
    switch(this->GetDevice())
    {
    case Device::CPU:
        return static_cast<Matrix<T,Device::CPU> const*>(this)->operator()(i,j);
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        return static_cast<Matrix<T,Device::GPU> const*>(this)->operator()(i,j);
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Unsupported device type.");
        throw std::logic_error("");
    }
}

template<typename T>
T& AbstractMatrix<T>::Ref(Int i, Int j)
    EL_NO_RELEASE_EXCEPT
{
    switch(this->GetDevice())
    {
    case Device::CPU:
        return static_cast<Matrix<T,Device::CPU>*>(this)->Ref(i,j);
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        return static_cast<Matrix<T,Device::GPU>*>(this)->Ref(i,j);
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Unsupported device type.");
    }
}

template<typename T>
T& AbstractMatrix<T>::operator()(Int i, Int j)
    EL_NO_RELEASE_EXCEPT
{
    switch(this->GetDevice())
    {
    case Device::CPU:
        return (static_cast<Matrix<T,Device::CPU>*>(this))->operator()(i,j);
#ifdef HYDROGEN_HAVE_CUDA
    case Device::GPU:
        return (static_cast<Matrix<T,Device::GPU>*>(this))->operator()(i,j);
#endif // HYDROGEN_HAVE_CUDA
    default:
        LogicError("Unsupported device type.");
    }
}
#endif // 0
}// namespace El
#endif // HYDROGEN_ABSTRACTMATRIX_HPP_
