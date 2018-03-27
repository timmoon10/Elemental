#ifndef EL_CORE_PROXYDEVICE_HPP_
#define EL_CORE_PROXYDEVICE_HPP_

namespace El
{

template <typename T, Device D>
class AbstractMatrixReadDeviceProxy
{
public:
    using proxy_type = Matrix<T,D>;
public:

    AbstractMatrixReadDeviceProxy(AbstractMatrix<T> const& A)
    {
        if (A.GetDevice() == D)
            proxy_ = const_cast<AbstractMatrix<T>*>(&A);
        else
        {
            switch (A.GetDevice())
            {
            case Device::CPU:
                proxy_ = new Matrix<T,D>{
                    static_cast<Matrix<T,Device::CPU> const&>(A)};
                break;
#ifdef HYDROGEN_HAVE_CUDA
            case Device::GPU:
                proxy_ = new Matrix<T,D>{
                    static_cast<Matrix<T,Device::GPU> const&>(A)};
                break;
#endif // HYDROGEN_HAVE_CUDA
            default:
                LogicError("AbstractMatrixReadDeviceProxy: Bad device.");
            }
            owned_ = true;
        }
    }

    ~AbstractMatrixReadDeviceProxy()
    {
        if (proxy_ && owned_)
            delete proxy_;
    }

    proxy_type const& GetLocked()
    {
        return *proxy_;
    }

private:

    proxy_type* proxy_ = nullptr;
    bool owned_ = false;

};// class AbstractMatrixReadDeviceProxy


template <typename T,Device D,typename=EnableIf<IsDeviceValidType<T,D>>>
class AbstractDistMatrixReadDeviceProxy;

template <typename T,Device D>
class AbstractDistMatrixReadDeviceProxy<T,D,void>
{
public:
    using proxy_type = AbstractDistMatrix<T>;
public:

    AbstractDistMatrixReadDeviceProxy(AbstractDistMatrix<T> const& A)
    {
        if (A.GetLocalDevice() == D)
        {
            proxy_ = const_cast<AbstractDistMatrix<T>*>(&A);
        }
        else
        {
            proxy_ = A.ConstructWithNewDevice(D).release();
            Copy(A,*proxy_);
            owned_ = true;
        }
    }
    ~AbstractDistMatrixReadDeviceProxy()
    {
        if (proxy_ && owned_)
            delete proxy_;
    }

    proxy_type const& GetLocked()
    {
        return *proxy_;
    }

private:
    proxy_type* proxy_ = nullptr;
    bool owned_ = false;
};// AbstractDistMatrixReadDeviceProxy

}// namespace El
#endif // EL_CORE_PROXYDEVICE_HPP_
