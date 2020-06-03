using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using MKE.ElementFragments;
using MKE.Matrix;

namespace MKE.Interface
{
    using Func1D = Expression<Func<double, double>>;
    using Func2D = Expression<Func<double, double, double>>;
    using Func3D = Expression<Func<double, double, double, double>>;

    public interface IBasisFunction
    {
        IEnumerable<Func1D> GetBasis1d(int order, double h, double left);
        IEnumerable<Func2D> GetBasis2d(int order, double hu, double hv, double leftU, double leftV);
        IEnumerable<Func3D> GetBasis3d(int order, double hu, double hv, double hw, double leftU, double leftV, double leftW);
        IEnumerable<Func1D> GetDeriveBasis1d(int order, double h, double left);
        IEnumerable<Func2D> GetDeriveBasis2dU(int order, double hu, double hv, double leftU, double leftV);
        IEnumerable<Func2D> GetDeriveBasis2dV(int order, double hu, double hv, double leftU, double leftV);
        IEnumerable<Func3D> GetDeriveBasis3dU(int order, double hu, double hv, double hw, double leftU, double leftV, double leftW);
        IEnumerable<Func3D> GetDeriveBasis3dV(int order, double hu, double hv, double hw, double leftU, double leftV, double leftW);
        IEnumerable<Func3D> GetDeriveBasis3dW(int order, double hu, double hv, double hw, double leftU, double leftV, double leftW);
        (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Func1D lambda, Func1D gamma, int order, double h, double left);
        (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Func2D lambda, Func2D gamma, int order, double hu, double hv, double leftU, double leftV);
        (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Func3D lambda, Func3D gamma, int order, double hu, double hv, double hw, double leftU, double leftV, double leftW);
        TemplateElementInformation Get3DFragments(int order);

        TemplateElementInformation Get2DFragments(int order);
    }
}