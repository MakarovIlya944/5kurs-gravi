using System;
using System.Collections.Immutable;
using MKE.ElementFragments;
using MKE.Matrix;

namespace MKE.Interface
{
    public interface IBasisFunction
    {
        ImmutableArray<Func<double, double>> GetBasis1d(int order);
        ImmutableArray<Func<double, double, double>> GetBasis2d(int order);
        ImmutableArray<Func<double, double, double, double>> GetBasis3d(int order);
        ImmutableArray<Func<double, double>> GetDeriveBasis1d(int order);
        ImmutableArray<Func<double, double, double>> GetDeriveBasis2dU(int order);
        ImmutableArray<Func<double, double, double>> GetDeriveBasis2dV(int order);
        ImmutableArray<Func<double, double, double, double>> GetDeriveBasis3dU(int order);
        ImmutableArray<Func<double, double, double, double>> GetDeriveBasis3dV(int order);
        ImmutableArray<Func<double, double, double, double>> GetDeriveBasis3dW(int order);
        (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Func<double, double> lambda, Func<double, double> gamma, int order, double h);
        (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Func<double, double, double> lambda, Func<double, double, double> gamma, int order, double hu, double hv);
        (ReadonlyStorageMatrix, ReadonlyStorageMatrix) GetMatrix(Func<double, double, double, double> lambda, Func<double, double, double, double> gamma, int order, double hu, double hv, double hw);
        TemplateElementInformation Get3DFragments(int order);

        TemplateElementInformation Get2DFragments(int order);
    }
}