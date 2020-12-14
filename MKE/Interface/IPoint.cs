using System.Collections.Generic;

namespace MKE.Interface
{
    public interface IPoint
    {
        int Dimension { get; }
        List<double> Coords { get; }
        IPoint Add(IPoint x);
        IPoint Subtract(IPoint x);
        IPoint Normalize();
        double Multiply(IPoint x);
        IPoint Multiply(double x);
        IPoint Divide(double x);
    }
}