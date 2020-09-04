using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using MKE.ElementFragments;
using MKE.Matrix;

namespace MKE.Interface
{
    public interface IElement
    {
        IBasisFunction Basis { get; }

        IEnumerable<IPointNumbered> Points { get; }

        Dictionary<int, int> LocalToGlobalEnumeration { get; }

        Dictionary<int, bool> SkipPoint { get; }
        TemplateElementInformation TemplateElementInformation { get; }
        void EvaluateLocal(Action<int, int, double> A);

        double Integrate(int i, Func<double, double, double, double> func);
        ReadonlyStorageMatrix GetMassMatrix();

        bool CheckElement(double x, double y, double z);

        double CalcOnElement(double[] solution, double x, double y, double z);
    }
}