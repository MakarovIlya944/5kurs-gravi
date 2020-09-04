using System.Collections.Generic;

namespace MKE.Interface
{
    public interface IProblem
    {
        void SolveProblem();

        double GetSolution(double x, double y, double z);
    }
}