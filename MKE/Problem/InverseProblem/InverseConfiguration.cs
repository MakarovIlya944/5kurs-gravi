using System.Collections.Generic;
using System.Linq;
using MKE.Point;

namespace MKE.Problem.InverseProblem
{
    public class InverseConfiguration
    {
        public class InverseDomainConfiguration
        {
            public bool FitGamma { get; set; }

            public bool FitLambda { get; set; }

            public FittingConfiguration GammaFittingConfiguration { get; set; }

            public FittingConfiguration LambdaFittingConfiguration { get; set; }
        }

        public Dictionary<int, InverseDomainConfiguration> DomainConfiguration { get; set; }

        public double Eps { get; set; } = 1E-14;

        public double MaxIteration { get; set; } = 100;

        public string ProblemPath { get; set; } = "problem.json";

        public double Regularization { get; set; } = 0;

        public List<double> Weights { get; set; } = Enumerable.Repeat(1d, 50).ToList();

        public List<ValuedPoint3D> ExperimentalPoints { get; set; }
    }
}