using System;
using System.Linq;
using MKE.Extenstion;
using MKE.Matrix;
using MKE.Utils;
using Newtonsoft.Json;

namespace MKE.Problem.InverseProblem
{
    public class GaussNewtonInverseProblem
    {
        public class CurrentFittingValue
        {
            private readonly Func<double, IDisposable> _function;

            private readonly Action<double> _changeValue;

            public FittingConfiguration FittingConfiguration { get; set; }

            public double CurrentValue { get; set; }

            public CurrentFittingValue(Func<double, IDisposable> function, Action<double> changeValue, FittingConfiguration fittingConfiguration)
            {
                _function = function;
                _changeValue = changeValue;
                CurrentValue = fittingConfiguration.StartedValue;
                FittingConfiguration = fittingConfiguration;
            }

            public IDisposable ChangeValueDisposable(double value) => _function(value);

            public void ChangeValue(double value) => _changeValue(value);
        }

        private BaseProblem Problem { get; set; }

        private InverseConfiguration Configuration { get; set; }

        public GaussNewtonInverseProblem(BaseProblem problem, InverseConfiguration configuration)
        {
            Problem = problem;
            Configuration = configuration;
        }

        private double[] CalculateDelta(double[] withOutChanges, CurrentFittingValue[] currentValues)
        {
            var solution = new double[currentValues.Length][];

            var matrix = new DenseMatrix(new double[currentValues.Length, currentValues.Length], currentValues.Length, currentValues.Length);
            var right = new double[currentValues.Length];

            //Not THREAD SAFE
            for (var i = 0; i < currentValues.Length; i++)
            {
                using (var _ = currentValues[i].ChangeValueDisposable(currentValues[i].CurrentValue + currentValues[i].FittingConfiguration.Step))
                {
                    Problem.SolveProblem();
                    solution[i] = Configuration.ExperimentalPoints.Select(x => Problem.GetSolution(x)).ToArray();
                }
            }

            for (var i = 0; i < matrix.RowsCount; i++)
            {
                // для каждой эксперементальной точки
                for (int j = 0; j < matrix.ColumnsCount; j++)
                {
                    for (int k = 0; k < Configuration.ExperimentalPoints.Count; k++)

                    {
                        matrix[i, j] += Math.Pow(Configuration.Weights[k], 2) * Derive(Difference(Configuration.ExperimentalPoints[k].Value, withOutChanges[k]),
                                                                                       Difference(Configuration.ExperimentalPoints[k].Value, solution[i][k]),
                                                                                       currentValues[i].FittingConfiguration.Step) *
                                        Derive(Difference(Configuration.ExperimentalPoints[k].Value, withOutChanges[k]),
                                               Difference(Configuration.ExperimentalPoints[k].Value, solution[j][k]),
                                               currentValues[j].FittingConfiguration.Step);
                    }
                }

                for (int k = 0; k < Configuration.ExperimentalPoints.Count; k++)
                {
                    right[i] += Math.Pow(Configuration.Weights[k], 2) * Derive(Difference(Configuration.ExperimentalPoints[k].Value, withOutChanges[k]),
                                                                               Difference(Configuration.ExperimentalPoints[k].Value, solution[i][k]),
                                                                               currentValues[i].FittingConfiguration.Step)
                                                                      * Difference(Configuration.ExperimentalPoints[k].Value, withOutChanges[k]);
                }
            }

            AddRegularization(withOutChanges, matrix, right, currentValues);

            return GaussExtension.Solve(matrix, right.ToArray());
        }

        public double[] Solve()
        {
            var currentValues = CreateCurrentValues();
            var iteration = 0;
            try
            {
                for (iteration = 0; iteration < Configuration.MaxIteration; iteration++)
                {
                    Problem.SolveProblem();
                    var withOutChanges = Configuration.ExperimentalPoints.Select(x => Problem.GetSolution(x)).ToArray();

                    var functional = CalculateFunctional(withOutChanges);
                    //Console.WriteLine(currentValues.Select(x => x.CurrentValue).Aggregate(functional.ToString() + ';', (p, c) => p + c.ToString() + ';'));

                    if (Math.Abs(functional) < Configuration.Eps)
                    {
                        break;
                    }

                    var delta = CalculateDelta(withOutChanges, currentValues);
                    var beta = 1.0;

                    while (beta > 1e-10)
                    {
                        using (_ = new DisposableContainer(currentValues.Select((x, i) => x.ChangeValueDisposable(ProcessDelta(x, beta * delta[i]))).ToList()))
                        {
                            Problem.SolveProblem();
                            var newFunctional = CalculateFunctional(Configuration.ExperimentalPoints.Select(x => Problem.GetSolution(x)).ToArray());

                            if (newFunctional < functional)
                            {
                                beta = 1;
                                break;
                            }

                            beta /= 5.0;
                        }
                    }

                    if (beta < 1e-10)
                    {
                        throw new Exception("Stagnation detected");
                    }

                    delta = delta.Select(x => beta * x).ToArray();
                    ApplyDeltaToCurrentValues(delta, currentValues);
                }
            }catch(Exception e) { }
            var withOutChanges1 = Configuration.ExperimentalPoints.Select(x => Problem.GetSolution(x)).ToArray();

            var functiona1l = CalculateFunctional(withOutChanges1);

            Console.WriteLine(currentValues.Select(x => x.CurrentValue).Aggregate(iteration.ToString()+';'+functiona1l.ToString() + ';', (p, c) => p + c.ToString() + ';'));

            return currentValues.Select(x => x.CurrentValue).ToArray();
        }

        private CurrentFittingValue[] CreateCurrentValues()
        {
            var currentValues = new CurrentFittingValue[Configuration.DomainConfiguration.Select(x =>
            {
                if (x.Value.FitGamma && x.Value.FitLambda) return 2;
                if (!x.Value.FitGamma && !x.Value.FitLambda) return 0;

                return 1;
            }).Sum()];

            var i = 0;

            foreach (var (index, configuration) in Configuration.DomainConfiguration)
            {
                if (configuration.FitLambda)
                {
                    currentValues[i++] = new CurrentFittingValue((value) =>
                                                                 {
                                                                     var old = Problem.Geometry.MapDomains[index].Lambda;
                                                                     Problem.Geometry.MapDomains[index].Lambda = (x, y, z) => value;
                                                                     Problem.Geometry.MapDomains[index].UpdateFunction();

                                                                     return new DisposableChanges(() =>
                                                                     {
                                                                         Problem.Geometry.MapDomains[index].Lambda = old;
                                                                         Problem.Geometry.MapDomains[index].UpdateFunction();
                                                                     });
                                                                 },
                                                                 (value) =>
                                                                 {
                                                                     Problem.Geometry.MapDomains[index].Lambda = (x, y, z) => value;
                                                                     Problem.Geometry.MapDomains[index].UpdateFunction();
                                                                 }, configuration.LambdaFittingConfiguration);
                }

                if (configuration.FitGamma)
                {
                    currentValues[i++] = new CurrentFittingValue((value) =>
                                                                 {
                                                                     var old = Problem.Geometry.MapDomains[index].Gamma;
                                                                     Problem.Geometry.MapDomains[index].Gamma = (x, y, z) => value;
                                                                     Problem.Geometry.MapDomains[index].UpdateFunction();

                                                                     return new DisposableChanges(() =>
                                                                     {
                                                                         Problem.Geometry.MapDomains[index].Gamma = old;
                                                                         Problem.Geometry.MapDomains[index].UpdateFunction();
                                                                     });
                                                                 },
                                                                 (value) =>
                                                                 {
                                                                     Problem.Geometry.MapDomains[index].Gamma = (x, y, z) => value;
                                                                     Problem.Geometry.MapDomains[index].UpdateFunction();
                                                                 }, configuration.GammaFittingConfiguration);
                }
            }

            foreach (var currentFittingValue in currentValues)
            {
                currentFittingValue.ChangeValue(currentFittingValue.CurrentValue);
            }

            return currentValues;
        }

        private double ProcessDelta(CurrentFittingValue currentFitting, double delta)
        {
            var currentValue = delta + currentFitting.CurrentValue;

            if (currentFitting.FittingConfiguration.MaxValue < currentValue)
            {
                return currentFitting.FittingConfiguration.MaxValue;
            }

            if (currentFitting.FittingConfiguration.MinValue > currentValue)
            {
                return currentFitting.FittingConfiguration.MinValue;
            }

            var percent = Math.Abs(delta - currentFitting.CurrentValue) / currentFitting.CurrentValue;

            if (double.IsInfinity(percent))
            {
                return delta;
            }

            if (delta < 0 && currentFitting.FittingConfiguration.MaxPercentDecrease > percent)
            {
                return currentFitting.CurrentValue - currentFitting.FittingConfiguration.MaxPercentDecrease * currentFitting.CurrentValue;
            }

            if (delta > 0 && currentFitting.FittingConfiguration.MaxPercentIncrease < percent)
            {
                return currentFitting.CurrentValue + currentFitting.FittingConfiguration.MaxPercentIncrease * currentFitting.CurrentValue;
            }

            return currentFitting.CurrentValue + delta;
        }

        private void ApplyDeltaToCurrentValues(double[] delta, CurrentFittingValue[] currentValues)
        {
            for (var i = 0; i < currentValues.Length; i++)
            {
                currentValues[i].CurrentValue = ProcessDelta(currentValues[i], delta[i]);
                currentValues[i].ChangeValue(currentValues[i].CurrentValue);
            }
        }

        private double CalculateFunctional(double[] withOutChanges) => Configuration.ExperimentalPoints.Select((t, i) => Math.Pow(t.Value - withOutChanges[i], 2)).Sum();

        private double Derive(double calculated, double calculatedWithStep, double step) =>
            (calculated - calculatedWithStep) / step;

        private double Difference(double expected, double calculated) => expected - calculated;

        private void AddRegularization(double[] withOutChanges, DenseMatrix a, double[] b, CurrentFittingValue[] currentFittingValues)
        {
            var alpha = withOutChanges.Select((t, i) => Configuration.Regularization * Math.Pow(Configuration.Weights[i] * Difference(Configuration.ExperimentalPoints[i].Value, t), 2)).Sum() /
                        currentFittingValues.Select((t, i) => Math.Pow(Difference(t.CurrentValue, 5), 2)).Sum();

            for (var i = 0; i < currentFittingValues.Length; i++)
            {
                a[i, i] += alpha;
                b[i] -= alpha * Difference(currentFittingValues[i].CurrentValue, 5);
            }
        }
    }
}