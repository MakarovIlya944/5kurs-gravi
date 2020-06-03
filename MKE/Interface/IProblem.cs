using System.Collections.Generic;
using System.Linq;
using MKE.Extenstion;
using MKE.Matrix;
using MKE.Point;
using MKE.Solver;

namespace MKE.Interface
{
    public interface IProblem
    {

    }
    public class BaseProblem
    {
        public GeometryParallelepiped Geometry { get; set; }

        public int ProblemSize => Geometry.LastSequenceIndex;
        public bool[] isDirichletCond { get; set; }
        public double[] Solution { get; set; }

        public void SolveProblem()
        {
            Geometry.ExtendsElementWeight();
            Geometry.FillConditionsElement();
            var _direchletData = CalcDirichletCondition();
            var _rightPart = CalcRightPart();
            var portrait = new SparsePortrait();
            foreach (var element in Geometry.MapDomains.Values.SelectMany(domain => domain.Elements))
            {
                portrait.Add(element.LocalToGlobalEnumeration.Values.ToList(), element.LocalToGlobalEnumeration.Values.ToList());
            }
            var matrix = new SparseMatrixReal(portrait.GetMappedLinks());
            var x = new double[matrix.Rows];
            void AddToMatrix(int i, int j, double value)
            {
                if (isDirichletCond[i])
                {
                    return;
                }

                if (isDirichletCond[j])
                {
                    _rightPart.ThreadSafeAdd(i, -value * _direchletData[j]);
                }
                else
                {
                    matrix.ThreadSafeAdd(i, j, value);
                }
            }
            foreach (var element in Geometry.MapDomains.Values.SelectMany(d => d.Elements))
            {
                element.EvaluateLocal(AddToMatrix);
            }
            var solve = new Los();
            for (var i = 0; i < _direchletData.Length; i++)
            {
                if (isDirichletCond[i])
                    matrix[i, i] = 1;
            }
            solve.Initialization(30000, 1e-14, FactorizationType.LUsq);
            Solution = solve.Solve(matrix, _rightPart, x);

            for (var i = 0; i < _direchletData.Length; i++)
            {
                var d = _direchletData[i];
                Solution[i] += d;
            }
        }
        protected double[] CalcRightPart()
        {
            var rightPart = new double[ProblemSize];

            foreach (var domain in Geometry.MapDomains.Values)
            {
                foreach (var elem in domain.Elements)
                {
                    var v = elem.LocalToGlobalEnumeration;

                    for (int i = 0; i < v.Count; ++i)
                    {
                        if (!isDirichletCond[v[i]])
                        {
                            rightPart.ThreadSafeAdd(v[i], elem.Integrate(i, domain.RightFunction));
                        }
                    }
                }
            }

            foreach (var cond in Geometry.NeumannConditions)
            {
                foreach (var elem in cond.Elements)
                {
                    var v = elem.LocalToGlobalEnumeration;

                    for (int i = 0; i < v.Count; i++)
                    {
                        if (!isDirichletCond[v[i]])
                        {
                            rightPart.ThreadSafeAdd(v[i], elem.Integrate(i, cond.F));
                        }
                    }
                }

            }

            return rightPart;
        }

        public double GetSolution(double x, double y, double z)
        {
            var sum = 0d;
            foreach (var element in Geometry.MapDomains.Values.SelectMany(d => d.Elements).Where(d=>d.CheckElement(x,y,z)))
            {
                sum += element.CalcOnElement(Solution, x, y, z);
            }

            return sum;
        }
        private double[] CalcDirichletCondition()
        {

            var portrait = new SparsePortrait();
            foreach (var elem in Geometry.DirichletConditions.SelectMany(x => x.Elements))
            {
                portrait.Add(elem.LocalToGlobalEnumeration.Values, elem.LocalToGlobalEnumeration.Values);
            }

            var matrix = new SymSparseMatrixReal(portrait.GetMappedLinks());
            var b = new double[matrix.Rows];
            var tempIndices = new int[matrix.Rows];
            foreach (var cond in Geometry.DirichletConditions)
            {
                foreach (var elem in cond.Elements)
                {
                    var mass = elem.GetMassMatrix();

                    for (int i = 0; i < mass.RowsCount; i++)
                    {
                        for (int j = 0; j < mass.ColumnsCount; j++)
                        {
                            matrix.ThreadSafeAdd(portrait.PremutationRowIndices.ElementAt(elem.LocalToGlobalEnumeration[i]),
                                                 portrait.PremutationColumnIndices.ElementAt(elem.LocalToGlobalEnumeration[j]), mass[i, j]);
                        }

                        var value = elem.Integrate(i, cond.F);
                        b.ThreadSafeAdd(portrait.PremutationRowIndices.ElementAt(elem.LocalToGlobalEnumeration[i]), value);
                        tempIndices[portrait.PremutationRowIndices.ElementAt(elem.LocalToGlobalEnumeration[i])] = elem.LocalToGlobalEnumeration[i];
                    }
                }
            }
            var solver = new CGSolver();
            solver.Initialization(30000, 1e-50, FactorizationType.LLt);
            var tempRightPart = solver.Solve(matrix, b, new double[matrix.Rows]).ToArray();
            var right = new double[ProblemSize];
            isDirichletCond = new bool[ProblemSize];
            for (var i = 0; i < tempRightPart.Length; i++)
            {
                isDirichletCond[tempIndices[i]] = true;
                right[tempIndices[i]] = tempRightPart[i];
            }

            return right;
        }
    }
}