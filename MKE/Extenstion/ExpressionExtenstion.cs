using System;
using System.Collections.Generic;
using System.Linq;
using System.Linq.Expressions;
using MKE.Interface;

namespace MKE.Extenstion
{
    using Func1D = Expression<Func<double, double>>;
    using Func2D = Expression<Func<double, double, double>>;
    using Func3D = Expression<Func<double, double, double, double>>;

    public static class ExpressionExtenstion
    {
        private class ParameterChanger : ExpressionVisitor
        {
            public Dictionary<ParameterExpression, Expression> dic = new Dictionary<ParameterExpression, Expression>();

            protected override Expression VisitInvocation(InvocationExpression node)
            {
                return Expression.Invoke(node.Expression, node.Arguments.Select(Visit));
            }

            protected override Expression VisitParameter(ParameterExpression node)
            {
                return dic[node];
            }
        }

        private static LambdaExpression ChangeVar(LambdaExpression func, params Expression[] parameters)
        {
            var changer = new ParameterChanger();
            for (int i = 0; i < func.Parameters.Count; i++)
                changer.dic[func.Parameters[i]] = parameters[i];
            return changer.VisitAndConvert(func, "change var");
        }

        public static IEnumerable<Func2D> Mult(IEnumerable<Func1D> basis1, IEnumerable<Func1D> basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            foreach (var func2 in basis2)
                foreach (var func1 in basis1)
                {
                    yield return Expression.Lambda<Func<double, double, double>>(
                                                                                 Expression.Multiply(ChangeVar(func1, u).Body, ChangeVar(func2, v).Body), u, v);
                }
        }

        public static IEnumerable<Func1D> Add(IEnumerable<Func1D> basis1, IEnumerable<Func1D> basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            foreach (var func2 in basis2)
                foreach (var func1 in basis1)
                {
                    yield return Expression.Lambda<Func<double, double>>(Expression.Add(ChangeVar(func1, u).Body, ChangeVar(func2, u).Body), u);
                }
        }

        public static IEnumerable<Func2D> Add(IEnumerable<Func2D> basis1, IEnumerable<Func2D> basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            foreach (var func2 in basis2)
                foreach (var func1 in basis1)
                {
                    yield return Expression.Lambda<Func<double, double, double>>(
                                                                                 Expression.Add(ChangeVar(func1, u, v).Body, ChangeVar(func2, u, v).Body), u, v);
                }
        }

        public static IEnumerable<Func3D> Add(IEnumerable<Func3D> basis1, IEnumerable<Func3D> basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            var w = Expression.Parameter(typeof(double), "w");
            foreach (var func2 in basis2)
                foreach (var func1 in basis1)
                {
                    yield return Expression.Lambda<Func<double, double, double, double>>(Expression.Add(ChangeVar(func1, u, v, w).Body, ChangeVar(func2, u, v, w).Body), u, v, w);
                }
        }

        public static Func3D Add(Func3D basis1, Func3D basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            var w = Expression.Parameter(typeof(double), "w");
            return Expression.Lambda<Func<double, double, double, double>>(Expression.Add(ChangeVar(basis1, u, v, w).Body, ChangeVar(basis2, u, v, w).Body), u, v, w);
        }

        public static Func2D Add(Func2D basis1, Func2D basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            return Expression.Lambda<Func<double, double, double>>(Expression.Add(ChangeVar(basis1, u, v).Body, ChangeVar(basis2, u, v).Body), u, v);
        }

        public static Func1D Add(Func1D basis1, Func1D basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            return Expression.Lambda<Func<double, double>>(Expression.Add(ChangeVar(basis1, u).Body, ChangeVar(basis2, u).Body), u);
        }

        public static Func2D Mult(Func2D basis1, Func2D basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            return Expression.Lambda<Func<double, double, double>>(Expression.Multiply(ChangeVar(basis1, u, v).Body, ChangeVar(basis2, u, v).Body), u, v);
        }

        public static Func1D Mult(Func1D basis1, Func1D basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            return Expression.Lambda<Func<double, double>>(Expression.Multiply(ChangeVar(basis1, u).Body, ChangeVar(basis2, u).Body), u);
        }

        public static Func3D Mult(Func3D basis1, Func3D basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            var w = Expression.Parameter(typeof(double), "w");
            return Expression.Lambda<Func<double, double, double, double>>(Expression.Multiply(ChangeVar(basis1, u, v, w).Body, ChangeVar(basis2, u, v, w).Body), u,v,w);
        }

        public static IEnumerable<Func3D> Mult(IEnumerable<Func1D> basis1, IEnumerable<Func1D> basis2,
            IEnumerable<Func1D> basis3)
        {
            return Mult(Mult(basis1, basis2), basis3);
        }

        public static IEnumerable<Func3D> Mult(IEnumerable<Func2D> basis1,
            IEnumerable<Func1D> basis2)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            var w = Expression.Parameter(typeof(double), "w");
            foreach (var func2 in basis2)
                foreach (var func1 in basis1)
                {
                    yield return Expression.Lambda<Func<double, double, double, double>>(
                                                                                         Expression.Multiply(ChangeVar(func1, u, v).Body, ChangeVar(func2, w).Body), u, v, w);
                }
        }

        public static IEnumerable<Func3D> FakeExtendsW(IEnumerable<Func2D> function)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            var w = Expression.Parameter(typeof(double), "w");
            Func1D fake = x => 1;
            foreach (var func1 in function)
            {
                yield return Expression.Lambda<Func<double, double, double, double>>(
                                                                                     Expression.Multiply(ChangeVar(func1, u, v).Body, ChangeVar(fake, w).Body), u, v, w);
            }
        }
        public static IEnumerable<Func3D> FakeExtendsV(IEnumerable<Func2D> function)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            var w = Expression.Parameter(typeof(double), "w");
            Func1D fake = x => 1;
            foreach (var func1 in function)
            {
                yield return Expression.Lambda<Func<double, double, double, double>>(
                                                                                     Expression.Multiply(ChangeVar(func1, u, w).Body, ChangeVar(fake, v).Body), u, v, w);
            }
        }
        public static IEnumerable<Func3D> FakeExtendsU(IEnumerable<Func2D> function)
        {
            var u = Expression.Parameter(typeof(double), "u");
            var v = Expression.Parameter(typeof(double), "v");
            var w = Expression.Parameter(typeof(double), "w");
            Func1D fake = x => 1;
            foreach (var func1 in function)
            {
                yield return Expression.Lambda<Func<double, double, double, double>>(
                                                                                     Expression.Multiply(ChangeVar(func1, v, w).Body, ChangeVar(fake, u).Body), u, v, w);
            }
        }
    }
}