using System;
using System.Collections.Generic;
using MKE.Interface;

namespace MKE.Point
{
    public class Point2D : IPoint
    {
        public double X { get; set; }
        public double Y { get; set; }
        public int Dimension => 2;
        public List<double> Coords => new List<double>(Dimension) {X, Y};

        public Point2D()
        {
        }

        public Point2D(List<double> x)
        {
            X = x[0];
            Y = x[1];
        }

        public Point2D(Point2D x)
        {
            X = x.X;
            Y = x.Y;
        }

        public Point2D(double x, double y)
        {
            X = x;
            Y = y;
        }

        public IPoint Add(IPoint x)
        {
            if (!(x is Point2D x1)) throw new ArgumentException();
            X += x1.X;
            Y += x1.Y;
            return this;
        }

        public IPoint Subtract(IPoint x)
        {
            if (!(x is Point2D x1)) throw new ArgumentException();
            X -= x1.X;
            Y -= x1.Y;
            return this;
        }

        public IPoint Normalize()
        {
            return Divide(Math.Abs(X) + Math.Abs(Y));
        }

        public double Multiply(IPoint x)
        {
            if (!(x is Point2D x1)) throw new ArgumentException();
            return X * x1.X + Y * x1.Y;
        }

        public IPoint Multiply(double x)
        {
            X *= x;
            Y *= x;
            return this;
        }

        public IPoint Divide(double x)
        {
            X /= x;
            Y /= x;
            return this;
        }

        public static Point2D operator +(Point2D x, Point2D y)
        {
            return (Point2D) new Point2D(x).Add(y);
        }

        public static double operator *(Point2D x, Point2D y)
        {
            return new Point2D(x).Multiply(y);
        }

        public static Point2D operator /(Point2D x, double y)
        {
            return (Point2D) new Point2D(x).Divide(y);
        }

        public static Point2D operator -(Point2D x, Point2D y)
        {
            return (Point2D) new Point2D(x).Subtract(y);
        }

        public static Point2D operator *(Point2D x, double y)
        {
            return (Point2D) new Point2D(x).Multiply(y);
        }
    }
}