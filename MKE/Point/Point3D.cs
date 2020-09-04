using System;
using System.Collections.Generic;
using MKE.Interface;

namespace MKE.Point
{
    public class Point3D : IPoint
    {
        public double X { get; set; }
        public double Y { get; set; }
        public double Z { get; set; }
        public int Dimension => 3;
        public List<double> GetCoords => new List<double>(3) { X, Y, Z };

        public Point3D()
        {
        }

        public Point3D(List<double> x)
        {
            X = x[0];
            Y = x[1];
            Z = x[2];
        }

        public Point3D(Point3D x)
        {
            X = x.X;
            Y = x.Y;
            Z = x.Z;
        }

        public Point3D(double x, double y, double z)
        {
            X = x;
            Y = y;
            Z = z;
        }

        public IPoint Add(IPoint x)
        {
            if (!(x is Point3D x1)) throw new ArgumentException();
            X += x1.X;
            Y += x1.Y;
            Z += x1.Z;
            return this;
        }

        public IPoint Subtract(IPoint x)
        {
            if (!(x is Point3D x1)) throw new ArgumentException();
            X -= x1.X;
            Y -= x1.Y;
            Z -= x1.Z;
            return this;
        }

        public IPoint Normalize()
        {
            return Divide(Math.Abs(X) + Math.Abs(Y) + Math.Abs(Z));
        }

        public double Multiply(IPoint x)
        {
            if (!(x is Point3D x1)) throw new ArgumentException();
            return X * x1.X + Y * x1.Y + Z * x1.Z;
        }

        public IPoint Multiply(double x)
        {
            X *= x;
            Y *= x;
            Z *= x;
            return this;
        }

        public IPoint Divide(double x)
        {
            X /= x;
            Y /= x;
            Z /= x;
            return this;
        }

        public static Point3D operator +(Point3D x, Point3D y)
        {
            return (Point3D)new Point3D(x).Add(y);
        }

        public static double operator *(Point3D x, Point3D y)
        {
            return new Point3D(x).Multiply(y);
        }

        public static Point3D operator /(Point3D x, double y)
        {
            return (Point3D)new Point3D(x).Divide(y);
        }

        public static Point3D operator -(Point3D x, Point3D y)
        {
            return (Point3D)new Point3D(x).Subtract(y);
        }

        public static Point3D operator *(Point3D x, double y)
        {
            return (Point3D)new Point3D(x).Multiply(y);
        }
    }
}