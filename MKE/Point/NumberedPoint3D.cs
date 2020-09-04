using System.Collections.Generic;
using MKE.Interface;

namespace MKE.Point {
    public class NumberedPoint3D : Point3D, IPointNumbered
    {
        public int Number { get; set; }

        public NumberedPoint3D(int number) : base()
        {
            Number = number;
        }

        public NumberedPoint3D(int number, List<double> x) : base(x)
        {
            Number = number;
        }

        public NumberedPoint3D(int number, Point3D x) : base(x)
        {
            Number = number;
        }

        public NumberedPoint3D(int number, double x, double y, double z) : base(x, y, z)
        {
            Number = number;
        }
    }
}