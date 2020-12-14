using System;
using System.Collections.Generic;
using System.Linq;
using MKE.Interface;

namespace MKE.Point {
    public class ComparerPoint : IEqualityComparer<IPoint>
    {
        public bool Equals(IPoint x, IPoint y)
        {
            if (ReferenceEquals(x, y)) return true;
            if (x == null || y == null) return false;
            if (x.Dimension != y.Dimension) return false;
            var yCoords = y.Coords;
            var xCoords = x.Coords;
            for (int i = 0; i < x.Dimension; i++)
            {
                if (Math.Abs(xCoords[i] - yCoords[i]) > 1e-14)
                {
                    return false;
                }
            }

            return true;
        }

        public int GetHashCode(IPoint obj)
        {
            return obj.Coords.Aggregate(0, HashCode.Combine);
        }
    }
}