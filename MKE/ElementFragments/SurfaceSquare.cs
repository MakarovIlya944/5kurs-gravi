using System;
using System.Collections.Generic;
using System.Linq;
using MKE.Interface;

namespace MKE.ElementFragments
{
    public class SurfaceSquare : ISurface,IEquatable<SurfaceSquare>
    {
        public Vertex A { get; set; }
        public Vertex B { get; set; }
        public Vertex C { get; set; }
        public Vertex D { get; set; }
        public int MinOrder { get; set; }
        public int[] Indexes => new[] { A.Index, B.Index, C.Index, D.Index };

        public SurfaceSquare(int indexLeft, int indexRight, int indexBottom, int indexTop, int minOrder)
        {
            MinOrder = minOrder;
            A = new Vertex(indexLeft,0);
            B = new Vertex(indexRight,0);
            C = new Vertex(indexBottom,0);
            D = new Vertex(indexTop,0);
        }

        public override string ToString()
        {
            return $"A:{A.Index}\t B:{B.Index} \t C:{C.Index}\t D:{D.Index}";
        }

        public bool Equals(SurfaceSquare other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Equals(A, other.A) && Equals(B, other.B) && Equals(C, other.C) && Equals(D, other.D);
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((SurfaceSquare) obj);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(A.GetHashCode(), B.GetHashCode(), C.GetHashCode(), D.GetHashCode());
        }
    }
}