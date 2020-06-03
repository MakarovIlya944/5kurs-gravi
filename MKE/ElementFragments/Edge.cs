using System;

namespace MKE.ElementFragments
{
    public class Edge:IEquatable<Edge>
    {
        public Vertex A { get; set; }
        public Vertex B { get; set; }
        public int MinOrder { get; set; }
        public int[] Indexes => new[] { A.Index, B.Index };
        public Edge(int indexLeft, int indexRight, int minOrder)
        {
            MinOrder = minOrder;
            A = new Vertex(indexLeft,0);
                B = new Vertex(indexRight,0);
        }
        public override string ToString()
        {
            return $"A:{A.Index}\t B:{B.Index}";
        }

        public bool Equals(Edge other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Equals(A, other.A) && Equals(B, other.B);
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((Edge) obj);
        }

        public override int GetHashCode()
        {
            return HashCode.Combine(A.GetHashCode(), B.GetHashCode());
        }
    }
}