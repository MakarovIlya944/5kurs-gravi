using System;

namespace MKE.ElementFragments
{
    public class Vertex:IEquatable<Vertex>
    {
        public int Index { get; set; }
        public int MinOrder { get; set; }
        public Vertex(int index, int minOrder)
        {
            Index = index;
            MinOrder = minOrder;
        }

        public bool Equals(Vertex other)
        {
            if (ReferenceEquals(null, other)) return false;
            if (ReferenceEquals(this, other)) return true;
            return Index == other.Index;
        }

        public override bool Equals(object obj)
        {
            if (ReferenceEquals(null, obj)) return false;
            if (ReferenceEquals(this, obj)) return true;
            if (obj.GetType() != this.GetType()) return false;
            return Equals((Vertex) obj);
        }

        public override int GetHashCode()
        {
            return Index.GetHashCode();
        }
    }
}