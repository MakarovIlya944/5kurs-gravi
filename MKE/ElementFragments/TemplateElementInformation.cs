using System.Collections.Generic;
using System.Drawing;
using System.Threading;
using MKE.Interface;

namespace MKE.ElementFragments
{
    public class TemplateElementInformation
    {
        public Dictionary<int, Vertex> Vertices { get; set; }

        public Dictionary<int, Edge> Edges { get; set; }

        public Dictionary<int, ISurface> Surfaces { get; set; }

        public int[] InnerIndexes { get; set; }

        public int Size => Vertices.Count + Edges.Count + Surfaces.Count + InnerIndexes.Length;

        public TemplateElementInformation(Dictionary<int, Vertex> vertices, Dictionary<int, Edge> edges, Dictionary<int, ISurface> surfaces, int[] innerIndexes)
        {
            Vertices = vertices;
            Edges = edges;
            Surfaces = surfaces;
            InnerIndexes = innerIndexes;
        }
    }
}