using System;
using System.Collections.Generic;
using MKE.ElementFragments;
using MKE.Interface;
using MKE.Point;

namespace MKE.Domain
{
    public class BaseDomain
    {
        public int DomainIndex { get; set; }
        public int XAxisIndex { get; set; }
        public int YAxisIndex { get; set; }
        public int ZAxisIndex { get; set; }
        public List<IElement> Elements { get; set; } = new List<IElement>();
    }

    public class HierarchicalDomain3D<T> : BaseDomain
        where T : IElement
    {
        public int Order { get; set; }
        public GeometryParallelepiped GeometryParallelepiped { get; set; }
        public Func<double,double,double,double> RightFunction { get; set; }
    }
}