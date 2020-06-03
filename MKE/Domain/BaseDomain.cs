using System;
using System.Collections.Generic;
using System.Linq.Expressions;
using MKE.ElementFragments;
using MKE.Interface;
using MKE.Point;
using Newtonsoft.Json;

namespace MKE.Domain
{
    public class BaseDomain
    {
        public int DomainIndex { get; set; }
        public int XAxisIndex { get; set; }
        public int YAxisIndex { get; set; }
        public int ZAxisIndex { get; set; }
        [JsonIgnore] public List<IElement> Elements { get; set; } = new List<IElement>();
        public string Function { get; set; } = "(x,y,z)=>0";
        public string Lambda { get; set; } = "(x,y,z)=>0";

        public string Gamma { get; set; } = "(x,y,z)=>0";
    }

    public class HierarchicalDomain3D<T> : BaseDomain
        where T : IElement
    {
        public int Order { get; set; }
        [JsonIgnore] public GeometryParallelepiped GeometryParallelepiped { get; set; }
        [JsonIgnore] public Func<double, double, double, double> RightFunction { get; set; }
        [JsonIgnore] public Func<double, double, double, double> Lambda { get; set; }
        [JsonIgnore] public Func<double, double, double, double> Gamma { get; set; }

    }
}