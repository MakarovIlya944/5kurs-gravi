using System;
using System.Collections.Generic;
using MKE.Geometry;
using MKE.Interface;
using MKE.Point;
using Newtonsoft.Json;

namespace MKE.Condition {
    public class BaseCondition
    {
        public Surface Surface { get; set; }

        public int XAxisIndex { get; set; }

        public int YAxisIndex { get; set; }

        public int ZAxisIndex { get; set; }
        public string Function { get; set; }
        [JsonIgnore] public List<IElement> Elements { get; set; } = new List<IElement>();
        public bool Check(Point3D point, GeometryParallelepiped geometry)
        {
            switch (Surface)
            {
                case Surface.Front:
                    return geometry.MapXAxisLines[XAxisIndex].Left <= point.X && point.X <= geometry.MapXAxisLines[XAxisIndex].Right &&
                           geometry.MapZAxisLines[ZAxisIndex].Left <= point.Z && point.Z <= geometry.MapZAxisLines[ZAxisIndex].Right &&
                           Math.Abs(geometry.MapYAxisLines[YAxisIndex].Left - point.Y) < 1e-14;
                case Surface.Back:
                    return geometry.MapXAxisLines[XAxisIndex].Left <= point.X && point.X <= geometry.MapXAxisLines[XAxisIndex].Right &&
                           geometry.MapZAxisLines[ZAxisIndex].Left <= point.Z && point.Z <= geometry.MapZAxisLines[ZAxisIndex].Right &&
                           Math.Abs(geometry.MapYAxisLines[YAxisIndex].Right - point.Y) < 1e-14;
                case Surface.Top:
                    return geometry.MapXAxisLines[XAxisIndex].Left <= point.X && point.X <= geometry.MapXAxisLines[XAxisIndex].Right &&
                           geometry.MapYAxisLines[YAxisIndex].Left <= point.Y && point.Y <= geometry.MapYAxisLines[YAxisIndex].Right &&
                           Math.Abs(geometry.MapZAxisLines[ZAxisIndex].Right - point.Z) < 1e-14;
                case Surface.Bottom:
                    return geometry.MapXAxisLines[XAxisIndex].Left <= point.X && point.X <= geometry.MapXAxisLines[XAxisIndex].Right &&
                           geometry.MapYAxisLines[YAxisIndex].Left <= point.Y && point.Y <= geometry.MapYAxisLines[YAxisIndex].Right &&
                           Math.Abs(geometry.MapZAxisLines[ZAxisIndex].Left - point.Z) < 1e-14;
                case Surface.Left:
                    return geometry.MapZAxisLines[ZAxisIndex].Left <= point.Z && point.Z <= geometry.MapZAxisLines[ZAxisIndex].Right &&
                           geometry.MapYAxisLines[YAxisIndex].Left <= point.Y && point.Y <= geometry.MapYAxisLines[YAxisIndex].Right &&
                           Math.Abs(geometry.MapXAxisLines[XAxisIndex].Left - point.X) < 1e-14;
                case Surface.Right:
                    return geometry.MapZAxisLines[ZAxisIndex].Left <= point.Z && point.Z <= geometry.MapZAxisLines[ZAxisIndex].Right &&
                           geometry.MapYAxisLines[YAxisIndex].Left <= point.Y && point.Y <= geometry.MapYAxisLines[YAxisIndex].Right &&
                           Math.Abs(geometry.MapXAxisLines[XAxisIndex].Right - point.X) < 1e-14;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }
}