using System.Collections.Generic;
using MKE.Interface;
using Newtonsoft.Json;

namespace MKE.Domain
{
    public class BaseDomain:IDomain
    {
        public int DomainIndex { get; set; }
        public int XAxisIndex { get; set; }
        public int YAxisIndex { get; set; }
        public int ZAxisIndex { get; set; }
        [JsonIgnore] public List<IElement> Elements { get; set; } = new List<IElement>();
        public string Function { get; set; }
        public string LambdaFunction { get; set; }

        public string GammaFunction { get; set; }
    }
}