using System.Collections;
using System.Collections.Generic;

namespace MKE.Interface
{
    public interface IDomain
    {
        int DomainIndex { get; set; }
        int XAxisIndex { get; set; }
        int YAxisIndex { get; set; }
        int ZAxisIndex { get; set; }
        List<IElement> Elements { get; set; }
        string Function { get; set; }
        string LambdaFunction { get; set; }
        string GammaFunction { get; set; }
    }
}