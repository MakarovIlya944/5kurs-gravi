using System.Collections;
using System.Collections.Generic;

namespace MKE.Interface
{
    public interface IDomain
    {
        
    }

    public enum Parameter
    {
        Lambda,
        Sigma,
        Gamma
    }
    public class FemDomain
    {
        public IEnumerable<IElement> Elements;
        public Dictionary<Parameter, double> Parameters;
    }
}