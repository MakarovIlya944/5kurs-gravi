using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace Mnist
{
    //{'iters': 1000,
    //'layers':[
    //    {'w': len_i},
    //    {'w': int ((len_i+len_o)/2)},
    //    {'w': len_o},
    //  ],
    //'lr':0.01}
    
    public class ModelConfig
    {
        public int iters;
        public int batch;
        public List<LayerConfig> layers;
        public float lr;
        public List<int> ToList()
        {
            return layers.Select(x => x.w).ToList();
        }
    }

    public class LayerConfig
    {
        public int w;
        public override string ToString()
        {
            return w.ToString();
        }
    }
}
