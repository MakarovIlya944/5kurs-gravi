using System;
using System.Collections.Generic;
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
        public List<LayerConfig> layers;
        public float lr;
    }

    public class LayerConfig
    {
        public int w;
    }
}
