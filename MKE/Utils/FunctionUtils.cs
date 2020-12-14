﻿namespace MKE.Utils
{
    public static class FunctionUtils
    {
        public static int FastPower(int x, int pow)
        {
            switch (pow)
            {
                case 0: return 1;
                case 1: return x;
                case 2: return x * x;
                case 3: return x * x * x;
                case 4: return x * x * x * x;
                case 5: return x * x * x * x * x;
                case 6: return x * x * x * x * x * x;
                case 7: return x * x * x * x * x * x * x;
                case 8: return x * x * x * x * x * x * x * x;
                case 9: return x * x * x * x * x * x * x * x * x;
                case 10: return x * x * x * x * x * x * x * x * x * x;
                case 11: return x * x * x * x * x * x * x * x * x * x * x;
                // up to 32 can be added 
                default: // Vilx's solution is used for default
                    int ret = 1;
                    while (pow != 0)
                    {
                        if ((pow & 1) == 1)
                            ret *= x;
                        x *= x;
                        pow >>= 1;
                    }
                    return ret;
            }
        }
    }
}